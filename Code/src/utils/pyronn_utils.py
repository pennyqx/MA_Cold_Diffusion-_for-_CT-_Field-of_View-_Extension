"""
PyRoNN Utilities for CT Projection and Backprojection

Provides a clean wrapper around PyRoNN with correct dimension handling.

Key convention:
- User inputs: detector_shape = (width, height)
- Output sinogram: (n_projections, width, height)
- This matches intuitive understanding where width is horizontal
"""

import numpy as np
import torch
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class CTGeometry:
    """CT geometry parameters"""
    # Volume
    volume_shape: Tuple[int, int, int] = (128, 512, 512)  # (Z, Y, X)
    volume_spacing: Tuple[float, float, float] = (1.5, 1.0, 1.0)  # mm

    # Detector
    detector_width: int = 640
    detector_height: int = 560
    detector_spacing_x: float = 0.5  # mm
    detector_spacing_y: float = 0.5  # mm

    # Projection
    n_projections: int = 360
    angular_range: float = 2 * np.pi

    # Source-detector
    source_detector_distance: float = 1200.0  # mm
    source_isocenter_distance: float = 750.0  # mm


def create_geometry(config: CTGeometry = None):
    """
    Create PyRoNN geometry object.

    Args:
        config: CT geometry configuration

    Returns:
        PyRoNN Geometry object
    """
    if config is None:
        config = CTGeometry()

    try:
        from pyronn.ct_reconstruction.geometry.geometry import Geometry
        from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import circular_trajectory_3d
    except ImportError:
        raise ImportError("PyRoNN is required. Install from: https://github.com/csyben/PYRO-NN")

    # Convert to PyRoNN convention (height, width)
    detector_shape_pyronn = (config.detector_height, config.detector_width)
    detector_spacing_pyronn = (config.detector_spacing_y, config.detector_spacing_x)

    geometry = Geometry()
    geometry.init_from_parameters(
        volume_shape=config.volume_shape,
        volume_spacing=config.volume_spacing,
        detector_shape=detector_shape_pyronn,
        detector_spacing=detector_spacing_pyronn,
        number_of_projections=config.n_projections,
        angular_range=config.angular_range,
        source_detector_distance=config.source_detector_distance,
        source_isocenter_distance=config.source_isocenter_distance,
        trajectory=circular_trajectory_3d,
        swap_detector_axis=True  # Critical for correct dimensions!
    )

    return geometry


class PyRoNNProjector:
    """
    PyRoNN projector wrapper with automatic dimension handling.

    Ensures consistent dimension convention:
    - Input volume: (Z, Y, X)
    - Output sinogram: (n_proj, width, height)
    - detector_shape specified as (width, height)
    """

    def __init__(self, config: Optional[CTGeometry] = None):
        """
        Args:
            config: CT geometry configuration
        """
        self.config = config or CTGeometry()
        self._geometry = None
        self._projector = None
        self._backprojector = None
        self._filter = None

    def _ensure_initialized(self):
        """Lazy initialization of PyRoNN components"""
        if self._geometry is not None:
            return

        try:
            from pyronn.ct_reconstruction.layers.projection_3d import ConeProjection3D
            from pyronn.ct_reconstruction.layers.backprojection_3d import ConeBackProjection3D
            from pyronn.ct_reconstruction.helpers.filters.filters import shepp_logan_3D as _fbp_filter
        except ImportError:
            raise ImportError("PyRoNN is required")

        self._geometry = create_geometry(self.config)
        self._projector = ConeProjection3D()
        self._backprojector = ConeBackProjection3D()

        self._filter = _fbp_filter(
            detector_shape=self._geometry.detector_shape,
            detector_spacing=self._geometry.detector_spacing,
            number_of_projections=self._geometry.number_of_projections
        )

    def forward(self, volume: np.ndarray) -> np.ndarray:
        """
        Forward projection: volume -> sinogram

        Args:
            volume: (Z, Y, X) volume array

        Returns:
            (n_proj, width, height) sinogram
        """
        self._ensure_initialized()

        assert volume.ndim == 3, f"Volume must be 3D, got shape {volume.shape}"
        assert volume.shape == self.config.volume_shape, \
            f"Volume shape {volume.shape} != config {self.config.volume_shape}"

        # Convert to tensor
        volume_tensor = torch.from_numpy(volume[np.newaxis, ...]).float()
        if torch.cuda.is_available():
            volume_tensor = volume_tensor.cuda()

        # Project
        sinogram = self._projector.forward(volume_tensor, **self._geometry)

        if torch.cuda.is_available():
            sinogram = sinogram.cpu()
        sinogram = sinogram.numpy()[0]  # (n_proj, height, width)

        # Transpose to user convention
        sinogram = np.transpose(sinogram, (0, 2, 1))  # -> (n_proj, width, height)

        return sinogram.astype(np.float32)

    def backward(self, sinogram: np.ndarray, apply_filter: bool = True) -> np.ndarray:
        """
        Backward projection: sinogram -> volume

        Args:
            sinogram: (n_proj, width, height) sinogram
            apply_filter: Whether to apply ramp filter (for FBP)

        Returns:
            (Z, Y, X) reconstructed volume
        """
        self._ensure_initialized()

        expected_shape = (self.config.n_projections,
                         self.config.detector_width,
                         self.config.detector_height)
        assert sinogram.shape == expected_shape, \
            f"Sinogram shape {sinogram.shape} != expected {expected_shape}"

        # Convert to PyRoNN convention
        sino_pyronn = np.transpose(sinogram, (0, 2, 1))  # -> (n_proj, height, width)

        # Convert to tensor
        sino_tensor = torch.from_numpy(sino_pyronn[np.newaxis, ...]).float()
        if torch.cuda.is_available():
            sino_tensor = sino_tensor.cuda()
            filter_tensor = torch.from_numpy(self._filter).float().cuda()
        else:
            filter_tensor = torch.from_numpy(self._filter).float()

        # Apply filter
        if apply_filter:
            sino_fft = torch.fft.fft(sino_tensor, dim=-1, norm="ortho")
            filtered = sino_fft * filter_tensor
            sino_tensor = torch.fft.ifft(filtered, dim=-1, norm="ortho").real

        # Backproject
        volume = self._backprojector.forward(sino_tensor.contiguous(), **self._geometry)

        if torch.cuda.is_available():
            volume = volume.cpu()

        return volume.numpy()[0].astype(np.float32)

    def fbp_reconstruct(self, sinogram: np.ndarray) -> np.ndarray:
        """
        Full FBP reconstruction (convenience method).

        Args:
            sinogram: (n_proj, width, height) sinogram

        Returns:
            (Z, Y, X) reconstructed volume
        """
        return self.backward(sinogram, apply_filter=True)


def test_dimension_consistency():
    """
    Test that dimensions are handled correctly.

    Expected:
    - detector_shape = (width=640, height=560)
    - sinogram.shape = (n_proj=360, width=640, height=560)
    """
    config = CTGeometry(
        volume_shape=(32, 128, 128),  # Small for testing
        detector_width=256,
        detector_height=128,
        n_projections=180
    )

    projector = PyRoNNProjector(config)

    # Create test volume
    volume = np.zeros(config.volume_shape, dtype=np.float32)
    z, y, x = config.volume_shape
    volume[z//4:3*z//4, y//4:3*y//4, x//4:3*x//4] = 1.0

    # Forward projection
    sinogram = projector.forward(volume)

    expected_shape = (config.n_projections, config.detector_width, config.detector_height)
    assert sinogram.shape == expected_shape, \
        f"Sinogram shape {sinogram.shape} != expected {expected_shape}"

    print(f"Volume shape: {volume.shape}")
    print(f"Sinogram shape: {sinogram.shape}")
    print(f"Expected: (n_proj={config.n_projections}, "
          f"width={config.detector_width}, height={config.detector_height})")

    # Backward projection
    recon = projector.backward(sinogram)
    assert recon.shape == config.volume_shape, \
        f"Reconstructed shape {recon.shape} != volume {config.volume_shape}"

    print(f"Reconstruction shape: {recon.shape}")
    print("Dimension consistency test passed!")

    return True


if __name__ == "__main__":
    print("Testing PyRoNN utilities...")

    # Check if PyRoNN is available
    try:
        from pyronn.ct_reconstruction.geometry.geometry import Geometry
        print("PyRoNN is available, running dimension test...")
        test_dimension_consistency()
    except ImportError:
        print("PyRoNN not available, skipping projection tests")
        print("Install PyRoNN for full functionality")

    # Test config creation
    config = CTGeometry()
    print(f"\nDefault geometry config:")
    print(f"  Volume: {config.volume_shape}")
    print(f"  Detector: {config.detector_width} x {config.detector_height}")
    print(f"  Projections: {config.n_projections}")
