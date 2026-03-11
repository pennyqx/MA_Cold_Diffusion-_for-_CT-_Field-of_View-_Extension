"""
Data Preprocessing Pipeline for CT Sinogram Generation

Converts DICOM/NIfTI CT volumes to sinograms using PyRoNN cone-beam projection.

Pipeline:
1. Load DICOM series or NIfTI volume
2. Resample to target spacing (default: 1.0 x 1.0 x 1.5 mm)
3. Crop/pad to target size (default: 512 x 512 x 128)
4. Clip HU values to [-1000, 1000]
5. Forward projection using PyRoNN
6. Save sinogram as .npy

Output sinogram shape: (n_projections, detector_width, detector_height)
Default: (360, 640, 128)
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import warnings


@dataclass
class GeometryConfig:
    """CT geometry configuration for PyRoNN projection"""
    # Volume parameters
    volume_shape: Tuple[int, int, int] = (128, 512, 512)  # (Z, Y, X)
    volume_spacing: Tuple[float, float, float] = (1.5, 1.0, 1.0)  # mm

    # Detector parameters
    detector_width: int = 640
    detector_height: int = 560
    detector_spacing_x: float = 0.5  # mm
    detector_spacing_y: float = 0.5  # mm

    # Projection parameters
    n_projections: int = 360
    angular_range: float = 2 * np.pi  # Full rotation

    # Source-detector geometry
    source_detector_distance: float = 1200.0  # mm
    source_isocenter_distance: float = 750.0  # mm


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration"""
    # Target spacing (mm)
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.5)  # (X, Y, Z)

    # Target XY size
    target_xy_size: int = 512

    # Slice thickness constraints
    min_slice_thickness: float = 0.8  # mm
    max_slice_thickness: float = 3.0  # mm

    # Minimum slices after preprocessing
    min_slices: int = 80

    # HU clipping range
    hu_min: float = -1000.0
    hu_max: float = 1000.0


def load_dicom_series(dicom_dir: str) -> Tuple[np.ndarray, Dict]:
    """
    Load DICOM series and convert to numpy volume.

    Args:
        dicom_dir: Directory containing DICOM files

    Returns:
        Tuple of (volume, metadata)
        - volume: (Z, Y, X) numpy array in HU
        - metadata: Dictionary with spacing, orientation, etc.
    """
    try:
        import pydicom
        from pydicom.filereader import dcmread
    except ImportError:
        raise ImportError("pydicom is required for DICOM loading. Install with: pip install pydicom")

    def _is_archive(path_str: str) -> bool:
        lower = path_str.lower()
        return lower.endswith('.zip') or lower.endswith('.tar') or lower.endswith('.tar.gz') or lower.endswith('.tgz')

    def _split_archive(path_str: str) -> Tuple[str, Optional[str]]:
        if '::' in path_str:
            archive_path, inner_prefix = path_str.split('::', 1)
            inner_prefix = inner_prefix.strip('/')
            return archive_path, inner_prefix or None
        return path_str, None

    def _is_dicom_name(name: str) -> bool:
        base = os.path.basename(name)
        return base.lower().endswith('.dcm') or '.' not in base

    def _list_archive_members(archive_path: str, inner_prefix: Optional[str]) -> List[str]:
        members: List[str] = []
        if archive_path.lower().endswith('.zip'):
            import zipfile
            with zipfile.ZipFile(archive_path, 'r') as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    name = info.filename
                    if inner_prefix and not name.startswith(inner_prefix + '/'):
                        continue
                    if _is_dicom_name(name):
                        members.append(name)
        else:
            import tarfile
            with tarfile.open(archive_path, 'r:*') as tf:
                for info in tf.getmembers():
                    if not info.isfile():
                        continue
                    name = info.name
                    if inner_prefix and not name.startswith(inner_prefix + '/'):
                        continue
                    if _is_dicom_name(name):
                        members.append(name)
        members.sort()
        return members

    def _iter_archive_files(archive_path: str, members: List[str]):
        import io
        if archive_path.lower().endswith('.zip'):
            import zipfile
            with zipfile.ZipFile(archive_path, 'r') as zf:
                for name in members:
                    with zf.open(name, 'r') as f:
                        data = f.read()
                    yield name, io.BytesIO(data)
        else:
            import tarfile
            with tarfile.open(archive_path, 'r:*') as tf:
                for name in members:
                    f = tf.extractfile(name)
                    if f is None:
                        continue
                    data = f.read()
                    f.close()
                    yield name, io.BytesIO(data)

    archive_path, inner_prefix = None, None
    if _is_archive(dicom_dir) or '::' in dicom_dir:
        archive_path, inner_prefix = _split_archive(dicom_dir)

    if archive_path and _is_archive(archive_path):
        dicom_files = _list_archive_members(archive_path, inner_prefix)
    else:
        # Collect all DICOM files
        dicom_files = []
        for root, _, files in os.walk(dicom_dir):
            for f in files:
                if f.endswith('.dcm') or '.' not in f:
                    dicom_files.append(os.path.join(root, f))

    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_dir}")

    # Load and sort by instance number
    slices = []
    if archive_path and _is_archive(archive_path):
        for name, bio in _iter_archive_files(archive_path, dicom_files):
            try:
                ds = dcmread(bio)
                if hasattr(ds, 'pixel_array'):
                    slices.append(ds)
            except Exception:
                continue
    else:
        for f in dicom_files:
            try:
                ds = dcmread(f)
                if hasattr(ds, 'pixel_array'):
                    slices.append(ds)
            except Exception:
                continue

    if not slices:
        raise ValueError("No valid DICOM slices found")

    # Sort by ImagePositionPatient or InstanceNumber
    slices.sort(key=lambda x: (
        float(x.ImagePositionPatient[2]) if hasattr(x, 'ImagePositionPatient')
        else float(x.InstanceNumber) if hasattr(x, 'InstanceNumber')
        else 0
    ))

    # Get metadata from first slice
    first = slices[0]
    pixel_spacing = list(map(float, first.PixelSpacing)) if hasattr(first, 'PixelSpacing') else [1.0, 1.0]
    slice_thickness = float(first.SliceThickness) if hasattr(first, 'SliceThickness') else 1.0

    # Stack slices into volume
    volume = np.stack([s.pixel_array for s in slices], axis=0)

    # Convert to HU
    slope = float(first.RescaleSlope) if hasattr(first, 'RescaleSlope') else 1.0
    intercept = float(first.RescaleIntercept) if hasattr(first, 'RescaleIntercept') else 0.0
    volume = volume.astype(np.float32) * slope + intercept

    metadata = {
        'spacing': (pixel_spacing[0], pixel_spacing[1], slice_thickness),
        'shape': volume.shape,
        'patient_id': str(first.PatientID) if hasattr(first, 'PatientID') else 'unknown'
    }

    return volume, metadata


def load_nifti_volume(nifti_path: str) -> Tuple[np.ndarray, Dict]:
    """
    Load NIfTI volume.

    Args:
        nifti_path: Path to .nii or .nii.gz file

    Returns:
        Tuple of (volume, metadata)
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel is required for NIfTI loading. Install with: pip install nibabel")

    img = nib.load(nifti_path)
    volume = np.asarray(img.dataobj, dtype=np.float32)

    # Get spacing from affine
    affine = img.affine
    spacing = tuple(np.abs(np.diag(affine)[:3]))

    # Ensure (Z, Y, X) orientation
    if volume.ndim == 3:
        # Assume input is (X, Y, Z), transpose to (Z, Y, X)
        volume = np.transpose(volume, (2, 1, 0))
        spacing = (spacing[2], spacing[1], spacing[0])

    metadata = {
        'spacing': spacing,
        'shape': volume.shape,
        'affine': affine
    }

    return volume, metadata


def resample_volume(
    volume: np.ndarray,
    original_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
    order: int = 1
) -> np.ndarray:
    """
    Resample volume to target spacing.

    Args:
        volume: (Z, Y, X) input volume
        original_spacing: (sz, sy, sx) original spacing in mm
        target_spacing: (sz, sy, sx) target spacing in mm
        order: Interpolation order (0=nearest, 1=linear, 3=cubic)

    Returns:
        Resampled volume
    """
    from scipy.ndimage import zoom

    # Calculate zoom factors
    zoom_factors = tuple(o / t for o, t in zip(original_spacing, target_spacing))

    # Resample
    resampled = zoom(volume, zoom_factors, order=order)

    return resampled.astype(np.float32)


def crop_or_pad_volume(
    volume: np.ndarray,
    target_shape: Tuple[int, int, int]
) -> np.ndarray:
    """
    Crop or pad volume to target shape.
    Centers the volume and pads with -1000 HU (air).

    Args:
        volume: Input volume
        target_shape: Target (Z, Y, X) shape

    Returns:
        Cropped/padded volume
    """
    current = volume.shape
    target = target_shape

    # Initialize output with air value
    output = np.full(target, -1000.0, dtype=np.float32)

    # Calculate crop/pad for each dimension
    slices_src = []
    slices_dst = []

    for c, t in zip(current, target):
        if c >= t:
            # Need to crop
            start = (c - t) // 2
            slices_src.append(slice(start, start + t))
            slices_dst.append(slice(None))
        else:
            # Need to pad
            start = (t - c) // 2
            slices_src.append(slice(None))
            slices_dst.append(slice(start, start + c))

    output[tuple(slices_dst)] = volume[tuple(slices_src)]

    return output


def clip_hu_values(volume: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    """Clip HU values to specified range"""
    return np.clip(volume, hu_min, hu_max)


def hu_to_mu(volume: np.ndarray, hu_air: float = -1000.0, hu_water: float = 0.0) -> np.ndarray:
    """
    Convert HU values to linear attenuation coefficients (mu).
    This makes the projection input non-negative for physical sinograms.
    """
    mu = (volume - hu_air) / (hu_water - hu_air)
    return np.clip(mu, 0.0, None).astype(np.float32)


def forward_projection_pyronn(
    volume: np.ndarray,
    geometry: GeometryConfig
) -> np.ndarray:
    """
    Perform cone-beam forward projection using PyRoNN.

    Args:
        volume: (Z, Y, X) CT volume
        geometry: Geometry configuration

    Returns:
        Sinogram of shape (n_projections, detector_width, detector_height)
    """
    try:
        import torch
        from pyronn.ct_reconstruction.geometry.geometry import Geometry
        from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import circular_trajectory_3d
        from pyronn.ct_reconstruction.layers.projection_3d import ConeProjection3D
    except ImportError:
        raise ImportError("PyRoNN is required for projection. Please install PyRoNN.")

    # Create geometry
    # Note: PyRoNN expects detector_shape as (height, width)
    detector_shape_pyronn = (geometry.detector_height, geometry.detector_width)
    detector_spacing_pyronn = (geometry.detector_spacing_y, geometry.detector_spacing_x)

    geo = Geometry()
    geo.init_from_parameters(
        volume_shape=geometry.volume_shape,
        volume_spacing=geometry.volume_spacing,
        detector_shape=detector_shape_pyronn,
        detector_spacing=detector_spacing_pyronn,
        number_of_projections=geometry.n_projections,
        angular_range=geometry.angular_range,
        source_detector_distance=geometry.source_detector_distance,
        source_isocenter_distance=geometry.source_isocenter_distance,
        trajectory=circular_trajectory_3d,
        swap_detector_axis=True  # Critical for correct dimension handling!
    )

    # Create projector
    projector = ConeProjection3D()

    # Convert to tensor
    volume_tensor = torch.from_numpy(volume[np.newaxis, ...]).float()
    if torch.cuda.is_available():
        volume_tensor = volume_tensor.cuda()

    # Project
    sinogram = projector.forward(volume_tensor, **geo)

    # Convert back to numpy
    if torch.cuda.is_available():
        sinogram = sinogram.cpu()
    sinogram = sinogram.numpy()[0]  # (n_proj, height, width)

    # Transpose to user convention: (n_proj, width, height)
    sinogram = np.transpose(sinogram, (0, 2, 1))

    return sinogram.astype(np.float32)


def preprocess_dicom_to_sinogram(
    input_path: str,
    output_dir: str,
    preprocess_config: Optional[PreprocessingConfig] = None,
    geometry_config: Optional[GeometryConfig] = None,
    save_volume: bool = False
) -> Optional[str]:
    """
    Full preprocessing pipeline: DICOM/NIfTI -> Sinogram.

    Args:
        input_path: Path to DICOM directory or NIfTI file
        output_dir: Directory to save output sinogram
        preprocess_config: Preprocessing configuration
        geometry_config: CT geometry configuration
        save_volume: Whether to also save the preprocessed volume

    Returns:
        Path to saved sinogram, or None if preprocessing failed
    """
    if preprocess_config is None:
        preprocess_config = PreprocessingConfig()
    if geometry_config is None:
        geometry_config = GeometryConfig()

    os.makedirs(output_dir, exist_ok=True)

    # Determine input type and load
    input_path = Path(input_path)
    if input_path.is_dir():
        # DICOM series
        volume, metadata = load_dicom_series(str(input_path))
        base_name = input_path.name
    elif input_path.suffix in ['.nii', '.gz']:
        # NIfTI file
        volume, metadata = load_nifti_volume(str(input_path))
        base_name = input_path.stem.replace('.nii', '')
    else:
        raise ValueError(f"Unsupported input format: {input_path}")

    original_spacing = metadata['spacing']

    # Check slice thickness constraint
    slice_thickness = original_spacing[2] if len(original_spacing) > 2 else original_spacing[0]
    if not (preprocess_config.min_slice_thickness <= slice_thickness <= preprocess_config.max_slice_thickness):
        warnings.warn(f"Slice thickness {slice_thickness:.2f}mm outside range, skipping")
        return None

    # Resample to target spacing
    # Note: volume is (Z, Y, X), spacing is (sz, sy, sx)
    target_spacing_zyx = (
        preprocess_config.target_spacing[2],  # Z
        preprocess_config.target_spacing[1],  # Y
        preprocess_config.target_spacing[0]   # X
    )
    volume = resample_volume(volume, original_spacing, target_spacing_zyx)

    # Crop/pad to target shape
    target_shape = (
        geometry_config.volume_shape[0],  # Z
        preprocess_config.target_xy_size,  # Y
        preprocess_config.target_xy_size   # X
    )
    volume = crop_or_pad_volume(volume, target_shape)

    # Check minimum slices
    if volume.shape[0] < preprocess_config.min_slices:
        warnings.warn(f"Volume has only {volume.shape[0]} slices, less than minimum {preprocess_config.min_slices}")

    # Clip HU values
    volume = clip_hu_values(volume, preprocess_config.hu_min, preprocess_config.hu_max)

    # Convert HU to mu (non-negative) before projection
    volume = hu_to_mu(volume)

    # Forward projection
    sinogram = forward_projection_pyronn(volume, geometry_config)

    # Save sinogram
    sino_path = os.path.join(
        output_dir,
        f"{base_name}_sinogram_{geometry_config.detector_width}x{geometry_config.detector_height}.npy"
    )
    np.save(sino_path, sinogram)

    # Optionally save volume
    if save_volume:
        vol_path = os.path.join(output_dir, f"{base_name}_volume.npy")
        np.save(vol_path, volume)

    return sino_path


def compute_global_stats(file_list: List[str]) -> Tuple[float, float]:
    """
    Compute global mean and std from a list of sinogram files.

    Args:
        file_list: List of .npy sinogram file paths

    Returns:
        Tuple of (mean, std)
    """
    import math

    total_sum = 0.0
    total_sq = 0.0
    total_cnt = 0

    for fp in file_list:
        arr = np.load(fp, mmap_mode='r')
        if arr.ndim == 4:
            arr = arr[0]

        total_sum += float(arr.sum())
        total_sq += float((arr ** 2).sum())
        total_cnt += arr.size
        mm = getattr(arr, "_mmap", None)
        if mm is not None:
            mm.close()

    mean = total_sum / total_cnt
    std = math.sqrt(max(total_sq / total_cnt - mean ** 2, 1e-12))

    return mean, std


if __name__ == "__main__":
    # Example usage
    print("Preprocessing Pipeline for CT Sinogram Generation")
    print("=" * 50)

    config = PreprocessingConfig()
    geo = GeometryConfig()

    print("\nPreprocessing Config:")
    print(f"  Target spacing: {config.target_spacing} mm")
    print(f"  Target XY size: {config.target_xy_size}")
    print(f"  HU range: [{config.hu_min}, {config.hu_max}]")

    print("\nGeometry Config:")
    print(f"  Volume shape: {geo.volume_shape}")
    print(f"  Detector: {geo.detector_width} x {geo.detector_height}")
    print(f"  Projections: {geo.n_projections}")
    print(f"  SDD: {geo.source_detector_distance} mm")
    print(f"  SID: {geo.source_isocenter_distance} mm")

    print("\nOutput sinogram shape: "
          f"({geo.n_projections}, {geo.detector_width}, {geo.detector_height})")
