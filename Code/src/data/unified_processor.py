"""
Unified preprocessing utilities for CT-ORG (NIfTI) and LIDC-IDRI (DICOM).
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .pipeline import (
    VolumeConfig,
    DetectorConfig,
    NIfTILoader,
    DICOMLoader,
    VolumePreprocessor,
    SinogramGenerator,
)


class UnifiedDataProcessor:
    """
    Unified loader + standardizer + sinogram generator.

    This keeps CT-ORG and LIDC in the same preprocessing flow after
    format-specific loading.
    """

    def __init__(self, volume_config: VolumeConfig, detector_config: DetectorConfig):
        self.vol_cfg = volume_config
        self.det_cfg = detector_config

        self.nifti_loader = NIfTILoader()
        self.dicom_loader = DICOMLoader(require_ct=True)
        self.preprocessor = VolumePreprocessor(self.vol_cfg)
        self.sino_generator = SinogramGenerator(self.vol_cfg, self.det_cfg)

    def load_volume(self, path: Path) -> Tuple[np.ndarray, Dict]:
        """Load a volume from NIfTI or DICOM directory."""
        path = Path(path)
        path_str = str(path)
        if path.is_dir():
            return self.dicom_loader.load(path_str)
        # DICOM archive or archive-with-prefix
        if '::' in path_str or path_str.lower().endswith(('.zip', '.tar', '.tar.gz', '.tgz')):
            return self.dicom_loader.load(path_str)
        if path.suffix in [".nii", ".gz"]:
            return self.nifti_loader.load(path_str)
        raise ValueError(f"Unsupported volume path: {path_str}")

    def standardize_volume(self, volume: np.ndarray, spacing: Tuple[float, float, float]) -> np.ndarray:
        """Apply resampling, crop/pad, and HU clipping."""
        return self.preprocessor.process(volume, spacing)

    def generate_sinogram(self, volume: np.ndarray) -> np.ndarray:
        """Forward project a standardized volume into a 3D sinogram."""
        return self.sino_generator.forward_project(volume)
