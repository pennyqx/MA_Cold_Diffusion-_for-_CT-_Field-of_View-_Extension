from .dataset import SinogramDataset2D, build_dataloaders
from .preprocessing import preprocess_dicom_to_sinogram, compute_global_stats

# 新增数据预处理模块
from .pipeline import (
    VolumeConfig,
    DetectorConfig,
    PreprocessingPipeline,
    NIfTILoader,
    DICOMLoader,
    VolumePreprocessor,
    SinogramGenerator,
    extract_2d_sinograms,
    print_dimension_info
)
from .ct_org_processor import CTORGConfig, CTORGProcessor, process_ct_org
from .lidc_processor import LIDCConfig, LIDCProcessor, process_lidc
from .split_dataset import (
    SplitConfig,
    DatasetSplitter,
    create_split_from_existing,
    load_split_files
)
from .validate_data import (
    ValidationConfig,
    SinogramValidator,
    validate_sinograms,
    check_dimension_consistency,
    print_data_summary
)

__all__ = [
    # 数据集
    'SinogramDataset2D',
    'build_dataloaders',

    # 预处理 (原有)
    'preprocess_dicom_to_sinogram',
    'compute_global_stats',

    # 预处理管道 (新增)
    'VolumeConfig',
    'DetectorConfig',
    'PreprocessingPipeline',
    'NIfTILoader',
    'DICOMLoader',
    'VolumePreprocessor',
    'SinogramGenerator',
    'extract_2d_sinograms',
    'print_dimension_info',

    # CT-ORG处理器
    'CTORGConfig',
    'CTORGProcessor',
    'process_ct_org',

    # LIDC-IDRI处理器
    'LIDCConfig',
    'LIDCProcessor',
    'process_lidc',

    # 数据集划分
    'SplitConfig',
    'DatasetSplitter',
    'create_split_from_existing',
    'load_split_files',

    # 数据验证
    'ValidationConfig',
    'SinogramValidator',
    'validate_sinograms',
    'check_dimension_consistency',
    'print_data_summary'
]
