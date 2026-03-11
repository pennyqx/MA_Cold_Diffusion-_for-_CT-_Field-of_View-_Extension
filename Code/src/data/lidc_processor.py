"""
=============================================================================
LIDC-IDRI数据集专用处理器
=============================================================================

LIDC-IDRI数据集概述:
- 来源: https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI
- 包含: 1018例胸部CT扫描，含肺结节标注
- 格式: DICOM
- 每个病例可能包含多个扫描序列

典型目录结构:
LIDC-IDRI/
├── LIDC-IDRI-0001/
│   ├── <study_date>/
│   │   └── <series_uid>/
│   │       ├── 00000001.dcm
│   │       ├── 00000002.dcm
│   │       └── ...
├── LIDC-IDRI-0002/
│   └── ...

维度说明:
- DICOM pixel_array: (rows, cols) = (Y, X)
- 堆叠后: (n_slices, rows, cols) = (Z, Y, X)
- 无需转置，直接符合医学影像约定

DICOM重要字段:
- ImagePositionPatient[2]: 切片Z位置
- PixelSpacing: [row_spacing, col_spacing] = (dy, dx)
- SliceThickness: 切片厚度 (sz)
- RescaleSlope/Intercept: HU值转换

注意事项:
1. 部分病例有多个系列，需要选择最完整的一个
2. 切片厚度变化较大 (0.5mm - 5mm)
3. 部分扫描可能包含定位片，需要过滤
=============================================================================
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Set
from dataclasses import dataclass
import logging
from tqdm import tqdm
import json
from collections import defaultdict
import warnings

# 导入主管道
from .pipeline import (
    PreprocessingPipeline,
    VolumeConfig,
    DetectorConfig,
    PreprocessingResult,
    DICOMLoader,
    extract_2d_sinograms
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class LIDCConfig:
    """LIDC-IDRI数据集配置"""
    # 数据根目录
    data_root: str = ""

    # 输出目录
    output_root: str = ""

    # 体积配置
    volume_config: VolumeConfig = None

    # 探测器配置
    detector_config: DetectorConfig = None

    # 是否保存中间体积
    save_volumes: bool = False

    # 是否提取2D切片
    extract_2d: bool = True

    # 2D切片步长
    slice_step: int = 4

    # 最小切片数阈值
    min_slices: int = 80

    # 有效切片厚度范围
    valid_slice_thickness: Tuple[float, float] = (0.5, 3.0)

    # 跳过已处理
    skip_existing: bool = True

    # 只处理特定病例ID (用于测试)
    patient_ids: Optional[List[str]] = None

    def __post_init__(self):
        if self.volume_config is None:
            self.volume_config = VolumeConfig()
        if self.detector_config is None:
            self.detector_config = DetectorConfig()


class LIDCSeriesScanner:
    """LIDC-IDRI系列扫描器"""

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)

    @staticmethod
    def _is_archive_path(path: Path) -> bool:
        name = str(path).lower()
        return name.endswith('.zip') or name.endswith('.tar') or name.endswith('.tar.gz') or name.endswith('.tgz')

    @staticmethod
    def _is_dicom_name(name: str) -> bool:
        base = os.path.basename(name)
        return base.lower().endswith('.dcm') or '.' not in base

    def _list_archive_dicom_members(self, archive_path: Path) -> List[str]:
        members: List[str] = []
        name = str(archive_path).lower()
        if name.endswith('.zip'):
            import zipfile
            with zipfile.ZipFile(archive_path, 'r') as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    if self._is_dicom_name(info.filename):
                        members.append(info.filename)
        else:
            import tarfile
            with tarfile.open(archive_path, 'r:*') as tf:
                for info in tf.getmembers():
                    if not info.isfile():
                        continue
                    if self._is_dicom_name(info.name):
                        members.append(info.name)
        members.sort()
        return members

    @staticmethod
    def _patient_id_from_path(path: Path) -> str:
        name = path.name
        for suffix in ['.tar.gz', '.tgz', '.tar', '.zip']:
            if name.lower().endswith(suffix):
                name = name[: -len(suffix)]
                break
        return name

    def find_patient_dirs(self) -> List[Path]:
        """查找所有病例目录"""
        patient_dirs = []

        # 方式1: 直接在根目录下
        for item in self.data_root.iterdir():
            if item.is_dir() and item.name.startswith('LIDC-IDRI-'):
                patient_dirs.append(item)
            elif item.is_file() and item.name.startswith('LIDC-IDRI-') and self._is_archive_path(item):
                patient_dirs.append(item)

        # 方式2: 在子目录中
        if not patient_dirs:
            for subdir in self.data_root.iterdir():
                if subdir.is_dir():
                    for item in subdir.iterdir():
                        if item.is_dir() and 'LIDC' in item.name:
                            patient_dirs.append(item)
                        elif item.is_file() and 'LIDC' in item.name and self._is_archive_path(item):
                            patient_dirs.append(item)

        return sorted(patient_dirs)

    def find_series_in_patient(self, patient_dir: Path) -> List[Dict]:
        """
        查找病例目录中的所有DICOM系列

        Returns:
            列表，每项包含 {series_dir, dicom_files, n_files}
        """
        series_list = []

        if self._is_archive_path(patient_dir):
            dicom_members = self._list_archive_dicom_members(patient_dir)
            if not dicom_members:
                return []

            # 按父目录分组 (使用归档内路径)
            series_groups = defaultdict(list)
            for name in dicom_members:
                parent = os.path.dirname(name)
                series_groups[parent].append(name)

            # 构建系列信息
            for series_dir, files in series_groups.items():
                if len(files) >= 20:  # 至少20张切片
                    series_list.append({
                        'series_dir': f"{patient_dir}::{series_dir}",
                        'dicom_files': sorted(files),
                        'n_files': len(files)
                    })
        else:
            # 递归查找所有DICOM文件
            dicom_files = list(patient_dir.rglob('*.dcm'))

            if not dicom_files:
                # 尝试无扩展名的文件
                for f in patient_dir.rglob('*'):
                    if f.is_file() and '.' not in f.name:
                        dicom_files.append(f)

            if not dicom_files:
                return []

            # 按父目录分组
            series_groups = defaultdict(list)
            for f in dicom_files:
                series_groups[f.parent].append(f)

            # 构建系列信息
            for series_dir, files in series_groups.items():
                if len(files) >= 20:  # 至少20张切片
                    series_list.append({
                        'series_dir': str(series_dir),
                        'dicom_files': sorted([str(f) for f in files]),
                        'n_files': len(files)
                    })

        # 按文件数排序 (选最多的)
        series_list.sort(key=lambda x: x['n_files'], reverse=True)

        return series_list

    def scan_all_patients(self) -> Dict[str, List[Dict]]:
        """
        扫描所有病例

        Returns:
            {patient_id: [series_info, ...]}
        """
        patient_dirs = self.find_patient_dirs()
        logger.info(f"找到 {len(patient_dirs)} 个病例目录")

        result = {}
        for pdir in tqdm(patient_dirs, desc="扫描LIDC系列"):
            patient_id = self._patient_id_from_path(pdir)
            series = self.find_series_in_patient(pdir)
            if series:
                result[patient_id] = series

        logger.info(f"有效病例数: {len(result)}")
        total_series = sum(len(s) for s in result.values())
        logger.info(f"总系列数: {total_series}")

        return result


class SeriesValidator:
    """DICOM系列验证器"""

    def __init__(self, config: LIDCConfig):
        self.config = config

    @staticmethod
    def _is_archive_path(path_str: str) -> bool:
        name = path_str.lower()
        return name.endswith('.zip') or name.endswith('.tar') or name.endswith('.tar.gz') or name.endswith('.tgz')

    @staticmethod
    def _split_archive_path(path_str: str) -> Tuple[str, Optional[str]]:
        if '::' in path_str:
            archive_path, inner_prefix = path_str.split('::', 1)
            inner_prefix = inner_prefix.strip('/')
            return archive_path, inner_prefix or None
        return path_str, None

    @staticmethod
    def _is_dicom_name(name: str) -> bool:
        base = os.path.basename(name)
        return base.lower().endswith('.dcm') or '.' not in base

    def _list_archive_members(self, archive_path: str, inner_prefix: Optional[str]) -> List[str]:
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
                    if self._is_dicom_name(name):
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
                    if self._is_dicom_name(name):
                        members.append(name)
        members.sort()
        return members

    def _read_first_dicom_from_archive(self, archive_path: str, member_name: str):
        import io
        from pydicom.filereader import dcmread
        if archive_path.lower().endswith('.zip'):
            import zipfile
            with zipfile.ZipFile(archive_path, 'r') as zf:
                with zf.open(member_name, 'r') as f:
                    data = f.read()
            return dcmread(io.BytesIO(data), force=True)
        import tarfile
        with tarfile.open(archive_path, 'r:*') as tf:
            f = tf.extractfile(member_name)
            if f is None:
                raise ValueError(f"无法读取DICOM: {member_name}")
            data = f.read()
            f.close()
        return dcmread(io.BytesIO(data), force=True)

    def validate_series(self, series_dir: str) -> Tuple[bool, str]:
        """
        验证系列是否满足条件

        Returns:
            (is_valid, message)
        """
        try:
            import pydicom
            from pydicom.filereader import dcmread
        except ImportError:
            return False, "pydicom未安装"

        archive_path = None
        inner_prefix = None
        if self._is_archive_path(series_dir) or '::' in series_dir:
            archive_path, inner_prefix = self._split_archive_path(series_dir)

        if archive_path and self._is_archive_path(archive_path):
            dicom_files = self._list_archive_members(archive_path, inner_prefix)
        else:
            dicom_files = list(Path(series_dir).glob('*.dcm'))
            if not dicom_files:
                dicom_files = [f for f in Path(series_dir).iterdir()
                              if f.is_file() and '.' not in f.name]

        # 1. 检查文件数
        if len(dicom_files) < self.config.min_slices:
            return False, f"切片数不足: {len(dicom_files)} < {self.config.min_slices}"

        # 2. 加载第一个文件检查元数据
        try:
            if archive_path and self._is_archive_path(archive_path):
                first_dcm = self._read_first_dicom_from_archive(archive_path, dicom_files[0])
            else:
                first_dcm = dcmread(str(dicom_files[0]), force=True)
        except Exception as e:
            return False, f"无法读取DICOM: {e}"

        # 3. 检查是否为CT
        if hasattr(first_dcm, 'Modality') and first_dcm.Modality != 'CT':
            return False, f"非CT模态: {first_dcm.Modality}"

        # 4. 检查切片厚度
        slice_thickness = 1.0
        if hasattr(first_dcm, 'SliceThickness'):
            slice_thickness = float(first_dcm.SliceThickness)

        min_t, max_t = self.config.valid_slice_thickness
        if not (min_t <= slice_thickness <= max_t):
            return False, f"切片厚度超出范围: {slice_thickness:.2f}mm"

        return True, f"有效: {len(dicom_files)} slices, {slice_thickness:.2f}mm thickness"


class LIDCProcessor:
    """LIDC-IDRI数据集处理器"""

    def __init__(self, config: LIDCConfig):
        self.config = config
        self.scanner = LIDCSeriesScanner(config.data_root)
        self.validator = SeriesValidator(config)
        self.pipeline = PreprocessingPipeline(
            volume_config=config.volume_config,
            detector_config=config.detector_config,
            save_intermediate=config.save_volumes
        )

    def select_best_series(self, series_list: List[Dict]) -> Optional[Dict]:
        """
        从多个系列中选择最佳系列

        选择标准:
        1. 切片数最多
        2. 满足验证条件
        """
        for series in series_list:  # 已按切片数排序
            is_valid, msg = self.validator.validate_series(series['series_dir'])
            if is_valid:
                return series
            else:
                logger.debug(f"  跳过系列: {msg}")
        return None

    def process_single_patient(
        self,
        patient_id: str,
        series_list: List[Dict],
        output_subdir: str = "sinograms_3d"
    ) -> PreprocessingResult:
        """处理单个病例"""
        output_dir = os.path.join(self.config.output_root, output_subdir)

        # 检查是否已处理
        if self.config.skip_existing:
            expected_file = f"{patient_id}_sinogram_{self.config.detector_config.detector_width}x{self.config.detector_config.detector_height}.npy"
            if os.path.exists(os.path.join(output_dir, expected_file)):
                logger.info(f"跳过已处理: {patient_id}")
                result = PreprocessingResult(success=True, patient_id=patient_id)
                result.sinogram_path = os.path.join(output_dir, expected_file)
                return result

        # 选择最佳系列
        best_series = self.select_best_series(series_list)
        if best_series is None:
            return PreprocessingResult(
                success=False,
                patient_id=patient_id,
                error_message="未找到有效系列"
            )

        logger.info(f"处理 {patient_id}: {best_series['n_files']} slices")

        # 使用pipeline处理
        result = self.pipeline.process_dicom(
            best_series['series_dir'],
            output_dir,
            patient_id
        )

        return result

    def process_all(
        self,
        patient_series: Optional[Dict[str, List[Dict]]] = None,
        output_subdir: str = "sinograms_3d"
    ) -> List[PreprocessingResult]:
        """
        处理所有病例

        Args:
            patient_series: {patient_id: [series_info]} 字典
            output_subdir: 输出子目录

        Returns:
            处理结果列表
        """
        if patient_series is None:
            patient_series = self.scanner.scan_all_patients()

        # 过滤指定病例
        if self.config.patient_ids:
            patient_series = {
                k: v for k, v in patient_series.items()
                if k in self.config.patient_ids
            }

        logger.info(f"开始处理 {len(patient_series)} 个病例...")

        results = []
        for patient_id, series_list in tqdm(patient_series.items(), desc="处理LIDC"):
            result = self.process_single_patient(
                patient_id, series_list, output_subdir
            )
            results.append(result)

        # 统计
        success_count = sum(1 for r in results if r.success)
        logger.info(f"处理完成: {success_count}/{len(results)} 成功")

        return results

    def extract_2d_slices(
        self,
        results: Optional[List[PreprocessingResult]] = None,
        sinogram_dir: Optional[str] = None,
        output_subdir: str = "sinograms_2d"
    ) -> List[str]:
        """从3D sinogram提取2D切片"""
        output_dir = os.path.join(self.config.output_root, output_subdir)
        os.makedirs(output_dir, exist_ok=True)

        # 获取3D sinogram文件列表
        if results is not None:
            sino_files = [r.sinogram_path for r in results if r.success and r.sinogram_path]
        elif sinogram_dir is not None:
            sino_files = list(Path(sinogram_dir).glob('*.npy'))
        else:
            sino_dir = os.path.join(self.config.output_root, "sinograms_3d")
            sino_files = list(Path(sino_dir).glob('*.npy'))

        logger.info(f"提取2D切片: {len(sino_files)} 个3D sinogram")

        all_2d_paths = []
        for sino_path in tqdm(sino_files, desc="提取2D切片"):
            paths = extract_2d_sinograms(
                str(sino_path),
                output_dir,
                slice_step=self.config.slice_step
            )
            all_2d_paths.extend(paths)

        logger.info(f"共提取 {len(all_2d_paths)} 个2D切片")

        return all_2d_paths

    def run_full_pipeline(self) -> Dict:
        """运行完整处理流程"""
        logger.info("=" * 60)
        logger.info("LIDC-IDRI数据集完整处理流程")
        logger.info("=" * 60)

        # 1. 扫描所有病例
        patient_series = self.scanner.scan_all_patients()

        # 2. 处理3D sinogram
        results_3d = self.process_all(patient_series)

        # 3. 提取2D切片
        paths_2d = []
        if self.config.extract_2d:
            paths_2d = self.extract_2d_slices(results_3d)

        # 4. 保存报告
        report = {
            'dataset': 'LIDC-IDRI',
            'total_patients': len(patient_series),
            'success_3d': sum(1 for r in results_3d if r.success),
            'failed_3d': sum(1 for r in results_3d if not r.success),
            'total_2d_slices': len(paths_2d),
            'volume_config': {
                'target_shape': self.config.volume_config.target_shape,
                'target_spacing': self.config.volume_config.target_spacing
            },
            'detector_config': {
                'width': self.config.detector_config.detector_width,
                'height': self.config.detector_config.detector_height,
                'n_projections': self.config.detector_config.n_projections
            },
            'failed_patients': [
                {'patient_id': r.patient_id, 'error': r.error_message}
                for r in results_3d if not r.success
            ]
        }

        report_path = os.path.join(self.config.output_root, 'lidc_processing_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"处理报告已保存: {report_path}")

        return report


def process_lidc(
    data_root: str,
    output_root: str,
    extract_2d: bool = True,
    slice_step: int = 4,
    min_slices: int = 80
) -> Dict:
    """
    处理LIDC-IDRI数据集的便捷函数

    Args:
        data_root: LIDC-IDRI数据目录
        output_root: 输出目录
        extract_2d: 是否提取2D切片
        slice_step: 2D切片步长
        min_slices: 最小切片数阈值

    Returns:
        处理报告字典
    """
    config = LIDCConfig(
        data_root=data_root,
        output_root=output_root,
        extract_2d=extract_2d,
        slice_step=slice_step,
        min_slices=min_slices
    )

    processor = LIDCProcessor(config)
    return processor.run_full_pipeline()


def scan_lidc_structure(data_root: str) -> None:
    """
    扫描并打印LIDC-IDRI目录结构

    用于了解数据集组织方式
    """
    scanner = LIDCSeriesScanner(data_root)
    patient_series = scanner.scan_all_patients()

    print(f"\n{'='*60}")
    print(f"LIDC-IDRI目录结构分析")
    print(f"{'='*60}")
    print(f"数据根目录: {data_root}")
    print(f"病例总数: {len(patient_series)}")

    # 统计系列数分布
    series_counts = [len(s) for s in patient_series.values()]
    print(f"\n每病例系列数统计:")
    print(f"  最小: {min(series_counts)}")
    print(f"  最大: {max(series_counts)}")
    print(f"  平均: {np.mean(series_counts):.1f}")

    # 统计切片数分布
    slice_counts = []
    for series_list in patient_series.values():
        for s in series_list:
            slice_counts.append(s['n_files'])

    print(f"\n切片数统计:")
    print(f"  最小: {min(slice_counts)}")
    print(f"  最大: {max(slice_counts)}")
    print(f"  平均: {np.mean(slice_counts):.1f}")

    # 显示前5个病例
    print(f"\n前5个病例示例:")
    for i, (pid, series) in enumerate(list(patient_series.items())[:5]):
        print(f"  {pid}:")
        for j, s in enumerate(series[:2]):  # 每个病例最多显示2个系列
            print(f"    系列{j+1}: {s['n_files']} slices")
            print(f"           {s['series_dir'][:60]}...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="处理LIDC-IDRI数据集")
    parser.add_argument('--data_root', type=str, required=True,
                        help='LIDC-IDRI数据目录')
    parser.add_argument('--output_root', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--scan_only', action='store_true',
                        help='仅扫描目录结构')
    parser.add_argument('--extract_2d', action='store_true', default=True,
                        help='是否提取2D切片')
    parser.add_argument('--slice_step', type=int, default=4,
                        help='2D切片步长')
    parser.add_argument('--min_slices', type=int, default=80,
                        help='最小切片数阈值')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='跳过已处理')

    args = parser.parse_args()

    if args.scan_only:
        scan_lidc_structure(args.data_root)
    else:
        config = LIDCConfig(
            data_root=args.data_root,
            output_root=args.output_root,
            extract_2d=args.extract_2d,
            slice_step=args.slice_step,
            min_slices=args.min_slices,
            skip_existing=args.skip_existing
        )

        processor = LIDCProcessor(config)
        report = processor.run_full_pipeline()

        print("\n" + "=" * 60)
        print("处理完成!")
        print("=" * 60)
        print(f"成功: {report['success_3d']}/{report['total_patients']}")
        print(f"2D切片: {report['total_2d_slices']}")
