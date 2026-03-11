"""
=============================================================================
CT-ORG数据集专用处理器
=============================================================================

CT-ORG数据集概述:
- 来源: https://wiki.cancerimagingarchive.net/display/Public/CT-ORG
- 包含: 140个腹部CT扫描 + 器官分割标签
- 格式: NIfTI (.nii.gz)
- 文件命名: volume-XX.nii.gz, labels-XX.nii.gz

数据集结构:
CT-ORG/
├── OrganSegmentations/
│   ├── volume-0.nii.gz      # CT体积
│   ├── labels-0.nii.gz      # 分割标签 (可选使用)
│   ├── volume-1.nii.gz
│   └── ...

维度说明:
- NIfTI存储: (X, Y, Z) - RAS坐标系
- 转换后: (Z, Y, X) - 标准医学影像坐标系

标签信息 (可用于ROI分析):
- 0: 背景
- 1: 肝脏
- 2: 膀胱
- 3: 肺
- 4: 肾脏
- 5: 骨骼
- 6: 脑
=============================================================================
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import logging
from tqdm import tqdm
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

# 导入主管道
from .pipeline import (
    PreprocessingPipeline,
    VolumeConfig,
    DetectorConfig,
    PreprocessingResult,
    NIfTILoader,
    extract_2d_sinograms
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class CTORGConfig:
    """CT-ORG数据集配置"""
    # 数据目录
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
    slice_step: int = 4  # 每4个切片取1个

    # 并行处理数
    n_workers: int = 1

    # 跳过已处理的文件
    skip_existing: bool = True

    def __post_init__(self):
        if self.volume_config is None:
            self.volume_config = VolumeConfig()
        if self.detector_config is None:
            self.detector_config = DetectorConfig()


class CTORGProcessor:
    """CT-ORG数据集处理器"""

    def __init__(self, config: CTORGConfig):
        self.config = config
        self.pipeline = PreprocessingPipeline(
            volume_config=config.volume_config,
            detector_config=config.detector_config,
            save_intermediate=config.save_volumes
        )

    def scan_dataset(self) -> List[Dict]:
        """
        扫描数据集，返回所有volume文件信息

        Returns:
            列表，每项包含 {path, patient_id}
        """
        data_root = Path(self.config.data_root)

        # 查找所有volume文件
        volume_files = list(data_root.glob('**/volume-*.nii.gz'))

        if not volume_files:
            # 尝试查找其他命名模式
            volume_files = list(data_root.glob('**/*.nii.gz'))
            volume_files = [f for f in volume_files if 'label' not in f.name.lower()]

        logger.info(f"找到 {len(volume_files)} 个NIfTI体积文件")

        # 构建文件信息
        file_infos = []
        for vf in sorted(volume_files):
            # 提取patient_id
            name = vf.stem.replace('.nii', '')  # 移除.nii后缀
            patient_id = name.replace('volume-', '').replace('volume', '')
            if not patient_id:
                patient_id = vf.parent.name

            file_infos.append({
                'path': str(vf),
                'patient_id': f"ctorg_{patient_id}",
                'label_path': str(vf).replace('volume', 'labels')  # 可能存在的标签
            })

        return file_infos

    def process_single(
        self,
        file_info: Dict,
        output_subdir: str = "sinograms_3d"
    ) -> PreprocessingResult:
        """处理单个文件"""
        nifti_path = file_info['path']
        patient_id = file_info['patient_id']

        output_dir = os.path.join(self.config.output_root, output_subdir)

        # 检查是否已存在
        if self.config.skip_existing:
            expected_file = f"{patient_id}_sinogram_{self.config.detector_config.detector_width}x{self.config.detector_config.detector_height}.npy"
            if os.path.exists(os.path.join(output_dir, expected_file)):
                logger.info(f"跳过已处理: {patient_id}")
                result = PreprocessingResult(success=True, patient_id=patient_id)
                result.sinogram_path = os.path.join(output_dir, expected_file)
                return result

        # 处理
        result = self.pipeline.process_nifti(nifti_path, output_dir, patient_id)

        return result

    def process_all(
        self,
        file_infos: Optional[List[Dict]] = None,
        output_subdir: str = "sinograms_3d"
    ) -> List[PreprocessingResult]:
        """
        处理所有文件

        Args:
            file_infos: 文件信息列表 (None则自动扫描)
            output_subdir: 输出子目录名

        Returns:
            处理结果列表
        """
        if file_infos is None:
            file_infos = self.scan_dataset()

        logger.info(f"开始处理 {len(file_infos)} 个文件...")

        results = []

        if self.config.n_workers <= 1:
            # 串行处理
            for info in tqdm(file_infos, desc="处理CT-ORG"):
                result = self.process_single(info, output_subdir)
                results.append(result)
        else:
            # 并行处理
            with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
                futures = {
                    executor.submit(self.process_single, info, output_subdir): info
                    for info in file_infos
                }

                for future in tqdm(as_completed(futures), total=len(futures), desc="处理CT-ORG"):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        info = futures[future]
                        logger.error(f"处理失败 {info['patient_id']}: {e}")
                        results.append(PreprocessingResult(
                            success=False,
                            patient_id=info['patient_id'],
                            error_message=str(e)
                        ))

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
        """
        从3D sinogram提取2D切片

        Args:
            results: 3D处理结果列表
            sinogram_dir: 3D sinogram目录 (如果results为None)
            output_subdir: 2D输出子目录

        Returns:
            所有2D切片路径列表
        """
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
        """
        运行完整处理流程

        Returns:
            包含处理统计信息的字典
        """
        logger.info("=" * 60)
        logger.info("CT-ORG数据集完整处理流程")
        logger.info("=" * 60)

        # 1. 扫描数据集
        file_infos = self.scan_dataset()

        # 2. 处理3D sinogram
        results_3d = self.process_all(file_infos)

        # 3. 提取2D切片 (可选)
        paths_2d = []
        if self.config.extract_2d:
            paths_2d = self.extract_2d_slices(results_3d)

        # 4. 保存处理报告
        report = {
            'dataset': 'CT-ORG',
            'total_files': len(file_infos),
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
            'failed_files': [
                {'patient_id': r.patient_id, 'error': r.error_message}
                for r in results_3d if not r.success
            ]
        }

        report_path = os.path.join(self.config.output_root, 'ctorg_processing_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"处理报告已保存: {report_path}")

        return report


def process_ct_org(
    data_root: str,
    output_root: str,
    extract_2d: bool = True,
    slice_step: int = 4,
    n_workers: int = 1
) -> Dict:
    """
    处理CT-ORG数据集的便捷函数

    Args:
        data_root: CT-ORG数据目录
        output_root: 输出目录
        extract_2d: 是否提取2D切片
        slice_step: 2D切片步长
        n_workers: 并行工作数

    Returns:
        处理报告字典
    """
    config = CTORGConfig(
        data_root=data_root,
        output_root=output_root,
        extract_2d=extract_2d,
        slice_step=slice_step,
        n_workers=n_workers
    )

    processor = CTORGProcessor(config)
    return processor.run_full_pipeline()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="处理CT-ORG数据集")
    parser.add_argument('--data_root', type=str, required=True,
                        help='CT-ORG数据目录路径')
    parser.add_argument('--output_root', type=str, required=True,
                        help='输出目录路径')
    parser.add_argument('--extract_2d', action='store_true', default=True,
                        help='是否提取2D切片')
    parser.add_argument('--slice_step', type=int, default=4,
                        help='2D切片步长')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='并行处理数')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='跳过已处理文件')

    args = parser.parse_args()

    config = CTORGConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        extract_2d=args.extract_2d,
        slice_step=args.slice_step,
        n_workers=args.n_workers,
        skip_existing=args.skip_existing
    )

    processor = CTORGProcessor(config)
    report = processor.run_full_pipeline()

    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"成功: {report['success_3d']}/{report['total_files']}")
    print(f"2D切片: {report['total_2d_slices']}")
