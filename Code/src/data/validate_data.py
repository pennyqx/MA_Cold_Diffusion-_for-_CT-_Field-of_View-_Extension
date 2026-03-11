"""
=============================================================================
数据验证工具
=============================================================================

功能:
1. 验证sinogram文件完整性和格式
2. 检查维度一致性
3. 统计数据分布
4. 生成质量报告
5. 可视化检查

验证项目:
- 文件是否可读
- 形状是否符合预期
- 值范围是否合理
- 是否包含NaN/Inf
- 数据分布是否正常
=============================================================================
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import logging
from tqdm import tqdm
import warnings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """验证配置"""
    # 数据目录
    data_dir: str = ""

    # 预期形状 (可选)
    expected_shape_3d: Optional[Tuple[int, int, int]] = (360, 640, 560)
    expected_shape_2d: Optional[Tuple[int, int]] = (360, 640)

    # 值范围检查
    check_value_range: bool = True
    expected_min: float = 0.0
    expected_max: float = 100.0  # sinogram值范围取决于投影参数

    # 是否生成可视化
    visualize: bool = False
    visualize_n_samples: int = 5

    # 输出目录
    output_dir: str = ""


@dataclass
class FileValidationResult:
    """单文件验证结果"""
    file_path: str = ""
    is_valid: bool = True
    shape: Optional[Tuple] = None
    dtype: str = ""
    min_value: float = 0.0
    max_value: float = 0.0
    mean_value: float = 0.0
    std_value: float = 0.0
    has_nan: bool = False
    has_inf: bool = False
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class SinogramValidator:
    """Sinogram数据验证器"""

    def __init__(self, config: ValidationConfig):
        self.config = config

    def validate_file(self, file_path: str) -> FileValidationResult:
        """验证单个文件"""
        result = FileValidationResult(file_path=file_path)

        try:
            # 尝试加载
            data = np.load(file_path)
            result.shape = data.shape
            result.dtype = str(data.dtype)

            # 统计值
            result.min_value = float(np.min(data))
            result.max_value = float(np.max(data))
            result.mean_value = float(np.mean(data))
            result.std_value = float(np.std(data))

            # NaN/Inf检查
            result.has_nan = bool(np.isnan(data).any())
            result.has_inf = bool(np.isinf(data).any())

            if result.has_nan:
                result.errors.append("包含NaN值")
                result.is_valid = False

            if result.has_inf:
                result.errors.append("包含Inf值")
                result.is_valid = False

            # 形状检查
            if len(data.shape) == 3:
                if self.config.expected_shape_3d and data.shape != self.config.expected_shape_3d:
                    result.errors.append(
                        f"3D形状不匹配: {data.shape} != {self.config.expected_shape_3d}"
                    )
                    result.is_valid = False
            elif len(data.shape) == 2:
                if self.config.expected_shape_2d and data.shape != self.config.expected_shape_2d:
                    result.errors.append(
                        f"2D形状不匹配: {data.shape} != {self.config.expected_shape_2d}"
                    )
                    result.is_valid = False
            else:
                result.errors.append(f"非预期维度: {len(data.shape)}D")
                result.is_valid = False

            # 值范围检查 (可选)
            if self.config.check_value_range:
                if result.min_value < self.config.expected_min:
                    result.errors.append(
                        f"最小值过低: {result.min_value:.4f} < {self.config.expected_min}"
                    )
                    # 不标记为无效，仅警告
                if result.max_value > self.config.expected_max:
                    result.errors.append(
                        f"最大值过高: {result.max_value:.4f} > {self.config.expected_max}"
                    )
                    # 不标记为无效，仅警告

        except Exception as e:
            result.is_valid = False
            result.errors.append(f"加载失败: {str(e)}")

        return result

    def validate_directory(self, progress: bool = True) -> Dict:
        """
        验证整个目录

        Returns:
            验证报告字典
        """
        data_path = Path(self.config.data_dir)
        all_files = list(data_path.glob('*.npy'))

        if not all_files:
            logger.warning(f"未找到.npy文件: {data_path}")
            return {'error': '未找到数据文件'}

        logger.info(f"验证 {len(all_files)} 个文件...")

        results = []
        iterator = tqdm(all_files, desc="验证") if progress else all_files

        for f in iterator:
            result = self.validate_file(str(f))
            results.append(result)

        return self._compile_report(results)

    def _compile_report(self, results: List[FileValidationResult]) -> Dict:
        """编译验证报告"""
        valid_results = [r for r in results if r.is_valid]
        invalid_results = [r for r in results if not r.is_valid]

        report = {
            'summary': {
                'total_files': len(results),
                'valid_files': len(valid_results),
                'invalid_files': len(invalid_results),
                'success_rate': len(valid_results) / len(results) * 100 if results else 0
            },
            'statistics': {},
            'shape_distribution': {},
            'invalid_files': []
        }

        # 统计有效文件的数值分布
        if valid_results:
            report['statistics'] = {
                'min_value': {
                    'min': min(r.min_value for r in valid_results),
                    'max': max(r.min_value for r in valid_results),
                    'mean': np.mean([r.min_value for r in valid_results])
                },
                'max_value': {
                    'min': min(r.max_value for r in valid_results),
                    'max': max(r.max_value for r in valid_results),
                    'mean': np.mean([r.max_value for r in valid_results])
                },
                'mean_value': {
                    'min': min(r.mean_value for r in valid_results),
                    'max': max(r.mean_value for r in valid_results),
                    'mean': np.mean([r.mean_value for r in valid_results])
                },
                'std_value': {
                    'min': min(r.std_value for r in valid_results),
                    'max': max(r.std_value for r in valid_results),
                    'mean': np.mean([r.std_value for r in valid_results])
                }
            }

        # 形状分布
        shape_counts = {}
        for r in results:
            if r.shape:
                shape_str = str(r.shape)
                shape_counts[shape_str] = shape_counts.get(shape_str, 0) + 1
        report['shape_distribution'] = shape_counts

        # 无效文件详情
        report['invalid_files'] = [
            {
                'file': Path(r.file_path).name,
                'errors': r.errors
            }
            for r in invalid_results
        ]

        return report

    def visualize_samples(
        self,
        n_samples: int = 5,
        output_dir: Optional[str] = None
    ) -> None:
        """
        可视化样本sinogram

        Args:
            n_samples: 可视化样本数
            output_dir: 输出目录
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib未安装，跳过可视化")
            return

        data_path = Path(self.config.data_dir)
        all_files = list(data_path.glob('*.npy'))

        if len(all_files) < n_samples:
            n_samples = len(all_files)

        sample_files = np.random.choice(all_files, n_samples, replace=False)

        output_dir = output_dir or self.config.output_dir or '.'
        os.makedirs(output_dir, exist_ok=True)

        for i, f in enumerate(sample_files):
            data = np.load(f)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # sinogram显示
            if data.ndim == 3:
                # 取中间切片
                mid_slice = data[:, :, data.shape[2] // 2]
                axes[0].imshow(mid_slice.T, cmap='gray', aspect='auto')
                axes[0].set_title(f'Middle Slice (z={data.shape[2]//2})')
            else:
                axes[0].imshow(data.T, cmap='gray', aspect='auto')
                axes[0].set_title('2D Sinogram')

            axes[0].set_xlabel('Projection Angle')
            axes[0].set_ylabel('Detector Column')

            # 直方图
            axes[1].hist(data.flatten(), bins=100, density=True)
            axes[1].set_xlabel('Value')
            axes[1].set_ylabel('Density')
            axes[1].set_title('Value Distribution')

            plt.suptitle(f'{f.name}\nShape: {data.shape}, Range: [{data.min():.2f}, {data.max():.2f}]')
            plt.tight_layout()

            save_path = os.path.join(output_dir, f'sample_{i+1}_{f.stem}.png')
            plt.savefig(save_path, dpi=100)
            plt.close()

            logger.info(f"保存可视化: {save_path}")


def validate_sinograms(
    data_dir: str,
    output_dir: Optional[str] = None,
    expected_shape: Optional[Tuple] = None,
    visualize: bool = False
) -> Dict:
    """
    验证sinogram数据的便捷函数

    Args:
        data_dir: 数据目录
        output_dir: 报告输出目录
        expected_shape: 预期形状
        visualize: 是否可视化

    Returns:
        验证报告
    """
    # 自动检测2D或3D
    sample_files = list(Path(data_dir).glob('*.npy'))[:1]
    if sample_files:
        sample = np.load(sample_files[0])
        is_3d = sample.ndim == 3
    else:
        is_3d = None

    config = ValidationConfig(
        data_dir=data_dir,
        output_dir=output_dir or data_dir,
        expected_shape_3d=expected_shape if is_3d else None,
        expected_shape_2d=expected_shape if not is_3d else None,
        visualize=visualize,
        check_value_range=False  # sinogram值范围变化大，默认不检查
    )

    validator = SinogramValidator(config)
    report = validator.validate_directory()

    # 保存报告
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, 'validation_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"报告保存至: {report_path}")

    # 可视化
    if visualize:
        validator.visualize_samples()

    return report


def check_dimension_consistency(data_dir: str) -> Dict:
    """
    快速检查维度一致性

    Args:
        data_dir: 数据目录

    Returns:
        {shape: count} 统计
    """
    data_path = Path(data_dir)
    all_files = list(data_path.glob('*.npy'))

    logger.info(f"检查 {len(all_files)} 个文件的维度...")

    shape_counts = {}
    for f in tqdm(all_files, desc="检查维度"):
        try:
            # 使用mmap_mode='r'避免完全加载
            data = np.load(f, mmap_mode='r')
            shape_str = str(data.shape)
            shape_counts[shape_str] = shape_counts.get(shape_str, 0) + 1
            mm = getattr(data, "_mmap", None)
            if mm is not None:
                mm.close()
        except Exception as e:
            shape_counts['ERROR'] = shape_counts.get('ERROR', 0) + 1

    logger.info("维度分布:")
    for shape, count in sorted(shape_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {shape}: {count} files")

    return shape_counts


def print_data_summary(data_dir: str) -> None:
    """打印数据摘要"""
    data_path = Path(data_dir)
    all_files = list(data_path.glob('*.npy'))

    if not all_files:
        print(f"未找到数据: {data_dir}")
        return

    print(f"\n{'='*60}")
    print(f"数据摘要: {data_dir}")
    print(f"{'='*60}")
    print(f"文件数: {len(all_files)}")

    # 采样检查
    sample = np.load(all_files[0])
    print(f"形状: {sample.shape}")
    print(f"数据类型: {sample.dtype}")
    print(f"值范围: [{sample.min():.4f}, {sample.max():.4f}]")

    # 维度说明
    if sample.ndim == 3:
        n_proj, width, height = sample.shape
        print(f"\n3D Sinogram维度解释:")
        print(f"  第0维 ({n_proj}): 投影角度数")
        print(f"  第1维 ({width}): 探测器宽度")
        print(f"  第2维 ({height}): 探测器高度")
    elif sample.ndim == 2:
        n_proj, width = sample.shape
        print(f"\n2D Sinogram维度解释:")
        print(f"  第0维 ({n_proj}): 投影角度数")
        print(f"  第1维 ({width}): 探测器宽度")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="数据验证工具")
    parser.add_argument('data_dir', type=str, help='数据目录')
    parser.add_argument('--output', type=str, help='报告输出目录')
    parser.add_argument('--visualize', action='store_true', help='生成可视化')
    parser.add_argument('--quick', action='store_true', help='快速维度检查')
    parser.add_argument('--summary', action='store_true', help='打印数据摘要')

    args = parser.parse_args()

    if args.summary:
        print_data_summary(args.data_dir)
    elif args.quick:
        check_dimension_consistency(args.data_dir)
    else:
        report = validate_sinograms(
            args.data_dir,
            args.output,
            visualize=args.visualize
        )

        print(f"\n验证结果:")
        print(f"  总文件数: {report['summary']['total_files']}")
        print(f"  有效文件: {report['summary']['valid_files']}")
        print(f"  无效文件: {report['summary']['invalid_files']}")
        print(f"  成功率: {report['summary']['success_rate']:.1f}%")
