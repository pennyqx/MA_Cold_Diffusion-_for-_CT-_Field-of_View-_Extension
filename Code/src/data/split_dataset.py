"""
=============================================================================
数据集划分工具
=============================================================================

功能:
1. 按病例级别划分训练/验证/测试集
2. 按切片级别划分 (保证同一病例的切片在同一子集)
3. 支持分层抽样 (按数据来源)
4. 生成划分清单文件

划分策略:
- 病例级划分: 保证同一病例的所有切片在同一子集
- 默认比例: 70% train / 15% val / 15% test
- 随机种子固定以保证可复现

输出文件:
- train.txt / val.txt / test.txt: 文件路径列表
- split_info.json: 划分统计信息
=============================================================================
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import json
import logging
from collections import defaultdict
import random
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    """数据集划分配置"""
    # 数据目录
    data_dir: str = ""

    # 输出目录
    output_dir: str = ""

    # 划分比例
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # 随机种子
    random_seed: int = 42

    # 是否按病例划分 (True) 或按文件划分 (False)
    split_by_patient: bool = True

    # 文件模式
    file_pattern: str = "*.npy"

    # 用于提取patient_id的正则表达式
    patient_id_pattern: str = r"^(.+?)_(?:slice|sinogram)"

    # 是否分层抽样 (按数据来源: ctorg, lidc等)
    stratify_by_source: bool = True


def extract_patient_id(filename: str, pattern: str) -> str:
    """
    从文件名提取病例ID

    例如:
    - ctorg_0001_slice0010_sino2d_640.npy -> ctorg_0001
    - LIDC-IDRI-0001_slice0010_sino2d_640.npy -> LIDC-IDRI-0001
    """
    match = re.match(pattern, filename)
    if match:
        return match.group(1)
    # 回退: 使用下划线分割的前几个部分
    parts = filename.split('_')
    if len(parts) >= 2:
        return '_'.join(parts[:2])
    return filename.split('.')[0]


def extract_source(patient_id: str) -> str:
    """
    从病例ID提取数据来源

    例如:
    - ctorg_0001 -> ctorg
    - LIDC-IDRI-0001 -> lidc
    """
    pid_lower = patient_id.lower()
    if 'ctorg' in pid_lower:
        return 'ctorg'
    elif 'lidc' in pid_lower:
        return 'lidc'
    else:
        return 'other'


class DatasetSplitter:
    """数据集划分器"""

    def __init__(self, config: SplitConfig):
        self.config = config
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)

    def scan_files(self) -> Dict[str, List[str]]:
        """
        扫描数据目录，按病例分组文件

        Returns:
            {patient_id: [file_path, ...]}
        """
        data_path = Path(self.config.data_dir)
        all_files = list(data_path.glob(self.config.file_pattern))

        logger.info(f"扫描目录: {data_path}")
        logger.info(f"找到文件: {len(all_files)}")

        if not all_files:
            raise ValueError(f"未找到匹配的文件: {self.config.file_pattern}")

        # 按病例ID分组
        patient_files = defaultdict(list)
        for f in all_files:
            pid = extract_patient_id(f.name, self.config.patient_id_pattern)
            patient_files[pid].append(str(f))

        # 按文件名排序
        for pid in patient_files:
            patient_files[pid].sort()

        logger.info(f"病例数: {len(patient_files)}")

        return dict(patient_files)

    def stratified_split_patients(
        self,
        patient_files: Dict[str, List[str]]
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        分层抽样划分病例

        Args:
            patient_files: {patient_id: [files]}

        Returns:
            (train_patients, val_patients, test_patients)
        """
        # 按来源分组
        source_patients = defaultdict(list)
        for pid in patient_files.keys():
            source = extract_source(pid)
            source_patients[source].append(pid)

        logger.info(f"数据来源分布:")
        for source, pids in source_patients.items():
            logger.info(f"  {source}: {len(pids)} patients")

        train_patients = []
        val_patients = []
        test_patients = []

        # 对每个来源分别划分
        for source, patients in source_patients.items():
            random.shuffle(patients)
            n = len(patients)

            n_train = int(n * self.config.train_ratio)
            n_val = int(n * self.config.val_ratio)

            train_patients.extend(patients[:n_train])
            val_patients.extend(patients[n_train:n_train + n_val])
            test_patients.extend(patients[n_train + n_val:])

        return train_patients, val_patients, test_patients

    def simple_split_patients(
        self,
        patient_files: Dict[str, List[str]]
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        简单随机划分病例

        Returns:
            (train_patients, val_patients, test_patients)
        """
        patients = list(patient_files.keys())
        random.shuffle(patients)

        n = len(patients)
        n_train = int(n * self.config.train_ratio)
        n_val = int(n * self.config.val_ratio)

        train_patients = patients[:n_train]
        val_patients = patients[n_train:n_train + n_val]
        test_patients = patients[n_train + n_val:]

        return train_patients, val_patients, test_patients

    def split_by_patient(
        self,
        patient_files: Dict[str, List[str]]
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        按病例划分，返回文件列表

        Returns:
            (train_files, val_files, test_files)
        """
        if self.config.stratify_by_source:
            train_p, val_p, test_p = self.stratified_split_patients(patient_files)
        else:
            train_p, val_p, test_p = self.simple_split_patients(patient_files)

        # 收集对应的文件
        train_files = []
        val_files = []
        test_files = []

        for pid in train_p:
            train_files.extend(patient_files[pid])
        for pid in val_p:
            val_files.extend(patient_files[pid])
        for pid in test_p:
            test_files.extend(patient_files[pid])

        # 排序
        train_files.sort()
        val_files.sort()
        test_files.sort()

        return train_files, val_files, test_files

    def split_by_file(
        self,
        patient_files: Dict[str, List[str]]
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        按文件随机划分 (不保证同病例在同子集)

        Returns:
            (train_files, val_files, test_files)
        """
        all_files = []
        for files in patient_files.values():
            all_files.extend(files)

        random.shuffle(all_files)

        n = len(all_files)
        n_train = int(n * self.config.train_ratio)
        n_val = int(n * self.config.val_ratio)

        train_files = sorted(all_files[:n_train])
        val_files = sorted(all_files[n_train:n_train + n_val])
        test_files = sorted(all_files[n_train + n_val:])

        return train_files, val_files, test_files

    def run(self) -> Dict:
        """
        执行划分

        Returns:
            划分统计信息
        """
        logger.info("=" * 60)
        logger.info("数据集划分")
        logger.info("=" * 60)

        # 扫描文件
        patient_files = self.scan_files()

        # 划分
        if self.config.split_by_patient:
            logger.info("按病例划分...")
            train_files, val_files, test_files = self.split_by_patient(patient_files)
        else:
            logger.info("按文件划分...")
            train_files, val_files, test_files = self.split_by_file(patient_files)

        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)

        # 保存文件列表
        self._save_file_list(train_files, 'train.txt')
        self._save_file_list(val_files, 'val.txt')
        self._save_file_list(test_files, 'test.txt')

        # 统计信息
        info = self._compute_statistics(patient_files, train_files, val_files, test_files)

        # 保存统计信息
        info_path = os.path.join(self.config.output_dir, 'split_info.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)

        logger.info(f"\n划分结果:")
        logger.info(f"  训练集: {info['train']['n_files']} files ({info['train']['n_patients']} patients)")
        logger.info(f"  验证集: {info['val']['n_files']} files ({info['val']['n_patients']} patients)")
        logger.info(f"  测试集: {info['test']['n_files']} files ({info['test']['n_patients']} patients)")
        logger.info(f"\n输出目录: {self.config.output_dir}")

        return info

    def _save_file_list(self, files: List[str], filename: str):
        """保存文件列表"""
        path = os.path.join(self.config.output_dir, filename)
        with open(path, 'w') as f:
            for fp in files:
                f.write(fp + '\n')
        logger.debug(f"保存: {path} ({len(files)} files)")

    def _compute_statistics(
        self,
        patient_files: Dict[str, List[str]],
        train_files: List[str],
        val_files: List[str],
        test_files: List[str]
    ) -> Dict:
        """计算划分统计信息"""

        def get_patients(files):
            patients = set()
            for f in files:
                pid = extract_patient_id(Path(f).name, self.config.patient_id_pattern)
                patients.add(pid)
            return patients

        def get_sources(patients):
            sources = defaultdict(int)
            for pid in patients:
                source = extract_source(pid)
                sources[source] += 1
            return dict(sources)

        train_patients = get_patients(train_files)
        val_patients = get_patients(val_files)
        test_patients = get_patients(test_files)

        info = {
            'config': {
                'train_ratio': self.config.train_ratio,
                'val_ratio': self.config.val_ratio,
                'test_ratio': self.config.test_ratio,
                'split_by_patient': self.config.split_by_patient,
                'stratify_by_source': self.config.stratify_by_source,
                'random_seed': self.config.random_seed
            },
            'total': {
                'n_patients': len(patient_files),
                'n_files': sum(len(f) for f in patient_files.values())
            },
            'train': {
                'n_patients': len(train_patients),
                'n_files': len(train_files),
                'sources': get_sources(train_patients)
            },
            'val': {
                'n_patients': len(val_patients),
                'n_files': len(val_files),
                'sources': get_sources(val_patients)
            },
            'test': {
                'n_patients': len(test_patients),
                'n_files': len(test_files),
                'sources': get_sources(test_patients)
            }
        }

        return info


def create_split_from_existing(
    sinogram_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict:
    """
    从已有sinogram目录创建数据划分

    Args:
        sinogram_dir: sinogram目录
        output_dir: 划分文件输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子

    Returns:
        划分统计信息
    """
    config = SplitConfig(
        data_dir=sinogram_dir,
        output_dir=output_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
        split_by_patient=True,
        stratify_by_source=True
    )

    splitter = DatasetSplitter(config)
    return splitter.run()


def load_split_files(split_dir: str, split_name: str = 'train') -> List[str]:
    """
    加载划分文件列表

    Args:
        split_dir: 划分文件目录
        split_name: 'train' / 'val' / 'test'

    Returns:
        文件路径列表
    """
    split_file = os.path.join(split_dir, f'{split_name}.txt')
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"划分文件不存在: {split_file}")

    with open(split_file, 'r') as f:
        files = [line.strip() for line in f if line.strip()]

    return files


def create_symlink_splits(
    split_dir: str,
    target_dir: str,
    create_copies: bool = False
) -> None:
    """
    根据划分文件创建符号链接或复制

    将文件组织为:
    target_dir/
    ├── train/
    ├── val/
    └── test/

    Args:
        split_dir: 划分文件目录
        target_dir: 目标目录
        create_copies: True=复制文件, False=创建符号链接
    """
    import shutil

    for split_name in ['train', 'val', 'test']:
        files = load_split_files(split_dir, split_name)

        target_subdir = os.path.join(target_dir, split_name)
        os.makedirs(target_subdir, exist_ok=True)

        for src_path in files:
            filename = os.path.basename(src_path)
            dst_path = os.path.join(target_subdir, filename)

            if os.path.exists(dst_path):
                continue

            if create_copies:
                shutil.copy2(src_path, dst_path)
            else:
                try:
                    os.symlink(src_path, dst_path)
                except OSError:
                    # Windows可能需要管理员权限
                    shutil.copy2(src_path, dst_path)

        logger.info(f"创建 {split_name}: {len(files)} files")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="数据集划分工具")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='sinogram数据目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='划分文件输出目录')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='测试集比例')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--no_stratify', action='store_true',
                        help='禁用分层抽样')
    parser.add_argument('--by_file', action='store_true',
                        help='按文件划分(而非按病例)')

    args = parser.parse_args()

    config = SplitConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        split_by_patient=not args.by_file,
        stratify_by_source=not args.no_stratify
    )

    splitter = DatasetSplitter(config)
    info = splitter.run()

    print("\n" + "=" * 60)
    print("划分完成!")
    print("=" * 60)
