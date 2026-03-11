#!/usr/bin/env python
"""
=============================================================================
Unified batch preprocessing script
=============================================================================

Pipeline overview:
1. Scan CT-ORG (NIfTI) and LIDC-IDRI (DICOM) into a unified case list
2. Split patients into train/val/test (stratified by source)
3. Standardize volumes with a unified processor
4. Save standardized volumes into a single folder
5. Forward project to sinograms and save 2D slices
6. Save split lists and normalization stats
7. Optional sanity visualizations
=============================================================================
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add src to path
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline import VolumeConfig, DetectorConfig, print_dimension_info
from src.data.ct_org_processor import CTORGConfig, CTORGProcessor
from src.data.lidc_processor import LIDCConfig, LIDCProcessor
from src.data.unified_processor import UnifiedDataProcessor
from src.data.dataset import compute_global_stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('batch_preprocess.log')]
)
logger = logging.getLogger(__name__)


def get_default_volume_config() -> VolumeConfig:
    return VolumeConfig(
        target_shape=(128, 512, 512),
        target_spacing=(1.5, 1.0, 1.0),
        hu_min=-1000.0,
        hu_max=1000.0,
        valid_slice_thickness_range=(0.5, 3.0),
        min_valid_slices=64,
    )


def get_default_detector_config() -> DetectorConfig:
    return DetectorConfig(
        detector_width=640,
        detector_height=560,
        detector_spacing_x=0.5,
        detector_spacing_y=0.5,
        n_projections=360,
        angular_range=2 * np.pi,
        source_detector_distance=1200.0,
        source_isocenter_distance=750.0,
    )


def scan_ctorg(data_root: str, volume_config: VolumeConfig, detector_config: DetectorConfig) -> List[Dict]:
    config = CTORGConfig(
        data_root=data_root,
        output_root=str(PROJECT_ROOT / 'data' / 'processed'),
        volume_config=volume_config,
        detector_config=detector_config,
    )
    processor = CTORGProcessor(config)
    return processor.scan_dataset()


def scan_lidc(data_root: str, volume_config: VolumeConfig, detector_config: DetectorConfig) -> List[Dict]:
    config = LIDCConfig(
        data_root=data_root,
        output_root=str(PROJECT_ROOT / 'data' / 'processed'),
        volume_config=volume_config,
        detector_config=detector_config,
        min_slices=80,
        valid_slice_thickness=(0.5, 3.0),
    )
    processor = LIDCProcessor(config)

    patient_series = processor.scanner.scan_all_patients()
    cases = []
    for patient_id, series_list in patient_series.items():
        best = processor.select_best_series(series_list)
        if best is None:
            continue
        cases.append(
            {
                "path": best["series_dir"],
                "patient_id": patient_id,
                "source": "lidc",
                "input_type": "dicom",
            }
        )
    return cases


def build_case_list(
    ctorg_root: Optional[str],
    lidc_root: Optional[str],
    volume_config: VolumeConfig,
    detector_config: DetectorConfig,
) -> List[Dict]:
    cases: List[Dict] = []

    if ctorg_root and os.path.exists(ctorg_root):
        ctorg_infos = scan_ctorg(ctorg_root, volume_config, detector_config)
        for info in ctorg_infos:
            cases.append(
                {
                    "path": info["path"],
                    "patient_id": info["patient_id"],
                    "source": "ctorg",
                    "input_type": "nifti",
                }
            )
    else:
        logger.warning(f"CT-ORG目录不存在或未指定: {ctorg_root}")

    if lidc_root and os.path.exists(lidc_root):
        cases.extend(scan_lidc(lidc_root, volume_config, detector_config))
    else:
        logger.warning(f"LIDC-IDRI目录不存在或未指定: {lidc_root}")

    if not cases:
        raise ValueError("未找到可处理的病例")

    return cases


def split_cases(
    cases: List[Dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    balance_sources: bool,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    rng = random.Random(seed)

    by_source: Dict[str, List[Dict]] = {}
    for case in cases:
        by_source.setdefault(case["source"], []).append(case)

    if balance_sources and len(by_source) > 1:
        min_n = min(len(v) for v in by_source.values())
        if min_n == 0:
            raise ValueError("数据来源之一为空，无法均衡分配")
        logger.info(f"按来源均衡采样: 每个来源保留 {min_n} 个病例")
        for source, items in by_source.items():
            rng.shuffle(items)
            by_source[source] = items[:min_n]

    train_cases: List[Dict] = []
    val_cases: List[Dict] = []
    test_cases: List[Dict] = []

    for source, items in by_source.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_cases.extend(items[:n_train])
        val_cases.extend(items[n_train:n_train + n_val])
        test_cases.extend(items[n_train + n_val:])

    rng.shuffle(train_cases)
    rng.shuffle(val_cases)
    rng.shuffle(test_cases)

    return train_cases, val_cases, test_cases


def _jsonify(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def save_volume(volume: np.ndarray, metadata: Dict, out_dir: Path, patient_id: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    vol_path = out_dir / f"{patient_id}.npy"
    np.save(vol_path, volume)

    meta_path = out_dir / f"{patient_id}.json"
    with open(meta_path, "w") as f:
        json.dump(_jsonify(metadata), f, indent=2)

    return vol_path


def hu_to_mu(volume: np.ndarray, hu_air: float = -1000.0, hu_water: float = 0.0) -> np.ndarray:
    """
    Convert HU values to linear attenuation coefficients (mu).
    This makes the projection input non-negative for more physical sinograms.
    """
    mu = (volume - hu_air) / (hu_water - hu_air)
    return np.clip(mu, 0.0, None).astype(np.float32)


def save_2d_slices(
    sinogram_3d: np.ndarray,
    output_dir: Path,
    patient_id: str,
    slice_step: int,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    n_proj, width, height = sinogram_3d.shape
    saved_paths: List[Path] = []

    for z_idx in range(0, height, slice_step):
        sino_2d = sinogram_3d[:, :, z_idx]
        filename = f"{patient_id}_slice{z_idx:04d}_sino2d_{width}.npy"
        out_path = output_dir / filename
        np.save(out_path, sino_2d)
        saved_paths.append(out_path)

    return saved_paths


def visualize_sinogram_slices(
    volume: np.ndarray,
    sinogram_3d: np.ndarray,
    save_path: Path,
    n_slices: int = 5,
) -> None:
    import matplotlib.pyplot as plt

    vol_depth = volume.shape[0]
    sino_height = sinogram_3d.shape[2]
    if vol_depth <= 0 or sino_height <= 0:
        return

    if vol_depth < n_slices:
        vol_indices = np.arange(vol_depth, dtype=int)
    else:
        vol_indices = np.linspace(0, vol_depth - 1, n_slices, dtype=int)

    if vol_depth > 1:
        sino_indices = np.round(vol_indices / (vol_depth - 1) * (sino_height - 1)).astype(int)
    else:
        sino_indices = np.zeros_like(vol_indices)

    fig, axes = plt.subplots(len(vol_indices), 2, figsize=(10, 2.5 * len(vol_indices)))
    if len(vol_indices) == 1:
        axes = [axes]

    for i, (z_vol, z_sino) in enumerate(zip(vol_indices, sino_indices)):
        axes[i][0].imshow(volume[z_vol], cmap="gray")
        axes[i][0].set_title(f"Volume z={z_vol}")
        axes[i][0].axis("off")

        sino_2d = sinogram_3d[:, :, z_sino]
        axes[i][1].imshow(sino_2d, cmap="gray", aspect="auto")
        axes[i][1].set_title(f"Sinogram z={z_sino}")
        axes[i][1].axis("off")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)


def compute_normalization_stats(files: List[Path], output_path: Path) -> Dict:
    if not files:
        logger.warning("未找到文件，跳过归一化统计")
        return {}

    mean, std = compute_global_stats([str(f) for f in files])
    stats = {
        "mean": float(mean),
        "std": float(std),
        "n_samples": len(files),
    }

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"统计量: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    logger.info(f"保存到: {output_path}")

    return stats


def run_full_pipeline(
    ctorg_root: Optional[str],
    lidc_root: Optional[str],
    output_root: str,
    slice_step: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    balance_sources: bool,
    save_volumes: bool,
    save_sino3d: bool,
    save_sino3d_max: int,
    skip_existing: bool,
    sanity_check: bool,
    sanity_slices: int,
    do_split: bool,
) -> Dict:
    logger.info("=" * 70)
    logger.info("Unified preprocessing pipeline")
    logger.info("=" * 70)

    print_dimension_info()

    vol_cfg = get_default_volume_config()
    det_cfg = get_default_detector_config()

    output_root = Path(output_root)
    volumes_dir = output_root / "volumes"
    train_val_dir = output_root / "train_val_sinograms"
    test_dir = output_root / "test_sinograms"
    train_val_3d_dir = output_root / "train_val_sinograms_3d"
    test_3d_dir = output_root / "test_sinograms_3d"
    splits_dir = output_root / "splits"
    sanity_dir = output_root / "sanity_checks"

    cases = build_case_list(ctorg_root, lidc_root, vol_cfg, det_cfg)

    if do_split:
        train_cases, val_cases, test_cases = split_cases(
            cases,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            balance_sources=balance_sources,
        )
    else:
        train_cases, val_cases, test_cases = cases, [], []

    logger.info(
        f"Split sizes (patients): train={len(train_cases)}, val={len(val_cases)}, test={len(test_cases)}"
    )

    processor = UnifiedDataProcessor(vol_cfg, det_cfg)

    # Choose one case per split for sanity visualization
    sanity_targets: Dict[str, Optional[str]] = {"train": None, "val": None, "test": None}
    if sanity_check:
        rng = random.Random(seed)
        if train_cases:
            sanity_targets["train"] = rng.choice(train_cases)["patient_id"]
        if val_cases:
            sanity_targets["val"] = rng.choice(val_cases)["patient_id"]
        if test_cases:
            sanity_targets["test"] = rng.choice(test_cases)["patient_id"]

    split_file_lists: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

    train_val_sources = {c["source"] for c in (train_cases + val_cases)}
    required_sources = set(train_val_sources)

    saved_3d_count: Dict[str, int] = {}

    def _remaining_slots() -> int:
        return max(save_sino3d_max - sum(saved_3d_count.values()), 0)

    def _can_save_3d(split_name: str, source: str) -> bool:
        if split_name == "test":
            return True
        if save_sino3d_max <= 0:
            return False

        remaining = _remaining_slots()
        if remaining <= 0:
            return False

        missing = [s for s in required_sources if saved_3d_count.get(s, 0) == 0]
        if missing:
            if remaining <= len(missing):
                return source in missing
            return True

        return True

    def _mark_saved(source: str) -> None:
        saved_3d_count[source] = saved_3d_count.get(source, 0) + 1

    def process_cases(
        case_list: List[Dict],
        split_name: str,
        out_dir: Path,
        out_3d_dir: Path,
    ) -> None:
        for case in case_list:
            patient_id = case["patient_id"]
            source = case["source"]

            if skip_existing:
                existing = list(out_dir.glob(f"{patient_id}_slice*_sino2d_*.npy"))
                if existing:
                    split_file_lists[split_name].extend([p.name for p in existing])
                    logger.info(f"跳过已处理: {patient_id}")
                    continue

            try:
                volume, metadata = processor.load_volume(Path(case["path"]))
                spacing = metadata["spacing"]
                sz = spacing[0]
                min_t, max_t = vol_cfg.valid_slice_thickness_range
                if not (min_t <= sz <= max_t):
                    logger.warning(
                        f"{patient_id} 切片厚度 {sz:.2f}mm 超出范围 [{min_t}, {max_t}]"
                    )
                    continue
                if volume.shape[0] < vol_cfg.min_valid_slices:
                    logger.warning(
                        f"{patient_id} 切片数不足: {volume.shape[0]} < {vol_cfg.min_valid_slices}"
                    )
                    continue

                volume = processor.standardize_volume(volume, spacing)
                volume = hu_to_mu(volume)

                if save_volumes:
                    meta = dict(metadata)
                    meta.update({"source": case["source"], "input_path": case["path"]})
                    save_volume(volume, meta, volumes_dir, patient_id)

                sinogram_3d = processor.generate_sinogram(volume)

                if save_sino3d and _can_save_3d(split_name, source):
                    out_3d_dir.mkdir(parents=True, exist_ok=True)
                    sino_name = f"{patient_id}_sinogram_{det_cfg.detector_width}x{det_cfg.detector_height}.npy"
                    np.save(out_3d_dir / sino_name, sinogram_3d)
                    if split_name != "test":
                        _mark_saved(source)

                if sanity_check and sanity_targets.get(split_name) == patient_id:
                    vis_path = sanity_dir / f"{split_name}_{patient_id}.png"
                    visualize_sinogram_slices(volume, sinogram_3d, vis_path, n_slices=sanity_slices)

                slice_paths = save_2d_slices(sinogram_3d, out_dir, patient_id, slice_step)
                split_file_lists[split_name].extend([p.name for p in slice_paths])

                logger.info(f"完成: {patient_id} -> {len(slice_paths)} slices")

            except Exception as exc:
                logger.error(f"处理失败 {patient_id}: {exc}")

    process_cases(train_cases, "train", train_val_dir, train_val_3d_dir)
    process_cases(val_cases, "val", train_val_dir, train_val_3d_dir)
    process_cases(test_cases, "test", test_dir, test_3d_dir)

    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ["train", "val", "test"]:
        split_path = splits_dir / f"{split_name}.txt"
        with open(split_path, "w") as f:
            for name in sorted(set(split_file_lists[split_name])):
                f.write(name + "\n")

    stats_path = output_root / "normalization_stats.json"
    train_files = [train_val_dir / name for name in split_file_lists["train"]]
    stats = compute_normalization_stats(train_files, stats_path)

    report = {
        "volume_config": {
            "target_shape": vol_cfg.target_shape,
            "target_spacing": vol_cfg.target_spacing,
        },
        "detector_config": {
            "width": det_cfg.detector_width,
            "height": det_cfg.detector_height,
            "n_projections": det_cfg.n_projections,
        },
        "splits": {
            "train_patients": len(train_cases),
            "val_patients": len(val_cases),
            "test_patients": len(test_cases),
        },
        "stats": stats,
        "outputs": {
            "train_val_2d": str(train_val_dir),
            "test_2d": str(test_dir),
            "train_val_3d": str(train_val_3d_dir) if save_sino3d else None,
            "test_3d": str(test_3d_dir) if save_sino3d else None,
        },
    }

    report_path = output_root / "preprocessing_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"报告保存至: {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified CT preprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", choices=["ctorg", "lidc", "all"], help="数据集类型")
    parser.add_argument("--data_root", type=str, help="单数据集模式的数据目录")
    parser.add_argument("--ctorg_root", type=str, help="CT-ORG数据目录")
    parser.add_argument("--lidc_root", type=str, help="LIDC-IDRI数据目录")
    parser.add_argument("--output_root", type=str, default="./data/processed", help="输出目录")
    parser.add_argument("--slice_step", type=int, default=4, help="2D切片步长")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--no_split", action="store_true", help="不进行划分")
    parser.add_argument("--no_balance_sources", action="store_true", help="不均衡来源样本数")
    parser.add_argument("--no_save_volumes", action="store_true", help="不保存标准化体积")
    parser.add_argument("--save_sino3d", action="store_true", help="保存3D sinogram")
    parser.add_argument(
        "--save_sino3d_max",
        type=int,
        default=10,
        help="训练/验证总共保存3D sinogram数量上限",
    )
    parser.add_argument("--no_skip_existing", action="store_true", help="不跳过已处理样本")
    parser.add_argument("--no_sanity", action="store_true", help="不生成可视化检查")
    parser.add_argument("--sanity_slices", type=int, default=5, help="可视化切片数")
    parser.add_argument("--print_dims", action="store_true", help="打印维度说明")

    args = parser.parse_args()

    if args.print_dims:
        print_dimension_info()
        return

    ctorg_root = args.ctorg_root
    lidc_root = args.lidc_root
    if args.dataset == "ctorg":
        ctorg_root = args.data_root
    elif args.dataset == "lidc":
        lidc_root = args.data_root

    run_full_pipeline(
        ctorg_root=ctorg_root,
        lidc_root=lidc_root,
        output_root=args.output_root,
        slice_step=args.slice_step,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        balance_sources=not args.no_balance_sources,
        save_volumes=not args.no_save_volumes,
        save_sino3d=args.save_sino3d,
        save_sino3d_max=args.save_sino3d_max,
        skip_existing=not args.no_skip_existing,
        sanity_check=not args.no_sanity,
        sanity_slices=args.sanity_slices,
        do_split=not args.no_split,
    )


if __name__ == "__main__":
    main()
