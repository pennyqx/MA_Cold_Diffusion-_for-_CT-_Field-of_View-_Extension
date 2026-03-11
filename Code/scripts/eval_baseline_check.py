#!/usr/bin/env python3
"""
Baseline evaluation with fixed truncation and clear FOV visualization.

Workflow:
1) Load test 3D sinograms
2) Run baseline models slice-wise to recover 3D sinograms
3) FBP reconstruct GT and predictions
4) Compute metrics (sinogram + FBP image)
5) Save FOV-cropped visualization grids (sinogram + recon rows)
"""

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation import compute_sinogram_metrics, compute_volume_metrics  # noqa: E402
from utils.pyronn_utils import PyRoNNProjector, CTGeometry  # noqa: E402
from evaluate_fbp import load_model, predict_sinogram_3d  # noqa: E402
from utils.config import resolve_keep_center  # noqa: E402
from utils.wce import water_cylinder_extrapolate_2d_batch  # noqa: E402


def _keep_center_to_level(width: int, keep_center: int) -> int:
    keep_center = max(0, min(keep_center, width))
    return (width - keep_center) // 2


def _build_mask_1d(width: int, level: int) -> np.ndarray:
    if level <= 0:
        return np.ones((width,), dtype=np.float32)

    left = level
    right = width - level
    if right <= left:
        return np.zeros((width,), dtype=np.float32)

    mask = np.zeros((width,), dtype=np.float32)
    mask[left:right] = 1.0

    return mask


def _build_truncated_sinogram(sino_3d: np.ndarray, level: int) -> np.ndarray:
    mask_1d = _build_mask_1d(sino_3d.shape[1], level)
    mask = mask_1d[np.newaxis, :, np.newaxis]
    return (sino_3d * mask).astype(np.float32)


def _water_cylinder_extrapolate_sino3d(trunc_sino: np.ndarray, level: int) -> np.ndarray:
    """
    Apply the shared torch WCE implementation to a 3D sinogram.
    Input/Output shape: (n_proj, width, height).
    """
    n_proj, width, height = trunc_sino.shape
    if level <= 0:
        return trunc_sino.astype(np.float32)

    left = level
    right = width - level
    if right <= left:
        return trunc_sino.astype(np.float32)

    trunc_4d = np.transpose(trunc_sino.astype(np.float32), (2, 0, 1))[:, None, :, :]  # [B=height, 1, n_proj, width]
    mask = np.zeros_like(trunc_4d, dtype=np.float32)
    mask[:, :, :, left:right] = 1.0

    x_t = torch.from_numpy(trunc_4d)
    m_t = torch.from_numpy(mask)
    out_t = water_cylinder_extrapolate_2d_batch(x_t, m_t)
    out_4d = out_t.numpy().astype(np.float32)
    out_3d = np.transpose(out_4d[:, 0, :, :], (1, 2, 0))
    return out_3d.astype(np.float32)


def _enforce_center_consistency(pred_sino: np.ndarray, trunc_sino: np.ndarray, level: int) -> np.ndarray:
    if level <= 0:
        return pred_sino.astype(np.float32)
    width = pred_sino.shape[1]
    right = width - level
    out = pred_sino.copy().astype(np.float32)
    out[:, level:right, :] = trunc_sino[:, level:right, :]
    return out.astype(np.float32)


def _smooth_sinogram_width(sino_3d: np.ndarray, kernel_width: int) -> np.ndarray:
    """Smooth along detector width axis using a Hann window."""
    if kernel_width is None or kernel_width <= 1:
        return sino_3d
    k = int(kernel_width)
    if k % 2 == 0:
        k += 1
    window = np.hanning(k).astype(np.float32)
    window /= max(window.sum(), 1e-8)
    pad = k // 2

    def _conv1d(x: np.ndarray) -> np.ndarray:
        x_pad = np.pad(x, (pad, pad), mode="reflect")
        return np.convolve(x_pad, window, mode="valid")

    return np.apply_along_axis(_conv1d, 1, sino_3d).astype(np.float32)


def _infer_source(name: str) -> str:
    name = name.lower()
    if "ctorg" in name:
        return "ctorg"
    if "lidc" in name:
        return "lidc"
    return "unknown"


def _select_cases_balanced(
    files: List[Path],
    fraction: float,
    max_cases: int,
    seed: int
) -> Tuple[List[Path], Dict[str, int]]:
    if not files:
        return [], {"ctorg": 0, "lidc": 0, "unknown": 0}

    rng = random.Random(seed)
    groups: Dict[str, List[Path]] = {"ctorg": [], "lidc": [], "unknown": []}
    for f in files:
        groups[_infer_source(f.name)].append(f)

    total = len(files)
    target = max(1, int(math.floor(total * fraction)))
    if max_cases is not None:
        target = min(target, max_cases)

    ctorg = groups["ctorg"]
    lidc = groups["lidc"]
    unknown = groups["unknown"]

    if ctorg and lidc:
        per_source = max(1, target // 2)
        rng.shuffle(ctorg)
        rng.shuffle(lidc)
        selected = ctorg[:per_source] + lidc[:per_source]
    else:
        all_files = ctorg + lidc
        rng.shuffle(all_files)
        selected = all_files[:target]

    remaining = max(0, target - len(selected))
    if remaining > 0 and unknown:
        rng.shuffle(unknown)
        selected += unknown[:remaining]

    counts = {
        "ctorg": sum(1 for f in selected if _infer_source(f.name) == "ctorg"),
        "lidc": sum(1 for f in selected if _infer_source(f.name) == "lidc"),
        "unknown": sum(1 for f in selected if _infer_source(f.name) == "unknown"),
    }
    return selected, counts


def _latest_checkpoint(ckpt_dir: Path, max_step: int | None = None) -> Path:
    ckpts = sorted(ckpt_dir.glob("ckpt_step*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    if max_step is None:
        return ckpts[-1]
    filtered = []
    for ckpt in ckpts:
        stem = ckpt.stem
        if "ckpt_step" not in stem:
            continue
        try:
            step = int(stem.split("ckpt_step", 1)[-1])
        except ValueError:
            continue
        if step <= max_step:
            filtered.append(ckpt)
    if not filtered:
        raise FileNotFoundError(f"No checkpoints <= {max_step} in {ckpt_dir}")
    return filtered[-1]


def _load_checkpoint(exp_dir: Path, max_step: int | None = None) -> Tuple[Path, Path]:
    ckpt_dir = exp_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Missing checkpoints: {ckpt_dir}")
    if max_step is None:
        best = ckpt_dir / "best.pt"
        if best.exists():
            return best, exp_dir / "config.yaml"
        return _latest_checkpoint(ckpt_dir), exp_dir / "config.yaml"
    return _latest_checkpoint(ckpt_dir, max_step=max_step), exp_dir / "config.yaml"


def _normalize_row(images: List[np.ndarray]) -> List[np.ndarray]:
    stack = np.stack(images, axis=0)
    vmin = float(stack.min())
    vmax = float(stack.max())
    if vmax - vmin < 1e-8:
        return [np.zeros_like(img) for img in images]
    return [(img - vmin) / (vmax - vmin) for img in images]


def _normalize_img(img: np.ndarray, p_low: float, p_high: float, exclude_zeros: bool) -> np.ndarray:
    img = img.astype(np.float32)
    if exclude_zeros:
        valid = img[np.abs(img) > 1e-8]
    else:
        valid = img.reshape(-1)
    if valid.size == 0:
        return np.zeros_like(img, dtype=np.float32)
    vmin, vmax = np.percentile(valid, [p_low, p_high])
    if vmax - vmin < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0)


def _compute_fov_bbox(
    gt_recon: np.ndarray,
    profile_frac: float = 0.2,
    min_threshold: float = 0.05,
    margin: int = 4,
    border_frac: float = 0.1,
) -> Tuple[int, int, int, int]:
    """Compute a bounding box from GT reconstruction using background-subtracted profiles."""
    h, w = gt_recon.shape
    bw = max(1, int(round(w * border_frac)))
    bh = max(1, int(round(h * border_frac)))
    top = gt_recon[:bh, :]
    bottom = gt_recon[-bh:, :]
    left = gt_recon[:, :bw]
    right = gt_recon[:, -bw:]
    border_vals = np.concatenate([top.ravel(), bottom.ravel(), left.ravel(), right.ravel()])
    bg = float(np.median(border_vals))

    delta = np.abs(gt_recon.astype(np.float32) - bg)
    norm = _normalize_img(delta, 1.0, 99.0, exclude_zeros=True)
    row_profile = norm.mean(axis=1)
    col_profile = norm.mean(axis=0)
    row_thresh = max(min_threshold, profile_frac * float(row_profile.max()))
    col_thresh = max(min_threshold, profile_frac * float(col_profile.max()))
    rows = np.where(row_profile > row_thresh)[0]
    cols = np.where(col_profile > col_thresh)[0]
    if rows.size == 0 or cols.size == 0:
        return (0, h, 0, w)
    r0, r1 = rows[0], rows[-1]
    c0, c1 = cols[0], cols[-1]
    r0 = max(0, r0 - margin)
    c0 = max(0, c0 - margin)
    r1 = min(h - 1, r1 + margin)
    c1 = min(w - 1, c1 + margin)
    return (r0, r1 + 1, c0, c1 + 1)


def _center_crop_bbox(shape: Tuple[int, int], crop_frac: float) -> Tuple[int, int, int, int]:
    h, w = shape
    crop_frac = max(0.2, min(1.0, crop_frac))
    ch = int(round(h * crop_frac))
    cw = int(round(w * crop_frac))
    ch = max(1, min(h, ch))
    cw = max(1, min(w, cw))
    r0 = (h - ch) // 2
    c0 = (w - cw) // 2
    return (r0, r0 + ch, c0, c0 + cw)


def _render_grid(save_path: Path, rows: List[List[np.ndarray]], col_labels: List[str], row_labels: List[str], dpi: int) -> None:
    n_rows = len(rows)
    n_cols = len(col_labels)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.2), dpi=dpi)
    if n_rows == 1:
        axs = np.expand_dims(axs, axis=0)
    if n_cols == 1:
        axs = np.expand_dims(axs, axis=1)

    for r in range(n_rows):
        for c in range(n_cols):
            axs[r, c].imshow(rows[r][c], cmap="gray", vmin=0.0, vmax=1.0)
            axs[r, c].axis("off")

    for col, label in enumerate(col_labels):
        axs[0, col].set_title(label, fontsize=10, pad=6)
    for row, label in enumerate(row_labels):
        axs[row, 0].set_ylabel(label, fontsize=9, rotation=0, labelpad=40, va="center")

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def _compute_sino_metrics(pred_sino: np.ndarray, gt_sino: np.ndarray, level: int) -> Dict[str, float]:
    data_range = float(gt_sino.max() - gt_sino.min())
    if data_range <= 0:
        data_range = 1.0

    metrics = compute_sinogram_metrics(pred_sino, gt_sino, data_range=data_range)

    if level <= 0:
        roi_metrics = {'ROI_MAE': 0.0, 'ROI_PSNR': 0.0, 'ROI_SSIM': 0.0}
    else:
        pred_roi = np.concatenate([pred_sino[:, :level, :], pred_sino[:, -level:, :]], axis=1)
        gt_roi = np.concatenate([gt_sino[:, :level, :], gt_sino[:, -level:, :]], axis=1)
        pred_flat = pred_roi.reshape(-1)
        gt_flat = gt_roi.reshape(-1)
        eps = 1e-8

        roi_mae = float(np.mean(np.abs(pred_flat - gt_flat)))
        roi_mse = float(np.mean((pred_flat - gt_flat) ** 2))
        roi_psnr = 10.0 * np.log10((data_range ** 2) / (roi_mse + eps))

        mu_x = float(np.mean(pred_flat))
        mu_y = float(np.mean(gt_flat))
        var_x = float(np.mean(pred_flat ** 2) - mu_x ** 2)
        var_y = float(np.mean(gt_flat ** 2) - mu_y ** 2)
        cov_xy = float(np.mean(pred_flat * gt_flat) - mu_x * mu_y)
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        roi_ssim = ((2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)) / \
            ((mu_x ** 2 + mu_y ** 2 + C1) * (var_x + var_y + C2) + eps)
        roi_metrics = {
            'ROI_MAE': float(roi_mae),
            'ROI_PSNR': float(roi_psnr),
            'ROI_SSIM': float(roi_ssim)
        }

    metrics.update(roi_metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline eval with FOV visualization")
    parser.add_argument("--test_sino3d_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--keep_center_eval", type=int, default=320)
    parser.add_argument("--truncation_ratio_eval", type=float, default=None,
                        help="Total removed detector ratio for eval (0.25 -> keep 75%% center)")
    parser.add_argument("--fbp_smooth_width", type=int, default=0,
                        help="Optional Hann smoothing width along detector for FBP visualization")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--slice_indices", type=int, nargs='+', default=None)
    parser.add_argument("--viz_percentiles", type=float, nargs=2, default=[1.0, 99.0])
    parser.add_argument("--viz_exclude_zeros", action="store_true")
    parser.add_argument("--fov_profile_frac", type=float, default=0.2)
    parser.add_argument("--fov_min_threshold", type=float, default=0.05)
    parser.add_argument("--fov_margin", type=int, default=4)
    parser.add_argument("--fov_border_frac", type=float, default=0.1)
    parser.add_argument("--fov_center_crop_frac", type=float, default=0.8,
                        help="Center-crop fraction for FOV (stable). Set to 0 to disable.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no_metrics", action="store_true")
    parser.add_argument("--no_viz", action="store_true")
    parser.add_argument("--no_truncation_for_cold_diffusion", action="store_true")

    parser.add_argument("--cold_dir", type=str, default=None, help="Baseline cold diffusion dir")
    parser.add_argument(
        "--cold_sampling_routine",
        type=str,
        choices=["default", "x0_step_down"],
        default=None,
        help="Override cold diffusion sampling routine at eval time using the same checkpoint.",
    )
    parser.add_argument("--ddpm_dir", type=str, default=None, help="Baseline DDPM dir")
    parser.add_argument("--unet_dir", type=str, default=None, help="Baseline UNet inpaint dir")
    parser.add_argument("--ddpm_max_step", type=int, default=None,
                        help="Use latest DDPM checkpoint at or before this step")
    parser.add_argument("--ddpm_sampling", type=str, default="ddpm",
                        choices=["ddpm", "ddim"], help="Sampling method for DDPM")
    parser.add_argument("--ddpm_steps", type=int, default=None,
                        help="DDIM steps for DDPM evaluation")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    models: Dict[str, Dict] = {}
    model_names: List[str] = []
    if args.cold_dir:
        ckpt, cfg = _load_checkpoint(Path(args.cold_dir))
        model, config, stats = load_model(str(ckpt), str(cfg), device)
        if args.cold_sampling_routine is not None:
            # Inference-only override: fixed weights, different sampling path.
            model.sampling_routine = args.cold_sampling_routine
            if hasattr(config, "diffusion"):
                config.diffusion.sampling_routine = args.cold_sampling_routine
            print(f"[Override] cold_diffusion.sampling_routine={args.cold_sampling_routine}")
        models["cold_diffusion"] = {"model": model, "cfg": config, "stats": stats}
        model_names.append("cold_diffusion")
    if args.ddpm_dir:
        ckpt, cfg = _load_checkpoint(Path(args.ddpm_dir), max_step=args.ddpm_max_step)
        model, config, stats = load_model(str(ckpt), str(cfg), device)
        models["ddpm"] = {"model": model, "cfg": config, "stats": stats}
        model_names.append("ddpm")
    if args.unet_dir:
        ckpt, cfg = _load_checkpoint(Path(args.unet_dir))
        model, config, stats = load_model(str(ckpt), str(cfg), device)
        models["unet_inpaint"] = {"model": model, "cfg": config, "stats": stats}
        model_names.append("unet_inpaint")

    if not model_names:
        raise ValueError("No models specified. Use --cold_dir/--ddpm_dir/--unet_dir.")

    test_files = sorted(Path(args.test_sino3d_dir).glob("*.npy"))
    test_files, source_counts = _select_cases_balanced(
        test_files, args.fraction, args.max_cases, args.sample_seed
    )
    if not test_files:
        raise ValueError(f"No .npy files found in {args.test_sino3d_dir}")

    case_list_path = output_dir / "eval_cases.txt"
    case_list_path.write_text("\n".join([str(p) for p in test_files]) + "\n")
    with open(output_dir / "eval_source_counts.json", "w") as f:
        json.dump(source_counts, f, indent=2)

    first = np.load(test_files[0])
    if first.ndim == 4:
        first = first[0]
    if first.ndim != 3:
        raise ValueError(f"Expected 3D sinogram, got shape {first.shape}")

    n_proj, width, height = first.shape
    geom = CTGeometry(
        volume_shape=(128, 512, 512),
        volume_spacing=(1.5, 1.0, 1.0),
        detector_width=width,
        detector_height=height,
        n_projections=n_proj
    )
    projector = PyRoNNProjector(geom)

    slice_indices = [height // 2] if not args.slice_indices else [
        min(max(0, idx), height - 1) for idx in args.slice_indices
    ]
    display_names = {
        "cold_diffusion": "ColdDiff",
        "ddpm": "DDPM",
        "unet_inpaint": "UNet",
    }
    col_labels = ["GT", "WCE-only"]
    for n in model_names:
        d = display_names.get(n, n)
        col_labels.append(d)
    vol_depth = geom.volume_shape[0]

    keep_center_eval = int(args.keep_center_eval)
    if args.truncation_ratio_eval is not None:
        ratio = max(0.0, min(0.99, float(args.truncation_ratio_eval)))
        keep_center_eval = int(round((1.0 - ratio) * float(width)))
        keep_center_eval = max(1, min(width, keep_center_eval))
    level = _keep_center_to_level(width, keep_center_eval)
    fbp_results: Dict[str, Dict] = {}
    sino_results: Dict[str, Dict] = {}
    fbp_results["truncated"] = {"per_case": {}}
    sino_results["truncated"] = {"per_case": {}}
    fbp_results["wce_only"] = {"per_case": {}}
    sino_results["wce_only"] = {"per_case": {}}
    for name in model_names:
        fbp_results[name] = {"per_case": {}}
        sino_results[name] = {"per_case": {}}

    for fp in tqdm(test_files, desc="Cases"):
        case_t0 = time.time()
        sino = np.load(fp)
        if sino.ndim == 4:
            sino = sino[0]
        if sino.ndim != 3:
            raise ValueError(f"Unexpected sinogram shape {sino.shape} in {fp}")

        case_name = Path(fp).stem
        gt_recon = projector.fbp_reconstruct(sino.astype(np.float32))
        trunc_sino = _build_truncated_sinogram(sino, level)

        wce_sino = _water_cylinder_extrapolate_sino3d(trunc_sino, level)
        preds: Dict[str, np.ndarray] = {}
        for name in model_names:
            model_t0 = time.time()
            cfg = models[name]["cfg"]
            stats = models[name]["stats"]
            pred = predict_sinogram_3d(
                model=models[name]["model"],
                model_type=name if name != "cold_diffusion" else "cold_diffusion",
                sinogram_3d=sino,
                data_mean=stats["mean"],
                data_std=stats["std"],
                truncation_level=level,
                batch_size=args.batch_size,
                device=device,
                use_truncation_for_cold_diffusion=(
                    False if args.no_truncation_for_cold_diffusion and name == "cold_diffusion" else True
                ),
                timesteps=cfg.diffusion.timesteps,
                keep_center=resolve_keep_center(cfg.diffusion, width),
                ddpm_sampling=args.ddpm_sampling if name == "ddpm" else "ddpm",
                ddpm_steps=args.ddpm_steps if name == "ddpm" else None
            )
            pred = _enforce_center_consistency(pred, trunc_sino, level)
            preds[name] = pred
            print(f"[Case {case_name}] {name} done in {(time.time()-model_t0)/60.0:.2f} min")

        trunc_recon = projector.fbp_reconstruct(trunc_sino.astype(np.float32))
        wce_recon = projector.fbp_reconstruct(wce_sino.astype(np.float32))
        recons: Dict[str, np.ndarray] = {
            name: projector.fbp_reconstruct(preds[name].astype(np.float32))
            for name in model_names
        }

        if not args.no_metrics:
            fbp_results["truncated"]["per_case"][case_name] = compute_volume_metrics(
                trunc_recon, gt_recon
            )
            sino_results["truncated"]["per_case"][case_name] = _compute_sino_metrics(
                trunc_sino, sino, level
            )
            fbp_results["wce_only"]["per_case"][case_name] = compute_volume_metrics(
                wce_recon, gt_recon
            )
            sino_results["wce_only"]["per_case"][case_name] = _compute_sino_metrics(
                wce_sino, sino, level
            )
            for name in model_names:
                fbp_results[name]["per_case"][case_name] = compute_volume_metrics(
                    recons[name], gt_recon
                )
                sino_results[name]["per_case"][case_name] = _compute_sino_metrics(
                    preds[name], sino, level
                )

        if args.no_viz:
            continue

        for slice_idx in slice_indices:
            sino_row = [sino[:, :, slice_idx], wce_sino[:, :, slice_idx]]
            for name in model_names:
                sino_row.append(preds[name][:, :, slice_idx])

            recon_idx = int(round((slice_idx / max(height - 1, 1)) * (vol_depth - 1)))
            # Optional smoothing for FBP visualization only
            if args.fbp_smooth_width and args.fbp_smooth_width > 1:
                trunc_vis_sino = _smooth_sinogram_width(trunc_sino, args.fbp_smooth_width)
                trunc_vis_recon = projector.fbp_reconstruct(trunc_vis_sino.astype(np.float32))
                wce_vis_sino = _smooth_sinogram_width(wce_sino, args.fbp_smooth_width)
                wce_vis_recon = projector.fbp_reconstruct(wce_vis_sino.astype(np.float32))
                recon_row = [gt_recon[recon_idx], wce_vis_recon[recon_idx]]
                for name in model_names:
                    pred_vis_sino = _smooth_sinogram_width(preds[name], args.fbp_smooth_width)
                    pred_vis_recon = projector.fbp_reconstruct(pred_vis_sino.astype(np.float32))
                    recon_row.append(pred_vis_recon[recon_idx])
            else:
                recon_row = [gt_recon[recon_idx], wce_recon[recon_idx]]
                for name in model_names:
                    recon_row.append(recons[name][recon_idx])

            if args.fov_center_crop_frac and args.fov_center_crop_frac > 0:
                r0, r1, c0, c1 = _center_crop_bbox(
                    gt_recon[recon_idx].shape, args.fov_center_crop_frac
                )
            else:
                r0, r1, c0, c1 = _compute_fov_bbox(
                    gt_recon[recon_idx],
                    profile_frac=args.fov_profile_frac,
                    min_threshold=args.fov_min_threshold,
                    margin=args.fov_margin,
                    border_frac=args.fov_border_frac,
                )
            recon_row = [img[r0:r1, c0:c1] for img in recon_row]

            sino_row = _normalize_row(sino_row)
            recon_row = [
                _normalize_img(img, args.viz_percentiles[0], args.viz_percentiles[1], args.viz_exclude_zeros)
                for img in recon_row
            ]

            rows = [sino_row, recon_row]
            row_labels = [f"keep_center={keep_center_eval}\nsinogram",
                          f"keep_center={keep_center_eval}\nrecon"]

            save_path = output_dir / f"{case_name}_kc{keep_center_eval}_slice{slice_idx}.png"
            _render_grid(save_path, rows, col_labels, row_labels, dpi=args.dpi)

        print(f"[Case {case_name}] Completed in {(time.time()-case_t0)/60.0:.2f} min")

    if not args.no_metrics:
        for result in (fbp_results, sino_results):
            for name, payload in result.items():
                per_case = payload.get("per_case", {})
                if not per_case:
                    payload["mean"] = {}
                    continue
                metric_keys = list(next(iter(per_case.values())).keys())
                payload["mean"] = {
                    key: float(np.mean([v.get(key, 0.0) for v in per_case.values()]))
                    for key in metric_keys
                }

        fbp_path = output_dir / "fbp_metrics.json"
        with open(fbp_path, "w") as f:
            json.dump(fbp_results, f, indent=2)
        print(f"[Save] Metrics saved to {fbp_path}")

        sino_path = output_dir / "sino_metrics.json"
        with open(sino_path, "w") as f:
            json.dump(sino_results, f, indent=2)
        print(f"[Save] Metrics saved to {sino_path}")


if __name__ == "__main__":
    main()
