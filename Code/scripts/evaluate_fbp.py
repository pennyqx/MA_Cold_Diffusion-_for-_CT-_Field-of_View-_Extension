#!/usr/bin/env python3
"""
Evaluate models in image domain via FBP backprojection.

Workflow:
1) Load 3D sinogram volumes (n_proj, width, height)
2) Run model slice-wise to get recovered 3D sinogram
3) Backproject (FBP) both GT and recovered sinograms
4) Report image-domain metrics (PSNR, SSIM)

Note: Requires test 3D sinograms (e.g., output_root/test_sinograms_3d).
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models import UNet, ColdDiffusion2D, DDPM2D, UNetInpaint
from models.cold_diffusion import build_cold_diffusion
from models.ddpm import build_ddpm
from models.unet_inpaint import build_unet_inpaint
from evaluation import compute_volume_metrics
from utils.config import load_config, resolve_keep_center
from utils.pyronn_utils import PyRoNNProjector, CTGeometry


def load_model(checkpoint_path: str, config_path: str = None, device: torch.device = None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_path = Path(checkpoint_path)

    # Load config
    if config_path is None:
        config_path = checkpoint_path.parent.parent / 'config.yaml'
    else:
        config_path = Path(config_path)

    if config_path.exists():
        config = load_config(config_path)
    else:
        print(f"[Warning] Config not found at {config_path}, using defaults")
        from utils.config import Config
        config = Config()

    # Load checkpoint
    print(f"[Load] Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Build model
    model_type = config.model.type.lower()
    image_size = tuple(config.data.image_size)
    keep_center = resolve_keep_center(config.diffusion, image_size[1])
    use_wce_input = bool(getattr(config.diffusion, "use_wce_input", False))

    if model_type == 'cold_diffusion':
        model = build_cold_diffusion(
            image_size=image_size,
            dim=config.model.dim,
            dim_mults=tuple(config.model.dim_mults),
            timesteps=config.diffusion.timesteps,
            keep_center=keep_center,
            sampling_routine=config.diffusion.sampling_routine,
            use_wce_input=use_wce_input
        )
    elif model_type == 'ddpm':
        model = build_ddpm(
            image_size=image_size,
            dim=config.model.dim,
            dim_mults=tuple(config.model.dim_mults),
            timesteps=config.diffusion.timesteps,
            beta_schedule=config.diffusion.beta_schedule,
            loss_type=config.diffusion.loss_type,
            keep_center=keep_center,
            cond_channels=2,
            use_wce_input=use_wce_input
        )
    elif model_type == 'unet_inpaint':
        model = build_unet_inpaint(
            image_size=image_size,
            dim=config.model.dim,
            dim_mults=tuple(config.model.dim_mults),
            keep_center=keep_center,
            use_wce_input=use_wce_input
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load weights (prefer EMA)
    if 'ema' in ckpt:
        model.load_state_dict(ckpt['ema'])
        print("[Load] Loaded EMA weights")
    else:
        model.load_state_dict(ckpt['model'])
        print("[Load] Loaded model weights")

    model = model.to(device)
    model.eval()

    # Data stats
    data_stats = {
        'mean': ckpt.get('data_mean', 0.0),
        'std': ckpt.get('data_std', 1.0)
    }
    print(f"[Load] keep_center={keep_center}, use_wce_input={use_wce_input}")

    return model, config, data_stats


def _build_mask_1d(width: int, level: int, device: torch.device) -> torch.Tensor:
    if level <= 0:
        return torch.ones((width,), device=device)

    left = level
    right = width - level
    if right <= left:
        return torch.zeros((width,), device=device)

    mask = torch.zeros((width,), device=device)
    mask[left:right] = 1.0

    return mask


def _build_mask(
    batch: int,
    n_proj: int,
    width: int,
    level: int,
    device: torch.device
) -> torch.Tensor:
    mask_1d = _build_mask_1d(width, level, device)
    return mask_1d.view(1, 1, 1, width).repeat(batch, 1, n_proj, 1)


def _truncation_to_t(level: int, width: int, timesteps: int, keep_center: int) -> int:
    if level <= 0:
        return 0
    max_n = max((width - keep_center) // 2, 1)
    level = min(level, max_n)
    t = int(round((level * timesteps) / max(width // 2, 1)))
    return max(1, min(timesteps, t))


@torch.no_grad()
def predict_sinogram_3d(
    model,
    model_type: str,
    sinogram_3d: np.ndarray,
    data_mean: float,
    data_std: float,
    truncation_level: int,
    batch_size: int,
    device: torch.device,
    use_truncation_for_cold_diffusion: bool,
    timesteps: int,
    keep_center: int,
    ddpm_sampling: str = "ddpm",
    ddpm_steps: int = None
) -> np.ndarray:
    """
    Run slice-wise inference to recover a 3D sinogram.

    Args:
        sinogram_3d: (n_proj, width, height)
    """
    n_proj, width, height = sinogram_3d.shape
    pred = np.zeros_like(sinogram_3d, dtype=np.float32)

    for start in range(0, height, batch_size):
        end = min(start + batch_size, height)
        # (n_proj, width, H) -> (H, n_proj, width)
        batch_np = np.transpose(sinogram_3d[:, :, start:end], (2, 0, 1)).astype(np.float32)
        batch_np = (batch_np - data_mean) / (data_std + 1e-6)
        batch = torch.from_numpy(batch_np).unsqueeze(1).to(device)  # [B, 1, n_proj, width]

        if model_type == 'cold_diffusion':
            if use_truncation_for_cold_diffusion and truncation_level > 0:
                mask = _build_mask(batch.size(0), n_proj, width, truncation_level, device)
                truncated = batch * mask
                t_start = _truncation_to_t(truncation_level, width, timesteps, keep_center)
                pred_t = model.sample_from_truncated(truncated, t_start)
            else:
                _, _, pred_t = model.sample(batch)
        elif model_type == 'unet_inpaint':
            mask = _build_mask(batch.size(0), n_proj, width, truncation_level, device)
            truncated = batch * mask
            pred_t = model.inpaint(truncated, mask)
        elif model_type == 'ddpm':
            mask = _build_mask(batch.size(0), n_proj, width, truncation_level, device)
            truncated = batch * mask
            if ddpm_sampling == "ddim":
                steps = ddpm_steps if ddpm_steps is not None else timesteps
                pred_t = model.sample_conditional_ddim(truncated, mask, steps=steps, eta=0.0)
            else:
                pred_t = model.sample_conditional(truncated, mask)
        else:
            pred_t = batch

        pred_np = pred_t.squeeze(1).cpu().numpy()
        pred_np = pred_np * (data_std + 1e-6) + data_mean
        pred[:, :, start:end] = np.transpose(pred_np, (1, 2, 0))

    return pred


def main() -> None:
    parser = argparse.ArgumentParser(description="FBP-based image-domain evaluation")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--test_sino3d_dir', type=str, required=True,
                        help='Directory containing 3D sinograms (.npy)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for metrics')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (number of slices per step)')
    parser.add_argument('--truncation_levels', type=int, nargs='+',
                        default=[50, 100, 150, 200],
                        help='Truncation levels to evaluate')
    parser.add_argument('--truncation_ratio', type=float, default=None,
                        help='If set, use ratio of detector width as truncation per side')
    parser.add_argument('--use_truncation_for_cold_diffusion', action='store_true',
                        help='Use sample_from_truncated for cold diffusion')
    parser.add_argument('--max_cases', type=int, default=None,
                        help='Optional cap on number of cases to evaluate')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] Using {device}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, config, data_stats = load_model(args.checkpoint, args.config, device)
    model_type = config.model.type.lower()
    print(f"[Model] Type: {model_type}")
    print(f"[Data] Mean: {data_stats['mean']:.4f}, Std: {data_stats['std']:.4f}")

    # Load 3D sinograms
    test_files = sorted(Path(args.test_sino3d_dir).glob("*.npy"))
    if args.max_cases:
        test_files = test_files[:args.max_cases]
    if not test_files:
        raise ValueError(f"No .npy files found in {args.test_sino3d_dir}")

    # Infer geometry from first sinogram
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

    results: Dict[str, Dict] = {}

    levels = args.truncation_levels
    if args.truncation_ratio is not None:
        levels = [None]

    for level in levels:
        if args.truncation_ratio is not None:
            print(f"\n[Eval] Truncation ratio: {args.truncation_ratio}")
        else:
            print(f"\n[Eval] Truncation level: {level}")
        per_case: Dict[str, Dict[str, float]] = {}
        psnr_vals: List[float] = []
        ssim_vals: List[float] = []

        for fp in tqdm(test_files, desc=f"FBP @ {level}"):
            sino = np.load(fp)
            if sino.ndim == 4:
                sino = sino[0]
            if sino.ndim != 3:
                raise ValueError(f"Unexpected sinogram shape {sino.shape} in {fp}")

            truncation_level = level
            if args.truncation_ratio is not None:
                truncation_level = int(sino.shape[1] * args.truncation_ratio)

            pred_sino = predict_sinogram_3d(
                model=model,
                model_type=model_type,
                sinogram_3d=sino,
                data_mean=data_stats['mean'],
                data_std=data_stats['std'],
                truncation_level=truncation_level,
                batch_size=args.batch_size,
                device=device,
                use_truncation_for_cold_diffusion=args.use_truncation_for_cold_diffusion,
                timesteps=config.diffusion.timesteps,
                keep_center=resolve_keep_center(config.diffusion, width),
            )

            # FBP reconstruction
            gt_vol = projector.fbp_reconstruct(sino.astype(np.float32))
            pred_vol = projector.fbp_reconstruct(pred_sino.astype(np.float32))

            metrics = compute_volume_metrics(pred_vol, gt_vol)
            per_case[Path(fp).stem] = metrics
            psnr_vals.append(metrics['PSNR'])
            ssim_vals.append(metrics['SSIM'])

        label_level = level
        if args.truncation_ratio is not None:
            label_level = int(width * args.truncation_ratio)

        results[f'level_{label_level}'] = {
            'mean': {
                'PSNR': float(np.mean(psnr_vals)) if psnr_vals else 0.0,
                'SSIM': float(np.mean(ssim_vals)) if ssim_vals else 0.0
            },
            'per_case': per_case
        }

        print(f"  Mean PSNR: {results[f'level_{label_level}']['mean']['PSNR']:.2f} dB")
        print(f"  Mean SSIM: {results[f'level_{label_level}']['mean']['SSIM']:.4f}")

    results_path = output_dir / 'fbp_metrics.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Save] Metrics saved to {results_path}")


if __name__ == '__main__':
    main()
