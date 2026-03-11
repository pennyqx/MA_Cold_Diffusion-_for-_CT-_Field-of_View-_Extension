"""
Evaluation Metrics for CT Sinogram and Reconstruction

Metrics:
- Sinogram domain: MAE, RMSE, PSNR, SSIM
- Reconstruction domain: PSNR, SSIM
- Masked region metrics: Evaluate only truncated regions
"""

import math
import numpy as np
import torch
from typing import Dict, Optional, Union, List
from dataclasses import dataclass, field


def _to_numpy(arr: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert to numpy array"""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().float().numpy()
    return arr.astype(np.float64)


def _to_torch(arr: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """Convert to torch tensor"""
    if isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr)
    return arr.float()


def compute_psnr(pred: np.ndarray, gt: np.ndarray, data_range: Optional[float] = None) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.

    PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)

    Args:
        pred: Predicted array
        gt: Ground truth array
        data_range: Dynamic range (max - min). If None, computed from gt.

    Returns:
        PSNR in dB
    """
    if data_range is None:
        data_range = float(gt.max() - gt.min())
    if data_range < 1e-10:
        data_range = 1.0

    mse = np.mean((pred - gt) ** 2)
    if mse < 1e-10:
        return float('inf')

    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def compute_ssim(
    pred: np.ndarray,
    gt: np.ndarray,
    data_range: Optional[float] = None,
    win_size: int = 7
) -> float:
    """
    Compute Structural Similarity Index.

    Args:
        pred: Predicted array (2D or 3D)
        gt: Ground truth array
        data_range: Dynamic range
        win_size: Window size for SSIM computation

    Returns:
        SSIM value in [0, 1]
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        raise ImportError("scikit-image is required. Install with: pip install scikit-image")

    if data_range is None:
        data_range = float(gt.max() - gt.min())

    if pred.ndim == 2:
        return ssim(gt, pred, data_range=data_range, win_size=win_size)
    elif pred.ndim == 3:
        # Compute SSIM for each slice and average
        ssim_values = []
        for i in range(pred.shape[0]):
            s = ssim(gt[i], pred[i], data_range=data_range, win_size=win_size)
            ssim_values.append(s)
        return float(np.mean(ssim_values))
    else:
        raise ValueError(f"Unsupported array dimension: {pred.ndim}")


def compute_sinogram_metrics(
    pred: Union[torch.Tensor, np.ndarray],
    gt: Union[torch.Tensor, np.ndarray],
    data_range: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute sinogram-domain metrics.

    Args:
        pred: Predicted sinogram
        gt: Ground truth sinogram
        data_range: Dynamic range for PSNR/SSIM

    Returns:
        Dictionary with MAE, RMSE, PSNR, SSIM
    """
    pred_np = _to_numpy(pred)
    gt_np = _to_numpy(gt)

    # Flatten for global metrics if multi-dimensional
    pred_flat = pred_np.flatten()
    gt_flat = gt_np.flatten()

    mae = float(np.mean(np.abs(pred_flat - gt_flat)))
    rmse = float(np.sqrt(np.mean((pred_flat - gt_flat) ** 2)))
    psnr = compute_psnr(pred_np, gt_np, data_range)

    # SSIM for 2D slices
    if pred_np.ndim == 2:
        ssim_val = compute_ssim(pred_np, gt_np, data_range)
    elif pred_np.ndim == 3:
        # Average SSIM over first dimension
        ssim_values = []
        for i in range(pred_np.shape[0]):
            s = compute_ssim(pred_np[i], gt_np[i], data_range)
            ssim_values.append(s)
        ssim_val = float(np.mean(ssim_values))
    else:
        ssim_val = 0.0

    return {
        'MAE': mae,
        'RMSE': rmse,
        'PSNR': psnr,
        'SSIM': ssim_val
    }


def compute_volume_metrics(
    pred: Union[torch.Tensor, np.ndarray],
    gt: Union[torch.Tensor, np.ndarray],
    data_range: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute reconstruction volume metrics.

    Args:
        pred: Predicted volume (Z, Y, X) or (B, Z, Y, X)
        gt: Ground truth volume

    Returns:
        Dictionary with PSNR, SSIM
    """
    pred_np = _to_numpy(pred)
    gt_np = _to_numpy(gt)

    # Handle batch dimension
    if pred_np.ndim == 4:
        psnr_values = []
        ssim_values = []
        for i in range(pred_np.shape[0]):
            psnr_values.append(compute_psnr(pred_np[i], gt_np[i], data_range))
            ssim_values.append(compute_ssim(pred_np[i], gt_np[i], data_range))
        return {
            'PSNR': float(np.mean(psnr_values)),
            'SSIM': float(np.mean(ssim_values))
        }

    return {
        'PSNR': compute_psnr(pred_np, gt_np, data_range),
        'SSIM': compute_ssim(pred_np, gt_np, data_range)
    }


def compute_masked_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    data_range: float = 1.0
) -> Dict[str, float]:
    """
    Compute metrics only on masked (truncated) regions.

    Args:
        pred: (B, C, H, W) predicted tensor
        gt: (B, C, H, W) ground truth tensor
        mask: (B, C, H, W) binary mask (1 = keep, 0 = truncated)
        data_range: Dynamic range for PSNR

    Returns:
        Dictionary with ROI_MAE, ROI_PSNR, ROI_SSIM
    """
    # Truncated region is where mask = 0
    truncated_region = 1.0 - mask
    eps = 1e-8

    # Compute only on truncated regions
    diff = (pred - gt) * truncated_region

    # MAE on truncated region
    roi_mae = diff.abs().sum() / (truncated_region.sum() + eps)

    # PSNR on truncated region
    mse = (diff ** 2).sum() / (truncated_region.sum() + eps)
    roi_psnr = 10.0 * torch.log10((data_range ** 2) / (mse + eps))

    # Simple SSIM approximation for ROI
    pred_roi = pred * truncated_region
    gt_roi = gt * truncated_region
    n_pixels = truncated_region.sum() + eps

    mu_x = pred_roi.sum() / n_pixels
    mu_y = gt_roi.sum() / n_pixels

    var_x = ((pred_roi ** 2).sum() / n_pixels) - mu_x ** 2
    var_y = ((gt_roi ** 2).sum() / n_pixels) - mu_y ** 2
    cov_xy = ((pred_roi * gt_roi).sum() / n_pixels) - mu_x * mu_y

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    roi_ssim = ((2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (var_x + var_y + C2) + eps)

    return {
        'ROI_MAE': float(roi_mae.item()),
        'ROI_PSNR': float(roi_psnr.item()),
        'ROI_SSIM': float(roi_ssim.item())
    }


@dataclass
class MetricsAccumulator:
    """
    Accumulator for computing average metrics over multiple batches.

    Usage:
        acc = MetricsAccumulator()
        for pred, gt in batches:
            acc.update(compute_sinogram_metrics(pred, gt))
        results = acc.compute()
    """
    _metrics: Dict[str, List[float]] = field(default_factory=dict)
    _count: int = 0

    def update(self, metrics: Dict[str, float]):
        """Add a batch of metrics"""
        for key, value in metrics.items():
            if key not in self._metrics:
                self._metrics[key] = []
            self._metrics[key].append(value)
        self._count += 1

    def compute(self) -> Dict[str, float]:
        """Compute average metrics"""
        return {key: float(np.mean(values)) for key, values in self._metrics.items()}

    def reset(self):
        """Reset accumulator"""
        self._metrics = {}
        self._count = 0

    @property
    def count(self) -> int:
        return self._count


class TruncationLevelEvaluator:
    """
    Evaluate model at different truncation levels.

    Useful for analyzing how performance degrades with more truncation.
    """
    def __init__(
        self,
        model,
        data_loader,
        truncation_levels: List[int],
        device: torch.device
    ):
        self.model = model
        self.data_loader = data_loader
        self.truncation_levels = truncation_levels
        self.device = device

    @torch.no_grad()
    def evaluate(self) -> Dict[int, Dict[str, float]]:
        """
        Evaluate at each truncation level.

        Returns:
            Dictionary mapping truncation_level -> metrics
        """
        self.model.eval()
        results = {}

        for level in self.truncation_levels:
            print(f"Evaluating at truncation level {level}...")
            acc = MetricsAccumulator()

            for batch in self.data_loader:
                gt = batch['sinogram'].to(self.device)

                # Apply truncation
                W = gt.shape[-1]
                mask = torch.ones_like(gt)
                if level > 0:
                    mask[:, :, :, :level] = 0
                    mask[:, :, :, -level:] = 0

                truncated = gt * mask

                # Recover
                if hasattr(self.model, 'sample_from_truncated'):
                    # Cold Diffusion
                    pred = self.model.sample_from_truncated(truncated, t_start=self.model.T)
                elif hasattr(self.model, 'inpaint'):
                    # UNet Inpaint
                    pred = self.model.inpaint(truncated, mask)
                else:
                    pred = truncated

                # Compute metrics
                metrics = compute_sinogram_metrics(pred, gt)
                masked_metrics = compute_masked_metrics(pred, gt, mask)
                metrics.update(masked_metrics)

                acc.update(metrics)

            results[level] = acc.compute()

        return results


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics computation...")

    # Create dummy data
    gt = np.random.randn(360, 640).astype(np.float32)
    pred = gt + 0.1 * np.random.randn(360, 640).astype(np.float32)

    metrics = compute_sinogram_metrics(pred, gt)
    print(f"Sinogram metrics: {metrics}")

    # Test with torch tensors
    gt_t = torch.randn(4, 1, 360, 640)
    pred_t = gt_t + 0.1 * torch.randn(4, 1, 360, 640)
    mask = torch.ones_like(gt_t)
    mask[:, :, :, :100] = 0
    mask[:, :, :, -100:] = 0

    masked_metrics = compute_masked_metrics(pred_t, gt_t, mask)
    print(f"Masked metrics: {masked_metrics}")

    # Test accumulator
    acc = MetricsAccumulator()
    for _ in range(5):
        acc.update({'PSNR': np.random.uniform(30, 35), 'SSIM': np.random.uniform(0.8, 0.9)})
    print(f"Accumulated metrics: {acc.compute()}")
