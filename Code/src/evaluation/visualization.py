"""
Visualization Utilities for CT Truncation Recovery

Provides functions for:
- Comparing sinograms (original, truncated, recovered)
- Comparing reconstructed volumes
- Plotting training/validation curves
- Creating summary figures for papers
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Optional, List, Dict, Union, Tuple
import torch


def _to_numpy(arr: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert to numpy array"""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    return arr


def _normalize_for_display(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] for display"""
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min < 1e-10:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)


def visualize_sinogram_comparison(
    original: Union[torch.Tensor, np.ndarray],
    truncated: Union[torch.Tensor, np.ndarray],
    recovered: Union[torch.Tensor, np.ndarray],
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    metrics: Optional[Dict[str, float]] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Create side-by-side comparison of sinograms.

    Args:
        original: Ground truth sinogram (H, W) or (1, H, W)
        truncated: Truncated sinogram
        recovered: Recovered sinogram
        save_path: Optional path to save figure
        title: Optional figure title
        metrics: Optional metrics dictionary to display

    Returns:
        Matplotlib figure
    """
    # Convert and squeeze
    original = _to_numpy(original).squeeze()
    truncated = _to_numpy(truncated).squeeze()
    recovered = _to_numpy(recovered).squeeze()

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # Original
    axes[0].imshow(original, cmap='gray', aspect='auto')
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Truncated
    axes[1].imshow(truncated, cmap='gray', aspect='auto')
    axes[1].set_title('Truncated')
    axes[1].axis('off')

    # Recovered
    axes[2].imshow(recovered, cmap='gray', aspect='auto')
    axes[2].set_title('Recovered')
    axes[2].axis('off')

    # Difference
    diff = np.abs(original - recovered)
    im = axes[3].imshow(diff, cmap='hot', aspect='auto')
    axes[3].set_title('|Difference|')
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=14)

    if metrics:
        metric_str = ' | '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        fig.text(0.5, 0.02, metric_str, ha='center', fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def visualize_reconstruction_comparison(
    original: Union[torch.Tensor, np.ndarray],
    truncated_recon: Union[torch.Tensor, np.ndarray],
    recovered_recon: Union[torch.Tensor, np.ndarray],
    slice_idx: int,
    save_path: Optional[str] = None,
    window: Optional[Tuple[float, float]] = None,  # (center, width) for HU windowing
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Compare reconstructed CT slices.

    Args:
        original: Ground truth volume (Z, Y, X)
        truncated_recon: Reconstruction from truncated sinogram
        recovered_recon: Reconstruction from recovered sinogram
        slice_idx: Axial slice index to display
        save_path: Optional save path
        window: Optional HU window (center, width)
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    original = _to_numpy(original)[slice_idx]
    truncated_recon = _to_numpy(truncated_recon)[slice_idx]
    recovered_recon = _to_numpy(recovered_recon)[slice_idx]

    # Apply windowing if specified
    if window:
        center, width = window
        vmin, vmax = center - width / 2, center + width / 2
    else:
        vmin = min(original.min(), truncated_recon.min(), recovered_recon.min())
        vmax = max(original.max(), truncated_recon.max(), recovered_recon.max())

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    axes[0].imshow(original, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(truncated_recon, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title('Truncated FBP')
    axes[1].axis('off')

    axes[2].imshow(recovered_recon, cmap='gray', vmin=vmin, vmax=vmax)
    axes[2].set_title('Recovered FBP')
    axes[2].axis('off')

    diff = np.abs(original - recovered_recon)
    im = axes[3].imshow(diff, cmap='hot')
    axes[3].set_title('|Difference|')
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    fig.suptitle(f'Slice {slice_idx}', fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_metrics_curve(
    csv_path: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """
    Plot training/validation metrics from CSV log.

    Args:
        csv_path: Path to CSV file with columns: step, metric1, metric2, ...
        save_path: Optional save path
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    columns = [c for c in df.columns if c != 'step' and c != 'lr']

    n_cols = len(columns)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        ax.plot(df['step'], df[col], linewidth=1)
        ax.set_xlabel('Step')
        ax.set_ylabel(col)
        ax.set_title(col)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_summary_figure(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Create summary bar chart comparing different methods.

    Args:
        results: Dictionary mapping method_name -> {metric_name: value}
        save_path: Optional save path
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    methods = list(results.keys())
    metrics = list(results[methods[0]].keys())

    n_methods = len(methods)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(n_methods)
    width = 0.6

    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))

    for ax, metric in zip(axes, metrics):
        values = [results[m][metric] for m in methods]
        bars = ax.bar(x, values, width, color=colors)

        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_diffusion_process(
    x0_list: List[torch.Tensor],
    n_show: int = 8,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 2)
) -> plt.Figure:
    """
    Visualize the diffusion/recovery process.

    Args:
        x0_list: List of intermediate predictions during sampling
        n_show: Number of steps to show
        save_path: Optional save path
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    n_steps = len(x0_list)
    step_indices = np.linspace(0, n_steps - 1, n_show, dtype=int)

    fig, axes = plt.subplots(1, n_show, figsize=figsize)

    for i, (ax, idx) in enumerate(zip(axes, step_indices)):
        img = _to_numpy(x0_list[idx]).squeeze()
        img = _normalize_for_display(img)
        ax.imshow(img, cmap='gray', aspect='auto')
        ax.set_title(f't={n_steps - 1 - idx}')
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def visualize_truncation_levels(
    model,
    gt_sinogram: torch.Tensor,
    truncation_levels: List[int],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 4)
) -> plt.Figure:
    """
    Visualize recovery at different truncation levels.

    Args:
        model: Trained model
        gt_sinogram: Ground truth sinogram (1, 1, H, W)
        truncation_levels: List of truncation levels to test
        save_path: Optional save path
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    n_levels = len(truncation_levels)
    fig, axes = plt.subplots(2, n_levels + 1, figsize=figsize)

    # Show original
    gt_np = _to_numpy(gt_sinogram).squeeze()
    gt_norm = _normalize_for_display(gt_np)

    axes[0, 0].imshow(gt_norm, cmap='gray', aspect='auto')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')

    device = next(model.parameters()).device

    for i, level in enumerate(truncation_levels):
        # Create mask
        W = gt_sinogram.shape[-1]
        mask = torch.ones_like(gt_sinogram)
        if level > 0:
            mask[:, :, :, :level] = 0
            mask[:, :, :, -level:] = 0

        # Truncate
        truncated = gt_sinogram * mask

        # Recover
        with torch.no_grad():
            if hasattr(model, 'sample'):
                _, _, recovered = model.sample(gt_sinogram.to(device))
            else:
                recovered = model.inpaint(truncated.to(device), mask.to(device))

        # Display
        trunc_np = _normalize_for_display(_to_numpy(truncated).squeeze())
        recov_np = _normalize_for_display(_to_numpy(recovered).squeeze())

        axes[0, i + 1].imshow(trunc_np, cmap='gray', aspect='auto')
        axes[0, i + 1].set_title(f'Trunc {level}')
        axes[0, i + 1].axis('off')

        axes[1, i + 1].imshow(recov_np, cmap='gray', aspect='auto')
        axes[1, i + 1].set_title('Recovered')
        axes[1, i + 1].axis('off')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization utilities...")

    # Create dummy data
    original = np.random.randn(360, 640).astype(np.float32)
    truncated = original.copy()
    truncated[:, :100] = 0
    truncated[:, -100:] = 0
    recovered = original + 0.1 * np.random.randn(360, 640).astype(np.float32)

    # Test comparison
    fig = visualize_sinogram_comparison(
        original, truncated, recovered,
        metrics={'PSNR': 35.2, 'SSIM': 0.92}
    )
    plt.close(fig)
    print("Sinogram comparison: OK")

    # Test summary figure
    results = {
        'FBP': {'PSNR': 18.2, 'SSIM': 0.41},
        'Linear': {'PSNR': 22.1, 'SSIM': 0.68},
        'UNet': {'PSNR': 25.4, 'SSIM': 0.79},
        'Cold Diff': {'PSNR': 28.9, 'SSIM': 0.88},
    }
    fig = create_summary_figure(results)
    plt.close(fig)
    print("Summary figure: OK")
