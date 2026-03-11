"""
Water Cylinder Extrapolation (WCE) helpers for sinogram preprocessing.

This module operates on 2D sinograms in tensor shape [B, 1, H, W], where:
- H: projection views
- W: detector width
"""

from __future__ import annotations

from typing import Tuple

import torch


def _mask_bounds(mask_1d: torch.Tensor) -> Tuple[int, int]:
    """
    Infer observed center bounds [left, right) from a binary 1D mask of length W.
    """
    w = int(mask_1d.numel())
    idx = torch.nonzero(mask_1d > 0.5, as_tuple=False).flatten()
    if idx.numel() == 0:
        return 0, 0
    left = int(idx[0].item())
    right = int(idx[-1].item()) + 1
    left = max(0, min(left, w))
    right = max(0, min(right, w))
    return left, right


def _fit_cylinder_profile(mean_profile: torch.Tensor, left: int, right: int) -> torch.Tensor:
    """
    Fit a simple water-cylinder profile over detector width.
    """
    w = int(mean_profile.numel())
    if left <= 0 and right >= w:
        return mean_profile.clone()
    if right <= left:
        return torch.zeros_like(mean_profile)

    device = mean_profile.device
    dtype = mean_profile.dtype
    u = torch.arange(w, device=device, dtype=dtype)
    u0 = (w - 1) * 0.5
    obs_u = u[left:right]
    y = torch.clamp(mean_profile[left:right], min=0.0)
    if y.numel() == 0:
        return torch.zeros_like(mean_profile)

    max_d = torch.max(torch.abs(obs_u - u0)).item()
    r_min = max(max_d + 1.0, w * 0.5)
    r_max = max(r_min + 1.0, w * 1.2)
    radii = torch.linspace(r_min, r_max, steps=10, device=device, dtype=dtype)

    best_mse = None
    best_profile = torch.zeros_like(mean_profile)
    eps = 1e-8

    for r in radii:
        basis_obs = torch.sqrt(torch.clamp(r * r - (obs_u - u0) ** 2, min=0.0))
        denom = torch.dot(basis_obs, basis_obs) + eps
        a = torch.clamp(torch.dot(y, basis_obs) / denom, min=0.0)
        pred_obs = a * basis_obs
        mse = torch.mean((pred_obs - y) ** 2)
        if best_mse is None or mse < best_mse:
            best_mse = mse
            basis_full = torch.sqrt(torch.clamp(r * r - (u - u0) ** 2, min=0.0))
            best_profile = a * basis_full

    return best_profile


@torch.no_grad()
def water_cylinder_extrapolate_2d_batch(x_truncated: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply WCE on masked sinograms.

    Args:
        x_truncated: [B, 1, H, W], truncated sinogram.
        mask: [B, 1, H, W], binary mask where 1 means observed center.

    Returns:
        [B, 1, H, W], WCE-extrapolated sinogram with observed center preserved.
    """
    if x_truncated.ndim != 4 or mask.ndim != 4:
        raise ValueError(f"Expected 4D tensors, got {x_truncated.shape} and {mask.shape}")
    if x_truncated.shape != mask.shape:
        raise ValueError(f"Shape mismatch: {x_truncated.shape} vs {mask.shape}")

    b, c, h, w = x_truncated.shape
    if c != 1:
        raise ValueError(f"Expected single channel sinogram, got channels={c}")

    out = x_truncated.clone()
    eps = 1e-6

    for i in range(b):
        m1d = mask[i, 0, 0, :]
        left, right = _mask_bounds(m1d)
        if right <= left:
            continue
        if left == 0 and right == w:
            continue

        obs = x_truncated[i, 0]  # [H, W]
        mean_profile = obs[:, left:right].mean(dim=0)
        mean_profile_full = torch.zeros((w,), dtype=obs.dtype, device=obs.device)
        mean_profile_full[left:right] = mean_profile
        cyl = _fit_cylinder_profile(mean_profile_full, left, right)

        left_ref = max(float(cyl[left].item()), eps)
        right_ref = max(float(cyl[right - 1].item()), eps)

        left_scale = obs[:, left] / left_ref
        right_scale = obs[:, right - 1] / right_ref

        if left > 0:
            out[i, 0, :, :left] = left_scale.unsqueeze(1) * cyl[:left].unsqueeze(0)
        if right < w:
            out[i, 0, :, right:] = right_scale.unsqueeze(1) * cyl[right:].unsqueeze(0)
        out[i, 0, :, left:right] = obs[:, left:right]

    return out
