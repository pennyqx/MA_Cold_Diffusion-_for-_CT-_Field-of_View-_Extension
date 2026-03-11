"""
U-Net Inpainting Model for CT Sinogram Truncation Recovery

Single-step baseline that directly learns to inpaint truncated regions.
Unlike diffusion models, this is a one-shot prediction without iterative refinement.

The model takes a truncated sinogram (with mask) and outputs the complete sinogram.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .unet import UNetSimple
from utils.wce import water_cylinder_extrapolate_2d_batch


class UNetInpaint(nn.Module):
    """
    U-Net based single-step inpainting model.

    Takes truncated sinogram and binary mask as input (2 channels),
    outputs the complete sinogram (1 channel).

    Args:
        dim: Base channel dimension
        dim_mults: Channel multipliers for each resolution level
        image_size: (H, W) size of input sinograms
        keep_center: Minimum center columns to keep (for generating mask during training)
    """
    def __init__(
        self,
        dim: int = 64,
        dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
        image_size: Tuple[int, int] = (360, 640),
        keep_center: int = 11,
        use_wce_input: bool = False
    ):
        super().__init__()

        self.H, self.W = image_size
        self.keep_center = keep_center
        self.use_wce_input = bool(use_wce_input)

        # 2-channel input: truncated sinogram + mask
        # 1-channel output: complete sinogram
        self.net = UNetSimple(
            dim=dim,
            dim_mults=dim_mults,
            channels=2,  # sinogram + mask
            out_dim=1    # predicted sinogram
        )

        # Pre-compute all possible masks for training
        self._register_masks()

    def _register_masks(self):
        """Pre-compute truncation masks for all possible truncation levels"""
        H, W = self.H, self.W
        max_truncation = (W - self.keep_center) // 2

        masks = []
        for n in range(max_truncation + 1):
            mask = torch.ones(1, H, W)
            if n > 0:
                mask[:, :, :n] = 0
                mask[:, :, -n:] = 0
            masks.append(mask)

        # Stack all masks: [num_levels, 1, H, W]
        self.register_buffer('all_masks', torch.stack(masks))
        self.num_truncation_levels = len(masks)

    def get_random_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Get random truncation masks for training.

        Args:
            batch_size: Number of masks needed
            device: Device to create tensor on

        Returns:
            (B, 1, H, W) binary masks (1 = keep, 0 = truncated)
        """
        indices = torch.randint(0, self.num_truncation_levels, (batch_size,), device=device)
        return self.all_masks[indices]

    def get_mask_at_level(self, truncation_level: int, batch_size: int = 1) -> torch.Tensor:
        """
        Get mask for a specific truncation level.

        Args:
            truncation_level: Number of columns to mask from each side
            batch_size: Number of masks

        Returns:
            (B, 1, H, W) binary masks
        """
        truncation_level = min(truncation_level, self.num_truncation_levels - 1)
        return self.all_masks[truncation_level].expand(batch_size, -1, -1, -1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            x: (B, 1, H, W) clean sinograms
            mask: (B, 1, H, W) optional mask (if None, random mask is used)

        Returns:
            Scalar loss (L1 loss between prediction and ground truth)
        """
        B = x.size(0)

        # Get or generate mask
        if mask is None:
            mask = self.get_random_mask(B, x.device)

        # Apply mask to create truncated input
        x_truncated = x * mask
        if self.use_wce_input:
            x_truncated = water_cylinder_extrapolate_2d_batch(x_truncated, mask)

        # Concatenate truncated sinogram and mask
        net_input = torch.cat([x_truncated, mask], dim=1)  # [B, 2, H, W]

        # Predict complete sinogram
        x_pred = self.net(net_input)

        # Compute loss
        loss = F.l1_loss(x_pred, x)

        return loss

    @torch.no_grad()
    def inpaint(
        self,
        x_truncated: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Inpaint truncated sinogram.

        Args:
            x_truncated: (B, 1, H, W) truncated sinograms
            mask: (B, 1, H, W) binary masks (1 = keep, 0 = truncated)

        Returns:
            (B, 1, H, W) inpainted sinograms
        """
        if self.use_wce_input:
            x_truncated = water_cylinder_extrapolate_2d_batch(x_truncated, mask)
        net_input = torch.cat([x_truncated, mask], dim=1)
        x_pred = self.net(net_input)
        # Hard replace observed region to preserve measured data
        return x_pred * (1.0 - mask) + x_truncated * mask

    @torch.no_grad()
    def inpaint_from_clean(
        self,
        x_clean: torch.Tensor,
        truncation_level: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply truncation and then inpaint (for evaluation).

        Args:
            x_clean: (B, 1, H, W) clean sinograms
            truncation_level: Number of columns to mask from each side

        Returns:
            Tuple of:
                - x_truncated: The truncated input
                - x_pred: The inpainted result
                - mask: The applied mask
        """
        B = x_clean.size(0)
        mask = self.get_mask_at_level(truncation_level, B).to(x_clean.device)
        x_truncated = x_clean * mask
        x_pred = self.inpaint(x_truncated, mask)

        return x_truncated, x_pred, mask


class UNetInpaintWithMaskedLoss(UNetInpaint):
    """
    Variant that only computes loss on the truncated (masked) regions.

    This focuses the learning on the inpainting task rather than
    reconstructing the already-visible regions.
    """
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with masked loss.

        Args:
            x: (B, 1, H, W) clean sinograms
            mask: (B, 1, H, W) optional mask

        Returns:
            Scalar loss (L1 loss on truncated regions only)
        """
        B = x.size(0)

        if mask is None:
            mask = self.get_random_mask(B, x.device)

        x_truncated = x * mask
        if self.use_wce_input:
            x_truncated = water_cylinder_extrapolate_2d_batch(x_truncated, mask)
        net_input = torch.cat([x_truncated, mask], dim=1)
        x_pred = self.net(net_input)

        # Loss only on truncated regions
        truncated_region = 1.0 - mask
        loss = (torch.abs(x_pred - x) * truncated_region).sum() / (truncated_region.sum() + 1e-8)

        return loss


class UNetInpaintHybridLoss(UNetInpaint):
    """
    Variant with hybrid loss: weighted combination of full image and masked region loss.

    loss = alpha * L1(pred, gt) + (1-alpha) * L1(pred[mask], gt[mask])
    """
    def __init__(
        self,
        dim: int = 64,
        dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
        image_size: Tuple[int, int] = (360, 640),
        keep_center: int = 11,
        alpha: float = 0.5,  # Weight for full image loss
        use_wce_input: bool = False
    ):
        super().__init__(dim, dim_mults, image_size, keep_center, use_wce_input=use_wce_input)
        self.alpha = alpha

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B = x.size(0)

        if mask is None:
            mask = self.get_random_mask(B, x.device)

        x_truncated = x * mask
        if self.use_wce_input:
            x_truncated = water_cylinder_extrapolate_2d_batch(x_truncated, mask)
        net_input = torch.cat([x_truncated, mask], dim=1)
        x_pred = self.net(net_input)

        # Full image loss
        loss_full = F.l1_loss(x_pred, x)

        # Masked region loss
        truncated_region = 1.0 - mask
        loss_masked = (torch.abs(x_pred - x) * truncated_region).sum() / (truncated_region.sum() + 1e-8)

        return self.alpha * loss_full + (1 - self.alpha) * loss_masked


def build_unet_inpaint(
    image_size: Tuple[int, int] = (360, 640),
    dim: int = 64,
    dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
    keep_center: int = 11,
    loss_type: str = 'full',  # 'full', 'masked', or 'hybrid'
    use_wce_input: bool = False
) -> UNetInpaint:
    """
    Factory function to build U-Net inpainting model.

    Args:
        image_size: (H, W) sinogram dimensions
        dim: Base channel dimension
        dim_mults: Channel multipliers
        keep_center: Minimum center columns to keep
        loss_type: Loss type ('full', 'masked', or 'hybrid')

    Returns:
        UNetInpaint model
    """
    if loss_type == 'full':
        return UNetInpaint(dim, dim_mults, image_size, keep_center, use_wce_input=use_wce_input)
    elif loss_type == 'masked':
        return UNetInpaintWithMaskedLoss(dim, dim_mults, image_size, keep_center, use_wce_input=use_wce_input)
    elif loss_type == 'hybrid':
        return UNetInpaintHybridLoss(dim, dim_mults, image_size, keep_center, use_wce_input=use_wce_input)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test UNetInpaint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_unet_inpaint(
        image_size=(360, 640),
        dim=64,
        keep_center=11
    ).to(device)

    # Test forward pass
    x = torch.randn(2, 1, 360, 640, device=device)
    loss = model(x)
    print(f"Training loss: {loss.item():.4f}")

    # Test inpainting
    x_trunc, x_pred, mask = model.inpaint_from_clean(x, truncation_level=100)
    print(f"Truncated: {x_trunc.shape}, Predicted: {x_pred.shape}")

    # Count parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {params:,}")
