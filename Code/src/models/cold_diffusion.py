"""
Cold Diffusion Model for CT Sinogram Truncation Recovery

Cold Diffusion uses deterministic degradation (masking) instead of Gaussian noise.
This is more suitable for CT truncation artifact recovery as it directly models
the physical truncation process.

Key differences from standard DDPM:
1. Forward process: Deterministic masking instead of noise addition
2. The mask progressively removes detector columns from edges to center
3. Reverse process: Network predicts clean sinogram from masked input

Reference: Bansal et al., "Cold Diffusion: Inverting Arbitrary Image Transforms
Without Noise", NeurIPS 2022
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from .unet import UNet
from utils.wce import water_cylinder_extrapolate_2d_batch


class ColdDiffusion2D(nn.Module):
    """
    Cold Diffusion model for 2D sinogram truncation recovery.

    The forward (degradation) process progressively masks the sinogram
    from the detector edges towards the center, simulating increasing
    levels of truncation.

    Args:
        denoise_fn: Neural network that predicts x_0 from x_t
        image_size: (H, W) size of input sinograms
        timesteps: Number of diffusion steps
        keep_center: Minimum number of center columns to keep (controls max truncation)
        loss_type: 'l1' or 'l2' loss
        sampling_routine: 'default' or 'x0_step_down' (recommended for stability)
    """
    def __init__(
        self,
        denoise_fn: nn.Module,
        image_size: Tuple[int, int],
        timesteps: int = 100,
        keep_center: int = 11,
        loss_type: str = 'l1',
        sampling_routine: str = 'x0_step_down',
        channels: int = 1,
        use_wce_input: bool = False
    ):
        super().__init__()

        self.denoise_fn = denoise_fn
        self.H, self.W = image_size
        self.channels = channels
        self.T = int(timesteps)
        self.keep_center = int(keep_center)
        self.loss_type = loss_type.lower()
        self.sampling_routine = sampling_routine
        self.use_wce_input = bool(use_wce_input)

        # Build and register masks
        fade_masks = self._build_fade_masks()  # List of [1, H, W] tensors
        fade = torch.stack(fade_masks)  # [T, 1, H, W]
        self.register_buffer("fade_factors", fade)

        # Cumulative masks: prefix_masks[t] = product of fade_factors[0:t+1]
        with torch.no_grad():
            prefix = torch.cumprod(fade, dim=0)  # [T, 1, H, W]
        self.register_buffer("prefix_masks", prefix)

        # All-ones mask for t=-1 case
        ones = torch.ones(1, 1, self.H, self.W, dtype=fade.dtype)
        self.register_buffer("ones_mask", ones)

    def _build_fade_masks(self) -> List[torch.Tensor]:
        """
        Build progressive truncation masks.

        At timestep t, we mask N = floor((W/2) * (t+1) / T) pixels from each side,
        but always keep at least `keep_center` columns in the center.

        Returns:
            List of [1, H, W] mask tensors
        """
        H, W = self.H, self.W
        keep_center = self.keep_center
        max_n = max(0, (W - keep_center) // 2)  # Maximum columns to mask from each side

        masks = []
        for t in range(self.T):
            # Progressive masking formula
            n = int((W // 2) * (t + 1) / self.T)
            n = min(n, max_n)  # Ensure we keep at least keep_center columns

            # Create 1D mask
            mask_1d = np.ones(W, dtype=np.float32)
            if n > 0:
                mask_1d[:n] = 0.0   # Mask left side
                mask_1d[-n:] = 0.0  # Mask right side

            # Broadcast to 2D and create tensor
            mask_2d = np.broadcast_to(mask_1d, (H, W)).copy()
            masks.append(torch.from_numpy(mask_2d).unsqueeze(0))  # [1, H, W]

        return masks

    def get_mask(self, t: int) -> torch.Tensor:
        """
        Get cumulative mask for timestep t.

        Args:
            t: Timestep (0 to T-1), or -1 for no masking

        Returns:
            [1, 1, H, W] mask tensor
        """
        if t < 0:
            return self.ones_mask
        return self.prefix_masks[t]

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward process: Apply truncation mask to clean sinogram.

        Args:
            x_start: (B, C, H, W) clean sinograms
            t: (B,) timesteps for each sample

        Returns:
            (B, C, H, W) masked sinograms
        """
        # Get masks for each sample in batch
        mask = self.prefix_masks.index_select(0, t)  # [B, 1, H, W]
        mask = mask.expand(-1, x_start.size(1), -1, -1)  # [B, C, H, W]
        x_t = x_start * mask
        if self.use_wce_input:
            x_t = water_cylinder_extrapolate_2d_batch(x_t, mask)
        return x_t

    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute training loss.

        Args:
            x_start: (B, C, H, W) clean sinograms
            t: (B,) random timesteps

        Returns:
            Scalar loss value
        """
        # Forward process: mask the sinogram
        x_t = self.q_sample(x_start, t)

        # Network predicts clean sinogram from masked input
        x0_pred = self.denoise_fn(x_t, t)

        # Compute loss
        if self.loss_type == 'l1':
            return F.l1_loss(x0_pred, x_start)
        elif self.loss_type == 'l2':
            return F.mse_loss(x0_pred, x_start)
        else:
            raise NotImplementedError(f"Unknown loss type: {self.loss_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass.

        Args:
            x: (B, C, H, W) batch of clean sinograms

        Returns:
            Scalar loss value
        """
        if x.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {x.shape}")

        B, C, H, W = x.shape
        assert (H, W) == (self.H, self.W), \
            f"Input size {(H, W)} != config {(self.H, self.W)}"

        # Random timesteps
        t = torch.randint(0, self.T, (B,), device=x.device, dtype=torch.long)

        return self.p_losses(x, t)

    @torch.no_grad()
    def sample(
        self,
        x_start: torch.Tensor,
        t_start: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reverse process: Recover clean sinogram from masked input.

        Args:
            x_start: (B, C, H, W) clean sinograms (will be masked first)
            t_start: Starting timestep (default: T)

        Returns:
            Tuple of:
                - x_T: Initial masked sinogram
                - first_pred: First network prediction
                - final_recon: Final reconstructed sinogram
        """
        if t_start is None:
            t_start = self.T

        B = x_start.size(0)
        device = x_start.device

        # Apply initial mask to get x_T
        mask_T = self.get_mask(t_start - 1)
        x_t = x_start * mask_T
        if self.use_wce_input:
            mask_T_batch = mask_T.expand(B, -1, -1, -1)
            x_t = water_cylinder_extrapolate_2d_batch(x_t, mask_T_batch)

        curr_x = x_t.clone()
        first_pred = None

        # Reverse sampling loop
        for t in range(t_start, 0, -1):
            step = torch.full((B,), t - 1, dtype=torch.long, device=device)
            x0_pred = self.denoise_fn(curr_x, step)

            if first_pred is None:
                first_pred = x0_pred

            # Update using chosen sampling routine
            if self.sampling_routine == 'default':
                # Simple: Just use prediction with less masking
                mask_tm1 = self.get_mask(t - 2)
                curr_x = x0_pred * mask_tm1

            elif self.sampling_routine == 'x0_step_down':
                # Residual update: More stable
                mask_t = self.get_mask(t - 1)
                mask_tm1 = self.get_mask(t - 2)
                x_t_pred = x0_pred * mask_t
                x_tm1_pred = x0_pred * mask_tm1
                curr_x = curr_x - x_t_pred + x_tm1_pred
            else:
                raise NotImplementedError(f"Unknown sampling routine: {self.sampling_routine}")

        # Hard replace observed region to preserve measured data
        curr_x = curr_x * (1.0 - mask_T) + x_t

        return x_t, first_pred, curr_x

    @torch.no_grad()
    def sample_from_truncated(
        self,
        x_truncated: torch.Tensor,
        t_start: int
    ) -> torch.Tensor:
        """
        Recover from an already-truncated sinogram.

        Args:
            x_truncated: (B, C, H, W) truncated sinograms
            t_start: Truncation level (timestep)

        Returns:
            (B, C, H, W) recovered sinograms
        """
        B = x_truncated.size(0)
        device = x_truncated.device

        if self.use_wce_input:
            mask_t_start = self.get_mask(t_start - 1).expand(B, -1, -1, -1)
            x_truncated = water_cylinder_extrapolate_2d_batch(x_truncated, mask_t_start)

        curr_x = x_truncated.clone()

        for t in range(t_start, 0, -1):
            step = torch.full((B,), t - 1, dtype=torch.long, device=device)
            x0_pred = self.denoise_fn(curr_x, step)

            if self.sampling_routine == 'default':
                mask_tm1 = self.get_mask(t - 2)
                curr_x = x0_pred * mask_tm1
            elif self.sampling_routine == 'x0_step_down':
                mask_t = self.get_mask(t - 1)
                mask_tm1 = self.get_mask(t - 2)
                x_t_pred = x0_pred * mask_t
                x_tm1_pred = x0_pred * mask_tm1
                curr_x = curr_x - x_t_pred + x_tm1_pred
            else:
                raise NotImplementedError(self.sampling_routine)

        # Hard replace observed region to preserve measured data
        mask_t = self.get_mask(t_start - 1)
        curr_x = curr_x * (1.0 - mask_t) + x_truncated * mask_t

        return curr_x

    @torch.no_grad()
    def all_sample_steps(
        self,
        x_start: torch.Tensor,
        t_start: Optional[int] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Return all intermediate steps during sampling.

        Args:
            x_start: (B, C, H, W) clean sinograms
            t_start: Starting timestep

        Returns:
            Tuple of:
                - x0_preds: List of x0 predictions at each step
                - x_ts: List of x_t at each step
        """
        if t_start is None:
            t_start = self.T

        B = x_start.size(0)
        device = x_start.device

        mask_T = self.get_mask(t_start - 1)
        curr_x = x_start * mask_T

        x0_preds, x_ts = [], []

        for t in range(t_start, 0, -1):
            x_ts.append(curr_x.clone())

            step = torch.full((B,), t - 1, dtype=torch.long, device=device)
            x0_pred = self.denoise_fn(curr_x, step)
            x0_preds.append(x0_pred.clone())

            if self.sampling_routine == 'default':
                mask_tm1 = self.get_mask(t - 2)
                curr_x = x0_pred * mask_tm1
            elif self.sampling_routine == 'x0_step_down':
                mask_t = self.get_mask(t - 1)
                mask_tm1 = self.get_mask(t - 2)
                curr_x = curr_x - x0_pred * mask_t + x0_pred * mask_tm1
            else:
                raise NotImplementedError(self.sampling_routine)

        return x0_preds, x_ts

    def get_truncation_info(self) -> dict:
        """
        Get information about the truncation configuration.

        Returns:
            Dictionary with truncation parameters
        """
        return {
            'timesteps': self.T,
            'keep_center': self.keep_center,
            'image_size': (self.H, self.W),
            'max_truncation_ratio': 1.0 - self.keep_center / self.W,
            'sampling_routine': self.sampling_routine
        }


def build_cold_diffusion(
    image_size: Tuple[int, int] = (360, 640),
    dim: int = 64,
    dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
    timesteps: int = 100,
    keep_center: int = 11,
    sampling_routine: str = 'x0_step_down',
    loss_type: str = 'l1',
    use_wce_input: bool = False
) -> ColdDiffusion2D:
    """
    Factory function to build a Cold Diffusion model.

    Args:
        image_size: (H, W) sinogram dimensions
        dim: Base channel dimension for UNet
        dim_mults: Channel multipliers
        timesteps: Number of diffusion steps
        keep_center: Minimum center columns to keep
        sampling_routine: 'default' or 'x0_step_down'
        loss_type: 'l1' or 'l2'

    Returns:
        ColdDiffusion2D model
    """
    unet = UNet(
        dim=dim,
        dim_mults=dim_mults,
        channels=1,
        with_time_emb=True,
        residual=False
    )

    model = ColdDiffusion2D(
        denoise_fn=unet,
        image_size=image_size,
        timesteps=timesteps,
        keep_center=keep_center,
        loss_type=loss_type,
        sampling_routine=sampling_routine,
        use_wce_input=use_wce_input
    )

    return model


if __name__ == "__main__":
    # Test Cold Diffusion
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_cold_diffusion(
        image_size=(360, 640),
        dim=64,
        timesteps=100,
        keep_center=11
    ).to(device)

    print(f"Model info: {model.get_truncation_info()}")

    # Test forward pass
    x = torch.randn(2, 1, 360, 640, device=device)
    loss = model(x)
    print(f"Training loss: {loss.item():.4f}")

    # Test sampling
    x_T, first_pred, final = model.sample(x)
    print(f"Sampling: x_T={x_T.shape}, final={final.shape}")
