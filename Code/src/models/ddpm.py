"""
Denoising Diffusion Probabilistic Model (DDPM) for CT Sinogram

Conditional DDPM baseline using Gaussian noise for the forward process.
Uses truncated sinogram + mask as condition for reconstruction.

Reference: Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from .unet import UNet
from utils.wce import water_cylinder_extrapolate_2d_batch


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule for beta values.
    Better than linear schedule for image generation.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """Linear schedule for beta values"""
    return torch.linspace(beta_start, beta_end, timesteps)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
    """Extract values from a at indices t, broadcast to x_shape"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class DDPM2D(nn.Module):
    """
    Denoising Diffusion Probabilistic Model for 2D sinograms.

    Standard DDPM with Gaussian noise forward process.
    Predicts noise (epsilon) from noisy input.

    Args:
        denoise_fn: UNet that predicts noise
        image_size: (H, W) size of input sinograms
        timesteps: Number of diffusion steps
        loss_type: 'l1' or 'l2' loss
        beta_schedule: 'linear' or 'cosine'
    """
    def __init__(
        self,
        denoise_fn: nn.Module,
        image_size: Tuple[int, int],
        timesteps: int = 1000,
        loss_type: str = 'l2',
        beta_schedule: str = 'cosine',
        channels: int = 1,
        keep_center: int = 11,
        cond_channels: int = 2,
        use_wce_input: bool = False
    ):
        super().__init__()

        self.denoise_fn = denoise_fn
        self.H, self.W = image_size
        self.channels = channels
        self.T = int(timesteps)
        self.loss_type = loss_type.lower()
        self.keep_center = keep_center
        self.cond_channels = cond_channels
        self.use_wce_input = bool(use_wce_input)

        # Pre-compute truncation masks for conditioning (if enabled)
        if self.cond_channels > 0:
            self._register_masks()

        # Set up diffusion schedule
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Calculations for diffusion q(x_t | x_0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                           betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                           (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    def _register_masks(self):
        """Pre-compute truncation masks for all possible truncation levels."""
        H, W = self.H, self.W
        max_truncation = (W - self.keep_center) // 2

        masks = []
        for n in range(max_truncation + 1):
            mask = torch.ones(1, H, W)
            if n > 0:
                mask[:, :, :n] = 0
                mask[:, :, -n:] = 0
            masks.append(mask)

        self.register_buffer('all_masks', torch.stack(masks))
        self.num_truncation_levels = len(masks)

    def get_random_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Get random truncation masks for training."""
        indices = torch.randint(0, self.num_truncation_levels, (batch_size,), device=device)
        return self.all_masks[indices]

    def get_mask_at_level(self, truncation_level: int, batch_size: int = 1) -> torch.Tensor:
        """Get mask for a specific truncation level (columns per side)."""
        truncation_level = min(truncation_level, self.num_truncation_levels - 1)
        return self.all_masks[truncation_level].expand(batch_size, -1, -1, -1)

    def _concat_condition(self, x_t: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        """Concatenate condition channels if enabled."""
        if self.cond_channels == 0:
            return x_t
        if cond is None:
            cond = torch.zeros(
                x_t.size(0),
                self.cond_channels,
                self.H,
                self.W,
                device=x_t.device,
                dtype=x_t.dtype
            )
        return torch.cat([x_t, cond], dim=1)

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward process: Add noise to clean image.

        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

        Args:
            x_start: (B, C, H, W) clean images
            t: (B,) timesteps
            noise: Optional pre-sampled noise

        Returns:
            (B, C, H, W) noisy images
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def predict_x0_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise"""
        sqrt_alpha = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_one_minus_alpha * noise) / sqrt_alpha

    def q_posterior(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute posterior q(x_{t-1} | x_t, x_0)

        Returns:
            Tuple of (posterior_mean, posterior_log_variance)
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_variance

    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute p(x_{t-1} | x_t) using the learned denoiser.

        Returns:
            Tuple of (model_mean, posterior_log_variance, x_0_pred)
        """
        # Predict noise
        net_input = self._concat_condition(x_t, cond)
        noise_pred = self.denoise_fn(net_input, t)

        # Reconstruct x_0
        x_0_pred = self.predict_x0_from_noise(x_t, t, noise_pred)
        x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)  # Clamp for stability

        # Compute posterior
        model_mean, posterior_log_variance = self.q_posterior(x_0_pred, x_t, t)

        return model_mean, posterior_log_variance, x_0_pred

    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute training loss.

        Args:
            x_start: (B, C, H, W) clean images
            t: (B,) random timesteps
            noise: Optional pre-sampled noise

        Returns:
            Scalar loss value
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        if self.cond_channels > 0 and cond is None:
            if mask is None:
                mask = self.get_random_mask(x_start.size(0), x_start.device)
            x_truncated = x_start * mask
            if self.use_wce_input:
                x_truncated = water_cylinder_extrapolate_2d_batch(x_truncated, mask)
            cond = torch.cat([x_truncated, mask], dim=1)

        # Forward process
        x_t = self.q_sample(x_start, t, noise)

        # Predict noise
        net_input = self._concat_condition(x_t, cond)
        noise_pred = self.denoise_fn(net_input, t)

        # Compute loss
        if self.loss_type == 'l1':
            return F.l1_loss(noise_pred, noise)
        elif self.loss_type == 'l2':
            return F.mse_loss(noise_pred, noise)
        else:
            raise NotImplementedError(f"Unknown loss type: {self.loss_type}")

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Training forward pass.

        Args:
            x: (B, C, H, W) batch of clean images

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

        return self.p_losses(x, t, cond=cond, mask=mask)

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Single reverse sampling step.

        Args:
            x_t: (B, C, H, W) noisy images
            t: (B,) timesteps (all same value)

        Returns:
            (B, C, H, W) slightly denoised images
        """
        model_mean, posterior_log_variance, _ = self.p_mean_variance(x_t, t, cond=cond)

        # Sample noise for t > 0
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))

        return model_mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance) * noise

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Full reverse sampling process (unconditional generation).

        Args:
            batch_size: Number of samples to generate
            device: Device to use

        Returns:
            (B, C, H, W) generated images
        """
        if device is None:
            device = self.betas.device

        shape = (batch_size, self.channels, self.H, self.W)
        x = torch.randn(shape, device=device)

        for t in reversed(range(self.T)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch)

        return x

    @torch.no_grad()
    def sample_with_cond(self, cond: torch.Tensor) -> torch.Tensor:
        """Full reverse sampling with conditioning."""
        batch_size = cond.size(0)
        device = cond.device

        shape = (batch_size, self.channels, self.H, self.W)
        x = torch.randn(shape, device=device)

        for t in reversed(range(self.T)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch, cond=cond)

        return x

    @torch.no_grad()
    def sample_conditional(
        self,
        x_truncated: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Sample with truncated sinogram + mask condition."""
        if self.use_wce_input:
            x_truncated = water_cylinder_extrapolate_2d_batch(x_truncated, mask)
        cond = torch.cat([x_truncated, mask], dim=1)
        x_pred = self.sample_with_cond(cond)
        # Hard replace observed region to preserve measured data
        return x_pred * (1.0 - mask) + x_truncated * mask

    @torch.no_grad()
    def _ddim_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        eta: float = 0.0
    ) -> torch.Tensor:
        """One DDIM step (eta=0 -> deterministic)."""
        net_input = self._concat_condition(x_t, cond)
        noise_pred = self.denoise_fn(net_input, t)

        alpha_t = extract(self.alphas_cumprod, t, x_t.shape)
        alpha_prev = extract(self.alphas_cumprod, t_prev, x_t.shape)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)

        x0_pred = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        if eta == 0.0:
            sigma_t = 0.0
        else:
            sigma_t = eta * torch.sqrt(
                (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
            )

        noise = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)
        dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_prev - sigma_t ** 2, min=0.0)) * noise_pred
        x_prev = torch.sqrt(alpha_prev) * x0_pred + dir_xt + sigma_t * noise
        return x_prev

    @torch.no_grad()
    def sample_conditional_ddim(
        self,
        x_truncated: torch.Tensor,
        mask: torch.Tensor,
        steps: int,
        eta: float = 0.0
    ) -> torch.Tensor:
        """DDIM sampling with truncated sinogram + mask condition."""
        if self.use_wce_input:
            x_truncated = water_cylinder_extrapolate_2d_batch(x_truncated, mask)
        cond = torch.cat([x_truncated, mask], dim=1)
        batch_size = cond.size(0)
        device = cond.device

        steps = int(steps)
        steps = max(2, min(self.T, steps))
        time_seq = np.linspace(0, self.T - 1, steps, dtype=np.int64)
        time_seq = np.unique(time_seq).astype(np.int64)
        time_seq = list(time_seq.tolist())
        if time_seq[-1] != self.T - 1:
            time_seq.append(self.T - 1)
        time_seq = sorted(set(time_seq))
        time_seq = list(reversed(time_seq))

        shape = (batch_size, self.channels, self.H, self.W)
        x = torch.randn(shape, device=device)

        for i, t_val in enumerate(time_seq):
            t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
            if i == len(time_seq) - 1:
                t_prev = torch.zeros_like(t)
            else:
                t_prev_val = time_seq[i + 1]
                t_prev = torch.full((batch_size,), t_prev_val, device=device, dtype=torch.long)
            x = self._ddim_step(x, t, t_prev, cond=cond, eta=eta)

        # Hard replace observed region to preserve measured data
        return x * (1.0 - mask) + x_truncated * mask

    @torch.no_grad()
    def denoise(
        self,
        x_noisy: torch.Tensor,
        t_start: int
    ) -> torch.Tensor:
        """
        Denoise from a given noise level.

        Args:
            x_noisy: (B, C, H, W) noisy images
            t_start: Starting timestep

        Returns:
            (B, C, H, W) denoised images
        """
        B = x_noisy.size(0)
        device = x_noisy.device
        x = x_noisy.clone()

        for t in reversed(range(t_start)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch)

        return x

    @torch.no_grad()
    def denoise_conditional(
        self,
        x_noisy: torch.Tensor,
        t_start: int,
        x_truncated: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Denoise from a given noise level with conditioning."""
        B = x_noisy.size(0)
        device = x_noisy.device
        x = x_noisy.clone()
        cond = torch.cat([x_truncated, mask], dim=1)

        for t in reversed(range(t_start)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch, cond=cond)

        return x

    @torch.no_grad()
    def sample_with_progress(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        save_interval: int = 100
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Sample with intermediate results.

        Args:
            batch_size: Number of samples
            device: Device
            save_interval: Save every N steps

        Returns:
            Tuple of (final_samples, intermediate_samples)
        """
        if device is None:
            device = self.betas.device

        shape = (batch_size, self.channels, self.H, self.W)
        x = torch.randn(shape, device=device)
        intermediates = []

        for t in reversed(range(self.T)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch)

            if t % save_interval == 0:
                intermediates.append(x.clone())

        return x, intermediates


def build_ddpm(
    image_size: Tuple[int, int] = (360, 640),
    dim: int = 64,
    dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
    timesteps: int = 1000,
    beta_schedule: str = 'cosine',
    loss_type: str = 'l2',
    keep_center: int = 11,
    cond_channels: int = 2,
    use_wce_input: bool = False
) -> DDPM2D:
    """
    Factory function to build a DDPM model.

    Args:
        image_size: (H, W) sinogram dimensions
        dim: Base channel dimension for UNet
        dim_mults: Channel multipliers
        timesteps: Number of diffusion steps
        beta_schedule: 'linear' or 'cosine'
        loss_type: 'l1' or 'l2'

    Returns:
        DDPM2D model
    """
    in_channels = 1 + cond_channels
    # DDPM predicts noise for the original sinogram (1 channel).
    # Conditioning is concatenated on the input only; output must stay 1 channel.
    unet = UNet(
        dim=dim,
        dim_mults=dim_mults,
        channels=in_channels,
        out_dim=1,
        with_time_emb=True,
        residual=False
    )

    model = DDPM2D(
        denoise_fn=unet,
        image_size=image_size,
        timesteps=timesteps,
        loss_type=loss_type,
        beta_schedule=beta_schedule,
        keep_center=keep_center,
        cond_channels=cond_channels,
        use_wce_input=use_wce_input
    )

    return model


if __name__ == "__main__":
    # Test DDPM
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_ddpm(
        image_size=(360, 640),
        dim=64,
        timesteps=1000
    ).to(device)

    # Test forward pass
    x = torch.randn(2, 1, 360, 640, device=device)
    loss = model(x)
    print(f"Training loss: {loss.item():.4f}")

    # Test sampling (partial for speed)
    print("Testing denoising...")
    noise = torch.randn(1, 1, 360, 640, device=device)
    denoised = model.denoise(noise, t_start=100)
    print(f"Denoised shape: {denoised.shape}")
