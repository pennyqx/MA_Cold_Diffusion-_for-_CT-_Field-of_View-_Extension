"""
UNet Architecture for CT Sinogram Processing

Architecture: ConvNeXt-based U-Net with:
- Sinusoidal time embedding for diffusion models
- Linear attention for efficient self-attention
- Skip connections between encoder and decoder

Input: (B, C, H, W) - Batch of 2D sinograms
Output: (B, C, H, W) - Predicted clean sinograms
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple, List


def exists(x):
    """Check if value exists (is not None)"""
    return x is not None


def default(val, d):
    """Return val if exists, otherwise return default d"""
    if exists(val):
        return val
    return d() if callable(d) else d


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embedding for timestep encoding.

    Maps scalar timestep t to a dim-dimensional embedding vector.
    Used to condition the network on the diffusion timestep.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) tensor of timesteps
        Returns:
            (B, dim) embedding vectors
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LayerNorm2d(nn.Module):
    """
    Layer normalization for 2D feature maps.
    Normalizes over the channel dimension.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma + self.beta


class PreNorm(nn.Module):
    """Pre-normalization wrapper for any module"""
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.fn(x)


class Residual(nn.Module):
    """Residual connection wrapper"""
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.fn(x, *args, **kwargs) + x


class ConvNextBlock(nn.Module):
    """
    ConvNeXt-style block for feature extraction.

    Architecture:
    1. Depthwise 7x7 conv
    2. Time embedding injection (optional)
    3. LayerNorm + 1x1 conv expansion + GELU + 1x1 conv compression
    4. Residual connection

    Reference: https://arxiv.org/abs/2201.03545
    """
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        time_emb_dim: Optional[int] = None,
        mult: int = 2,
        use_norm: bool = True
    ):
        super().__init__()

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, dim_in)
        ) if exists(time_emb_dim) else None

        # Depthwise convolution
        self.ds_conv = nn.Conv2d(dim_in, dim_in, 7, padding=3, groups=dim_in)

        # Main network
        self.net = nn.Sequential(
            LayerNorm2d(dim_in) if use_norm else nn.Identity(),
            nn.Conv2d(dim_in, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1)
        )

        # Residual projection
        self.res_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        time_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input features
            time_emb: (B, time_emb_dim) time embedding
        Returns:
            (B, C_out, H, W) output features
        """
        h = self.ds_conv(x)

        if exists(self.time_mlp) and exists(time_emb):
            condition = self.time_mlp(time_emb)
            h = h + rearrange(condition, 'b c -> b c 1 1')

        h = self.net(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    """
    Linear attention mechanism for efficient self-attention.

    Complexity: O(N * d^2) instead of O(N^2 * d) for standard attention.
    Suitable for high-resolution feature maps.
    """
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input features
        Returns:
            (B, C, H, W) attended features
        """
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads),
            qkv
        )

        q = q * self.scale
        k = k.softmax(dim=-1)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)

        return self.to_out(out)


class Downsample(nn.Module):
    """2x spatial downsampling"""
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """2x spatial upsampling"""
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for diffusion models.

    Architecture:
    - Encoder: ConvNextBlocks with downsampling
    - Bottleneck: ConvNextBlocks with attention
    - Decoder: ConvNextBlocks with upsampling and skip connections
    - Time embedding: Sinusoidal embedding injected into all blocks

    Args:
        dim: Base channel dimension
        out_dim: Output channels (default: same as input)
        dim_mults: Channel multipliers for each resolution level
        channels: Input channels
        with_time_emb: Whether to use time conditioning
        residual: Whether to add input to output (for residual learning)
    """
    def __init__(
        self,
        dim: int = 64,
        out_dim: Optional[int] = None,
        dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
        channels: int = 1,
        with_time_emb: bool = True,
        residual: bool = False
    ):
        super().__init__()

        self.channels = channels
        self.residual = residual

        # Channel dimensions at each level
        dims = [channels, *[dim * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        # Time embedding
        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        # Encoder
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ConvNextBlock(dim_in, dim_out, time_emb_dim=time_dim, use_norm=ind != 0),
                ConvNextBlock(dim_out, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        # Bottleneck
        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        # Decoder
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                ConvNextBlock(dim_out * 2, dim_in, time_emb_dim=time_dim),
                ConvNextBlock(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        # Output projection
        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            ConvNextBlock(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of U-Net.

        Args:
            x: (B, C, H, W) input sinogram
            time: (B,) diffusion timesteps

        Returns:
            (B, C, H, W) predicted clean sinogram
        """
        orig_x = x

        # Time embedding
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        # Encoder with skip connections
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # Bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # Decoder with skip connections
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat([x, h.pop()], dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        # Output
        out = self.final_conv(x)

        if self.residual:
            return out + orig_x
        return out


class UNetSimple(nn.Module):
    """
    Simplified U-Net without time embedding.
    Used for single-step inpainting baselines.
    """
    def __init__(
        self,
        dim: int = 64,
        out_dim: Optional[int] = None,
        dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
        channels: int = 1
    ):
        super().__init__()

        self.channels = channels
        dims = [channels, *[dim * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        # Encoder
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ConvNextBlock(dim_in, dim_out, use_norm=ind != 0),
                ConvNextBlock(dim_out, dim_out),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        # Bottleneck
        mid_dim = dims[-1]
        self.mid = nn.Sequential(
            ConvNextBlock(mid_dim, mid_dim),
            ConvNextBlock(mid_dim, mid_dim)
        )

        # Decoder
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                ConvNextBlock(dim_out * 2, dim_in),
                ConvNextBlock(dim_in, dim_in),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        # Output
        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            ConvNextBlock(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input
        Returns:
            (B, C, H, W) output
        """
        h = []

        # Encoder
        for block1, block2, downsample in self.downs:
            x = block1(x)
            x = block2(x)
            h.append(x)
            x = downsample(x)

        # Bottleneck
        x = self.mid(x)

        # Decoder
        for block1, block2, upsample in self.ups:
            x = torch.cat([x, h.pop()], dim=1)
            x = block1(x)
            x = block2(x)
            x = upsample(x)

        return self.final_conv(x)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test UNet
    model = UNet(dim=64, channels=1, dim_mults=(1, 2, 4, 8))
    print(f"UNet parameters: {count_parameters(model):,}")

    x = torch.randn(2, 1, 360, 640)
    t = torch.randint(0, 100, (2,))
    y = model(x, t)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    # Test UNetSimple
    model_simple = UNetSimple(dim=64, channels=1)
    print(f"UNetSimple parameters: {count_parameters(model_simple):,}")
    y_simple = model_simple(x)
    print(f"Simple Input: {x.shape} -> Output: {y_simple.shape}")
