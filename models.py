"""
models.py

Model backbones used by EDMPrecond (see edm.py).

Key points for EDM compatibility:
- The EDM wrapper calls: model(x_scaled, c_noise, class_labels=None, **kwargs)
  where:
    x_scaled: Tensor[B, C, H, W]
    c_noise:  Tensor[B]  ( = log(sigma)/4 )
- The backbone must return a Tensor[B, C, H, W] with the same spatial shape.
- We do NOT use class conditioning for now; class_labels is accepted but ignored.

This file includes:
  - DiTBackbone (vanilla DiT-style, Peebles & Xie 2023)
  - UNetBackbone (standard U-Net with residual blocks + GroupNorm)
  - ResNetBackbone (plain CNN ResNet with GroupNorm)

Defaults are tuned for the 28x28x1 parameter-matched trio:
  - DiT:   patch=2, dim=256, depth=8, heads=8, mlp_ratio=4    (~10.2M params)
  - UNet:  3 scales (47/94/188), 3 resblocks/scale, 3 bottleneck (~10.1M params)
  - ResNet: width=404, 6 residual blocks (1 conv/block)         (~10.0M params)

Parameter counts are printed by default when constructing each backbone
(set cfg.print_params=False to silence).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def maybe_print_params(name: str, model: nn.Module, enabled: bool = True) -> None:
    if enabled:
        n = count_parameters(model)
        print(f"[models] {name} params = {n/1e6:.3f} M")


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply AdaLN modulation: (1+scale) * x + shift.

    x:     [B, N, D]
    shift: [B, D]
    scale: [B, D]
    """
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Sinusoidal timestep embedding.
    t -> [cos(w_1 t), ..., cos(w_{d/2} t), sin(w_1 t), ..., sin(w_{d/2} t)]

    t:   [B] float tensor (in EDM we pass c_noise = log(sigma)/4)
    dim: embedding dimension

    Returns: [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def gn(num_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    """GroupNorm with a safe number of groups."""
    g = min(num_groups, num_channels)
    while num_channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, num_channels)


# -----------------------------
# Shared time embedding
# -----------------------------

class TimestepEmbedder(nn.Module):
    """Map scalar timestep/noise embedding to model width."""

    def __init__(self, model_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or model_dim * 4
        self.model_dim = model_dim
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, model_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = timestep_embedding(t, self.model_dim)
        return self.mlp(emb)


# -----------------------------
# DiT building blocks
# -----------------------------

class MultiHeadSelfAttention(nn.Module):
    """Simple multi-head self-attention with batch-first tensors.
    B: batch size, N: sequence length (number of tokens), D: model dimension
    """

    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x)  # [B, N, 3D]
        qkv = qkv.view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, H, N, Hd]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # [B, H, N, Hd]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class FusedMultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention using PyTorch fused SDPA kernels when available.

    Its parameters and initialization match ``MultiHeadSelfAttention``. This
    separate class lets memory-based DiT variants opt into SDPA without
    changing existing DiT checkpoints or numerical behavior.
    """

    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_drop = float(attn_drop)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(b, n, d)
        return self.proj_drop(self.proj(out))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DiTBlock(nn.Module):
    """A single DiT block with AdaLN-Zero conditioning."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = MultiHeadSelfAttention(dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = MLP(dim, hidden_dim=int(dim * mlp_ratio), drop=mlp_drop)

        self.adaLN = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        params = self.adaLN(c)  # [B, 6D]
        shift1, scale1, gate1, shift2, scale2, gate2 = params.chunk(6, dim=-1)

        h = self.norm1(x)
        h = _modulate(h, shift1, scale1)
        h = self.attn(h)
        x = x + gate1[:, None, :] * h

        h = self.norm2(x)
        h = _modulate(h, shift2, scale2)
        h = self.mlp(h)
        x = x + gate2[:, None, :] * h
        return x


# -----------------------------
# DiT Backbone
# -----------------------------

@dataclass
class DiTConfig:
    img_resolution: int = 28
    img_channels: int = 1
    patch_size: int = 2
    embed_dim: int = 256
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    mlp_drop: float = 0.0
    print_params: bool = True


class DiTBackbone(nn.Module):
    """Vanilla DiT-style Transformer backbone for EDM."""

    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.img_resolution % cfg.patch_size == 0, "img_resolution must be divisible by patch_size"
        self.grid_size = cfg.img_resolution // cfg.patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.patch_embed = nn.Conv2d(
            in_channels=cfg.img_channels,
            out_channels=cfg.embed_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
            padding=0,
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, cfg.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.time_embed = TimestepEmbedder(cfg.embed_dim)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    attn_drop=cfg.attn_drop,
                    proj_drop=cfg.proj_drop,
                    mlp_drop=cfg.mlp_drop,
                )
                for _ in range(cfg.depth)
            ]
        )

        self.final_norm = nn.LayerNorm(cfg.embed_dim, elementwise_affine=False)
        self.final_adaLN = nn.Linear(cfg.embed_dim, 2 * cfg.embed_dim)
        nn.init.zeros_(self.final_adaLN.weight)
        nn.init.zeros_(self.final_adaLN.bias)

        out_dim = cfg.img_channels * (cfg.patch_size ** 2)
        self.head = nn.Linear(cfg.embed_dim, out_dim)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        maybe_print_params("DiTBackbone", self, enabled=cfg.print_params)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        P = self.cfg.patch_size
        C = self.cfg.img_channels
        G = self.grid_size
        assert N == G * G
        x = x.view(B, G, G, C, P, P)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, C, G * P, G * P)
        return x

    def forward(self, x: torch.Tensor, c_noise: torch.Tensor, class_labels=None, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        if H != self.cfg.img_resolution or W != self.cfg.img_resolution or C != self.cfg.img_channels:
            raise ValueError(
                f"Expected [B,{self.cfg.img_channels},{self.cfg.img_resolution},{self.cfg.img_resolution}] "
                f"but got [B,{C},{H},{W}]."
            )

        h = self.patch_embed(x)                            # [B, D, G, G]
        h = h.flatten(2).transpose(1, 2).contiguous()      # [B, N, D]
        h = h + self.pos_embed

        c = self.time_embed(c_noise)                       # [B, D]
        for blk in self.blocks:
            h = blk(h, c)

        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        h = self.final_norm(h)
        h = _modulate(h, shift, scale)
        h = self.head(h)                                   # [B, N, C*P*P]
        y = self.unpatchify(h)
        return y


# -----------------------------
# CNN building blocks (UNet/ResNet)
# -----------------------------

class TimeMLP(nn.Module):
    """Embed c_noise into a time vector (shared across CNNs)."""

    def __init__(self, time_dim: int):
        super().__init__()
        self.embed = TimestepEmbedder(time_dim)

    def forward(self, c_noise: torch.Tensor) -> torch.Tensor:
        return self.embed(c_noise)  # [B, time_dim]


class ResBlockGN(nn.Module):
    """2-conv residual block with GroupNorm + SiLU + timestep conditioning."""

    def __init__(self, c_in: int, c_out: int, time_dim: int, groups: int = 32):
        super().__init__()
        self.norm1 = gn(c_in, groups)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_dim, c_out)

        self.norm2 = gn(c_out, groups)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, padding=1)

        self.skip = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, kernel_size=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return self.skip(x) + h


class Downsample(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Conv2d(c, c, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# -----------------------------
# UNet Backbone (28x28 tuned)
# -----------------------------

@dataclass
class UNetConfig:
    img_resolution: int = 28
    img_channels: int = 1
    base_channels: int = 47
    channel_mults: Tuple[int, ...] = (1, 2, 4)   # 64/128/256
    num_res_blocks: int = 3                      # per scale
    num_mid_blocks: int = 3                      # bottleneck
    time_dim: int = 256
    gn_groups: int = 32
    print_params: bool = True


class UNetBackbone(nn.Module):
    """Standard UNet with residual blocks (no self-attention), EDM-compatible."""

    def __init__(self, cfg: UNetConfig):
        super().__init__()
        self.cfg = cfg
        C = cfg.img_channels
        B = cfg.base_channels
        chs = [B * m for m in cfg.channel_mults]  # [B,2B,4B,...]

        self.time = TimeMLP(cfg.time_dim)
        self.in_conv = nn.Conv2d(C, chs[0], kernel_size=3, padding=1)

        # Down path
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_ch = chs[0]
        skips_channels: List[int] = []

        for si, out_ch in enumerate(chs):
            blocks = nn.ModuleList()
            for _ in range(cfg.num_res_blocks):
                blocks.append(ResBlockGN(in_ch, out_ch, cfg.time_dim, groups=cfg.gn_groups))
                in_ch = out_ch
                skips_channels.append(in_ch)
            self.down_blocks.append(blocks)
            self.downsamples.append(Downsample(in_ch) if si != len(chs) - 1 else nn.Identity())

        # Bottleneck
        self.mid_blocks = nn.ModuleList(
            [ResBlockGN(in_ch, in_ch, cfg.time_dim, groups=cfg.gn_groups) for _ in range(cfg.num_mid_blocks)]
        )

        # Up path
        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for si in reversed(range(len(chs))):
            out_ch = chs[si]
            self.upsamples.append(Upsample(in_ch) if si != len(chs) - 1 else nn.Identity())

            blocks = nn.ModuleList()
            for _ in range(cfg.num_res_blocks):
                skip_ch = skips_channels.pop()
                blocks.append(ResBlockGN(in_ch + skip_ch, out_ch, cfg.time_dim, groups=cfg.gn_groups))
                in_ch = out_ch
            self.up_blocks.append(blocks)

        self.out_norm = gn(in_ch, cfg.gn_groups)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(in_ch, C, kernel_size=3, padding=1)

        maybe_print_params("UNetBackbone", self, enabled=cfg.print_params)

    def forward(self, x: torch.Tensor, c_noise: torch.Tensor, class_labels=None, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        if H != self.cfg.img_resolution or W != self.cfg.img_resolution or C != self.cfg.img_channels:
            raise ValueError(
                f"Expected [B,{self.cfg.img_channels},{self.cfg.img_resolution},{self.cfg.img_resolution}] "
                f"but got [B,{C},{H},{W}]."
            )

        t_emb = self.time(c_noise)  # [B, time_dim]
        h = self.in_conv(x)

        skips: List[torch.Tensor] = []
        for blocks, down in zip(self.down_blocks, self.downsamples):
            for blk in blocks:
                h = blk(h, t_emb)
                skips.append(h)
            h = down(h)

        for blk in self.mid_blocks:
            h = blk(h, t_emb)

        for up, blocks in zip(self.upsamples, self.up_blocks):
            h = up(h)
            for blk in blocks:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = blk(h, t_emb)

        return self.out_conv(self.out_act(self.out_norm(h)))


# -----------------------------
# ResNet Backbone (28x28 tuned)
# -----------------------------

@dataclass
class ResNetConfig:
    img_resolution: int = 28
    img_channels: int = 1
    width: int = 404
    num_blocks: int = 6
    time_dim: int = 256
    gn_groups: int = 32
    print_params: bool = True


class ResNetBlock1Conv(nn.Module):
    """1-conv residual block with GN+SiLU and time conditioning."""

    def __init__(self, width: int, time_dim: int, groups: int = 32):
        super().__init__()
        self.norm = gn(width, groups)
        self.act = nn.SiLU()
        self.time_proj = nn.Linear(time_dim, width)
        self.conv = nn.Conv2d(width, width, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm(x))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv(h)
        return x + h


class ResNetBackbone(nn.Module):
    """Plain CNN ResNet (no self-attention), EDM-compatible."""

    def __init__(self, cfg: ResNetConfig):
        super().__init__()
        self.cfg = cfg
        self.time = TimeMLP(cfg.time_dim)

        C = cfg.img_channels
        W = cfg.width

        self.in_conv = nn.Conv2d(C, W, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([ResNetBlock1Conv(W, cfg.time_dim, groups=cfg.gn_groups) for _ in range(cfg.num_blocks)])
        self.out_norm = gn(W, cfg.gn_groups)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(W, C, kernel_size=3, padding=1)

        maybe_print_params("ResNetBackbone", self, enabled=cfg.print_params)

    def forward(self, x: torch.Tensor, c_noise: torch.Tensor, class_labels=None, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        if H != self.cfg.img_resolution or W != self.cfg.img_resolution or C != self.cfg.img_channels:
            raise ValueError(
                f"Expected [B,{self.cfg.img_channels},{self.cfg.img_resolution},{self.cfg.img_resolution}] "
                f"but got [B,{C},{H},{W}]."
            )

        t_emb = self.time(c_noise)
        h = self.in_conv(x)
        for blk in self.blocks:
            h = blk(h, t_emb)
        return self.out_conv(self.out_act(self.out_norm(h)))


# -----------------------------
# Convenience factory
# -----------------------------

def build_backbone(name: str, img_resolution: int = 28, img_channels: int = 1, print_params: bool = True) -> nn.Module:
    """Build default backbones for the 28x28 trio."""
    name = name.lower()
    if name == "dit":
        return DiTBackbone(DiTConfig(img_resolution=img_resolution, img_channels=img_channels, print_params=print_params))
    if name == "unet":
        return UNetBackbone(UNetConfig(img_resolution=img_resolution, img_channels=img_channels, print_params=print_params))
    if name == "resnet":
        return ResNetBackbone(ResNetConfig(img_resolution=img_resolution, img_channels=img_channels, print_params=print_params))
    raise ValueError("name must be one of: dit | unet | resnet")
