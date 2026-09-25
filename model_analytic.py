"""
model_analytic.py

Two-stage analytic denoiser for the image-denoising project / Mode A training.

This file supports both the original single analytic denoiser and a stacked
Euler-style refinement model.  The single-layer case is preserved by default:

    num_layers = 1  =>  output = D_theta(x_sigma; sigma)

For multiple layers, each layer computes a two-stage analytic denoiser and then
updates the current state through a denoiser-form Euler/residual step:

    d_l       = D_{theta_l}(x_l; sigma)
    x_{l+1}   = (1 - alpha_l) x_l + alpha_l d_l

where alpha_l is learnable by default for num_layers > 1 and initialized to
1 / num_layers.  This is meant to approximate repeated denoising/refinement while
keeping each layer's Stage-2 retrieval same-position over clean empirical
patches.

Normalization note:
    The legacy config names still say "l2" for CLI/backward compatibility, but
    enabled normalization now uses RMS normalization.  Stage 1 uses a learned
    RMSNorm gain on patch tokens before Q/K/V projection.  Stage 2 uses a shared
    learned RMSNorm gain for Q/K features before same-position RBF retrieval.

Core invariants:
    - The model directly predicts x0 from x_sigma. Do not wrap it in EDMPrecond.
    - Stage-2 keys and queries use identity-initialized per-position Q/K
      projections.
    - Stage-2 values use identity-initialized per-position V projections of raw
      clean memory patches.
    - At initialization, the model exactly recovers the previous analytic
      denoiser.
    - After training, if V projections differ from identity, the output is no
      longer guaranteed to be a convex combination of raw clean memory patches.
    - Stage-2 weights are a softmax over memory images for each same-position
      patch, so weights sum to 1.
    - Same-position retrieval is intentional: patch m only retrieves patch m from
      memory images.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
try:
    from torch.utils.checkpoint import checkpoint as torch_checkpoint
except ModuleNotFoundError:
    torch_checkpoint = None


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters in a module."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def _as_scalar_sigma(sigma, *, batch_size: int, device: torch.device) -> Optional[float]:
    """Convert sigma argument to a single Python float."""
    if sigma is None:
        return None
    if not torch.is_tensor(sigma):
        return float(sigma)

    sigma = sigma.to(device=device, dtype=torch.float32).flatten()
    if sigma.numel() == 1:
        return float(sigma.item())
    if sigma.numel() != batch_size:
        raise ValueError(
            f"sigma must be scalar or shape [B]; got shape {tuple(sigma.shape)} for B={batch_size}"
        )

    sigma_mean = sigma.mean()
    if not torch.allclose(sigma, sigma_mean.expand_as(sigma), rtol=1e-5, atol=1e-6):
        raise ValueError("This model assumes a single fixed sigma per batch.")
    return float(sigma_mean.item())


def rms_normalize_last(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS-normalize along the final dimension with stable fp32 statistics."""
    x_float = x.float()
    rms = x_float.pow(2).mean(dim=-1, keepdim=True).add(float(eps)).sqrt()
    return (x_float / rms).to(dtype=x.dtype)


class RMSNorm(nn.Module):
    """RMSNorm over the final dimension with a learned channel-wise gain."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.dim))

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"RMSNorm expected final dim {self.dim}, got {x.shape[-1]}")
        x_float = x.float()
        rms = x_float.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        y = x_float / rms
        weight = self.weight.to(device=x.device, dtype=y.dtype)
        return (y * weight).to(dtype=x.dtype)


def safe_logit(p: float) -> float:
    """Numerically stable logit for initializing alpha parameters."""
    p = min(max(float(p), 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


# -----------------------------------------------------------------------------
# Patchify / unpatchify
# -----------------------------------------------------------------------------


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert images [B,C,H,W] to patch tokens [B,N,D]."""
    if x.ndim != 4:
        raise ValueError(f"Expected x with shape [B,C,H,W], got {tuple(x.shape)}")

    b, c, h, w = x.shape
    p = int(patch_size)
    if h % p != 0 or w % p != 0:
        raise ValueError(f"H and W must be divisible by patch_size={p}; got H={h}, W={w}")

    gh, gw = h // p, w // p
    x = x.reshape(b, c, gh, p, gw, p)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    x = x.reshape(b, gh * gw, c * p * p)
    return x


def unpatchify(
    tokens: torch.Tensor,
    *,
    img_channels: int,
    img_resolution: int,
    patch_size: int,
) -> torch.Tensor:
    """Convert patch tokens [B,N,D] back to images [B,C,H,W]."""
    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens [B,N,D], got {tuple(tokens.shape)}")

    b, n, d = tokens.shape
    p = int(patch_size)
    c = int(img_channels)
    h = w = int(img_resolution)
    gh, gw = h // p, w // p

    expected_n = gh * gw
    expected_d = c * p * p
    if n != expected_n or d != expected_d:
        raise ValueError(
            f"Token shape mismatch: got [B,{n},{d}], expected [B,{expected_n},{expected_d}] "
            f"for C={c}, resolution={h}, patch_size={p}"
        )

    x = tokens.reshape(b, gh, gw, c, p, p)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.reshape(b, c, h, w)
    return x


# -----------------------------------------------------------------------------
# Fixed 2D sinusoidal positional encoding
# -----------------------------------------------------------------------------


def _build_1d_sincos_from_positions(pos: torch.Tensor, dim: int) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError(f"1D sin/cos embedding dim must be even, got dim={dim}")
    half_dim = dim // 2
    if half_dim == 0:
        raise ValueError("Embedding dim too small")
    omega = torch.arange(half_dim, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(half_dim - 1, 1)))
    out = pos[:, None] * omega[None, :]
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def build_2d_sincos_pos_embed(grid_h: int, grid_w: int, dim: int) -> torch.Tensor:
    """Build fixed 2D sin/cos positional encoding of shape [grid_h*grid_w, dim]."""
    if dim % 4 != 0:
        raise ValueError(
            f"2D sin/cos embedding dim must be divisible by 4, got dim={dim}. "
            "For RGB patch_size=2, token_dim=12 works."
        )

    y = torch.arange(grid_h, dtype=torch.float32)
    x = torch.arange(grid_w, dtype=torch.float32)
    if grid_h > 1:
        y = y / (grid_h - 1)
    if grid_w > 1:
        x = x / (grid_w - 1)

    yy, xx = torch.meshgrid(y, x, indexing="ij")
    yy = yy.reshape(-1)
    xx = xx.reshape(-1)
    emb_y = _build_1d_sincos_from_positions(yy, dim // 2)
    emb_x = _build_1d_sincos_from_positions(xx, dim // 2)
    return torch.cat([emb_y, emb_x], dim=1)


# -----------------------------------------------------------------------------
# Stage 1: no-residual RBF self-attention window selector
# -----------------------------------------------------------------------------


class RBFSelfAttentionWindow(nn.Module):
    """Stage-1 window selector W_theta built from no-residual RBF MHA."""

    def __init__(
        self,
        token_dim: int,
        num_heads: int = 8,
        key_dim: Optional[int] = None,
        query_chunk_size: int = 64,
        init_noise: float = 1e-3,
        l2_normalize_inputs: bool = True,
        l2_eps: float = 1e-6,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim) if key_dim is not None else int(token_dim)
        self.query_chunk_size = int(query_chunk_size)
        self.init_noise = float(init_noise)
        # Legacy name kept for compatibility; enabled path now means RMSNorm.
        self.l2_normalize_inputs = bool(l2_normalize_inputs)
        self.l2_eps = float(l2_eps)

        h = self.num_heads
        d = self.token_dim
        k = self.key_dim
        self.input_norm = RMSNorm(d, eps=self.l2_eps) if self.l2_normalize_inputs else nn.Identity()
        self.W_q = nn.Parameter(torch.empty(h, d, k))
        self.W_k = nn.Parameter(torch.empty(h, d, k))
        self.W_vo = nn.Parameter(torch.empty(h, d, d))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if isinstance(self.input_norm, RMSNorm):
                self.input_norm.reset_parameters()

            if self.token_dim == self.key_dim:
                eye = torch.eye(self.token_dim, dtype=self.W_q.dtype, device=self.W_q.device)
                self.W_q.copy_(eye[None].expand_as(self.W_q))
                self.W_k.copy_(eye[None].expand_as(self.W_k))
                if self.init_noise > 0:
                    self.W_q.add_(self.init_noise * torch.randn_like(self.W_q))
                    self.W_k.add_(self.init_noise * torch.randn_like(self.W_k))
            else:
                nn.init.xavier_uniform_(self.W_q)
                nn.init.xavier_uniform_(self.W_k)

            eye_v = torch.eye(self.token_dim, dtype=self.W_vo.dtype, device=self.W_vo.device)
            self.W_vo.copy_(eye_v[None].expand_as(self.W_vo))
            if self.init_noise > 0:
                self.W_vo.add_(self.init_noise * torch.randn_like(self.W_vo))

    def forward(self, tokens: torch.Tensor, pos_embed: torch.Tensor, *, tau) -> torch.Tensor:
        """Compute W_theta(tokens)."""
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens [B,M,D], got {tuple(tokens.shape)}")
        if pos_embed.ndim != 3 or pos_embed.shape[0] != 1:
            raise ValueError(f"Expected pos_embed [1,M,D], got {tuple(pos_embed.shape)}")

        _b, m, d = tokens.shape
        if d != self.token_dim:
            raise ValueError(f"Token dim mismatch: got D={d}, expected {self.token_dim}")
        if pos_embed.shape[1] != m or pos_embed.shape[2] != d:
            raise ValueError(f"pos_embed shape mismatch: got {tuple(pos_embed.shape)}, expected [1,{m},{d}]")

        tau_t = torch.as_tensor(tau, device=tokens.device, dtype=torch.float32).clamp_min(1e-8)

        x = tokens + pos_embed.to(device=tokens.device, dtype=tokens.dtype)
        if self.l2_normalize_inputs:
            x = self.input_norm(x)

        q = torch.einsum("bmd,hdk->bhmk", x, self.W_q)
        k = torch.einsum("bmd,hdk->bhmk", x, self.W_k)
        v = torch.einsum("bmd,hde->bhme", x, self.W_vo)

        out_chunks = []
        k_float = k.float()
        v_dtype = v.dtype
        k_norm2 = (k_float ** 2).sum(dim=-1)

        for start in range(0, m, self.query_chunk_size):
            end = min(start + self.query_chunk_size, m)
            q_chunk = q[:, :, start:end, :]
            q_float = q_chunk.float()
            q_norm2 = (q_float ** 2).sum(dim=-1, keepdim=True)
            dot = torch.einsum("bhqk,bhmk->bhqm", q_float, k_float)
            dist2 = (q_norm2 + k_norm2[:, :, None, :] - 2.0 * dot).clamp_min(0.0)
            logits = -dist2 / (2.0 * (tau_t ** 2))
            attn = torch.softmax(logits, dim=-1).to(dtype=v_dtype)
            out_chunk_h = torch.einsum("bhqm,bhme->bhqe", attn, v)
            out_chunks.append(out_chunk_h)

        out_h = torch.cat(out_chunks, dim=2)
        return out_h.sum(dim=1)

    def attention_weights(self, tokens: torch.Tensor, pos_embed: torch.Tensor, *, tau) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return Stage-1 RBF attention weights and value features.

        The returned attention has shape [B,H,M,M].  This is intended for
        diagnostics; the regular forward path stays chunked to keep memory low.
        """
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens [B,M,D], got {tuple(tokens.shape)}")
        if pos_embed.ndim != 3 or pos_embed.shape[0] != 1:
            raise ValueError(f"Expected pos_embed [1,M,D], got {tuple(pos_embed.shape)}")

        _b, m, d = tokens.shape
        if d != self.token_dim:
            raise ValueError(f"Token dim mismatch: got D={d}, expected {self.token_dim}")
        if pos_embed.shape[1] != m or pos_embed.shape[2] != d:
            raise ValueError(f"pos_embed shape mismatch: got {tuple(pos_embed.shape)}, expected [1,{m},{d}]")

        tau_t = torch.as_tensor(tau, device=tokens.device, dtype=torch.float32).clamp_min(1e-8)
        x = tokens + pos_embed.to(device=tokens.device, dtype=tokens.dtype)
        if self.l2_normalize_inputs:
            x = self.input_norm(x)

        q = torch.einsum("bmd,hdk->bhmk", x, self.W_q)
        k = torch.einsum("bmd,hdk->bhmk", x, self.W_k)
        v = torch.einsum("bmd,hde->bhme", x, self.W_vo)
        q_float = q.float()
        k_float = k.float()
        q_norm2 = (q_float ** 2).sum(dim=-1, keepdim=True)
        k_norm2 = (k_float ** 2).sum(dim=-1)
        dot = torch.einsum("bhqk,bhmk->bhqm", q_float, k_float)
        dist2 = (q_norm2 + k_norm2[:, :, None, :] - 2.0 * dot).clamp_min(0.0)
        logits = -dist2 / (2.0 * (tau_t ** 2))
        return torch.softmax(logits, dim=-1), v


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class AnalyticDenoiserConfig:
    img_resolution: int = 32
    img_channels: int = 3
    patch_size: int = 2
    num_layers: int = 1
    learn_update_alpha: bool = True
    init_update_alpha: Optional[float] = None
    single_layer_exact: bool = True
    num_heads: int = 8
    key_dim: Optional[int] = None
    stage1_query_chunk_size: int = 256
    init_noise: float = 1e-3
    # Legacy names retained for train/eval CLI compatibility. When True, these
    # now enable RMS normalization rather than unit L2 normalization.
    stage1_l2_normalize_inputs: bool = True
    sigma: float = 0.5
    strict_sigma: bool = False
    stage1_rbf_sigma: Optional[float] = None
    stage2_rbf_sigma: Optional[float] = None
    learn_stage1_temperature: bool = True
    learn_stage2_temperature: bool = True
    init_stage1_log_tau_scale: float = 0.0
    init_stage2_log_tau_scale: float = 0.0
    min_tau: float = 1e-4
    stage2_l2_normalize_qk: bool = True
    stage2_learn_q_proj: bool = True
    stage2_learn_k_proj: bool = True
    stage2_learn_v_proj: bool = True
    l2_eps: float = 1e-6
    memory_chunk_size: int = 256
    clamp_output: bool = False
    memory_bank_size: int = 2048
    learned_memory: bool = False
    learned_memory_init: str = "random_train_images"
    print_params: bool = True


class AnalyticDenoiserLayer(nn.Module):
    """One analytic denoising layer plus optional residual update coefficient."""

    def __init__(self, cfg: AnalyticDenoiserConfig, *, token_dim: int, num_patches: int, layer_idx: int, init_alpha: float):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = int(layer_idx)
        self.num_patches = int(num_patches)
        self.token_dim = int(token_dim)
        self.window_encoder = RBFSelfAttentionWindow(
            token_dim=token_dim,
            num_heads=cfg.num_heads,
            key_dim=cfg.key_dim,
            query_chunk_size=cfg.stage1_query_chunk_size,
            init_noise=cfg.init_noise,
            l2_normalize_inputs=cfg.stage1_l2_normalize_inputs,
            l2_eps=cfg.l2_eps,
        )
        self.stage2_qk_norm = RMSNorm(token_dim, eps=cfg.l2_eps) if cfg.stage2_l2_normalize_qk else nn.Identity()
        self.stage2_W_q = nn.Parameter(
            torch.empty(self.num_patches, self.token_dim, self.token_dim),
            requires_grad=cfg.stage2_learn_q_proj,
        )
        self.stage2_W_k = nn.Parameter(
            torch.empty(self.num_patches, self.token_dim, self.token_dim),
            requires_grad=cfg.stage2_learn_k_proj,
        )
        self.stage2_W_v = nn.Parameter(
            torch.empty(self.num_patches, self.token_dim, self.token_dim),
            requires_grad=cfg.stage2_learn_v_proj,
        )
        self.reset_stage2_projections()
        if cfg.learn_stage1_temperature:
            self.log_stage1_tau_scale = nn.Parameter(torch.tensor(float(cfg.init_stage1_log_tau_scale)))
        else:
            self.register_buffer("log_stage1_tau_scale", torch.tensor(float(cfg.init_stage1_log_tau_scale)))
        if cfg.learn_stage2_temperature:
            self.log_stage2_tau_scale = nn.Parameter(torch.tensor(float(cfg.init_stage2_log_tau_scale)))
        else:
            self.register_buffer("log_stage2_tau_scale", torch.tensor(float(cfg.init_stage2_log_tau_scale)))
        self.exact_alpha_one = bool(cfg.single_layer_exact and cfg.num_layers == 1 and init_alpha >= 1.0)
        if self.exact_alpha_one:
            self.register_buffer("logit_update_alpha", torch.tensor(safe_logit(1.0 - 1e-4)))
        elif cfg.learn_update_alpha:
            self.logit_update_alpha = nn.Parameter(torch.tensor(safe_logit(init_alpha)))
        else:
            self.register_buffer("logit_update_alpha", torch.tensor(safe_logit(init_alpha)))

    def reset_stage2_projections(self) -> None:
        """Initialize all per-position Stage-2 projections exactly to identity."""
        with torch.no_grad():
            eye = torch.eye(self.token_dim, dtype=self.stage2_W_q.dtype, device=self.stage2_W_q.device)
            self.stage2_W_q.copy_(eye[None].expand(self.num_patches, self.token_dim, self.token_dim))
            eye = eye.to(dtype=self.stage2_W_k.dtype, device=self.stage2_W_k.device)
            self.stage2_W_k.copy_(eye[None].expand(self.num_patches, self.token_dim, self.token_dim))
            eye = eye.to(dtype=self.stage2_W_v.dtype, device=self.stage2_W_v.device)
            self.stage2_W_v.copy_(eye[None].expand(self.num_patches, self.token_dim, self.token_dim))

    def update_alpha(self, *, device=None, dtype=torch.float32) -> torch.Tensor:
        if device is None:
            device = self.logit_update_alpha.device
        if self.exact_alpha_one:
            return torch.ones((), device=device, dtype=dtype)
        return torch.sigmoid(self.logit_update_alpha.to(device=device, dtype=dtype))

    def stage2_normalize_qk(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize Stage-2 Q/K features using the configured RMSNorm path."""
        if self.cfg.stage2_l2_normalize_qk:
            return self.stage2_qk_norm(x)
        return x

    def stage2_project_q(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...md,mde->...me", x, self.stage2_W_q.to(device=x.device, dtype=x.dtype))

    def stage2_project_k(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...md,mde->...me", x, self.stage2_W_k.to(device=x.device, dtype=x.dtype))

    def stage2_project_v(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...md,mde->...me", x, self.stage2_W_v.to(device=x.device, dtype=x.dtype))

    def resolve_effective_taus(self, *, sigma_in: float, base1: float, base2: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = self.log_stage1_tau_scale.dtype
        tau1 = torch.as_tensor(base1, device=device, dtype=dtype) * self.log_stage1_tau_scale.to(device=device).exp()
        tau2 = torch.as_tensor(base2, device=device, dtype=dtype) * self.log_stage2_tau_scale.to(device=device).exp()
        return tau1.clamp_min(float(self.cfg.min_tau)), tau2.clamp_min(float(self.cfg.min_tau))

    def temperature_info(self, *, sigma_in: float, base1: float, base2: float, device: torch.device) -> Dict[str, float]:
        with torch.no_grad():
            tau1, tau2 = self.resolve_effective_taus(sigma_in=sigma_in, base1=base1, base2=base2, device=device)
            alpha = self.update_alpha(device=device, dtype=torch.float32)
            return {
                "layer": int(self.layer_idx),
                "tau_stage1": float(tau1.detach().float().cpu()),
                "tau_stage2": float(tau2.detach().float().cpu()),
                "stage1_tau_scale": float(self.log_stage1_tau_scale.detach().float().exp().cpu()),
                "stage2_tau_scale": float(self.log_stage2_tau_scale.detach().float().exp().cpu()),
                "alpha": float(alpha.detach().float().cpu()),
            }


class TwoStageAnalyticDenoiser(nn.Module):
    """Stacked analytic denoiser with Euler-style denoiser residual updates."""

    def __init__(self, cfg: AnalyticDenoiserConfig = AnalyticDenoiserConfig()):
        super().__init__()
        self.cfg = cfg
        if cfg.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if cfg.patch_size == 1:
            raise ValueError("patch_size=1 is not supported by current 2D sin/cos positional encoding.")
        if cfg.img_resolution % cfg.patch_size != 0:
            raise ValueError("img_resolution must be divisible by patch_size")
        if cfg.memory_chunk_size <= 0:
            raise ValueError("memory_chunk_size must be positive")
        if cfg.sigma <= 0:
            raise ValueError("sigma must be positive")
        if cfg.min_tau <= 0:
            raise ValueError("min_tau must be positive")

        self.grid_size = cfg.img_resolution // cfg.patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.token_dim = cfg.img_channels * (cfg.patch_size ** 2)
        pos_embed = build_2d_sincos_pos_embed(self.grid_size, self.grid_size, self.token_dim)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0), persistent=True)
        init_alpha = 1.0 if cfg.init_update_alpha is None and cfg.num_layers == 1 else (
            1.0 / float(cfg.num_layers) if cfg.init_update_alpha is None else float(cfg.init_update_alpha)
        )
        if init_alpha <= 0:
            raise ValueError("init_update_alpha must be positive")
        self.layers = nn.ModuleList([
            AnalyticDenoiserLayer(cfg, token_dim=self.token_dim, num_patches=self.num_patches, layer_idx=i, init_alpha=init_alpha)
            for i in range(cfg.num_layers)
        ])
        self.register_buffer("memory_bank", torch.empty(0), persistent=False)
        if cfg.learned_memory:
            if cfg.memory_bank_size <= 0:
                raise ValueError("memory_bank_size must be positive when learned_memory=True")
            self.learned_memory = nn.Parameter(
                torch.empty(cfg.memory_bank_size, self.num_patches, self.token_dim)
            )
            nn.init.uniform_(self.learned_memory, -1.0, 1.0)
        else:
            self.register_parameter("learned_memory", None)
        if cfg.print_params:
            n = count_parameters(self)
            print(
                f"[model_analytic] TwoStageAnalyticDenoiser params = {n} ({n / 1e6:.6f} M); "
                f"layers={cfg.num_layers}, token_dim={self.token_dim}, patches={self.num_patches}, "
                f"heads={cfg.num_heads}, stage1_rms={cfg.stage1_l2_normalize_inputs}, "
                f"stage2_rms_qk={cfg.stage2_l2_normalize_qk}, "
                f"stage2_qkv_proj=({cfg.stage2_learn_q_proj}, {cfg.stage2_learn_k_proj}, {cfg.stage2_learn_v_proj}), "
                f"learned_memory={cfg.learned_memory}"
            )

    @property
    def window_encoder(self) -> RBFSelfAttentionWindow:
        return self.layers[0].window_encoder

    @torch.no_grad()
    def set_memory_bank(self, memory: torch.Tensor) -> None:
        self._validate_images(memory, name="memory")
        self.memory_bank = memory.detach().clone()

    def clear_memory_bank(self) -> None:
        self.memory_bank = torch.empty(0, device=self.pos_embed.device)

    @torch.no_grad()
    def initialize_learned_memory(self, memory: torch.Tensor) -> None:
        if self.learned_memory is None:
            raise ValueError("Cannot initialize learned memory when cfg.learned_memory=False.")
        self._validate_images(memory, name="learned_memory_init")
        if int(memory.shape[0]) != int(self.cfg.memory_bank_size):
            raise ValueError(
                f"learned_memory_init must contain {self.cfg.memory_bank_size} images, got {memory.shape[0]}"
            )
        memory_tokens = patchify(
            memory.to(device=self.learned_memory.device, dtype=self.learned_memory.dtype),
            self.cfg.patch_size,
        )
        self.learned_memory.copy_(memory_tokens)

    def learned_memory_size(self) -> int:
        if self.learned_memory is None:
            return 0
        return int(self.learned_memory.shape[0])

    def _get_memory(self, memory: Optional[torch.Tensor], *, device, dtype) -> torch.Tensor:
        if memory is None:
            if self.learned_memory is not None:
                return self.learned_memory.to(device=device, dtype=dtype)
            elif self.memory_bank.numel() == 0:
                raise ValueError("No memory passed to forward() and no registered memory_bank found.")
            else:
                memory = self.memory_bank
        if memory.ndim == 3:
            if memory.shape[1:] != (self.num_patches, self.token_dim):
                raise ValueError(
                    f"memory tokens must have shape [R,{self.num_patches},{self.token_dim}], got {tuple(memory.shape)}"
                )
        else:
            self._validate_images(memory, name="memory")
        return memory.to(device=device, dtype=dtype, non_blocking=True)

    def _validate_images(self, x: torch.Tensor, *, name: str) -> None:
        if x.ndim != 4:
            raise ValueError(f"{name} must have shape [B,C,H,W], got {tuple(x.shape)}")
        _, c, h, w = x.shape
        if c != self.cfg.img_channels or h != self.cfg.img_resolution or w != self.cfg.img_resolution:
            raise ValueError(
                f"{name} must have shape [B,{self.cfg.img_channels},{self.cfg.img_resolution},{self.cfg.img_resolution}], "
                f"got {tuple(x.shape)}"
            )

    def _resolve_base_sigmas(self, sigma, batch_size: int, device: torch.device) -> Tuple[float, float, float]:
        sigma_in = _as_scalar_sigma(sigma, batch_size=batch_size, device=device)
        if sigma_in is None:
            sigma_in = float(self.cfg.sigma)
        if sigma_in <= 0:
            raise ValueError("sigma must be positive")
        if self.cfg.strict_sigma and not math.isclose(sigma_in, float(self.cfg.sigma), rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError(f"Model is configured for sigma={self.cfg.sigma}, but got sigma={sigma_in}")
        base1 = float(self.cfg.stage1_rbf_sigma) if self.cfg.stage1_rbf_sigma is not None else sigma_in
        base2 = float(self.cfg.stage2_rbf_sigma) if self.cfg.stage2_rbf_sigma is not None else sigma_in
        if base1 <= 0 or base2 <= 0:
            raise ValueError("RBF base bandwidths must be positive")
        return sigma_in, base1, base2

    def resolve_effective_taus(self, sigma=None, *, batch_size: int = 1, device=None, layer_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor, float]:
        if device is None:
            device = self.pos_embed.device
        sigma_in, base1, base2 = self._resolve_base_sigmas(sigma, batch_size=batch_size, device=device)
        tau1, tau2 = self.layers[int(layer_idx)].resolve_effective_taus(sigma_in=sigma_in, base1=base1, base2=base2, device=device)
        return tau1, tau2, sigma_in

    def resolve_all_effective_taus(self, sigma=None, *, batch_size: int = 1, device=None) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], float]:
        if device is None:
            device = self.pos_embed.device
        sigma_in, base1, base2 = self._resolve_base_sigmas(sigma, batch_size=batch_size, device=device)
        return [layer.resolve_effective_taus(sigma_in=sigma_in, base1=base1, base2=base2, device=device) for layer in self.layers], sigma_in

    def temperature_info(self, sigma=None, *, batch_size: int = 1, device=None) -> dict:
        if device is None:
            device = self.pos_embed.device
        sigma_in, base1, base2 = self._resolve_base_sigmas(sigma, batch_size=batch_size, device=device)
        layer_infos = [layer.temperature_info(sigma_in=sigma_in, base1=base1, base2=base2, device=device) for layer in self.layers]
        tau1s = [d["tau_stage1"] for d in layer_infos]
        tau2s = [d["tau_stage2"] for d in layer_infos]
        alphas = [d["alpha"] for d in layer_infos]
        return {
            "sigma": float(sigma_in),
            "tau_stage1": float(sum(tau1s) / len(tau1s)),
            "tau_stage2": float(sum(tau2s) / len(tau2s)),
            "stage1_tau_scale": float(sum(d["stage1_tau_scale"] for d in layer_infos) / len(layer_infos)),
            "stage2_tau_scale": float(sum(d["stage2_tau_scale"] for d in layer_infos) / len(layer_infos)),
            "alpha": float(sum(alphas) / len(alphas)),
            "num_layers": int(len(self.layers)),
            "layers": layer_infos,
        }

    def _same_position_rbf_cross_attention(
        self,
        *,
        layer: AnalyticDenoiserLayer,
        q_tokens: torch.Tensor,
        mem_tokens: torch.Tensor,
        tau_stage1,
        tau_stage2,
    ) -> torch.Tensor:
        if q_tokens.ndim != 3 or mem_tokens.ndim != 3:
            raise ValueError("q_tokens and mem_tokens must be rank-3 tensors")
        b, m, d = q_tokens.shape
        r, m_mem, d_mem = mem_tokens.shape
        if m != m_mem or d != d_mem:
            raise ValueError(f"Shape mismatch: q_tokens [B,{m},{d}] vs mem_tokens [R,{m_mem},{d_mem}]")
        tau2 = torch.as_tensor(tau_stage2, device=q_tokens.device, dtype=torch.float32).clamp_min(float(self.cfg.min_tau))
        running_max = torch.full((b, m), -float("inf"), device=q_tokens.device, dtype=torch.float32)
        running_sum = torch.zeros((b, m), device=q_tokens.device, dtype=torch.float32)
        running_num = torch.zeros((b, m, d), device=q_tokens.device, dtype=torch.float32)

        q_base = layer.stage2_normalize_qk(q_tokens).float()
        q_feat = layer.stage2_project_q(q_base).float()
        q_norm2 = (q_feat ** 2).sum(dim=-1, keepdim=True)
        chunk_size = int(self.cfg.memory_chunk_size)
        for start in range(0, r, chunk_size):
            end = min(start + chunk_size, r)
            mem_chunk = mem_tokens[start:end]
            k_chunk = layer.window_encoder(mem_chunk, self.pos_embed, tau=tau_stage1)
            v_chunk = mem_chunk
            k_base = layer.stage2_normalize_qk(k_chunk).float()
            k_feat = layer.stage2_project_k(k_base).float()
            v_float = layer.stage2_project_v(v_chunk.float()).float()
            k_norm2 = (k_feat ** 2).sum(dim=-1).transpose(0, 1).unsqueeze(0)
            dot = torch.einsum("bmd,rmd->bmr", q_feat, k_feat)
            dist2 = (q_norm2 + k_norm2 - 2.0 * dot).clamp_min(0.0)
            logits = -dist2 / (2.0 * (tau2 ** 2))
            block_max = logits.max(dim=-1).values
            new_max = torch.maximum(running_max, block_max)
            old_scale = torch.exp(running_max - new_max)
            exp_logits = torch.exp(logits - new_max[..., None])
            block_sum = exp_logits.sum(dim=-1)
            block_num = torch.einsum("bmr,rmd->bmd", exp_logits, v_float)
            running_num = running_num * old_scale[..., None] + block_num
            running_sum = running_sum * old_scale + block_sum
            running_max = new_max
        pred_tokens = running_num / running_sum.clamp_min(1e-20)[..., None]
        return pred_tokens.to(dtype=q_tokens.dtype)

    def denoise_one_layer(self, x_state: torch.Tensor, mem_tokens: torch.Tensor, *, layer_idx: int, sigma) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        layer = self.layers[int(layer_idx)]
        b = x_state.shape[0]
        tau1, tau2, sigma_in = self.resolve_effective_taus(sigma, batch_size=b, device=x_state.device, layer_idx=layer_idx)
        dtype = layer.window_encoder.W_q.dtype
        x_tokens = patchify(x_state.to(dtype=dtype), self.cfg.patch_size)
        q_tokens = layer.window_encoder(x_tokens, self.pos_embed, tau=tau1)
        pred_tokens = self._same_position_rbf_cross_attention(layer=layer, q_tokens=q_tokens, mem_tokens=mem_tokens, tau_stage1=tau1, tau_stage2=tau2)
        denoised = unpatchify(pred_tokens, img_channels=self.cfg.img_channels, img_resolution=self.cfg.img_resolution, patch_size=self.cfg.patch_size).float()
        alpha = layer.update_alpha(device=x_state.device, dtype=torch.float32)
        updated = (1.0 - alpha) * x_state.float() + alpha * denoised
        if self.cfg.clamp_output:
            updated = updated.clamp(-1.0, 1.0)
        info = {
            "layer": int(layer_idx),
            "sigma": float(sigma_in),
            "tau_stage1": float(tau1.detach().float().cpu()),
            "tau_stage2": float(tau2.detach().float().cpu()),
            "alpha": float(alpha.detach().float().cpu()),
        }
        return denoised, updated, info

    def forward(
        self,
        x: torch.Tensor,
        sigma=None,
        class_labels=None,
        *,
        memory: Optional[torch.Tensor] = None,
        return_all: bool = False,
        checkpoint_layers: bool = False,
        **kwargs,
    ):
        del class_labels, kwargs
        self._validate_images(x, name="x")
        if checkpoint_layers and return_all:
            raise ValueError("checkpoint_layers=True is only supported with return_all=False.")
        inferred_checkpoint_layers = bool(checkpoint_layers)
        if (not inferred_checkpoint_layers) and self.training and (not return_all) and len(self.layers) > 1 and x.requires_grad:
            inferred_checkpoint_layers = True
        dtype = self.layers[0].window_encoder.W_q.dtype
        mem = self._get_memory(memory, device=x.device, dtype=dtype)
        if mem.ndim == 3:
            mem_tokens = mem
        else:
            mem_tokens = patchify(mem, self.cfg.patch_size)
        x_state = x.float()
        if inferred_checkpoint_layers and self.training and not x_state.requires_grad:
            x_state = x_state.detach().requires_grad_(True)
        states: List[torch.Tensor] = [x_state]
        denoised_outputs: List[torch.Tensor] = []
        layer_infos: List[dict] = []
        for layer_idx in range(len(self.layers)):
            if inferred_checkpoint_layers and self.training:
                if torch_checkpoint is None:
                    raise RuntimeError("torch.utils.checkpoint is unavailable; rerun without checkpoint_layers.")

                def _layer_forward(inp: torch.Tensor, idx: int = layer_idx) -> torch.Tensor:
                    _denoised, updated, _info = self.denoise_one_layer(inp, mem_tokens, layer_idx=idx, sigma=sigma)
                    return updated
                x_state = torch_checkpoint(_layer_forward, x_state, use_reentrant=False, preserve_rng_state=False)
            else:
                denoised, x_state, info = self.denoise_one_layer(x_state, mem_tokens, layer_idx=layer_idx, sigma=sigma)
                denoised_outputs.append(denoised)
                states.append(x_state)
                layer_infos.append(info)
        final = x_state
        if self.cfg.clamp_output:
            final = final.clamp(-1.0, 1.0)
            if states:
                states[-1] = final
        if return_all:
            return {"final": final.float(), "states": states, "denoised": denoised_outputs, "layer_infos": layer_infos}
        return final.float()


def upgrade_legacy_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map old single-layer keys and synthesize missing identity/RMSNorm tensors."""
    prefix = "_orig_mod."
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith(prefix):
            k = k[len(prefix):]
        if k.startswith("layers.") or k == "pos_embed":
            out[k] = v
        elif k.startswith("window_encoder."):
            out[f"layers.0.{k}"] = v
        elif k in {"log_stage1_tau_scale", "log_stage2_tau_scale", "logit_update_alpha"}:
            out[f"layers.0.{k}"] = v
        else:
            out[k] = v

    pos_embed = out.get("pos_embed", None)
    num_patches = int(pos_embed.shape[1]) if torch.is_tensor(pos_embed) and pos_embed.ndim == 3 else None

    # Old checkpoints do not contain RMSNorm gains or Stage-2 projection
    # matrices.  Initialize gains to ones and projections exactly to identity.
    for k, v in list(out.items()):
        if not k.endswith("window_encoder.W_q"):
            continue
        if v.ndim != 3:
            continue
        d = int(v.shape[1])
        layer_prefix = k[: -len("window_encoder.W_q")]
        window_prefix = layer_prefix + "window_encoder."
        dtype = v.dtype if torch.is_floating_point(v) else torch.float32
        device = v.device
        out.setdefault(f"{window_prefix}input_norm.weight", torch.ones(d, dtype=dtype, device=device))
        out.setdefault(f"{layer_prefix}stage2_qk_norm.weight", torch.ones(d, dtype=dtype, device=device))
        if num_patches is not None:
            eye = torch.eye(d, dtype=dtype, device=device)[None].expand(num_patches, d, d).clone()
            out.setdefault(f"{layer_prefix}stage2_W_q", eye.clone())
            out.setdefault(f"{layer_prefix}stage2_W_k", eye.clone())
            out.setdefault(f"{layer_prefix}stage2_W_v", eye.clone())
    return out


def build_analytic_denoiser(
    img_resolution: int = 32,
    img_channels: int = 3,
    patch_size: int = 2,
    num_heads: int = 8,
    sigma: float = 0.5,
    key_dim: Optional[int] = None,
    stage1_rbf_sigma: Optional[float] = None,
    stage2_rbf_sigma: Optional[float] = None,
    memory_chunk_size: int = 256,
    print_params: bool = True,
    **kwargs,
) -> TwoStageAnalyticDenoiser:
    cfg = AnalyticDenoiserConfig(
        img_resolution=img_resolution,
        img_channels=img_channels,
        patch_size=patch_size,
        num_heads=num_heads,
        sigma=sigma,
        key_dim=key_dim,
        stage1_rbf_sigma=stage1_rbf_sigma,
        stage2_rbf_sigma=stage2_rbf_sigma,
        memory_chunk_size=memory_chunk_size,
        print_params=print_params,
        **kwargs,
    )
    return TwoStageAnalyticDenoiser(cfg)


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = AnalyticDenoiserConfig(
        img_resolution=32,
        img_channels=3,
        patch_size=2,
        num_heads=8,
        sigma=0.5,
        num_layers=2,
        strict_sigma=False,
        stage1_query_chunk_size=256,
        memory_chunk_size=16,
        print_params=True,
    )
    model = TwoStageAnalyticDenoiser(cfg)
    x0 = torch.randn(4, 3, 32, 32).clamp(-1, 1)
    mem = torch.randn(32, 3, 32, 32).clamp(-1, 1)
    sigma = 0.5
    x_sigma = x0 + sigma * torch.randn_like(x0)
    out = model(x_sigma, sigma=sigma, memory=mem, return_all=True)
    x_sigma_req = x_sigma.detach().requires_grad_(True)
    out_ckpt = model(x_sigma_req, sigma=sigma, memory=mem, checkpoint_layers=True)
    print("final shape:", tuple(out["final"].shape))
    print("ckpt final shape:", tuple(out_ckpt.shape))
    print("num states:", len(out["states"]))
    print("temperature info:", model.temperature_info(sigma=sigma))
    print("smoke-test mse:", float(((out["final"] - x0) ** 2).mean()))
