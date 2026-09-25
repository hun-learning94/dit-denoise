"""
model_dpd.py

Deep patchwise denoiser (DPD).

This is a time-conditioned learned-memory variant of the analytic patchwise
denoiser.  It directly predicts clean images from noisy images and sigma:

    D_theta(x_sigma; sigma) ~= x0

No EDM preconditioning is applied.  The model assumes one scalar sigma per
minibatch.  Memory is a fixed-size learned bank of patch tokens.
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

from model_analytic import (
    build_2d_sincos_pos_embed,
    count_parameters,
    patchify,
    safe_logit,
    unpatchify,
)


def _as_scalar_sigma(sigma, *, batch_size: int, device: torch.device) -> float:
    if sigma is None:
        raise ValueError("DPD requires sigma.")
    if not torch.is_tensor(sigma):
        sigma_in = float(sigma)
    else:
        sigma_t = sigma.to(device=device, dtype=torch.float32).flatten()
        if sigma_t.numel() == 1:
            sigma_in = float(sigma_t.item())
        elif sigma_t.numel() == batch_size:
            sigma_mean = sigma_t.mean()
            if not torch.allclose(sigma_t, sigma_mean.expand_as(sigma_t), rtol=1e-5, atol=1e-6):
                raise ValueError("DPD currently assumes one scalar sigma per minibatch.")
            sigma_in = float(sigma_mean.item())
        else:
            raise ValueError(f"sigma must be scalar or shape [B], got {tuple(sigma_t.shape)}")
    if sigma_in <= 0:
        raise ValueError("sigma must be positive")
    return sigma_in


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal embedding for scalar log-sigma coordinate."""
    if t.ndim == 0:
        t = t[None]
    half = dim // 2
    freqs = torch.exp(
        -math.log(float(max_period)) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half, 1)
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() / x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (y * self.weight.to(device=x.device, dtype=y.dtype)).to(dtype=x.dtype)


class LayerNormNoAffine(nn.Module):
    """LayerNorm over channels without trainable affine parameters."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float()
        y = (y - y.mean(dim=-1, keepdim=True)) / y.var(dim=-1, keepdim=True, unbiased=False).add(self.eps).sqrt()
        return y.to(dtype=x.dtype)


class AdaRMSNorm(nn.Module):
    def __init__(self, dim: int, time_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.mod = nn.Linear(int(time_dim), 2 * int(dim))
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        y = x.float() / x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        scale, shift = self.mod(cond.float()).chunk(2, dim=-1)
        while scale.ndim < y.ndim:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        y = y * (1.0 + scale.to(device=x.device, dtype=y.dtype)) + shift.to(device=x.device, dtype=y.dtype)
        return y.to(dtype=x.dtype)


class AdaLayerNorm(nn.Module):
    """AdaLN over channels: LayerNorm without base affine, plus time shift/scale."""

    def __init__(self, dim: int, time_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.mod = nn.Linear(int(time_dim), 2 * int(dim))
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        y = x.float()
        y = (y - y.mean(dim=-1, keepdim=True)) / y.var(dim=-1, keepdim=True, unbiased=False).add(self.eps).sqrt()
        shift, scale = self.mod(cond.float()).chunk(2, dim=-1)
        while scale.ndim < y.ndim:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        y = y * (1.0 + scale.to(device=x.device, dtype=y.dtype)) + shift.to(device=x.device, dtype=y.dtype)
        return y.to(dtype=x.dtype)


def make_norm(
    *,
    dim: int,
    eps: float,
    scheme: str,
    time_dim: int,
    time_adaptive: bool,
) -> nn.Module:
    if scheme == "rms":
        return AdaRMSNorm(dim, time_dim, eps=eps) if time_adaptive else RMSNorm(dim, eps=eps)
    if scheme == "layernorm":
        return AdaLayerNorm(dim, time_dim, eps=eps) if time_adaptive else LayerNormNoAffine(dim, eps=eps)
    raise ValueError("normalization_scheme must be rms | layernorm")


class TimeRBFSelfAttentionWindow(nn.Module):
    """No-residual RBF self-attention window selector with optional AdaRMS."""

    def __init__(
        self,
        *,
        token_dim: int,
        num_heads: int,
        key_dim: Optional[int],
        query_chunk_size: int,
        init_noise: float,
        normalize_inputs: bool,
        normalization_scheme: str,
        time_dim: int,
        time_adarms: bool,
        eps: float,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim) if key_dim is not None else int(token_dim)
        self.query_chunk_size = int(query_chunk_size)
        self.normalize_inputs = bool(normalize_inputs)
        self.normalization_scheme = str(normalization_scheme)
        self.time_adarms = bool(time_adarms)
        self.eps = float(eps)

        if self.normalize_inputs:
            self.input_norm = make_norm(
                dim=token_dim,
                eps=eps,
                scheme=self.normalization_scheme,
                time_dim=time_dim,
                time_adaptive=time_adarms,
            )
        else:
            self.input_norm = nn.Identity()

        h, d, k = self.num_heads, self.token_dim, self.key_dim
        self.W_q = nn.Parameter(torch.empty(h, d, k))
        self.W_k = nn.Parameter(torch.empty(h, d, k))
        self.W_vo = nn.Parameter(torch.empty(h, d, d))
        self.reset_parameters(float(init_noise))

    def reset_parameters(self, init_noise: float) -> None:
        with torch.no_grad():
            if self.token_dim == self.key_dim:
                eye = torch.eye(self.token_dim, dtype=self.W_q.dtype, device=self.W_q.device)
                self.W_q.copy_(eye[None].expand_as(self.W_q))
                self.W_k.copy_(eye[None].expand_as(self.W_k))
                if init_noise > 0:
                    self.W_q.add_(init_noise * torch.randn_like(self.W_q))
                    self.W_k.add_(init_noise * torch.randn_like(self.W_k))
            else:
                nn.init.xavier_uniform_(self.W_q)
                nn.init.xavier_uniform_(self.W_k)
            eye_v = torch.eye(self.token_dim, dtype=self.W_vo.dtype, device=self.W_vo.device)
            self.W_vo.copy_(eye_v[None].expand_as(self.W_vo))
            if init_noise > 0:
                self.W_vo.add_(init_noise * torch.randn_like(self.W_vo))

    def _norm(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if not self.normalize_inputs:
            return x
        if self.time_adarms:
            return self.input_norm(x, cond)
        return self.input_norm(x)

    def forward(self, tokens: torch.Tensor, pos_embed: torch.Tensor, *, tau, cond: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens [B,M,D], got {tuple(tokens.shape)}")
        b, m, d = tokens.shape
        if d != self.token_dim:
            raise ValueError(f"Token dim mismatch: got {d}, expected {self.token_dim}")
        if pos_embed.shape != (1, m, d):
            raise ValueError(f"pos_embed shape mismatch: got {tuple(pos_embed.shape)}, expected {(1, m, d)}")
        if cond.shape[0] not in (1, b):
            raise ValueError(f"cond batch mismatch: cond={tuple(cond.shape)} tokens={tuple(tokens.shape)}")
        if cond.shape[0] == 1 and b != 1:
            cond = cond.expand(b, -1)

        tau_t = torch.as_tensor(tau, device=tokens.device, dtype=torch.float32).clamp_min(1e-8)
        x = tokens + pos_embed.to(device=tokens.device, dtype=tokens.dtype)
        x = self._norm(x, cond)
        q = torch.einsum("bmd,hdk->bhmk", x, self.W_q)
        k = torch.einsum("bmd,hdk->bhmk", x, self.W_k)
        v = torch.einsum("bmd,hde->bhme", x, self.W_vo)

        out_chunks = []
        k_float = k.float()
        k_norm2 = (k_float ** 2).sum(dim=-1)
        for start in range(0, m, self.query_chunk_size):
            end = min(start + self.query_chunk_size, m)
            q_chunk = q[:, :, start:end, :]
            q_float = q_chunk.float()
            q_norm2 = (q_float ** 2).sum(dim=-1, keepdim=True)
            dot = torch.einsum("bhqk,bhmk->bhqm", q_float, k_float)
            dist2 = (q_norm2 + k_norm2[:, :, None, :] - 2.0 * dot).clamp_min(0.0)
            attn = torch.softmax(-dist2 / (2.0 * (tau_t ** 2)), dim=-1).to(dtype=v.dtype)
            out_chunks.append(torch.einsum("bhqm,bhme->bhqe", attn, v))
        return torch.cat(out_chunks, dim=2).sum(dim=1)


@dataclass
class DPDConfig:
    img_resolution: int = 32
    img_channels: int = 3
    patch_size: int = 2
    num_layers: int = 8
    memory_bank_size: int = 256
    learned_memory_init: str = "random_train_images"
    num_heads: int = 8
    key_dim: Optional[int] = None
    stage1_query_chunk_size: int = 256
    memory_chunk_size: int = 256
    init_noise: float = 1e-3
    min_tau: float = 1e-4
    l2_eps: float = 1e-6
    stage1_rms: bool = True
    stage2_rms_qk: bool = True
    normalization_scheme: str = "rms"
    stage2_learn_q_proj: bool = True
    stage2_learn_k_proj: bool = True
    stage2_learn_v_proj: bool = False
    init_stage1_log_tau_scale: float = 0.0
    init_stage2_log_tau_scale: float = 0.0
    init_update_alpha: Optional[float] = None
    alpha_mode: str = "tied"
    time_dim: int = 128
    time_embed_dim: int = 128
    time_adarms: bool = False
    clamp_output: bool = False
    print_params: bool = True


class DPDLayer(nn.Module):
    def __init__(self, cfg: DPDConfig, *, token_dim: int, num_patches: int, layer_idx: int, init_alpha: float):
        super().__init__()
        if cfg.alpha_mode not in {"tied", "patch"}:
            raise ValueError("alpha_mode must be tied | patch")
        if cfg.normalization_scheme not in {"rms", "layernorm"}:
            raise ValueError("normalization_scheme must be rms | layernorm")
        self.cfg = cfg
        self.layer_idx = int(layer_idx)
        self.token_dim = int(token_dim)
        self.num_patches = int(num_patches)
        self.window_encoder = TimeRBFSelfAttentionWindow(
            token_dim=token_dim,
            num_heads=cfg.num_heads,
            key_dim=cfg.key_dim,
            query_chunk_size=cfg.stage1_query_chunk_size,
            init_noise=cfg.init_noise,
            normalize_inputs=cfg.stage1_rms,
            normalization_scheme=cfg.normalization_scheme,
            time_dim=cfg.time_dim,
            time_adarms=cfg.time_adarms,
            eps=cfg.l2_eps,
        )
        if cfg.stage2_rms_qk:
            self.stage2_qk_norm = make_norm(
                dim=token_dim,
                eps=cfg.l2_eps,
                scheme=cfg.normalization_scheme,
                time_dim=cfg.time_dim,
                time_adaptive=cfg.time_adarms,
            )
        else:
            self.stage2_qk_norm = nn.Identity()

        self.stage2_W_q = nn.Parameter(torch.empty(num_patches, token_dim, token_dim), requires_grad=cfg.stage2_learn_q_proj)
        self.stage2_W_k = nn.Parameter(torch.empty(num_patches, token_dim, token_dim), requires_grad=cfg.stage2_learn_k_proj)
        self.stage2_W_v = nn.Parameter(torch.empty(num_patches, token_dim, token_dim), requires_grad=cfg.stage2_learn_v_proj)
        self.reset_stage2_projections()

        self.log_stage1_tau_scale = nn.Parameter(torch.tensor(float(cfg.init_stage1_log_tau_scale)))
        self.log_stage2_tau_scale = nn.Parameter(torch.tensor(float(cfg.init_stage2_log_tau_scale)))
        self.tau_offset = nn.Linear(cfg.time_dim, 2)
        nn.init.zeros_(self.tau_offset.weight)
        nn.init.zeros_(self.tau_offset.bias)

        alpha_shape = (num_patches,) if cfg.alpha_mode == "patch" else ()
        self.logit_update_alpha = nn.Parameter(torch.full(alpha_shape, safe_logit(init_alpha)))
        alpha_out = num_patches if cfg.alpha_mode == "patch" else 1
        self.alpha_offset = nn.Linear(cfg.time_dim, alpha_out)
        nn.init.zeros_(self.alpha_offset.weight)
        nn.init.zeros_(self.alpha_offset.bias)

    def reset_stage2_projections(self) -> None:
        with torch.no_grad():
            eye = torch.eye(self.token_dim, dtype=self.stage2_W_q.dtype, device=self.stage2_W_q.device)
            self.stage2_W_q.copy_(eye[None].expand_as(self.stage2_W_q))
            self.stage2_W_k.copy_(eye[None].expand_as(self.stage2_W_k))
            self.stage2_W_v.copy_(eye[None].expand_as(self.stage2_W_v))

    def resolve_taus(self, sigma: float, cond: torch.Tensor, *, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        offsets = self.tau_offset(cond.float()).mean(dim=0)
        tau1 = float(sigma) * torch.exp(self.log_stage1_tau_scale.to(device=device) + offsets[0].to(device=device))
        tau2 = float(sigma) * torch.exp(self.log_stage2_tau_scale.to(device=device) + offsets[1].to(device=device))
        return tau1.clamp_min(float(self.cfg.min_tau)), tau2.clamp_min(float(self.cfg.min_tau))

    def update_alpha(self, cond: torch.Tensor, *, device: torch.device, dtype=torch.float32) -> torch.Tensor:
        offset = self.alpha_offset(cond.float()).mean(dim=0).to(device=device)
        if self.cfg.alpha_mode == "tied":
            logits = self.logit_update_alpha.to(device=device) + offset.squeeze(0)
            return torch.sigmoid(logits).to(dtype=dtype)
        logits = self.logit_update_alpha.to(device=device) + offset
        return torch.sigmoid(logits).to(dtype=dtype).view(1, self.num_patches, 1)

    def stage2_normalize_qk(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if not self.cfg.stage2_rms_qk:
            return x
        if self.cfg.time_adarms:
            return self.stage2_qk_norm(x, cond)
        return self.stage2_qk_norm(x)

    def stage2_project_q(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...md,mde->...me", x, self.stage2_W_q.to(device=x.device, dtype=x.dtype))

    def stage2_project_k(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...md,mde->...me", x, self.stage2_W_k.to(device=x.device, dtype=x.dtype))

    def stage2_project_v(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...md,mde->...me", x, self.stage2_W_v.to(device=x.device, dtype=x.dtype))


class DeepPatchwiseDenoiser(nn.Module):
    def __init__(self, cfg: DPDConfig = DPDConfig()):
        super().__init__()
        self.cfg = cfg
        if cfg.img_resolution % cfg.patch_size != 0:
            raise ValueError("img_resolution must be divisible by patch_size")
        if cfg.patch_size == 1:
            raise ValueError("patch_size=1 is not supported by current 2D sin/cos positional encoding.")
        if cfg.memory_bank_size <= 0:
            raise ValueError("memory_bank_size must be positive")
        if cfg.normalization_scheme not in {"rms", "layernorm"}:
            raise ValueError("normalization_scheme must be rms | layernorm")
        self.grid_size = cfg.img_resolution // cfg.patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.token_dim = cfg.img_channels * cfg.patch_size * cfg.patch_size
        pos = build_2d_sincos_pos_embed(self.grid_size, self.grid_size, self.token_dim)
        self.register_buffer("pos_embed", pos.unsqueeze(0), persistent=True)

        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_embed_dim, cfg.time_dim),
            nn.SiLU(),
            nn.Linear(cfg.time_dim, cfg.time_dim),
        )
        nn.init.normal_(self.time_mlp[0].weight, std=0.02)
        nn.init.zeros_(self.time_mlp[0].bias)
        nn.init.normal_(self.time_mlp[2].weight, std=0.02)
        nn.init.zeros_(self.time_mlp[2].bias)

        init_alpha = 1.0 / float(cfg.num_layers) if cfg.init_update_alpha is None else float(cfg.init_update_alpha)
        self.layers = nn.ModuleList([
            DPDLayer(cfg, token_dim=self.token_dim, num_patches=self.num_patches, layer_idx=i, init_alpha=init_alpha)
            for i in range(cfg.num_layers)
        ])
        self.learned_memory = nn.Parameter(torch.empty(cfg.memory_bank_size, self.num_patches, self.token_dim))
        nn.init.uniform_(self.learned_memory, -1.0, 1.0)
        if cfg.print_params:
            n = count_parameters(self)
            print(
                f"[model_dpd] DeepPatchwiseDenoiser params={n:,} ({n/1e6:.3f} M); "
                f"layers={cfg.num_layers} memory={cfg.memory_bank_size} alpha_mode={cfg.alpha_mode} "
                f"norm={cfg.normalization_scheme} time_adaptive_norm={cfg.time_adarms} "
                f"V_learn={cfg.stage2_learn_v_proj}"
            )

    @torch.no_grad()
    def initialize_learned_memory(self, memory: torch.Tensor) -> None:
        self._validate_images(memory, name="learned_memory_init")
        if memory.shape[0] != self.cfg.memory_bank_size:
            raise ValueError(f"expected {self.cfg.memory_bank_size} memory images, got {memory.shape[0]}")
        self.learned_memory.copy_(patchify(memory.to(device=self.learned_memory.device, dtype=self.learned_memory.dtype), self.cfg.patch_size))

    def _validate_images(self, x: torch.Tensor, *, name: str) -> None:
        if x.ndim != 4:
            raise ValueError(f"{name} must be [B,C,H,W], got {tuple(x.shape)}")
        if x.shape[1:] != (self.cfg.img_channels, self.cfg.img_resolution, self.cfg.img_resolution):
            raise ValueError(
                f"{name} must be [B,{self.cfg.img_channels},{self.cfg.img_resolution},{self.cfg.img_resolution}], "
                f"got {tuple(x.shape)}"
            )

    def time_condition(self, sigma: float, *, device: torch.device) -> torch.Tensor:
        s = torch.tensor([math.log(float(sigma)) / 4.0], device=device, dtype=torch.float32)
        return self.time_mlp(timestep_embedding(s, self.cfg.time_embed_dim))

    def _same_position_cross_attention(
        self,
        *,
        layer: DPDLayer,
        q_tokens: torch.Tensor,
        mem_tokens: torch.Tensor,
        tau_stage1,
        tau_stage2,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        b, m, d = q_tokens.shape
        r = mem_tokens.shape[0]
        tau2 = torch.as_tensor(tau_stage2, device=q_tokens.device, dtype=torch.float32).clamp_min(float(self.cfg.min_tau))
        q_cond = cond.expand(b, -1)
        q_base = layer.stage2_normalize_qk(q_tokens, q_cond).float()
        q_feat = layer.stage2_project_q(q_base).float()
        q_norm2 = (q_feat ** 2).sum(dim=-1, keepdim=True)
        running_max = torch.full((b, m), -float("inf"), device=q_tokens.device, dtype=torch.float32)
        running_sum = torch.zeros((b, m), device=q_tokens.device, dtype=torch.float32)
        running_num = torch.zeros((b, m, d), device=q_tokens.device, dtype=torch.float32)
        chunk = int(self.cfg.memory_chunk_size)
        mem_cond = cond.expand(min(chunk, r), -1)
        for start in range(0, r, chunk):
            end = min(start + chunk, r)
            mem_chunk = mem_tokens[start:end]
            if mem_cond.shape[0] != end - start:
                mem_cond = cond.expand(end - start, -1)
            k_chunk = layer.window_encoder(mem_chunk, self.pos_embed, tau=tau_stage1, cond=mem_cond)
            k_base = layer.stage2_normalize_qk(k_chunk, mem_cond).float()
            k_feat = layer.stage2_project_k(k_base).float()
            v_float = layer.stage2_project_v(mem_chunk.float()).float()
            k_norm2 = (k_feat ** 2).sum(dim=-1).transpose(0, 1).unsqueeze(0)
            dot = torch.einsum("bmd,rmd->bmr", q_feat, k_feat)
            dist2 = (q_norm2 + k_norm2 - 2.0 * dot).clamp_min(0.0)
            logits = -dist2 / (2.0 * (tau2 ** 2))
            block_max = logits.max(dim=-1).values
            new_max = torch.maximum(running_max, block_max)
            old_scale = torch.exp(running_max - new_max)
            exp_logits = torch.exp(logits - new_max[..., None])
            running_num = running_num * old_scale[..., None] + torch.einsum("bmr,rmd->bmd", exp_logits, v_float)
            running_sum = running_sum * old_scale + exp_logits.sum(dim=-1)
            running_max = new_max
        return (running_num / running_sum.clamp_min(1e-20)[..., None]).to(dtype=q_tokens.dtype)

    def denoise_one_layer(self, x_state: torch.Tensor, mem_tokens: torch.Tensor, *, layer_idx: int, sigma: float, cond: torch.Tensor):
        layer = self.layers[layer_idx]
        tau1, tau2 = layer.resolve_taus(sigma, cond, device=x_state.device)
        x_tokens = patchify(x_state.to(dtype=layer.window_encoder.W_q.dtype), self.cfg.patch_size)
        q_tokens = layer.window_encoder(x_tokens, self.pos_embed, tau=tau1, cond=cond.expand(x_state.shape[0], -1))
        pred_tokens = self._same_position_cross_attention(layer=layer, q_tokens=q_tokens, mem_tokens=mem_tokens, tau_stage1=tau1, tau_stage2=tau2, cond=cond)
        denoised = unpatchify(pred_tokens, img_channels=self.cfg.img_channels, img_resolution=self.cfg.img_resolution, patch_size=self.cfg.patch_size).float()
        alpha = layer.update_alpha(cond, device=x_state.device, dtype=torch.float32)
        alpha_img = alpha if alpha.ndim == 0 else unpatchify(alpha.expand(x_state.shape[0], -1, self.token_dim), img_channels=self.cfg.img_channels, img_resolution=self.cfg.img_resolution, patch_size=self.cfg.patch_size)
        updated = (1.0 - alpha_img) * x_state.float() + alpha_img * denoised
        if self.cfg.clamp_output:
            updated = updated.clamp(-1.0, 1.0)
        info = {
            "layer": int(layer_idx),
            "tau_stage1": float(tau1.detach().float().cpu()),
            "tau_stage2": float(tau2.detach().float().cpu()),
            "alpha_mean": float(alpha.detach().float().mean().cpu()),
            "alpha_min": float(alpha.detach().float().min().cpu()),
            "alpha_max": float(alpha.detach().float().max().cpu()),
        }
        return denoised, updated, info

    def forward(self, x: torch.Tensor, sigma=None, class_labels=None, *, return_all: bool = False, checkpoint_layers: bool = False, **kwargs):
        del class_labels, kwargs
        self._validate_images(x, name="x")
        sigma_in = _as_scalar_sigma(sigma, batch_size=x.shape[0], device=x.device)
        cond = self.time_condition(sigma_in, device=x.device)
        mem_tokens = self.learned_memory.to(device=x.device, dtype=self.layers[0].window_encoder.W_q.dtype)
        x_state = x.float()
        if checkpoint_layers and self.training and not x_state.requires_grad:
            x_state = x_state.detach().requires_grad_(True)
        states: List[torch.Tensor] = [x_state]
        denoised_outputs: List[torch.Tensor] = []
        layer_infos: List[dict] = []
        for layer_idx in range(len(self.layers)):
            if checkpoint_layers and self.training:
                if torch_checkpoint is None:
                    raise RuntimeError("torch.utils.checkpoint is unavailable; rerun without checkpoint_layers.")

                def _layer_forward(inp: torch.Tensor, idx: int = layer_idx) -> torch.Tensor:
                    _d, upd, _info = self.denoise_one_layer(inp, mem_tokens, layer_idx=idx, sigma=sigma_in, cond=cond)
                    return upd
                x_state = torch_checkpoint(_layer_forward, x_state, use_reentrant=False, preserve_rng_state=False)
            else:
                denoised, x_state, info = self.denoise_one_layer(x_state, mem_tokens, layer_idx=layer_idx, sigma=sigma_in, cond=cond)
                denoised_outputs.append(denoised)
                states.append(x_state)
                layer_infos.append(info)
        final = x_state.clamp(-1.0, 1.0) if self.cfg.clamp_output else x_state
        if return_all:
            return {"final": final.float(), "states": states, "denoised": denoised_outputs, "layer_infos": layer_infos}
        return final.float()

    @torch.no_grad()
    def temperature_info(self, sigma: float, *, device=None) -> Dict[str, object]:
        if device is None:
            device = self.pos_embed.device
        cond = self.time_condition(float(sigma), device=device)
        layers = []
        for layer in self.layers:
            tau1, tau2 = layer.resolve_taus(float(sigma), cond, device=device)
            alpha = layer.update_alpha(cond, device=device, dtype=torch.float32)
            layers.append({
                "layer": int(layer.layer_idx),
                "tau_stage1": float(tau1.detach().float().cpu()),
                "tau_stage2": float(tau2.detach().float().cpu()),
                "alpha_mean": float(alpha.detach().float().mean().cpu()),
                "alpha_min": float(alpha.detach().float().min().cpu()),
                "alpha_max": float(alpha.detach().float().max().cpu()),
            })
        return {
            "sigma": float(sigma),
            "tau_stage1": float(sum(x["tau_stage1"] for x in layers) / len(layers)),
            "tau_stage2": float(sum(x["tau_stage2"] for x in layers) / len(layers)),
            "alpha_mean": float(sum(x["alpha_mean"] for x in layers) / len(layers)),
            "alpha_min": float(min(x["alpha_min"] for x in layers)),
            "alpha_max": float(max(x["alpha_max"] for x in layers)),
            "num_layers": len(layers),
            "layers": layers,
        }


# Backward-compatible aliases for older checkpoints/scripts that imported the
# previous NPD class names.
NPDConfig = DPDConfig
NPDLayer = DPDLayer
NeuralPatchwiseDenoiser = DeepPatchwiseDenoiser
