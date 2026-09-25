"""Deep Patchwise Denoiser 2 (DPD2), matching the paper-slide equations."""

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
        raise ValueError("DPD2 requires sigma.")
    sigma_t = torch.as_tensor(sigma, device=device, dtype=torch.float32).flatten()
    if sigma_t.numel() == 1:
        value = float(sigma_t.item())
    elif sigma_t.numel() == batch_size:
        mean = sigma_t.mean()
        if not torch.allclose(sigma_t, mean.expand_as(sigma_t), rtol=1e-5, atol=1e-6):
            raise ValueError("DPD2 assumes one scalar sigma per minibatch.")
        value = float(mean.item())
    else:
        raise ValueError(f"sigma must be scalar or [B], got {tuple(sigma_t.shape)}")
    if value <= 0:
        raise ValueError("sigma must be positive")
    return value


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    if t.ndim == 0:
        t = t[None]
    half = dim // 2
    freqs = torch.exp(
        -math.log(float(max_period))
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / max(half, 1)
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimeLocationFeature(nn.Module):
    """psi(x,e,t) = alpha(t) + unit(x)*beta_x(t) + e*beta_e(t)."""

    def __init__(self, token_dim: int, time_dim: int, eps: float):
        super().__init__()
        self.token_dim = int(token_dim)
        self.eps = float(eps)
        self.affine = nn.Linear(int(time_dim), 3 * self.token_dim)
        nn.init.zeros_(self.affine.weight)
        with torch.no_grad():
            self.affine.bias.zero_()
            self.affine.bias[self.token_dim :].fill_(1.0)

    def forward(self, x: torch.Tensor, pos_embed: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.token_dim:
            raise ValueError(f"expected [B,M,{self.token_dim}], got {tuple(x.shape)}")
        if pos_embed.shape != (1, x.shape[1], self.token_dim):
            raise ValueError("positional embedding shape does not match tokens")
        if cond.shape[0] == 1 and x.shape[0] != 1:
            cond = cond.expand(x.shape[0], -1)
        if cond.shape[0] != x.shape[0]:
            raise ValueError("time condition batch does not match tokens")
        alpha, beta_x, beta_e = self.affine(cond.float()).chunk(3, dim=-1)
        x_float = x.float()
        unit_x = x_float / x_float.square().sum(dim=-1, keepdim=True).clamp_min(self.eps).sqrt()
        return (
            alpha[:, None, :]
            + unit_x * beta_x[:, None, :]
            + pos_embed.float() * beta_e[:, None, :]
        )


@dataclass
class DPD2Config:
    img_resolution: int = 32
    img_channels: int = 3
    patch_size: int = 2
    num_layers: int = 8
    memory_bank_size: int = 256
    learned_memory_init: str = "random_uniform"
    num_heads: int = 8
    key_dim: Optional[int] = None
    stage2_patch_specific_qk: bool = False
    stage2_patch_qk_sym: bool = False
    stage2_patch_specific_v: bool = False
    stage1_query_chunk_size: int = 256
    memory_chunk_size: int = 64
    init_noise: float = 1e-3
    feature_eps: float = 1e-12
    init_eta: float = 0.1
    init_gamma: Optional[float] = None
    gate_mode: str = "patch"
    euler_faithful: bool = True
    time_dim: int = 128
    time_embed_dim: int = 128
    clamp_output: bool = False
    print_params: bool = True


class DPD2Layer(nn.Module):
    def __init__(self, cfg: DPD2Config, *, token_dim: int, num_patches: int, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = int(layer_idx)
        self.token_dim = int(token_dim)
        self.num_patches = int(num_patches)
        self.num_heads = int(cfg.num_heads)
        self.key_dim = int(cfg.key_dim) if cfg.key_dim is not None else self.token_dim
        if cfg.gate_mode not in {"patch", "tied"}:
            raise ValueError("gate_mode must be patch | tied")
        if cfg.stage2_patch_specific_qk and cfg.stage2_patch_qk_sym:
            raise ValueError("stage2_patch_specific_qk and stage2_patch_qk_sym are mutually exclusive")
        if cfg.stage2_patch_qk_sym and self.key_dim != self.token_dim:
            raise ValueError("stage2_patch_qk_sym requires key_dim to equal token_dim")

        self.stage1_feature = TimeLocationFeature(token_dim, cfg.time_dim, cfg.feature_eps)
        self.stage2_feature = TimeLocationFeature(token_dim, cfg.time_dim, cfg.feature_eps)

        h, d, k = self.num_heads, self.token_dim, self.key_dim
        self.stage1_W_q = nn.Parameter(torch.empty(h, d, k))
        self.stage1_W_k = nn.Parameter(torch.empty(h, d, k))
        self.stage1_B = nn.Parameter(torch.empty(h, d, d))
        if cfg.stage2_patch_qk_sym:
            qk_triu = torch.triu_indices(d, d)
            self.register_buffer("stage2_qk_triu_rows", qk_triu[0], persistent=False)
            self.register_buffer("stage2_qk_triu_cols", qk_triu[1], persistent=False)
            self.stage2_W_qk_packed = nn.Parameter(torch.empty(num_patches, qk_triu.shape[1]))
        else:
            stage2_shape = (num_patches, d, k) if cfg.stage2_patch_specific_qk else (d, k)
            self.stage2_W_q = nn.Parameter(torch.empty(stage2_shape))
            self.stage2_W_k = nn.Parameter(torch.empty(stage2_shape))
        if cfg.stage2_patch_specific_v:
            self.stage2_W_v = nn.Parameter(torch.empty(num_patches, d, d))
        self._reset_projections(float(cfg.init_noise))

        gate_size = num_patches if cfg.gate_mode == "patch" else 1
        self.eta_base_logit = nn.Parameter(torch.full((gate_size,), safe_logit(cfg.init_eta)))
        self.eta_offset = nn.Linear(cfg.time_dim, gate_size)
        nn.init.zeros_(self.eta_offset.weight)
        nn.init.zeros_(self.eta_offset.bias)

        init_gamma = 1.0 / float(cfg.num_layers) if cfg.init_gamma is None else float(cfg.init_gamma)
        self.gamma_base_logit = nn.Parameter(torch.full((gate_size,), safe_logit(init_gamma)))
        self.gamma_offset = nn.Linear(cfg.time_dim, gate_size)
        nn.init.zeros_(self.gamma_offset.weight)
        nn.init.zeros_(self.gamma_offset.bias)

    def _reset_projections(self, noise: float) -> None:
        with torch.no_grad():
            if self.key_dim == self.token_dim:
                eye = torch.eye(self.token_dim)
                self.stage1_W_q.copy_(eye[None].expand_as(self.stage1_W_q))
                self.stage1_W_k.copy_(eye[None].expand_as(self.stage1_W_k))
                if self.cfg.stage2_patch_qk_sym:
                    self.stage2_W_qk_packed.copy_(eye[self.stage2_qk_triu_rows, self.stage2_qk_triu_cols])
                else:
                    self.stage2_W_q.copy_(eye.expand_as(self.stage2_W_q))
                    self.stage2_W_k.copy_(eye.expand_as(self.stage2_W_k))
                if noise > 0:
                    self.stage1_W_q.add_(noise * torch.randn_like(self.stage1_W_q))
                    self.stage1_W_k.add_(noise * torch.randn_like(self.stage1_W_k))
                    if self.cfg.stage2_patch_qk_sym:
                        self.stage2_W_qk_packed.add_(noise * torch.randn_like(self.stage2_W_qk_packed))
                    else:
                        self.stage2_W_q.add_(noise * torch.randn_like(self.stage2_W_q))
                        self.stage2_W_k.add_(noise * torch.randn_like(self.stage2_W_k))
            else:
                nn.init.xavier_uniform_(self.stage1_W_q)
                nn.init.xavier_uniform_(self.stage1_W_k)
                nn.init.xavier_uniform_(self.stage2_W_q)
                nn.init.xavier_uniform_(self.stage2_W_k)
            eye_b = torch.eye(self.token_dim)
            self.stage1_B.copy_(eye_b[None].expand_as(self.stage1_B))
            if self.cfg.stage2_patch_specific_v:
                self.stage2_W_v.copy_(eye_b.expand_as(self.stage2_W_v))
            if noise > 0:
                self.stage1_B.add_(noise * torch.randn_like(self.stage1_B))

    def gates(self, cond: torch.Tensor, *, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        eta = torch.sigmoid(self.eta_base_logit + self.eta_offset(cond.float()).squeeze(0))
        gamma = torch.sigmoid(self.gamma_base_logit + self.gamma_offset(cond.float()).squeeze(0))
        gate_patches = self.num_patches if self.cfg.gate_mode == "patch" else 1
        return eta.to(dtype=dtype).view(1, gate_patches, 1), gamma.to(dtype=dtype).view(1, gate_patches, 1)

    def window_select(
        self, tokens: torch.Tensor, pos_embed: torch.Tensor, cond: torch.Tensor, eta: torch.Tensor
    ) -> torch.Tensor:
        b, m, d = tokens.shape
        psi = self.stage1_feature(tokens, pos_embed, cond)
        q = torch.einsum("bmd,hdk->bhmk", psi, self.stage1_W_q.float())
        k = torch.einsum("bmd,hdk->bhmk", psi, self.stage1_W_k.float())
        values = torch.einsum("bmd,hde->bhme", tokens.float(), self.stage1_B.float())
        k_norm2 = k.square().sum(dim=-1)
        outputs = []
        chunk = int(self.cfg.stage1_query_chunk_size)
        for start in range(0, m, chunk):
            end = min(start + chunk, m)
            q_part = q[:, :, start:end]
            dist2 = (
                q_part.square().sum(dim=-1, keepdim=True)
                + k_norm2[:, :, None, :]
                - 2.0 * torch.einsum("bhqk,bhmk->bhqm", q_part, k)
            ).clamp_min(0.0)
            weights = torch.softmax(-0.5 * dist2, dim=-1)
            outputs.append(torch.einsum("bhqm,bhme->bhqe", weights, values).mean(dim=1))
        attended = torch.cat(outputs, dim=1)
        return ((1.0 - eta.float()) * tokens.float() + eta.float() * attended).to(tokens.dtype)

    def stage2_query(self, selected: torch.Tensor, pos_embed: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        psi = self.stage2_feature(selected, pos_embed, cond)
        if self.cfg.stage2_patch_qk_sym:
            qk = self.stage2_symmetric_qk()
            return torch.einsum("bmd,mdk->bmk", psi, qk)
        if self.cfg.stage2_patch_specific_qk:
            return torch.einsum("bmd,mdk->bmk", psi, self.stage2_W_q.float())
        return torch.einsum("bmd,dk->bmk", psi, self.stage2_W_q.float())

    def stage2_key(self, selected: torch.Tensor, pos_embed: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        psi = self.stage2_feature(selected, pos_embed, cond)
        if self.cfg.stage2_patch_qk_sym:
            qk = self.stage2_symmetric_qk()
            return torch.einsum("rmd,mdk->rmk", psi, qk)
        if self.cfg.stage2_patch_specific_qk:
            return torch.einsum("rmd,mdk->rmk", psi, self.stage2_W_k.float())
        return torch.einsum("rmd,dk->rmk", psi, self.stage2_W_k.float())

    def stage2_value(self, raw_memory: torch.Tensor) -> torch.Tensor:
        if self.cfg.stage2_patch_specific_v:
            return torch.einsum("rmd,mde->rme", raw_memory.float(), self.stage2_W_v.float())
        return raw_memory.float()

    def stage2_symmetric_qk(self) -> torch.Tensor:
        """Unpack the independent upper triangle into a symmetric Q=K matrix."""
        packed = self.stage2_W_qk_packed.float()
        upper = packed.new_zeros(self.num_patches, self.token_dim, self.token_dim)
        upper[:, self.stage2_qk_triu_rows, self.stage2_qk_triu_cols] = packed
        return upper + upper.transpose(-1, -2) - torch.diag_embed(upper.diagonal(dim1=-2, dim2=-1))


class DeepPatchwiseDenoiser2(nn.Module):
    def __init__(self, cfg: DPD2Config = DPD2Config()):
        super().__init__()
        self.cfg = cfg
        if cfg.img_resolution % cfg.patch_size:
            raise ValueError("img_resolution must be divisible by patch_size")
        if cfg.patch_size == 1:
            raise ValueError("patch_size=1 is unsupported by the current positional encoding")
        if cfg.memory_bank_size <= 0 or cfg.num_layers <= 0 or cfg.num_heads <= 0:
            raise ValueError("memory_bank_size, num_layers, and num_heads must be positive")
        if cfg.stage1_query_chunk_size <= 0 or cfg.memory_chunk_size <= 0:
            raise ValueError("chunk sizes must be positive")
        self.grid_size = cfg.img_resolution // cfg.patch_size
        self.num_patches = self.grid_size**2
        self.token_dim = cfg.img_channels * cfg.patch_size**2
        pos = build_2d_sincos_pos_embed(self.grid_size, self.grid_size, self.token_dim)
        self.register_buffer("pos_embed", pos.unsqueeze(0), persistent=True)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_embed_dim, cfg.time_dim), nn.SiLU(), nn.Linear(cfg.time_dim, cfg.time_dim)
        )
        nn.init.normal_(self.time_mlp[0].weight, std=0.02)
        nn.init.zeros_(self.time_mlp[0].bias)
        nn.init.normal_(self.time_mlp[2].weight, std=0.02)
        nn.init.zeros_(self.time_mlp[2].bias)
        self.layers = nn.ModuleList(
            [DPD2Layer(cfg, token_dim=self.token_dim, num_patches=self.num_patches, layer_idx=i) for i in range(cfg.num_layers)]
        )
        self.learned_memory = nn.Parameter(torch.empty(cfg.memory_bank_size, self.num_patches, self.token_dim))
        nn.init.uniform_(self.learned_memory, -1.0, 1.0)
        if cfg.print_params:
            n = count_parameters(self)
            print(
                f"[model_dpd2] params={n:,} ({n / 1e6:.3f} M); "
                f"layers={cfg.num_layers} memory={cfg.memory_bank_size} gate_mode={cfg.gate_mode} "
                f"stage2_patch_specific_qk={cfg.stage2_patch_specific_qk} "
                f"stage2_patch_qk_sym={cfg.stage2_patch_qk_sym} "
                f"stage2_patch_specific_v={cfg.stage2_patch_specific_v} "
                f"euler_faithful={cfg.euler_faithful}"
            )

    def _validate_images(self, x: torch.Tensor, *, name: str) -> None:
        expected = (self.cfg.img_channels, self.cfg.img_resolution, self.cfg.img_resolution)
        if x.ndim != 4 or x.shape[1:] != expected:
            raise ValueError(f"{name} must be [B,{expected[0]},{expected[1]},{expected[2]}], got {tuple(x.shape)}")

    @torch.no_grad()
    def initialize_learned_memory(self, memory: torch.Tensor) -> None:
        self._validate_images(memory, name="learned_memory_init")
        if memory.shape[0] != self.cfg.memory_bank_size:
            raise ValueError(f"expected {self.cfg.memory_bank_size} memory images, got {memory.shape[0]}")
        self.learned_memory.copy_(patchify(memory.to(self.learned_memory), self.cfg.patch_size))

    def time_condition(self, sigma: float, *, device: torch.device) -> torch.Tensor:
        coordinate = torch.tensor([math.log(float(sigma)) / 4.0], device=device)
        return self.time_mlp(timestep_embedding(coordinate, self.cfg.time_embed_dim))

    def _memory_retrieval(
        self, layer: DPD2Layer, selected_query: torch.Tensor, cond: torch.Tensor, eta: torch.Tensor
    ) -> torch.Tensor:
        b, m, d = selected_query.shape
        q = layer.stage2_query(selected_query, self.pos_embed, cond.expand(b, -1))
        q_norm2 = q.square().sum(dim=-1, keepdim=True)
        running_max = torch.full((b, m), -float("inf"), device=q.device)
        running_sum = torch.zeros((b, m), device=q.device)
        running_num = torch.zeros((b, m, d), device=q.device)
        chunk = int(self.cfg.memory_chunk_size)
        memory = self.learned_memory
        for start in range(0, memory.shape[0], chunk):
            raw = memory[start : start + chunk]
            mem_cond = cond.expand(raw.shape[0], -1)
            selected = layer.window_select(raw, self.pos_embed, mem_cond, eta)
            keys = layer.stage2_key(selected, self.pos_embed, mem_cond)
            values = layer.stage2_value(raw)
            dist2 = (
                q_norm2
                + keys.square().sum(dim=-1).transpose(0, 1).unsqueeze(0)
                - 2.0 * torch.einsum("bmk,rmk->bmr", q, keys)
            ).clamp_min(0.0)
            logits = -0.5 * dist2
            block_max = logits.max(dim=-1).values
            new_max = torch.maximum(running_max, block_max)
            old_scale = torch.exp(running_max - new_max)
            exp_logits = torch.exp(logits - new_max[..., None])
            running_num = running_num * old_scale[..., None] + torch.einsum("bmr,rmd->bmd", exp_logits, values)
            running_sum = running_sum * old_scale + exp_logits.sum(dim=-1)
            running_max = new_max
        return running_num / running_sum.clamp_min(1e-20)[..., None]

    def denoise_one_layer(self, tokens: torch.Tensor, *, layer_idx: int, cond: torch.Tensor):
        layer = self.layers[layer_idx]
        eta, gamma = layer.gates(cond, dtype=tokens.dtype)
        selected = layer.window_select(tokens, self.pos_embed, cond.expand(tokens.shape[0], -1), eta)
        proposal = self._memory_retrieval(layer, selected, cond, eta)
        residual_state = tokens if self.cfg.euler_faithful else selected
        updated = (1.0 - gamma.float()) * residual_state.float() + gamma.float() * proposal
        info = {
            "layer": layer_idx,
            "eta_mean": float(eta.detach().float().mean().cpu()),
            "eta_min": float(eta.detach().float().min().cpu()),
            "eta_max": float(eta.detach().float().max().cpu()),
            "gamma_mean": float(gamma.detach().float().mean().cpu()),
            "gamma_min": float(gamma.detach().float().min().cpu()),
            "gamma_max": float(gamma.detach().float().max().cpu()),
        }
        return selected, proposal, updated.to(tokens.dtype), info

    def forward(self, x: torch.Tensor, sigma=None, class_labels=None, *, return_all=False, checkpoint_layers=False, **kwargs):
        del class_labels, kwargs
        self._validate_images(x, name="x")
        if return_all and checkpoint_layers:
            raise ValueError("return_all and checkpoint_layers cannot be used together")
        sigma_value = _as_scalar_sigma(sigma, batch_size=x.shape[0], device=x.device)
        cond = self.time_condition(sigma_value, device=x.device)
        tokens = patchify(x.float(), self.cfg.patch_size)
        states: List[torch.Tensor] = [tokens]
        selected_all: List[torch.Tensor] = []
        proposals: List[torch.Tensor] = []
        infos: List[dict] = []
        for layer_idx in range(len(self.layers)):
            if checkpoint_layers and self.training:
                if torch_checkpoint is None:
                    raise RuntimeError("torch.utils.checkpoint is unavailable")
                if not tokens.requires_grad:
                    tokens = tokens.detach().requires_grad_(True)

                def layer_forward(inp: torch.Tensor, idx: int = layer_idx) -> torch.Tensor:
                    return self.denoise_one_layer(inp, layer_idx=idx, cond=cond)[2]

                tokens = torch_checkpoint(layer_forward, tokens, use_reentrant=False, preserve_rng_state=False)
            else:
                selected, proposal, tokens, info = self.denoise_one_layer(tokens, layer_idx=layer_idx, cond=cond)
                states.append(tokens)
                selected_all.append(selected)
                proposals.append(proposal)
                infos.append(info)
        output = unpatchify(tokens.float(), img_channels=self.cfg.img_channels, img_resolution=self.cfg.img_resolution, patch_size=self.cfg.patch_size)
        if self.cfg.clamp_output:
            output = output.clamp(-1.0, 1.0)
        if return_all:
            return {"final": output, "states": states, "selected": selected_all, "proposals": proposals, "layer_infos": infos}
        return output

    @torch.no_grad()
    def gate_info(self, sigma: float, *, device=None) -> Dict[str, object]:
        device = self.pos_embed.device if device is None else device
        cond = self.time_condition(float(sigma), device=device)
        rows = []
        for layer in self.layers:
            eta, gamma = layer.gates(cond, dtype=torch.float32)
            rows.append({
                "layer": layer.layer_idx,
                "eta_mean": float(eta.mean()), "eta_min": float(eta.min()), "eta_max": float(eta.max()),
                "gamma_mean": float(gamma.mean()), "gamma_min": float(gamma.min()), "gamma_max": float(gamma.max()),
            })
        return {
            "sigma": float(sigma), "layers": rows,
            "eta_mean": sum(r["eta_mean"] for r in rows) / len(rows),
            "eta_min": min(r["eta_min"] for r in rows), "eta_max": max(r["eta_max"] for r in rows),
            "gamma_mean": sum(r["gamma_mean"] for r in rows) / len(rows),
            "gamma_min": min(r["gamma_min"] for r in rows), "gamma_max": max(r["gamma_max"] for r in rows),
        }
