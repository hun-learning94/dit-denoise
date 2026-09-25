"""DPD3: DPD2 memory attention with DiT-style additive residual streams."""

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
    unpatchify,
)
from model_dpd2 import TimeLocationFeature, _as_scalar_sigma, timestep_embedding


@dataclass
class DPD3Config:
    img_resolution: int = 32
    img_channels: int = 3
    patch_size: int = 2
    # Set to None (or the raw patch width) for legacy checkpoint compatibility.
    hidden_dim: Optional[int] = 24
    projected_readout: str = "linear"
    num_layers: int = 8
    memory_bank_size: int = 256
    learned_memory_init: str = "random_uniform"
    num_heads: int = 8
    key_dim: Optional[int] = None
    # Best DPD2 configuration is the DPD3 default.
    stage2_patch_specific_qk: bool = True
    stage2_patch_qk_sym: bool = False
    stage2_patch_specific_v: bool = True
    stage1_query_chunk_size: int = 256
    memory_chunk_size: int = 64
    init_noise: float = 1e-3
    feature_eps: float = 1e-12
    gate_mode: str = "patch"
    residual_gate_init: float = 0.0
    time_dim: int = 128
    time_embed_dim: int = 128
    residual_output_head: bool = False
    clamp_output: bool = False
    print_params: bool = True


class DPD3Layer(nn.Module):
    def __init__(self, cfg: DPD3Config, *, token_dim: int, num_patches: int, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = int(layer_idx)
        self.token_dim = int(token_dim)
        self.num_patches = int(num_patches)
        self.num_heads = int(cfg.num_heads)
        self.key_dim = int(cfg.key_dim) if cfg.key_dim is not None else self.token_dim
        if cfg.gate_mode not in {"patch", "channel", "patch_channel"}:
            raise ValueError("gate_mode must be patch | channel | patch_channel")
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
            qk_shape = (num_patches, d, k) if cfg.stage2_patch_specific_qk else (d, k)
            self.stage2_W_q = nn.Parameter(torch.empty(qk_shape))
            self.stage2_W_k = nn.Parameter(torch.empty(qk_shape))
        v_shape = (num_patches, d, d) if cfg.stage2_patch_specific_v else (d, d)
        self.stage2_W_v = nn.Parameter(torch.empty(v_shape))
        self._reset_projections(float(cfg.init_noise))

        if cfg.gate_mode == "patch":
            gate_shape, gate_out = (num_patches,), num_patches
        elif cfg.gate_mode == "channel":
            gate_shape, gate_out = (d,), d
        else:
            gate_shape, gate_out = (num_patches, d), num_patches * d
        self.gate_shape = gate_shape
        init = float(cfg.residual_gate_init)
        self.window_gate_base = nn.Parameter(torch.full(gate_shape, init))
        self.memory_gate_base = nn.Parameter(torch.full(gate_shape, init))
        self.window_gate_offset = nn.Linear(cfg.time_dim, gate_out)
        self.memory_gate_offset = nn.Linear(cfg.time_dim, gate_out)
        nn.init.zeros_(self.window_gate_offset.weight)
        nn.init.zeros_(self.window_gate_offset.bias)
        nn.init.zeros_(self.memory_gate_offset.weight)
        nn.init.zeros_(self.memory_gate_offset.bias)

    def _reset_projections(self, noise: float) -> None:
        with torch.no_grad():
            d = self.token_dim
            if self.key_dim == d:
                eye = torch.eye(d)
                self.stage1_W_q.copy_(eye[None].expand_as(self.stage1_W_q))
                self.stage1_W_k.copy_(eye[None].expand_as(self.stage1_W_k))
                if self.cfg.stage2_patch_qk_sym:
                    self.stage2_W_qk_packed.copy_(
                        eye[self.stage2_qk_triu_rows, self.stage2_qk_triu_cols]
                    )
                else:
                    self.stage2_W_q.copy_(eye.expand_as(self.stage2_W_q))
                    self.stage2_W_k.copy_(eye.expand_as(self.stage2_W_k))
            else:
                nn.init.xavier_uniform_(self.stage1_W_q)
                nn.init.xavier_uniform_(self.stage1_W_k)
                nn.init.xavier_uniform_(self.stage2_W_q)
                nn.init.xavier_uniform_(self.stage2_W_k)
            eye = torch.eye(d)
            self.stage1_B.copy_(eye[None].expand_as(self.stage1_B))
            self.stage2_W_v.copy_(eye.expand_as(self.stage2_W_v))
            if noise > 0:
                self.stage1_W_q.add_(noise * torch.randn_like(self.stage1_W_q))
                self.stage1_W_k.add_(noise * torch.randn_like(self.stage1_W_k))
                self.stage1_B.add_(noise * torch.randn_like(self.stage1_B))
                if self.cfg.stage2_patch_qk_sym:
                    self.stage2_W_qk_packed.add_(noise * torch.randn_like(self.stage2_W_qk_packed))
                else:
                    self.stage2_W_q.add_(noise * torch.randn_like(self.stage2_W_q))
                    self.stage2_W_k.add_(noise * torch.randn_like(self.stage2_W_k))

    def residual_gates(self, cond: torch.Tensor, *, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        window = self.window_gate_base + self.window_gate_offset(cond.float()).squeeze(0).view(self.gate_shape)
        memory = self.memory_gate_base + self.memory_gate_offset(cond.float()).squeeze(0).view(self.gate_shape)
        if self.cfg.gate_mode == "patch":
            shape = (1, self.num_patches, 1)
        elif self.cfg.gate_mode == "channel":
            shape = (1, 1, self.token_dim)
        else:
            shape = (1, self.num_patches, self.token_dim)
        return window.to(dtype=dtype).view(shape), memory.to(dtype=dtype).view(shape)

    def window_attention(
        self, tokens: torch.Tensor, pos_embed: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        b, m, _d = tokens.shape
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
        return torch.cat(outputs, dim=1).to(tokens.dtype)

    def stage2_query(self, tokens: torch.Tensor, pos_embed: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        psi = self.stage2_feature(tokens, pos_embed, cond)
        if self.cfg.stage2_patch_qk_sym:
            return torch.einsum("bmd,mdk->bmk", psi, self.stage2_symmetric_qk())
        if self.cfg.stage2_patch_specific_qk:
            return torch.einsum("bmd,mdk->bmk", psi, self.stage2_W_q.float())
        return torch.einsum("bmd,dk->bmk", psi, self.stage2_W_q.float())

    def stage2_key(self, tokens: torch.Tensor, pos_embed: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        psi = self.stage2_feature(tokens, pos_embed, cond)
        if self.cfg.stage2_patch_qk_sym:
            return torch.einsum("rmd,mdk->rmk", psi, self.stage2_symmetric_qk())
        if self.cfg.stage2_patch_specific_qk:
            return torch.einsum("rmd,mdk->rmk", psi, self.stage2_W_k.float())
        return torch.einsum("rmd,dk->rmk", psi, self.stage2_W_k.float())

    def stage2_value(self, raw_memory: torch.Tensor) -> torch.Tensor:
        if self.cfg.stage2_patch_specific_v:
            return torch.einsum("rmd,mde->rme", raw_memory.float(), self.stage2_W_v.float())
        return torch.einsum("rmd,de->rme", raw_memory.float(), self.stage2_W_v.float())

    def stage2_symmetric_qk(self) -> torch.Tensor:
        packed = self.stage2_W_qk_packed.float()
        upper = packed.new_zeros(self.num_patches, self.token_dim, self.token_dim)
        upper[:, self.stage2_qk_triu_rows, self.stage2_qk_triu_cols] = packed
        return upper + upper.transpose(-1, -2) - torch.diag_embed(
            upper.diagonal(dim1=-2, dim2=-1)
        )


class DeepPatchwiseDenoiser3(nn.Module):
    def __init__(self, cfg: DPD3Config = DPD3Config()):
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
        self.raw_token_dim = cfg.img_channels * cfg.patch_size**2
        self.token_dim = self.raw_token_dim if cfg.hidden_dim is None else int(cfg.hidden_dim)
        if self.token_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if cfg.projected_readout not in {"linear", "residual_delta"}:
            raise ValueError("projected_readout must be linear | residual_delta")
        self.uses_projection = self.token_dim != self.raw_token_dim
        if self.uses_projection:
            self.input_projection = nn.Linear(self.raw_token_dim, self.token_dim)
            self.output_projection = nn.Linear(self.token_dim, self.raw_token_dim)
            nn.init.xavier_uniform_(self.input_projection.weight)
            nn.init.zeros_(self.input_projection.bias)
            nn.init.xavier_uniform_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)
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
        self.layers = nn.ModuleList(
            [
                DPD3Layer(
                    cfg, token_dim=self.token_dim, num_patches=self.num_patches, layer_idx=i
                )
                for i in range(cfg.num_layers)
            ]
        )
        self.learned_memory = nn.Parameter(
            torch.empty(cfg.memory_bank_size, self.num_patches, self.token_dim)
        )
        nn.init.uniform_(self.learned_memory, -1.0, 1.0)

        if cfg.residual_output_head:
            self.output_affine = nn.Linear(self.token_dim, self.token_dim)
            with torch.no_grad():
                self.output_affine.weight.copy_(torch.eye(self.token_dim))
                self.output_affine.bias.zero_()
            self.output_gate_base = nn.Parameter(torch.zeros(self.num_patches))
            self.output_gate_offset = nn.Linear(cfg.time_dim, self.num_patches)
            nn.init.zeros_(self.output_gate_offset.weight)
            nn.init.zeros_(self.output_gate_offset.bias)

        if cfg.print_params:
            n = count_parameters(self)
            print(
                f"[model_dpd3] params={n:,} ({n / 1e6:.3f} M); "
                f"layers={cfg.num_layers} memory={cfg.memory_bank_size} "
                f"raw_dim={self.raw_token_dim} hidden_dim={self.token_dim} "
                f"gate_mode={cfg.gate_mode} patch_qk={cfg.stage2_patch_specific_qk} "
                f"patch_v={cfg.stage2_patch_specific_v} output_head={cfg.residual_output_head}"
            )

    def _validate_images(self, x: torch.Tensor, *, name: str) -> None:
        expected = (self.cfg.img_channels, self.cfg.img_resolution, self.cfg.img_resolution)
        if x.ndim != 4 or x.shape[1:] != expected:
            raise ValueError(
                f"{name} must be [B,{expected[0]},{expected[1]},{expected[2]}], "
                f"got {tuple(x.shape)}"
            )

    @torch.no_grad()
    def initialize_learned_memory(self, memory: torch.Tensor) -> None:
        self._validate_images(memory, name="learned_memory_init")
        if memory.shape[0] != self.cfg.memory_bank_size:
            raise ValueError(
                f"expected {self.cfg.memory_bank_size} memory images, got {memory.shape[0]}"
            )
        tokens = patchify(memory.to(self.learned_memory), self.cfg.patch_size)
        if self.uses_projection:
            tokens = self.input_projection(tokens.float()).to(self.learned_memory)
        self.learned_memory.copy_(tokens)

    def time_condition(self, sigma: float, *, device: torch.device) -> torch.Tensor:
        coordinate = torch.tensor([math.log(float(sigma)) / 4.0], device=device)
        return self.time_mlp(timestep_embedding(coordinate, self.cfg.time_embed_dim))

    def _memory_retrieval(
        self,
        layer: DPD3Layer,
        query_tokens: torch.Tensor,
        cond: torch.Tensor,
        window_gate: torch.Tensor,
    ) -> torch.Tensor:
        b, m, d = query_tokens.shape
        q = layer.stage2_query(query_tokens, self.pos_embed, cond.expand(b, -1))
        q_norm2 = q.square().sum(dim=-1, keepdim=True)
        running_max = torch.full((b, m), -float("inf"), device=q.device)
        running_sum = torch.zeros((b, m), device=q.device)
        running_num = torch.zeros((b, m, d), device=q.device)
        for start in range(0, self.learned_memory.shape[0], int(self.cfg.memory_chunk_size)):
            raw = self.learned_memory[start : start + int(self.cfg.memory_chunk_size)]
            mem_cond = cond.expand(raw.shape[0], -1)
            memory_window = layer.window_attention(raw, self.pos_embed, mem_cond)
            selected = raw.float() + window_gate.float() * memory_window.float()
            keys = layer.stage2_key(selected.to(raw.dtype), self.pos_embed, mem_cond)
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
            running_num = (
                running_num * old_scale[..., None]
                + torch.einsum("bmr,rmd->bmd", exp_logits, values)
            )
            running_sum = running_sum * old_scale + exp_logits.sum(dim=-1)
            running_max = new_max
        return running_num / running_sum.clamp_min(1e-20)[..., None]

    def denoise_one_layer(
        self, tokens: torch.Tensor, *, layer_idx: int, cond: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        layer = self.layers[layer_idx]
        window_gate, memory_gate = layer.residual_gates(cond, dtype=tokens.dtype)
        window_branch = layer.window_attention(
            tokens, self.pos_embed, cond.expand(tokens.shape[0], -1)
        )
        selected = tokens.float() + window_gate.float() * window_branch.float()
        proposal = self._memory_retrieval(
            layer, selected.to(tokens.dtype), cond, window_gate
        )
        updated = selected + memory_gate.float() * proposal
        info = {
            "layer": int(layer_idx),
            "window_gate_mean": float(window_gate.detach().float().mean().cpu()),
            "window_gate_min": float(window_gate.detach().float().min().cpu()),
            "window_gate_max": float(window_gate.detach().float().max().cpu()),
            "memory_gate_mean": float(memory_gate.detach().float().mean().cpu()),
            "memory_gate_min": float(memory_gate.detach().float().min().cpu()),
            "memory_gate_max": float(memory_gate.detach().float().max().cpu()),
        }
        return selected.to(tokens.dtype), proposal, updated.to(tokens.dtype), info

    def _apply_output_head(self, tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if not self.cfg.residual_output_head:
            return tokens
        gate = self.output_gate_base + self.output_gate_offset(cond.float()).squeeze(0)
        delta = self.output_affine(tokens.float())
        return (tokens.float() + gate.view(1, self.num_patches, 1) * delta).to(tokens.dtype)

    def forward(
        self,
        x: torch.Tensor,
        sigma=None,
        class_labels=None,
        *,
        return_all: bool = False,
        checkpoint_layers: bool = False,
        **kwargs,
    ):
        del class_labels, kwargs
        self._validate_images(x, name="x")
        if return_all and checkpoint_layers:
            raise ValueError("return_all and checkpoint_layers cannot be used together")
        sigma_value = _as_scalar_sigma(sigma, batch_size=x.shape[0], device=x.device)
        cond = self.time_condition(sigma_value, device=x.device)
        raw_tokens = patchify(x.float(), self.cfg.patch_size)
        tokens = self.input_projection(raw_tokens) if self.uses_projection else raw_tokens
        input_tokens = tokens
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

                tokens = torch_checkpoint(
                    layer_forward,
                    tokens,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                selected, proposal, tokens, info = self.denoise_one_layer(
                    tokens, layer_idx=layer_idx, cond=cond
                )
                states.append(tokens)
                selected_all.append(selected)
                proposals.append(proposal)
                infos.append(info)
        tokens = self._apply_output_head(tokens, cond)
        if self.uses_projection:
            if self.cfg.projected_readout == "residual_delta":
                tokens = raw_tokens + self.output_projection(tokens - input_tokens)
            else:
                tokens = self.output_projection(tokens)
        output = unpatchify(
            tokens.float(),
            img_channels=self.cfg.img_channels,
            img_resolution=self.cfg.img_resolution,
            patch_size=self.cfg.patch_size,
        )
        if self.cfg.clamp_output:
            output = output.clamp(-1.0, 1.0)
        if return_all:
            return {
                "final": output,
                "states": states,
                "selected": selected_all,
                "proposals": proposals,
                "layer_infos": infos,
            }
        return output
    @torch.no_grad()
    def gate_info(self, sigma: float, *, device=None) -> Dict[str, object]:
        device = self.pos_embed.device if device is None else device
        cond = self.time_condition(float(sigma), device=device)
        rows = []
        for layer in self.layers:
            window, memory = layer.residual_gates(cond, dtype=torch.float32)
            rows.append(
                {
                    "layer": layer.layer_idx,
                    "window_gate_mean": float(window.mean()),
                    "window_gate_min": float(window.min()),
                    "window_gate_max": float(window.max()),
                    "memory_gate_mean": float(memory.mean()),
                    "memory_gate_min": float(memory.min()),
                    "memory_gate_max": float(memory.max()),
                }
            )
        result = {
            "sigma": float(sigma),
            "layers": rows,
            "window_gate_mean": sum(r["window_gate_mean"] for r in rows) / len(rows),
            "window_gate_min": min(r["window_gate_min"] for r in rows),
            "window_gate_max": max(r["window_gate_max"] for r in rows),
            "memory_gate_mean": sum(r["memory_gate_mean"] for r in rows) / len(rows),
            "memory_gate_min": min(r["memory_gate_min"] for r in rows),
            "memory_gate_max": max(r["memory_gate_max"] for r in rows),
        }
        if self.cfg.residual_output_head:
            gate = self.output_gate_base + self.output_gate_offset(cond.float()).squeeze(0)
            result.update(
                output_gate_mean=float(gate.mean()),
                output_gate_min=float(gate.min()),
                output_gate_max=float(gate.max()),
            )
        return result


class DPD3EDMBackbone(nn.Module):
    """Adapt EDM's c_noise input to DPD3's sigma-conditioned interface."""

    def __init__(self, denoiser: DeepPatchwiseDenoiser3):
        super().__init__()
        self.denoiser = denoiser

    def forward(self, x: torch.Tensor, c_noise: torch.Tensor, class_labels=None, **kwargs):
        sigma = torch.exp(4.0 * torch.as_tensor(c_noise, device=x.device, dtype=torch.float32))
        return self.denoiser(x, sigma=sigma, class_labels=class_labels, **kwargs)
