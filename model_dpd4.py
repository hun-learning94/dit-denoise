"""DPD4: contextual memory keys retrieving raw same-position memory values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import FusedMultiHeadSelfAttention, TimestepEmbedder, count_parameters


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


def _modulate_memory(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Modulate an [R,N,D] or [B,R,N,D] bank per batch element."""
    if x.ndim == 3:
        x = x.unsqueeze(0)
    return x * (1.0 + scale[:, None, None, :]) + shift[:, None, None, :]


class SamePositionMemoryAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_head_dim: int,
        value_head_dim: int,
        gain_mode: str = "none",
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        value_input_dim: int | None = None,
    ):
        super().__init__()
        for name, value in (
            ("dim", dim),
            ("num_heads", num_heads),
            ("qk_head_dim", qk_head_dim),
            ("value_head_dim", value_head_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        self.dim = dim
        self.num_heads = num_heads
        self.qk_head_dim = qk_head_dim
        self.value_head_dim = value_head_dim
        if gain_mode not in {"none", "per_head"}:
            raise ValueError("gain_mode must be 'none' or 'per_head'")
        self.gain_mode = gain_mode
        # Diagnostic escape hatch for fused-SDPA backward instability. This
        # changes no parameters or checkpoint keys.
        self.use_explicit_attention = False
        self.scale = qk_head_dim ** -0.5
        self.q_proj = nn.Linear(dim, num_heads * qk_head_dim)
        self.k_proj = nn.Linear(dim, num_heads * qk_head_dim)
        self.v_proj = nn.Linear(dim if value_input_dim is None else value_input_dim,
                                num_heads * value_head_dim)
        self.out_proj = nn.Linear(num_heads * value_head_dim, dim)
        if gain_mode == "per_head":
            self.gain_weight = nn.Parameter(torch.zeros(num_heads, qk_head_dim))
            # softplus(log(exp(1) - 1)) == 1, so enabling gains does not
            # perturb the initial function.
            initial_bias = torch.log(torch.expm1(torch.tensor(1.0)))
            self.gain_bias = nn.Parameter(initial_bias.expand(num_heads).clone())
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        *,
        value_memory: torch.Tensor | None = None,
        memory_group_ids: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        """Attend over R at matching positions.

        value_memory: raw [R,N,W] bank, or [G,R,N,W] with W=v_proj.in_features.
            None retains the historical contextualized-value operator.
        query:  [B,N,D]
        memory: [B,R,N,D], or [G,R,N,D] with ``memory_group_ids`` mapping B to G.
        """
        b, n, d = query.shape
        bm, r, nm, dm = memory.shape
        if n != nm or d != dm or (memory_group_ids is None and b != bm):
            raise ValueError(f"query/memory mismatch: {tuple(query.shape)} vs {tuple(memory.shape)}")

        q = self.q_proj(query).view(b, n, self.num_heads, self.qk_head_dim)
        # Project conditioned keys per noise group. Raw values are projected
        # once for the whole batch, before broadcasting across noise groups.
        k = self.k_proj(memory).view(bm, r, n, self.num_heads, self.qk_head_dim)
        if value_memory is None:
            value_memory = memory
        w = self.v_proj.in_features
        if value_memory.ndim == 3 and value_memory.shape == (r, n, w):
            v = self.v_proj(value_memory).view(1, r, n, self.num_heads, self.value_head_dim)
            v = v.expand(bm, -1, -1, -1, -1)
        elif value_memory.shape == (bm, r, n, w):
            v = self.v_proj(value_memory).view(bm, r, n, self.num_heads, self.value_head_dim)
        else:
            raise ValueError(
                f"value memory must be {(r, n, w)} or {(bm, r, n, w)}, "
                f"got {tuple(value_memory.shape)}"
            )
        if memory_group_ids is not None:
            memory_group_ids = memory_group_ids.to(device=query.device, dtype=torch.long)
            k = k.index_select(0, memory_group_ids)
            v = v.index_select(0, memory_group_ids)
        k_by_position = k.permute(0, 2, 3, 1, 4)  # [B,N,H,R,d]
        v_by_position = v.permute(0, 2, 3, 1, 4)
        if return_attention or self.use_explicit_attention:
            logits = torch.einsum("bnhq,bnhrq->bnhr", q.float(), k_by_position.float())
            weights = (logits * self.scale).softmax(dim=-1)  # [B,N,H,R]
            dropped = self.attn_drop(weights)
            out = torch.einsum(
                "bnhr,bnhrv->bnhv", dropped, v_by_position.float()
            ).to(query.dtype)
        else:
            weights = None
            # Flatten B and N to use CUDA's four-dimensional fused SDPA path.
            q_sdpa = q.reshape(b * n, self.num_heads, 1, self.qk_head_dim)
            k_sdpa = k_by_position.reshape(b * n, self.num_heads, r, self.qk_head_dim)
            v_sdpa = v_by_position.reshape(b * n, self.num_heads, r, self.value_head_dim)
            out = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            ).squeeze(-2).reshape(b, n, self.num_heads, self.value_head_dim)
        if self.gain_mode == "per_head":
            gain = F.softplus(
                torch.einsum("bnhq,hq->bnh", q.float(), self.gain_weight.float())
                + self.gain_bias.float()
            ).to(out.dtype)
            out = out * gain.unsqueeze(-1)
        out = out.reshape(b, n, self.num_heads * self.value_head_dim)
        out = self.proj_drop(self.out_proj(out))
        return (out, weights) if return_attention else out


class DPD4Block(nn.Module):
    def __init__(
        self,
        dim: int,
        self_attention_heads: int,
        memory_attention_heads: int,
        memory_qk_head_dim: int,
        memory_value_head_dim: int,
        memory_gain_mode: str = "none",
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        shared_memory_projection: bool = True,
        memory_value_source: str = "raw",
        memory_token_dim: int | None = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = FusedMultiHeadSelfAttention(
            dim, self_attention_heads, attn_drop=attn_drop, proj_drop=proj_drop
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.memory_attn = SamePositionMemoryAttention(
            dim,
            memory_attention_heads,
            memory_qk_head_dim,
            memory_value_head_dim,
            gain_mode=memory_gain_mode,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            value_input_dim=memory_token_dim if memory_value_source == "raw" else dim,
        )
        self.adaLN = nn.Linear(dim, 6 * dim)
        self.shared_memory_projection = shared_memory_projection
        if memory_value_source not in {"raw", "contextualized"}:
            raise ValueError("memory_value_source must be 'raw' or 'contextualized'")
        self.memory_value_source = memory_value_source
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

    def forward(
        self,
        x: torch.Tensor,
        memory_base: torch.Tensor,
        c: torch.Tensor,
        c_groups: torch.Tensor,
        noise_group_ids: torch.Tensor,
        *,
        return_attention: bool = False,
        memory_values: torch.Tensor | None = None,
    ):
        if self.memory_value_source == "raw" and memory_values is None:
            raise ValueError("raw-value DPD4 requires the unconditioned memory_values bank")
        value_memory = memory_values if self.memory_value_source == "raw" else None
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(c).chunk(6, dim=-1)
        gshift1, gscale1, ggate1, gshift2, gscale2, _ = self.adaLN(c_groups).chunk(6, dim=-1)

        query_input = _modulate(self.norm1(x), shift1, scale1)
        memory_input = _modulate_memory(self.norm1(memory_base), gshift1, gscale1)
        # Query and memory use identical spatial-attention weights and sequence
        # length, so one larger SDPA call is more efficient than two calls.
        groups, r, n, d = memory_input.shape
        b = x.shape[0]
        combined_sa = self.attn(
            torch.cat([query_input, memory_input.reshape(groups * r, n, d)], dim=0)
        )
        query_sa = combined_sa[:b]
        memory_sa = combined_sa[b:].reshape(groups, r, n, d)
        x = x + gate1[:, None, :] * query_sa
        memory = memory_base[None] + ggate1[:, None, None, :] * memory_sa

        query = _modulate(self.norm2(x), shift2, scale2)
        memory = _modulate_memory(self.norm2(memory), gshift2, gscale2)
        if self.shared_memory_projection:
            result = self.memory_attn(
                query, memory, value_memory=value_memory, memory_group_ids=noise_group_ids,
                return_attention=return_attention,
            )
        else:
            selected_memory = memory.index_select(0, noise_group_ids)
            result = self.memory_attn(
                query, selected_memory, value_memory=value_memory,
                return_attention=return_attention,
            )
        if return_attention:
            update, weights = result
        else:
            update, weights = result, None
        x = x + gate2[:, None, :] * update
        return (x, weights) if return_attention else x


@dataclass
class DPD4Config:
    img_resolution: int = 32
    img_channels: int = 3
    patch_size: int = 2
    embed_dim: int = 256
    depth: int = 8
    num_heads: int = 8
    memory_size: int = 32
    memory_attention_heads: int = 8
    # Direct bank width. A depth-shared, key-only compatibility map is used
    # when this differs from embed_dim; raw values never pass through it.
    memory_token_dim: Optional[int] = None
    # Legacy checkpoint metadata only. New models use direct memory exclusively.
    memory_mode: Optional[str] = None
    memory_code_dim: Optional[int] = None
    memory_qk_head_dim: Optional[int] = None
    memory_value_head_dim: Optional[int] = None
    # None preserves the layout and behavior of checkpoints created before
    # tokenwise gains were introduced.
    memory_gain_mode: Optional[str] = None
    # New models retrieve raw same-position tokens. Checkpoint readers must
    # use from_checkpoint() so pre-correction checkpoints retain their behavior.
    memory_value_source: str = "raw"
    memory_code_init: str = "uniform_width"
    memory_lift_init: str = "orthogonal"
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    shared_memory_projection: bool = True
    print_params: bool = True

    @classmethod
    def from_checkpoint(cls, config: dict) -> "DPD4Config":
        values = dict(config)
        values.setdefault("memory_value_source", "contextualized")
        cfg = cls(**values)
        # Evaluation-only compatibility: never expose factorization as a new
        # model/training option. This runtime marker is not serialized.
        cfg._legacy_factorized_checkpoint = cfg.memory_mode == "factorized"
        return cfg


class DPD4Backbone(nn.Module):
    """EDM-compatible DPD4 backbone with per-example noise conditioning."""

    def __init__(self, cfg: DPD4Config):
        super().__init__()
        self.cfg = cfg
        if cfg.img_resolution % cfg.patch_size != 0:
            raise ValueError("img_resolution must be divisible by patch_size")
        if cfg.embed_dim % cfg.num_heads:
            raise ValueError("embed_dim must be divisible by self-attention heads")
        if cfg.memory_size <= 0 or cfg.memory_attention_heads <= 0:
            raise ValueError("memory size and memory attention heads must be positive")
        cfg.memory_mode = cfg.memory_mode or "direct"
        if cfg.memory_token_dim is None:
            cfg.memory_token_dim = cfg.embed_dim
        if cfg.memory_token_dim <= 0:
            raise ValueError("memory_token_dim must be positive")
        if cfg.memory_token_dim != cfg.embed_dim and (
            cfg.memory_mode != "direct" or cfg.memory_value_source != "raw"
        ):
            raise ValueError("Independent memory width requires direct memory and raw values")
        if cfg.memory_mode == "factorized" and not getattr(cfg, "_legacy_factorized_checkpoint", False):
            raise ValueError("Memory factorization is retired; new Learned PLD models use direct memory. "
                             "Legacy evaluation must load a checkpoint config via from_checkpoint().")
        if cfg.memory_code_dim is None:
            cfg.memory_code_dim = cfg.embed_dim
        if cfg.memory_qk_head_dim is None:
            cfg.memory_qk_head_dim = cfg.embed_dim // cfg.memory_attention_heads
        if cfg.memory_value_head_dim is None:
            cfg.memory_value_head_dim = cfg.embed_dim // cfg.memory_attention_heads
        if cfg.memory_mode not in {"direct", "factorized"}:
            raise ValueError("memory_mode must be 'direct' or 'factorized'")
        cfg.memory_gain_mode = cfg.memory_gain_mode or "none"
        if cfg.memory_gain_mode not in {"none", "per_head"}:
            raise ValueError("memory_gain_mode must be 'none' or 'per_head'")
        if cfg.memory_code_init != "uniform_width" or cfg.memory_lift_init != "orthogonal":
            raise ValueError("supported memory initializers are uniform_width and orthogonal")
        for name in ("memory_code_dim", "memory_qk_head_dim", "memory_value_head_dim"):
            if getattr(cfg, name) <= 0:
                raise ValueError(f"{name} must be positive")
        self.grid_size = cfg.img_resolution // cfg.patch_size
        self.num_patches = self.grid_size**2
        self.patch_embed = nn.Conv2d(
            cfg.img_channels, cfg.embed_dim, cfg.patch_size, stride=cfg.patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, cfg.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if cfg.memory_mode == "direct":
            bound = cfg.memory_token_dim ** -0.5
            # Preserve the original parameter name and shape for strict old
            # checkpoint loading.
            self.latent_memory = nn.Parameter(
                torch.empty(cfg.memory_size, self.num_patches, cfg.memory_token_dim).uniform_(
                    -bound, bound
                )
            )
        else:
            bound = cfg.memory_code_dim ** -0.5
            self.memory_codes = nn.Parameter(
                torch.empty(cfg.memory_size, self.num_patches, cfg.memory_code_dim).uniform_(
                    -bound, bound
                )
            )
            self.memory_lift = nn.Linear(cfg.memory_code_dim, cfg.embed_dim, bias=False)
            nn.init.orthogonal_(self.memory_lift.weight)
        # Identity adds no parameters or state keys, preserving the existing
        # width-256 model exactly (including its initialization sequence).
        if cfg.memory_token_dim == cfg.embed_dim:
            self.memory_key_adapter = nn.Identity()
        else:
            self.memory_key_adapter = nn.Linear(cfg.memory_token_dim, cfg.embed_dim, bias=False)
            nn.init.orthogonal_(self.memory_key_adapter.weight)
        self.time_embed = TimestepEmbedder(cfg.embed_dim)
        self.blocks = nn.ModuleList(
            DPD4Block(
                cfg.embed_dim,
                cfg.num_heads,
                cfg.memory_attention_heads,
                cfg.memory_qk_head_dim,
                cfg.memory_value_head_dim,
                memory_gain_mode=cfg.memory_gain_mode,
                attn_drop=cfg.attn_drop,
                proj_drop=cfg.proj_drop,
                shared_memory_projection=cfg.shared_memory_projection,
                memory_value_source=cfg.memory_value_source,
                memory_token_dim=cfg.memory_token_dim,
            )
            for _ in range(cfg.depth)
        )
        self.final_norm = nn.LayerNorm(cfg.embed_dim, elementwise_affine=False)
        self.final_adaLN = nn.Linear(cfg.embed_dim, 2 * cfg.embed_dim)
        nn.init.zeros_(self.final_adaLN.weight)
        nn.init.zeros_(self.final_adaLN.bias)
        self.head = nn.Linear(cfg.embed_dim, cfg.img_channels * cfg.patch_size**2)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        if cfg.print_params:
            print(f"[model_dpd4] DPD4Backbone params = {count_parameters(self) / 1e6:.3f} M")

    def compile_blocks(self):
        """Compile block callables without changing checkpoint parameter names."""
        for block in self.blocks:
            if not hasattr(block.forward, "_torchdynamo_orig_callable"):
                block.forward = torch.compile(block.forward)
        return self

    def unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        b, n, _ = tokens.shape
        p, c, g = self.cfg.patch_size, self.cfg.img_channels, self.grid_size
        if n != g * g:
            raise ValueError(f"expected {g*g} tokens, got {n}")
        x = tokens.view(b, g, g, c, p, p).permute(0, 3, 1, 4, 2, 5)
        return x.contiguous().view(b, c, g * p, g * p)

    def forward(
        self,
        x: torch.Tensor,
        c_noise: torch.Tensor,
        class_labels=None,
        *,
        noise_group_ids: torch.Tensor | None = None,
        return_attention: bool = False,
        return_all: bool = False,
        **kwargs,
    ):
        del class_labels, kwargs
        b, c, h, w = x.shape
        if (c, h, w) != (self.cfg.img_channels, self.cfg.img_resolution, self.cfg.img_resolution):
            raise ValueError(f"unexpected image shape {tuple(x.shape)}")
        c_noise = torch.as_tensor(c_noise, device=x.device, dtype=torch.float32).flatten()
        if c_noise.numel() == 1:
            c_noise = c_noise.expand(b)
        if c_noise.numel() != b:
            raise ValueError(f"c_noise must be scalar or [B], got {tuple(c_noise.shape)}")
        if noise_group_ids is None:
            group_noise, noise_group_ids = torch.unique(
                c_noise, sorted=True, return_inverse=True
            )
        else:
            noise_group_ids = torch.as_tensor(
                noise_group_ids, device=x.device, dtype=torch.long
            ).flatten()
            if noise_group_ids.numel() != b:
                raise ValueError(
                    f"noise_group_ids must have B elements, got {tuple(noise_group_ids.shape)}"
                )
            if noise_group_ids.min() < 0:
                raise ValueError("noise_group_ids must be nonnegative")
            groups = int(noise_group_ids.max().item()) + 1
            counts = torch.bincount(noise_group_ids, minlength=groups)
            if torch.any(counts == 0):
                raise ValueError("noise_group_ids must be contiguous and every group must be used")
            group_noise = c_noise.new_zeros(groups)
            group_noise.scatter_add_(0, noise_group_ids, c_noise)
            group_noise = group_noise / counts.to(group_noise.dtype)
            reconstructed = group_noise.index_select(0, noise_group_ids)
            if not torch.allclose(c_noise, reconstructed, rtol=1e-5, atol=1e-6):
                raise ValueError("all samples assigned to a noise group must share c_noise")
        cond_groups = self.time_embed(group_noise)
        cond = cond_groups.index_select(0, noise_group_ids)
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2).contiguous()
        tokens = tokens + self.pos_embed
        memory_latent = (
            self.latent_memory
            if self.cfg.memory_mode == "direct"
            else self.memory_lift(self.memory_codes)
        )
        memory_base = self.memory_key_adapter(memory_latent) + self.pos_embed
        attentions = []
        states = [tokens]
        for block in self.blocks:
            call = block
            if self.training and getattr(self, "activation_checkpointing", False) and not return_attention:
                from torch.utils.checkpoint import checkpoint
                from functools import partial
                call = partial(checkpoint, block, use_reentrant=False)
            result = call(
                tokens,
                memory_base,
                cond,
                cond_groups,
                noise_group_ids,
                return_attention=return_attention,
                memory_values=memory_latent if self.cfg.memory_value_source == "raw" else None,
            )
            if return_attention:
                tokens, weights = result
                attentions.append(weights)
            else:
                tokens = result
            if return_all:
                states.append(tokens)
        shift, scale = self.final_adaLN(cond).chunk(2, dim=-1)
        tokens = _modulate(self.final_norm(tokens), shift, scale)
        output = self.unpatchify(self.head(tokens))
        if return_all:
            return {"final": output, "states": states, "attentions": attentions}
        return (output, attentions) if return_attention else output
