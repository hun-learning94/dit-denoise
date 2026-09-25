"""DPD5: DiT with each MLP replaced by shared-prototype attention."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import FusedMultiHeadSelfAttention, TimestepEmbedder, count_parameters


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class PrototypeAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        key_width: int,
        value_width: int,
        qk_head_dim: int,
        value_head_dim: int,
        gain_mode: str,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        for name, value in (
            ("dim", dim),
            ("num_heads", num_heads),
            ("key_width", key_width),
            ("value_width", value_width),
            ("qk_head_dim", qk_head_dim),
            ("value_head_dim", value_head_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if gain_mode not in {"scalar", "per_head"}:
            raise ValueError("prototype_gain_mode must be 'scalar' or 'per_head'")
        self.dim = dim
        self.num_heads = num_heads
        self.key_width = key_width
        self.value_width = value_width
        self.qk_head_dim = qk_head_dim
        self.value_head_dim = value_head_dim
        self.gain_mode = gain_mode
        self.dropout_p = float(attn_drop)
        self.q_proj = nn.Linear(dim, num_heads * qk_head_dim)
        self.k_proj = nn.Linear(key_width, num_heads * qk_head_dim)
        self.v_proj = nn.Linear(value_width, num_heads * value_head_dim)
        self.out_proj = nn.Linear(num_heads * value_head_dim, dim)
        if gain_mode == "per_head":
            self.gain_weight = nn.Parameter(torch.zeros(num_heads, qk_head_dim))
            self.gain_bias = nn.Parameter(
                torch.full((num_heads,), math.log(math.e - 1.0))
            )
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        query: torch.Tensor,
        key_prototypes: torch.Tensor,
        value_prototypes: torch.Tensor,
        *,
        return_attention: bool = False,
    ):
        b, n, _ = query.shape
        r = key_prototypes.shape[0]
        q = self.q_proj(query).view(b, n, self.num_heads, self.qk_head_dim)
        k = self.k_proj(key_prototypes).view(r, self.num_heads, self.qk_head_dim)
        v = self.v_proj(value_prototypes).view(r, self.num_heads, self.value_head_dim)
        q_sdpa = q.transpose(1, 2)
        k_heads = k.permute(1, 0, 2)
        v_heads = v.permute(1, 0, 2)
        k_batch = k_heads.unsqueeze(0).expand(b, -1, -1, -1)
        v_batch = v_heads.unsqueeze(0).expand(b, -1, -1, -1)

        if return_attention:
            logits = torch.einsum("bnhq,rhq->bnhr", q.float(), k.float())
            logits = logits * (self.qk_head_dim ** -0.5)
            weights = logits.softmax(dim=-1)
            dropped = F.dropout(weights, p=self.dropout_p, training=self.training)
            heads = torch.einsum("bnhr,rhv->bnhv", dropped, v.float()).to(query.dtype)
        else:
            weights = None
            heads = F.scaled_dot_product_attention(
                q_sdpa,
                k_batch,
                v_batch,
                dropout_p=self.dropout_p if self.training else 0.0,
            ).transpose(1, 2)
        gain = None
        if self.gain_mode == "per_head":
            gain = F.softplus(
                torch.einsum("bnhq,hq->bnh", q.float(), self.gain_weight.float())
                + self.gain_bias.float()
            ).to(heads.dtype)
            heads = heads * gain[..., None]
        out = heads.contiguous().view(b, n, self.num_heads * self.value_head_dim)
        out = self.proj_drop(self.out_proj(out))
        return (out, weights, gain) if return_attention else out


class DPD5Block(nn.Module):
    def __init__(
        self,
        dim: int,
        self_attention_heads: int,
        prototype_attention_heads: int,
        prototype_key_width: int,
        prototype_value_width: int,
        prototype_qk_head_dim: int,
        prototype_value_head_dim: int,
        prototype_gain_mode: str,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = FusedMultiHeadSelfAttention(
            dim, self_attention_heads, attn_drop=attn_drop, proj_drop=proj_drop
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.prototype_attn = PrototypeAttention(
            dim,
            prototype_attention_heads,
            prototype_key_width,
            prototype_value_width,
            prototype_qk_head_dim,
            prototype_value_head_dim,
            prototype_gain_mode,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.prototype_gain_mode = prototype_gain_mode
        if prototype_gain_mode == "scalar":
            # Preserve the original module name and shape for strict loading of
            # pre-compression DPD5 checkpoints.
            self.gain = nn.Linear(dim, 1)
            nn.init.zeros_(self.gain.weight)
            nn.init.constant_(self.gain.bias, math.log(math.e - 1.0))
        self.adaLN = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        key_prototypes: torch.Tensor,
        value_prototypes: torch.Tensor,
        *,
        return_attention: bool = False,
    ):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(c).chunk(6, dim=-1)
        update = self.attn(_modulate(self.norm1(x), shift1, scale1))
        x = x + gate1[:, None, :] * update
        query = _modulate(self.norm2(x), shift2, scale2)
        result = self.prototype_attn(
            query,
            key_prototypes,
            value_prototypes,
            return_attention=return_attention,
        )
        if return_attention:
            update, weights, gain = result
        else:
            update, weights, gain = result, None, None
        if self.prototype_gain_mode == "scalar":
            gain = F.softplus(self.gain(query))
            update = gain * update
        x = x + gate2[:, None, :] * update
        return (x, weights, gain) if return_attention else x


@dataclass
class DPD5Config:
    img_resolution: int = 32
    img_channels: int = 3
    patch_size: int = 2
    embed_dim: int = 256
    depth: int = 8
    num_heads: int = 8
    prototype_count: int = 1024
    prototype_attention_heads: int = 8
    # None means the original full-width, scalar-gain architecture. New runs
    # created by the CLI pass the compressed defaults explicitly.
    prototype_key_width: Optional[int] = None
    prototype_value_width: Optional[int] = None
    prototype_qk_head_dim: Optional[int] = None
    prototype_value_head_dim: Optional[int] = None
    prototype_gain_mode: Optional[str] = None
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    print_params: bool = True


class DPD5Backbone(nn.Module):
    """EDM-compatible DPD5 backbone with per-example noise conditioning."""

    def __init__(self, cfg: DPD5Config):
        super().__init__()
        self.cfg = cfg
        if cfg.img_resolution % cfg.patch_size != 0:
            raise ValueError("img_resolution must be divisible by patch_size")
        if cfg.embed_dim % cfg.num_heads:
            raise ValueError("embed_dim must be divisible by self-attention heads")
        if cfg.prototype_attention_heads <= 0 or cfg.prototype_count <= 0:
            raise ValueError("prototype head and prototype counts must be positive")
        # Missing fields identify an old checkpoint/config and reconstruct its
        # architecture exactly. No full-width tensor is silently compressed.
        cfg.prototype_key_width = cfg.prototype_key_width or cfg.embed_dim
        cfg.prototype_value_width = cfg.prototype_value_width or cfg.embed_dim
        cfg.prototype_qk_head_dim = (
            cfg.prototype_qk_head_dim or cfg.embed_dim // cfg.prototype_attention_heads
        )
        cfg.prototype_value_head_dim = (
            cfg.prototype_value_head_dim or cfg.embed_dim // cfg.prototype_attention_heads
        )
        cfg.prototype_gain_mode = cfg.prototype_gain_mode or "scalar"
        self.grid_size = cfg.img_resolution // cfg.patch_size
        self.num_patches = self.grid_size**2
        self.patch_embed = nn.Conv2d(
            cfg.img_channels, cfg.embed_dim, cfg.patch_size, stride=cfg.patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, cfg.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        key_bound = cfg.prototype_key_width ** -0.5
        value_bound = cfg.prototype_value_width ** -0.5
        self.key_prototypes = nn.Parameter(
            torch.empty(cfg.prototype_count, cfg.prototype_key_width).uniform_(-key_bound, key_bound)
        )
        self.value_prototypes = nn.Parameter(
            torch.empty(cfg.prototype_count, cfg.prototype_value_width).uniform_(
                -value_bound, value_bound
            )
        )
        self.time_embed = TimestepEmbedder(cfg.embed_dim)
        self.blocks = nn.ModuleList(
            DPD5Block(
                cfg.embed_dim,
                cfg.num_heads,
                cfg.prototype_attention_heads,
                cfg.prototype_key_width,
                cfg.prototype_value_width,
                cfg.prototype_qk_head_dim,
                cfg.prototype_value_head_dim,
                cfg.prototype_gain_mode,
                attn_drop=cfg.attn_drop,
                proj_drop=cfg.proj_drop,
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
            print(f"[model_dpd5] DPD5Backbone params = {count_parameters(self) / 1e6:.3f} M")

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
        cond = self.time_embed(c_noise)
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2).contiguous()
        tokens = tokens + self.pos_embed
        attentions, gains = [], []
        states = [tokens]
        for block in self.blocks:
            result = block(
                tokens,
                cond,
                self.key_prototypes,
                self.value_prototypes,
                return_attention=return_attention,
            )
            if return_attention:
                tokens, weights, gain = result
                attentions.append(weights)
                gains.append(gain)
            else:
                tokens = result
            if return_all:
                states.append(tokens)
        shift, scale = self.final_adaLN(cond).chunk(2, dim=-1)
        tokens = _modulate(self.final_norm(tokens), shift, scale)
        output = self.unpatchify(self.head(tokens))
        if return_all:
            return {
                "final": output,
                "states": states,
                "attentions": attentions,
                "gains": gains,
            }
        return (output, attentions, gains) if return_attention else output
