"""Grouped DPD6: dense GELU coefficient groups over shared prototype banks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import FusedMultiHeadSelfAttention, TimestepEmbedder, count_parameters
from fused_grouped_gelu import fused_grouped_gelu


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class GroupedGELUExpansion(nn.Module):
    """H factorized GELU expansions over R shared K/V identities.

    Each key row is projected to ``H * (d + 1)`` entries so the detector and
    its affine bias are generated jointly from the same shared identity.
    Value rows are projected to ``H * d`` entries.  All groups are dense; this
    module has no expert router.
    """

    def __init__(self, dim: int, groups: int, group_dim: int, key_width: int,
                 value_width: int, fused: bool = False):
        super().__init__()
        for name, value in (("dim", dim), ("groups", groups), ("group_dim", group_dim),
                            ("key_width", key_width), ("value_width", value_width)):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        self.dim = dim
        self.groups = groups
        self.group_dim = group_dim
        self.fused = fused
        self.q_proj = nn.Linear(dim, groups * group_dim)
        # No projection bias: every detector and coefficient bias is generated
        # from its base prototype rather than receiving a prototype-independent offset.
        self.kb_proj = nn.Linear(key_width, groups * (group_dim + 1), bias=False)
        self.v_proj = nn.Linear(value_width, groups * group_dim, bias=False)
        self.out_proj = nn.Linear(groups * group_dim, dim)

    def project_library(self, key_prototypes: torch.Tensor, value_prototypes: torch.Tensor):
        r = key_prototypes.shape[0]
        if value_prototypes.shape[0] != r:
            raise ValueError("key and value prototype counts must match")
        kb = self.kb_proj(key_prototypes).view(r, self.groups, self.group_dim + 1)
        keys, biases = kb[..., : self.group_dim], kb[..., self.group_dim]
        values = self.v_proj(value_prototypes).view(r, self.groups, self.group_dim)
        return keys, biases, values

    def forward(self, query: torch.Tensor, key_prototypes: torch.Tensor | None = None,
                value_prototypes: torch.Tensor | None = None, *, projected=None,
                return_coefficients: bool = False):
        if projected is None:
            if key_prototypes is None or value_prototypes is None:
                raise ValueError("prototype banks are required without projected libraries")
            keys, biases, values = self.project_library(key_prototypes, value_prototypes)
        else:
            keys, biases, values = projected
        b, n, _ = query.shape
        q = self.q_proj(query).view(b, n, self.groups, self.group_dim)
        if self.fused and not return_coefficients:
            heads = fused_grouped_gelu(q, keys, biases, values)
            coefficients = None
        else:
            logits = torch.einsum("bnhd,rhd->bnhr", q, keys)
            logits = logits * (self.group_dim ** -0.5) + biases.permute(1, 0)[None, None]
            coefficients = F.gelu(logits)
            heads = torch.einsum("bnhr,rhd->bnhd", coefficients, values)
        output = self.out_proj(heads.reshape(b, n, self.groups * self.group_dim))
        return (output, coefficients) if return_coefficients else output


class GroupedDPD6Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, groups: int, group_dim: int,
                 key_width: int, value_width: int, attn_drop: float = 0.0,
                 proj_drop: float = 0.0, fused_expansion: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = FusedMultiHeadSelfAttention(dim, num_heads, attn_drop=attn_drop,
                                                proj_drop=proj_drop)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.expansion = GroupedGELUExpansion(
            dim, groups, group_dim, key_width, value_width, fused=fused_expansion
        )
        self.adaLN = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

    def forward(self, x, c, key_prototypes=None, value_prototypes=None, *, projected=None,
                return_coefficients=False):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(c).chunk(6, dim=-1)
        x = x + gate1[:, None, :] * self.attn(_modulate(self.norm1(x), shift1, scale1))
        result = self.expansion(_modulate(self.norm2(x), shift2, scale2), key_prototypes,
                                value_prototypes, projected=projected,
                                return_coefficients=return_coefficients)
        update, coefficients = result if return_coefficients else (result, None)
        x = x + gate2[:, None, :] * update
        return (x, coefficients) if return_coefficients else x


@dataclass
class GroupedDPD6Config:
    img_resolution: int = 32
    img_channels: int = 3
    patch_size: int = 2
    embed_dim: int = 256
    depth: int = 8
    num_heads: int = 8
    prototype_count: int = 1024
    coefficient_groups: int = 8
    group_dim: int = 32
    prototype_key_width: int = 256
    prototype_value_width: int = 256
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    fused_expansion: bool = False
    print_params: bool = True


class GroupedDPD6Backbone(nn.Module):
    """EDM-compatible grouped DPD6 with globally shared base K/V banks."""

    def __init__(self, cfg: GroupedDPD6Config):
        super().__init__()
        self.cfg = cfg
        if cfg.img_resolution % cfg.patch_size:
            raise ValueError("img_resolution must be divisible by patch_size")
        if cfg.embed_dim % cfg.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.grid_size = cfg.img_resolution // cfg.patch_size
        self.num_patches = self.grid_size ** 2
        self.patch_embed = nn.Conv2d(cfg.img_channels, cfg.embed_dim, cfg.patch_size,
                                     stride=cfg.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, cfg.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.key_prototypes = nn.Parameter(torch.empty(cfg.prototype_count,
                                                        cfg.prototype_key_width))
        self.value_prototypes = nn.Parameter(torch.empty(cfg.prototype_count,
                                                          cfg.prototype_value_width))
        nn.init.normal_(self.key_prototypes, std=cfg.prototype_key_width ** -0.5)
        nn.init.normal_(self.value_prototypes, std=cfg.prototype_value_width ** -0.5)
        self.time_embed = TimestepEmbedder(cfg.embed_dim)
        self.blocks = nn.ModuleList([
            GroupedDPD6Block(cfg.embed_dim, cfg.num_heads, cfg.coefficient_groups,
                             cfg.group_dim, cfg.prototype_key_width,
                             cfg.prototype_value_width, cfg.attn_drop, cfg.proj_drop,
                             fused_expansion=cfg.fused_expansion)
            for _ in range(cfg.depth)
        ])
        self.final_norm = nn.LayerNorm(cfg.embed_dim, elementwise_affine=False)
        self.final_adaLN = nn.Linear(cfg.embed_dim, 2 * cfg.embed_dim)
        nn.init.zeros_(self.final_adaLN.weight)
        nn.init.zeros_(self.final_adaLN.bias)
        self.head = nn.Linear(cfg.embed_dim, cfg.img_channels * cfg.patch_size ** 2)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self._projected_cache = None
        if cfg.print_params:
            print(f"[model_grouped_dpd6] params = {count_parameters(self) / 1e6:.3f} M")

    def clear_projected_cache(self):
        self._projected_cache = None

    @torch.no_grad()
    def prepare_projected_cache(self):
        """Cache per-layer K/V projections after final inference weights are loaded."""
        if self.training:
            raise RuntimeError("projected libraries may only be cached in eval mode")
        self._projected_cache = tuple(
            tuple(t.detach() for t in block.expansion.project_library(
                self.key_prototypes, self.value_prototypes)) for block in self.blocks
        )
        return self

    def train(self, mode: bool = True):
        if mode:
            self.clear_projected_cache()
        return super().train(mode)

    def _apply(self, fn):
        self.clear_projected_cache()
        return super()._apply(fn)

    def load_state_dict(self, *args, **kwargs):
        self.clear_projected_cache()
        return super().load_state_dict(*args, **kwargs)

    def unpatchify(self, tokens):
        b, n, _ = tokens.shape
        p, c, g = self.cfg.patch_size, self.cfg.img_channels, self.grid_size
        if n != g * g:
            raise ValueError(f"expected {g*g} tokens, got {n}")
        x = tokens.view(b, g, g, c, p, p).permute(0, 3, 1, 4, 2, 5)
        return x.contiguous().view(b, c, g * p, g * p)

    def forward(self, x, c_noise, class_labels=None, *, return_coefficients=False,
                return_all=False, **kwargs):
        del class_labels, kwargs
        b, c, h, w = x.shape
        expected = (self.cfg.img_channels, self.cfg.img_resolution, self.cfg.img_resolution)
        if (c, h, w) != expected:
            raise ValueError(f"expected [B,{expected[0]},{expected[1]},{expected[2]}], got {tuple(x.shape)}")
        c_noise = torch.as_tensor(c_noise, device=x.device, dtype=torch.float32).flatten()
        if c_noise.numel() == 1:
            c_noise = c_noise.expand(b)
        if c_noise.numel() != b:
            raise ValueError(f"c_noise must be scalar or [B], got {tuple(c_noise.shape)}")
        cond = self.time_embed(c_noise)
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2).contiguous() + self.pos_embed
        states, coefficients = [tokens], []
        for index, block in enumerate(self.blocks):
            projected = None if self._projected_cache is None else self._projected_cache[index]
            result = block(tokens, cond, self.key_prototypes, self.value_prototypes,
                           projected=projected, return_coefficients=return_coefficients)
            tokens, layer_coefficients = result if return_coefficients else (result, None)
            if return_coefficients:
                coefficients.append(layer_coefficients)
            if return_all:
                states.append(tokens)
        shift, scale = self.final_adaLN(cond).chunk(2, dim=-1)
        output = self.unpatchify(self.head(_modulate(self.final_norm(tokens), shift, scale)))
        if return_all:
            return {"final": output, "states": states, "coefficients": coefficients}
        return (output, coefficients) if return_coefficients else output


__all__ = ["GroupedDPD6Backbone", "GroupedDPD6Config", "GroupedGELUExpansion"]
