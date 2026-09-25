"""DPD6: DiT MLPs generated from depth-shared key/value libraries."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import FusedMultiHeadSelfAttention, TimestepEmbedder, count_parameters


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class SharedLibraryMLP(nn.Module):
    """DiT MLP with optionally shared, layer-projected key and value weights."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        key_library_dim: int,
        value_library_dim: int,
        sharing_mode: str,
        drop: float = 0.0,
    ):
        super().__init__()
        if sharing_mode not in {"both", "key", "value"}:
            raise ValueError("sharing_mode must be one of: both, key, value")
        for name, value in (
            ("dim", dim),
            ("hidden_dim", hidden_dim),
            ("key_library_dim", key_library_dim),
            ("value_library_dim", value_library_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.sharing_mode = sharing_mode
        self.key_projection = (
            nn.Linear(key_library_dim, dim + 1, bias=False)
            if sharing_mode in {"both", "key"}
            else None
        )
        self.value_projection = (
            nn.Linear(value_library_dim, dim, bias=False)
            if sharing_mode in {"both", "value"}
            else None
        )
        self.fc1 = nn.Linear(dim, hidden_dim) if sharing_mode == "value" else None
        self.fc2 = nn.Linear(hidden_dim, dim) if sharing_mode == "key" else None
        self.output_bias = (
            nn.Parameter(torch.zeros(dim)) if sharing_mode in {"both", "value"} else None
        )
        if self.key_projection is not None:
            nn.init.uniform_(
                self.key_projection.weight[:dim, :],
                -dim**-0.5,
                dim**-0.5,
            )
            nn.init.zeros_(self.key_projection.weight[dim, :])
        if self.value_projection is not None:
            value_bound = hidden_dim**-0.5
            nn.init.uniform_(self.value_projection.weight, -value_bound, value_bound)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def generated_parameters(
        self,
        key_library: torch.Tensor | None,
        value_library: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.key_projection is not None:
            if key_library is None:
                raise ValueError("key_library is required for this sharing mode")
            input_parameters = self.key_projection(key_library)
            weight1 = input_parameters[:, : self.dim]
            bias1 = input_parameters[:, self.dim]
        else:
            weight1, bias1 = self.fc1.weight, self.fc1.bias

        if self.value_projection is not None:
            if value_library is None:
                raise ValueError("value_library is required for this sharing mode")
            # F.linear expects [out_features, in_features].
            weight2 = self.value_projection(value_library).transpose(0, 1)
            bias2 = self.output_bias
        else:
            weight2, bias2 = self.fc2.weight, self.fc2.bias
        return weight1, bias1, weight2, bias2

    def forward(
        self,
        x: torch.Tensor,
        key_library: torch.Tensor | None,
        value_library: torch.Tensor | None,
        *,
        return_coefficients: bool = False,
    ):
        weight1, bias1, weight2, bias2 = self.generated_parameters(
            key_library, value_library
        )
        coefficients = self.act(F.linear(x, weight1, bias1))
        output = self.drop(coefficients)
        output = F.linear(output, weight2, bias2)
        output = self.drop(output)
        return (output, coefficients) if return_coefficients else output


class DPD6Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        hidden_dim: int,
        key_library_dim: int,
        value_library_dim: int,
        sharing_mode: str,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = FusedMultiHeadSelfAttention(
            dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = SharedLibraryMLP(
            dim,
            hidden_dim,
            key_library_dim,
            value_library_dim,
            sharing_mode,
            drop=mlp_drop,
        )
        self.adaLN = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        key_library: torch.Tensor | None,
        value_library: torch.Tensor | None,
        *,
        return_coefficients: bool = False,
    ):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(c).chunk(6, dim=-1)
        update = self.attn(_modulate(self.norm1(x), shift1, scale1))
        x = x + gate1[:, None, :] * update
        mlp_input = _modulate(self.norm2(x), shift2, scale2)
        result = self.mlp(
            mlp_input,
            key_library,
            value_library,
            return_coefficients=return_coefficients,
        )
        if return_coefficients:
            update, coefficients = result
        else:
            update, coefficients = result, None
        x = x + gate2[:, None, :] * update
        return (x, coefficients) if return_coefficients else x


@dataclass
class DPD6Config:
    img_resolution: int = 32
    img_channels: int = 3
    patch_size: int = 2
    embed_dim: int = 256
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    key_library_dim: int = 256
    value_library_dim: int = 256
    sharing_mode: str = "both"
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    mlp_drop: float = 0.0
    print_params: bool = True


class DPD6Backbone(nn.Module):
    """EDM-compatible DiT with depth-shared MLP K/V libraries."""

    def __init__(self, cfg: DPD6Config):
        super().__init__()
        self.cfg = cfg
        if cfg.img_resolution % cfg.patch_size:
            raise ValueError("img_resolution must be divisible by patch_size")
        if cfg.embed_dim % cfg.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if cfg.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if cfg.sharing_mode not in {"both", "key", "value"}:
            raise ValueError("sharing_mode must be one of: both, key, value")

        self.grid_size = cfg.img_resolution // cfg.patch_size
        self.num_patches = self.grid_size**2
        self.hidden_dim = int(cfg.embed_dim * cfg.mlp_ratio)
        if self.hidden_dim <= 0:
            raise ValueError("embed_dim * mlp_ratio must be positive")

        self.patch_embed = nn.Conv2d(
            cfg.img_channels, cfg.embed_dim, cfg.patch_size, stride=cfg.patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, cfg.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.time_embed = TimestepEmbedder(cfg.embed_dim)

        if cfg.sharing_mode in {"both", "key"}:
            self.key_library = nn.Parameter(
                torch.empty(self.hidden_dim, cfg.key_library_dim)
            )
            nn.init.normal_(self.key_library, std=cfg.key_library_dim**-0.5)
        else:
            self.register_parameter("key_library", None)
        if cfg.sharing_mode in {"both", "value"}:
            self.value_library = nn.Parameter(
                torch.empty(self.hidden_dim, cfg.value_library_dim)
            )
            nn.init.normal_(self.value_library, std=cfg.value_library_dim**-0.5)
        else:
            self.register_parameter("value_library", None)

        self.blocks = nn.ModuleList(
            DPD6Block(
                cfg.embed_dim,
                cfg.num_heads,
                self.hidden_dim,
                cfg.key_library_dim,
                cfg.value_library_dim,
                cfg.sharing_mode,
                attn_drop=cfg.attn_drop,
                proj_drop=cfg.proj_drop,
                mlp_drop=cfg.mlp_drop,
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
            print(f"[model_dpd6] DPD6Backbone params = {count_parameters(self) / 1e6:.3f} M")

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
        return_coefficients: bool = False,
        return_all: bool = False,
        **kwargs,
    ):
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
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2).contiguous()
        tokens = tokens + self.pos_embed
        states, coefficients = [tokens], []
        for block in self.blocks:
            result = block(
                tokens,
                cond,
                self.key_library,
                self.value_library,
                return_coefficients=return_coefficients,
            )
            if return_coefficients:
                tokens, layer_coefficients = result
                coefficients.append(layer_coefficients)
            else:
                tokens = result
            if return_all:
                states.append(tokens)

        shift, scale = self.final_adaLN(cond).chunk(2, dim=-1)
        tokens = _modulate(self.final_norm(tokens), shift, scale)
        output = self.unpatchify(self.head(tokens))
        if return_all:
            return {"final": output, "states": states, "coefficients": coefficients}
        return (output, coefficients) if return_coefficients else output


__all__ = ["DPD6Backbone", "DPD6Config", "SharedLibraryMLP"]
