"""Shared, checkpoint-compatible model construction for the scale campaign."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import random

import numpy as np
import torch

from models import DiTBackbone, DiTConfig, FusedMultiHeadSelfAttention
from model_grouped_dpd6 import GroupedDPD6Backbone, GroupedDPD6Config


DATASETS = {
    "celeba128": dict(resolution=128, count=162770, patch=8),
    "imagenet128": dict(resolution=128, count=1281167, patch=8),
    "imagenet256": dict(resolution=256, count=1281167, patch=16),
}


def seed_training(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(arch, resolution, seed):
    """Copy common tensors explicitly; predictor RNG draws cannot change them."""
    torch.manual_seed(seed)
    patch = resolution // 16
    cfg = DiTConfig(img_resolution=resolution, img_channels=3, patch_size=patch,
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, print_params=False)
    ordinary = DiTBackbone(cfg)
    for block in ordinary.blocks:
        attention = FusedMultiHeadSelfAttention(768, 12)
        attention.load_state_dict(block.attn.state_dict())
        block.attn = attention
    source = ordinary.state_dict()
    signature = hashlib.sha256()
    for name, value in sorted(source.items()):
        if ".mlp." not in name:
            signature.update(name.encode())
            signature.update(value.detach().numpy().tobytes())
    if arch == "mlp":
        raw = ordinary
    elif arch == "smlp":
        cfg = GroupedDPD6Config(img_resolution=resolution, img_channels=3, patch_size=patch,
            embed_dim=768, depth=12, num_heads=12, prototype_count=2304,
            coefficient_groups=8, group_dim=96, prototype_key_width=1840,
            prototype_value_width=1840, fused_expansion=True, print_params=False)
        raw = GroupedDPD6Backbone(cfg)
        destination = raw.state_dict()
        shared = {k: v for k, v in source.items() if k in destination and v.shape == destination[k].shape}
        raw.load_state_dict(shared, strict=False)
        assert all(torch.equal(raw.state_dict()[k], v) for k, v in shared.items())
    else:
        raise ValueError(arch)
    count = sum(p.numel() for p in raw.parameters())
    expected = {128: dict(mlp=133936320, smlp=134011584),
                256: dict(mlp=134821632, smlp=134896896)}
    assert count == expected[resolution][arch], count
    return raw, asdict(cfg), dict(parameters=count, common_initialization_sha256=signature.hexdigest(),
        attention="PyTorch SDPA", predictor="fused wide GELU" if arch == "smlp" else "GELU MLP")


class EMA:
    def __init__(self, canonical):
        self.shadow = {k: v.detach().float().clone() for k, v in canonical.state_dict().items()}

    @torch.no_grad()
    def update(self, canonical):
        for name, value in canonical.state_dict().items():
            self.shadow[name].mul_(0.999).add_(value.detach().float(), alpha=0.001)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state(state["cuda"])
