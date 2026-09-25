"""train.py

Single-GPU EDM training script with:
  - Backbones: DiT / UNet / ResNet (EDM-compatible signature)
  - Optional fixed patch shuffling (shared permutation for all images)
  - Robust resume:
      * Restores backbone config from checkpoint (including legacy DiT checkpoints).
      * If checkpoint was trained with shuffle, resume AUTOMATICALLY enables shuffle and restores the exact permutation.
      * Refuses to resume into a different regime unless --resume_override_args is set.

Usage examples:
python train.py --dataset folder --data_root ./datasets/cifar10_32/train --resolution 32 --channels 3 --arch dit --patch 2 --dim 256 --depth 8 --heads 8 --precision bf16 --batch 256 --steps 300000 --outdir runs/cifar10_dit --hflip

python train.py --dataset folder --data_root ./datasets/cifar10_32/train --resolution 32 --channels 3 --arch unet --unet_base 47 --unet_resblocks 3 --unet_midblocks 3 --precision bf16 --batch 256 --steps 300000 --outdir runs/cifar10_unet --hflip --resume runs/cifar10_unet/checkpoints/ckpt_step15000.pt

python train.py --dataset folder --data_root ./datasets/cifar10_32/train --resolution 32 --channels 3 --arch resnet --resnet_width 404 --resnet_blocks 6 --precision bf16 --batch 256 --steps 300000 --outdir runs/cifar10_resnet --hflip --resume runs/cifar10_resnet/checkpoints/ckpt_step15000.pt


python train.py --dataset mnist --data_root ./datasets --outdir runs/mnist_resnet \
    --arch resnet --precision bf16 --batch 256 --steps 300000
python train.py --dataset mnist --data_root ./datasets --outdir runs/mnist_resnet \
    --resume runs/mnist_resnet/checkpoints/ckpt_step285000.pt \
    --precision bf16 --batch 256 --steps 300000

python train.py --dataset fashionmnist --data_root ./datasets --outdir runs/fashionmnist_resnet \
    --arch resnet --precision bf16 --batch 256 --steps 300000
python train.py --dataset fashionmnist --data_root ./datasets --outdir runs/fashionmnist_resnet \
    --resume runs/fashionmnist_resnet/checkpoints/ckpt_step275000.pt \
    --precision bf16 --batch 256 --steps 300000

python train.py --dataset mnist --data_root ./datasets --outdir runs/mnist_unet \
    --arch unet --precision bf16 --batch 256 --steps 300000
python train.py --dataset mnist --data_root ./datasets --outdir runs/mnist_unet \
    --resume runs/mnist_unet/checkpoints/ckpt_step195000.pt \
    --precision bf16 --batch 256 --steps 300000

python train.py --dataset fashionmnist --data_root ./datasets --outdir runs/fashionmnist_unet \
    --arch unet --precision bf16 --batch 256 --steps 300000
python train.py --dataset fashionmnist --data_root ./datasets --outdir runs/fashionmnist_unet \
    --resume runs/fashionmnist_unet/checkpoints/ckpt_step195000.pt \
    --precision bf16 --batch 256 --steps 300000


python train.py --dataset mnist --data_root ./datasets --outdir runs/mnist_dit \
    --arch dit --precision bf16 --batch 256 --steps 300000
python train.py --dataset fashionmnist --data_root ./datasets --outdir runs/fashionmnist_dit \
    --arch dit --precision bf16 --batch 256 --steps 300000

python train.py --dataset fashionmnist --data_root ./datasets --outdir runs/fashionmnist_dit \
    --resume runs/fashionmnist_dit/checkpoints/ckpt_step270000.pt \
    --precision bf16 --batch 256 --steps 300000


  # Fresh shuffled run (DiT)
  python train.py --dataset mnist --data_root ./datasets --outdir runs/mnist_dit_shuf2 \
    --arch dit --shuffle --shuffle_p 2 --precision bf16 --batch 256 --steps 300000

  # Resume (you may omit --shuffle; it will auto-enable if ckpt had shuffle)
  python train.py --dataset mnist --data_root ./datasets --outdir runs/mnist_dit_shuf2 \
    --resume runs/mnist_dit_shuf2/checkpoints/ckpt_step160000.pt \
    --precision bf16 --batch 256 --steps 300000

python train.py \
  --dataset folder \
  --data_root ./datasets/celeba32/train \
  --resolution 32 \
  --channels 3 \
  --arch dit \
  --patch 2 --dim 256 --depth 8 --heads 8 \
  --outdir runs/celeba32_dit

python train.py \
  --dataset folder \
  --data_root ./datasets/celeba32/train \
  --resolution 32 \
  --channels 3 \
  --arch dit \
  --patch 2 --dim 256 --depth 8 --heads 8 \
  --shuffle --shuffle_p 2 \
  --shuffle_perm_path ./shuffle_specs/celeba32_p2_shared.pt \
  --outdir runs/celeba32_dit_shuf2

python train.py \
  --resume runs/celeba32_dit_mix/checkpoints/ckpt_step135000.pt \
  --dataset folder \
  --data_root ./datasets/celeba32/train \
  --resolution 32 \
  --channels 3 \
  --arch dit \
  --patch 2 --dim 256 --depth 8 --heads 8 \
  --channel_mix \
  --mix_spec_path ./mix_specs/celeba32_mix_p2.pt \
  --outdir runs/celeba32_dit_mix

python train.py \
  --dataset folder \
  --data_root ./datasets/celeba32/train \
  --resolution 32 \
  --channels 3 \
  --arch unet \
  --unet_base 47 --unet_resblocks 3 --unet_midblocks 3 \
  --shuffle --shuffle_p 2 \
  --shuffle_perm_path ./shuffle_specs/celeba32_p2_shared.pt \
  --outdir runs/celeba32_unet_shuf2

python train.py \
  --dataset folder \
  --data_root ./datasets/celeba32/train \
  --resolution 32 \
  --channels 3 \
  --arch unet \
  --unet_base 47 --unet_resblocks 3 --unet_midblocks 3 \
  --outdir runs/celeba32_unet

python train.py \
  --dataset folder \
  --data_root ./datasets/celeba32/train \
  --resolution 32 \
  --channels 3 \
  --arch unet \
  --unet_base 47 --unet_resblocks 3 --unet_midblocks 3 \
  --channel_mix \
  --mix_spec_path ./mix_specs/celeba32_mix_p2.pt \
  --outdir runs/celeba32_unet_mix

python train.py \
  --resume runs/celeba32_resnet_shuf2/checkpoints/ckpt_step120000.pt \
  --dataset folder \
  --data_root ./datasets/celeba32/train \
  --resolution 32 \
  --channels 3 \
  --arch resnet \
  --resnet_width 404 --resnet_blocks 6 \
  --shuffle --shuffle_p 2 \
  --shuffle_perm_path ./shuffle_specs/celeba32_p2_shared.pt \
  --outdir runs/celeba32_resnet_shuf2

python train.py \
  --resume runs/celeba32_resnet/checkpoints/ckpt_step110000.pt \
  --dataset folder \
  --data_root ./datasets/celeba32/train \
  --resolution 32 \
  --channels 3 \
  --arch resnet \
  --resnet_width 404 --resnet_blocks 6 \
  --outdir runs/celeba32_resnet

python train.py \
  --resume runs/celeba32_resnet_mix/checkpoints/ckpt_step135000.pt \
  --dataset folder \
  --data_root ./datasets/celeba32/train \
  --resolution 32 \
  --channels 3 \
  --arch resnet \
  --resnet_width 404 --resnet_blocks 6 \
  --channel_mix \
  --mix_spec_path ./mix_specs/celeba32_mix_p2.pt \
  --outdir runs/celeba32_resnet_mix
"""

from __future__ import annotations

import argparse
import math
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from edm import (
    EDMPrecond, EDMLoss, VelocityXPredLoss, VPVelocityXPredLoss,
    VPXPred, XPredPrecond, edm_sampler, geometric_heun_sampler, vp_heun_sampler,
)
from models import DiTConfig, UNetConfig, ResNetConfig, DiTBackbone, UNetBackbone, ResNetBackbone
import data as data_utils


# -----------------------------
# Utilities
# -----------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def checkpoint_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_initial_weights(net: nn.Module, checkpoint: dict, source: str) -> None:
    state = checkpoint[source]
    target = net.state_dict()
    if set(state) != set(target):
        missing = sorted(set(target) - set(state))
        unexpected = sorted(set(state) - set(target))
        raise ValueError(
            f"initial {source} state does not exactly match model: "
            f"missing={missing}, unexpected={unexpected}"
        )
    mismatched = [key for key in target if target[key].shape != state[key].shape]
    if mismatched:
        raise ValueError(f"initial {source} tensors have mismatched shapes: {mismatched}")
    net.load_state_dict(state, strict=True)


def configure_dit_trainable_groups(net: nn.Module, group: str):
    """Freeze a DiT by functional group and mask branch-specific AdaLN rows."""
    if group not in {"all", "block", "block_weights", "attn", "attn_weights", "mlp", "mlp_weights", "qk_weights", "mlp_vo_weights"}:
        raise ValueError(f"unknown trainable group: {group}")
    backbone = net.model
    if not isinstance(backbone, DiTBackbone):
        if group != "all":
            raise ValueError("restricted trainable groups are implemented only for DiT")
        trainable = list(net.parameters())
        return trainable, {}, {}, []

    dim = int(backbone.cfg.embed_dim)
    row_masks = {}
    hooks = []
    counts = {"global": 0, "attn": 0, "mlp": 0}
    trainable_counts = {"global": 0, "attn": 0, "mlp": 0}
    classified = set()

    for name, parameter in net.named_parameters():
        if not name.startswith("model.blocks."):
            family = "global"
        elif ".attn." in name:
            family = "attn"
        elif ".mlp." in name:
            family = "mlp"
        elif ".adaLN." in name:
            family = "adaln"
        else:
            raise RuntimeError(f"unclassified DiT block parameter: {name}")
        classified.add(name)

        if family != "adaln":
            counts[family] += parameter.numel()
            if group in {"qk_weights", "mlp_vo_weights"}:
                is_qkv = family == "attn" and ".attn.qkv." in name
                enabled = is_qkv or (group == "mlp_vo_weights" and
                    (family == "mlp" or (family == "attn" and ".attn.proj." in name)))
                parameter.requires_grad_(enabled)
                if is_qkv:
                    if parameter.shape[0] != 3 * dim:
                        raise RuntimeError(f"unexpected QKV shape: {name}")
                    mask = torch.zeros_like(parameter)
                    if group == "qk_weights":
                        mask[:2 * dim] = 1
                    else:
                        mask[2 * dim:] = 1
                    row_masks[name] = mask
                    hooks.append(parameter.register_hook(lambda grad, mask=mask: grad * mask))
                    trainable_counts[family] += int(mask.sum().item())
                elif enabled:
                    trainable_counts[family] += parameter.numel()
                continue
            enabled = (
                group == "all"
                or (group == "block" and family != "global")
                or (group == "block_weights" and family in {"attn", "mlp"})
                or group == family
                or (group == "attn_weights" and family == "attn")
                or (group == "mlp_weights" and family == "mlp")
            )
            parameter.requires_grad_(enabled)
            if enabled:
                trainable_counts[family] += parameter.numel()
            continue

        if parameter.shape[0] != 6 * dim:
            raise RuntimeError(f"unexpected AdaLN shape for {name}: {tuple(parameter.shape)}")
        counts["attn"] += 3 * dim * (parameter.numel() // (6 * dim))
        counts["mlp"] += 3 * dim * (parameter.numel() // (6 * dim))
        enabled = group in {"all", "block", "attn", "mlp"}
        parameter.requires_grad_(enabled)
        if group in {"all", "block"}:
            trainable_counts["attn"] += 3 * dim * (parameter.numel() // (6 * dim))
            trainable_counts["mlp"] += 3 * dim * (parameter.numel() // (6 * dim))
        elif group in {"attn", "mlp"}:
            mask = torch.zeros_like(parameter)
            if group == "attn":
                mask[:3 * dim] = 1
            else:
                mask[3 * dim:] = 1
            row_masks[name] = mask
            hooks.append(parameter.register_hook(lambda grad, mask=mask: grad * mask))
            trainable_counts[group] += int(mask.sum().item())

    expected = {name for name, _ in net.named_parameters()}
    if classified != expected:
        raise RuntimeError(f"parameter partition is incomplete: {sorted(expected - classified)}")
    trainable = [parameter for parameter in net.parameters() if parameter.requires_grad]
    manifest = {
        "group": group,
        "total": counts,
        "trainable": trainable_counts,
        "total_parameters": sum(counts.values()),
        "trainable_parameters": sum(trainable_counts.values()),
        "adaln_row_split": [0, 3 * dim, 6 * dim],
    }
    print("[trainable] " + json.dumps(manifest, sort_keys=True))

    frozen_snapshot = {}
    for name, parameter in net.named_parameters():
        if not parameter.requires_grad:
            frozen_snapshot[name] = (parameter.detach().cpu().clone(), None)
        elif name in row_masks:
            frozen_snapshot[name] = (parameter.detach().cpu().clone(), row_masks[name].detach().cpu().bool())
    return trainable, manifest, frozen_snapshot, hooks


@torch.no_grad()
def assert_frozen_unchanged(net: nn.Module, snapshot) -> None:
    current = dict(net.named_parameters())
    for name, (initial, trainable_mask) in snapshot.items():
        value = current[name].detach().cpu()
        if trainable_mask is None:
            unchanged = torch.equal(value, initial)
        else:
            unchanged = torch.equal(value[~trainable_mask], initial[~trainable_mask])
        if not unchanged:
            raise RuntimeError(f"frozen parameter coordinates changed: {name}")


def refresh_frozen_snapshot(net: nn.Module, snapshot):
    current = dict(net.named_parameters())
    return {
        name: (current[name].detach().cpu().clone(), trainable_mask)
        for name, (_initial, trainable_mask) in snapshot.items()
    }


def _apply_patch_perm(x: torch.Tensor, p: int, perm: torch.Tensor) -> torch.Tensor:
    """Apply a fixed permutation over non-overlapping p×p patches. x: [B,C,H,W]."""
    B, C, H, W = x.shape
    if H % p != 0 or W % p != 0:
        raise ValueError(f"shuffle_p={p} must divide H={H} and W={W}")
    gh, gw = H // p, W // p
    num = gh * gw
    if perm.numel() != num:
        raise ValueError(f"perm has {perm.numel()} elems but expected {num} for H=W={H}, p={p}")

    blocks = x.view(B, C, gh, p, gw, p).permute(0, 2, 4, 1, 3, 5).contiguous()
    blocks = blocks.view(B, num, C, p, p)
    blocks = blocks[:, perm, :, :, :]
    blocks = blocks.view(B, gh, gw, C, p, p).permute(0, 3, 1, 4, 2, 5).contiguous()
    return blocks.view(B, C, H, W)


def _invert_perm(perm: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype)
    return inv


def load_or_make_shuffle_perm(
    img_res: int,
    shuffle_p: int,
    seed: int,
    perm_path: Path,
    fraction: float = 1.0,
    force_new: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Path]:
    """Create or load a saved patch permutation at an explicit path."""
    perm_path = Path(perm_path)
    perm_path.parent.mkdir(parents=True, exist_ok=True)

    if perm_path.exists() and not force_new:
        d = torch.load(perm_path, map_location="cpu")
        if int(d.get("img_res")) != int(img_res) or int(d.get("shuffle_p")) != int(shuffle_p):
            raise ValueError(
                f"Existing perm at {perm_path} has mismatched "
                f"img_res/shuffle_p: got ({d.get('img_res')}, {d.get('shuffle_p')}), "
                f"expected ({img_res}, {shuffle_p})."
            )
        if abs(float(d.get("fraction", 1.0)) - float(fraction)) > 1e-12:
            raise ValueError(
                f"Existing perm at {perm_path} has fraction={d.get('fraction', 1.0)}, "
                f"expected {fraction}."
            )
        perm = d["perm"].long()
        inv_perm = d.get("inv_perm", None)
        inv_perm = _invert_perm(perm) if inv_perm is None else inv_perm.long()
        return perm, inv_perm, perm_path

    if img_res % shuffle_p != 0:
        raise ValueError(f"--shuffle_p {shuffle_p} must divide image resolution {img_res}")

    gh = img_res // shuffle_p
    num = gh * gh
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"--shuffle_fraction must be in (0,1], got {fraction}")
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    order = torch.randperm(num, generator=g)
    count = max(2, int(round(float(fraction) * num)))
    selected = order[:count]
    perm = torch.arange(num)
    perm[selected] = selected.roll(1)
    inv_perm = _invert_perm(perm)

    payload = dict(
        img_res=int(img_res),
        shuffle_p=int(shuffle_p),
        perm=perm.cpu(),
        inv_perm=inv_perm.cpu(),
        seed=int(seed),
        fraction=float(fraction),
    )
    torch.save(payload, perm_path)
    return perm, inv_perm, perm_path


def _normalize_basis(B: torch.Tensor, target_norm: float = 1.0) -> torch.Tensor:
    out = []
    for k in range(B.shape[0]):
        svals = torch.linalg.svdvals(B[k])
        smax = torch.clamp(svals.max(), min=1e-8)
        out.append(B[k] * (target_norm / smax))
    return torch.stack(out, dim=0)


def _make_basis_matrices() -> torch.Tensor:
    B1 = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
    B2 = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
    B3 = torch.tensor([[0.8, -0.8, 0.0], [-0.8, 0.8, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
    B4 = torch.tensor([[0.6, 0.0, -0.6], [0.0, 0.0, 0.0], [-0.6, 0.0, 0.6]], dtype=torch.float32)
    B = torch.stack([B1, B2, B3, B4], dim=0)
    return _normalize_basis(B, target_norm=1.0)


def _patch_centers(grid_h: int, grid_w: int) -> Tuple[torch.Tensor, torch.Tensor]:
    ys = torch.linspace(-1.0, 1.0, steps=grid_h)
    xs = torch.linspace(-1.0, 1.0, steps=grid_w)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return xx, yy


def _dist_to_skewed_arc(
    xx: torch.Tensor,
    yy: torch.Tensor,
    center: Tuple[float, float],
    radius: float,
    arc_mid_deg: float,
    arc_span_deg: float,
    sx: float = 1.0,
    sy: float = 1.0,
    shear: float = 0.0,
) -> torch.Tensor:
    cx, cy = center
    x = xx - cx
    y = yy - cy
    xw = (x + shear * y) / sx
    yw = y / sy
    rr = torch.sqrt(xw**2 + yw**2 + 1e-12)
    ang = torch.atan2(yw, xw)
    mid = torch.tensor(arc_mid_deg * torch.pi / 180.0, device=xx.device, dtype=xx.dtype)
    half = torch.tensor(0.5 * arc_span_deg * torch.pi / 180.0, device=xx.device, dtype=xx.dtype)
    dang = torch.atan2(torch.sin(ang - mid), torch.cos(ang - mid)).abs()
    d_rad = torch.abs(rr - radius)
    d_ang = torch.clamp(dang - half, min=0.0)
    return torch.sqrt(d_rad**2 + (0.8 * radius * d_ang)**2 + 1e-12)


def _weights_two_moons_center_bg(
    grid_h: int,
    grid_w: int,
    tau: float = 1.0,
    moon_sigma: float = 0.20,
    center_sigma: float = 0.20,
    bg_radius: float = 0.85,
    bg_sigma: float = 0.18,
    moon_span_deg: float = 180.0,
    moon_radius: float = 0.42,
    moon_shear: float = 0.0,
) -> torch.Tensor:
    xx, yy = _patch_centers(grid_h, grid_w)
    rr = torch.sqrt(xx**2 + yy**2 + 1e-12)
    d1 = _dist_to_skewed_arc(xx, yy, center=(-0.42, 0.10), radius=moon_radius, arc_mid_deg=140.0, arc_span_deg=moon_span_deg, sx=0.85, sy=1.15, shear=moon_shear)
    d2 = _dist_to_skewed_arc(xx, yy, center=(0.42, -0.10), radius=moon_radius, arc_mid_deg=-40.0, arc_span_deg=moon_span_deg, sx=0.85, sy=1.15, shear=-moon_shear)
    s1 = -(d1 ** 2) / (2 * moon_sigma ** 2)
    s2 = -(d2 ** 2) / (2 * moon_sigma ** 2)
    s3 = -(rr ** 2) / (2 * center_sigma ** 2)
    s4 = -((rr - bg_radius) ** 2) / (2 * bg_sigma ** 2)
    scores = torch.stack([1.0 * s1, 1.0 * s2, 2.0 * s3, 2.0 * s4], dim=-1)
    return torch.softmax(tau * scores, dim=-1)


def _weights_gaussian_bumps(
    grid_h: int,
    grid_w: int,
    tau: float = 1.0,
    sigma: float = 0.28,
) -> torch.Tensor:
    centers = [(-0.6, 0.55), (0.6, 0.55), (-0.6, -0.55), (0.6, -0.55)]
    xx, yy = _patch_centers(grid_h, grid_w)
    scores = []
    for cx, cy in centers:
        s = -(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        scores.append(s)
    scores = torch.stack(scores, dim=-1)
    return torch.softmax(tau * scores, dim=-1)


def _build_channel_mix_mats(
    img_res: int,
    mix_patch: int,
    mode: str,
    lam: float,
    tau: float,
    moon_sigma: float,
    center_sigma: float,
    bg_radius: float,
    bg_sigma: float,
    moon_span_deg: float,
    moon_radius: float,
    moon_shear: float,
    bump_sigma: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if img_res % mix_patch != 0:
        raise ValueError(f"--mix_patch {mix_patch} must divide image resolution {img_res}")
    gh, gw = img_res // mix_patch, img_res // mix_patch
    B = _make_basis_matrices()
    if mode == "global_orthogonal_rgb":
        # A spatially constant orthogonal rotation about the grayscale axis.
        # Therefore A @ (c,c,c) = (c,c,c) and A_inv = A.T.
        axis = torch.ones(3, dtype=torch.float32) / math.sqrt(3.0)
        x, y, z = axis
        K = torch.tensor([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        R = math.cos(lam) * torch.eye(3) + (1.0 - math.cos(lam)) * torch.outer(axis, axis) + math.sin(lam) * K
        A = R.view(1, 1, 3, 3).expand(gh, gw, 3, 3).contiguous()
        A_inv = R.T.view(1, 1, 3, 3).expand(gh, gw, 3, 3).contiguous()
        w = torch.ones(gh, gw, 1)
        return A, A_inv, w
    if mode == "two_moons_center_bg":
        w = _weights_two_moons_center_bg(gh, gw, tau=tau, moon_sigma=moon_sigma, center_sigma=center_sigma, bg_radius=bg_radius, bg_sigma=bg_sigma, moon_span_deg=moon_span_deg, moon_radius=moon_radius, moon_shear=moon_shear)
    elif mode == "gaussian_bumps":
        w = _weights_gaussian_bumps(gh, gw, tau=tau, sigma=bump_sigma)
    else:
        raise ValueError(f"Unknown channel-mix mode: {mode}")
    M = torch.einsum("hwk,kij->hwij", w, B)
    A = torch.matrix_exp(lam * M)
    A_inv = torch.matrix_exp(-lam * M)
    return A, A_inv, w


def _apply_patchwise_channel_mix(x: torch.Tensor, A: torch.Tensor, patch: int) -> torch.Tensor:
    """
    x: [B,C,H,W], C must be 3
    A: [Gh,Gw,3,3]
    """
    Bsz, C, H, W = x.shape
    if C != 3:
        raise ValueError("Channel mix requires 3-channel images.")
    if H % patch != 0 or W % patch != 0:
        raise ValueError(f"mix_patch={patch} must divide H={H} and W={W}")

    gh, gw = H // patch, W // patch
    if tuple(A.shape[:2]) != (gh, gw):
        raise ValueError(f"A has spatial shape {tuple(A.shape[:2])} but expected {(gh, gw)}")

    A = A.to(device=x.device, dtype=x.dtype)

    x_blocks = x.view(Bsz, C, gh, patch, gw, patch).permute(0, 2, 4, 3, 5, 1)
    y_blocks = torch.einsum("hwij,nhwabj->nhwabi", A, x_blocks)
    y = y_blocks.permute(0, 5, 1, 3, 2, 4).contiguous().view(Bsz, C, H, W)
    return y


def load_or_make_channel_mix_spec(
    img_res: int,
    mix_patch: int,
    spec_path: Path,
    mode: str,
    lam: float,
    tau: float,
    moon_sigma: float,
    center_sigma: float,
    bg_radius: float,
    bg_sigma: float,
    moon_span_deg: float,
    moon_radius: float,
    moon_shear: float,
    bump_sigma: float,
    force_new: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Path]:
    spec_path = Path(spec_path)
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    expected = dict(
        img_res=int(img_res), mix_patch=int(mix_patch), mode=str(mode), lam=float(lam), tau=float(tau),
        moon_sigma=float(moon_sigma), center_sigma=float(center_sigma), bg_radius=float(bg_radius), bg_sigma=float(bg_sigma),
        moon_span_deg=float(moon_span_deg), moon_radius=float(moon_radius), moon_shear=float(moon_shear), bump_sigma=float(bump_sigma),
    )
    if spec_path.exists() and not force_new:
        d = torch.load(spec_path, map_location="cpu")
        for k, v in expected.items():
            if k not in d:
                raise ValueError(f"Existing mix spec at {spec_path} is missing field {k}.")
            dv = d[k]
            if isinstance(v, float):
                if abs(float(dv) - v) > 1e-12:
                    raise ValueError(f"Existing mix spec at {spec_path} has mismatched {k}: got {dv}, expected {v}.")
            else:
                if dv != v:
                    raise ValueError(f"Existing mix spec at {spec_path} has mismatched {k}: got {dv}, expected {v}.")
        return d["A"].float(), d["A_inv"].float(), d["w"].float(), spec_path
    A, A_inv, w = _build_channel_mix_mats(img_res=img_res, mix_patch=mix_patch, mode=mode, lam=lam, tau=tau, moon_sigma=moon_sigma, center_sigma=center_sigma, bg_radius=bg_radius, bg_sigma=bg_sigma, moon_span_deg=moon_span_deg, moon_radius=moon_radius, moon_shear=moon_shear, bump_sigma=bump_sigma)
    payload = dict(**expected, A=A.cpu(), A_inv=A_inv.cpu(), w=w.cpu())
    torch.save(payload, spec_path)
    return A, A_inv, w, spec_path


@torch.no_grad()
def save_samples(
    net: nn.Module,
    out_path: str,
    n: int,
    img_resolution: int,
    img_channels: int,
    num_steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device: str = "cuda",
    shuffle_p: Optional[int] = None,
    inv_perm: Optional[torch.Tensor] = None,
    mix_patch: Optional[int] = None,
    A_inv_mix: Optional[torch.Tensor] = None,
    prediction_type: str = "edm_preconditioned",
    noise_process: str = "ve",
    vp_denominator_floor: float = 0.05,
) -> None:
    """Generate and save a small image grid; optionally invert channel mix and shuffle for visualization."""
    net.eval()
    latents = torch.randn([n, img_channels, img_resolution, img_resolution], device=device)
    if prediction_type == "x" and noise_process == "vp":
        samples = vp_heun_sampler(
            net=net, latents=latents, num_steps=num_steps,
            denominator_floor=vp_denominator_floor,
        )
    elif prediction_type == "x":
        samples = geometric_heun_sampler(
            net=net, latents=latents, num_steps=num_steps,
            t_min=sigma_min, t_max=sigma_max,
        )
    else:
        samples = edm_sampler(
            net=net, latents=latents, num_steps=num_steps,
            sigma_min=sigma_min, sigma_max=sigma_max, rho=rho,
        )

    if (mix_patch is not None) and (A_inv_mix is not None):
        samples = _apply_patchwise_channel_mix(samples, A=A_inv_mix.to(device), patch=int(mix_patch))

    if (shuffle_p is not None) and (inv_perm is not None):
        samples = _apply_patch_perm(samples, p=int(shuffle_p), perm=inv_perm.to(device))

    x01 = (samples.float() + 1.0) / 2.0
    x01 = x01.clamp(0.0, 1.0)

    nrow = int(max(1, round(n ** 0.5)))
    grid = make_grid(x01, nrow=nrow, padding=2)
    save_image(grid, out_path)
    net.train()


class EMA:
    """Simple EMA of model parameters (fp32 shadow)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k] = v.detach().float().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        msd = model.state_dict()
        for k, v in msd.items():
            if k not in self.shadow:
                continue
            self.shadow[k].mul_(d).add_(v.detach().to(self.shadow[k].device).float(), alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.shadow.items():
            if k in msd:
                msd[k].copy_(v.to(msd[k].dtype))


def build_dataloader(args) -> Tuple[DataLoader, int, int, int]:
    """Return (dataloader, img_resolution, img_channels, dataset_size)."""
    name = args.dataset.lower()

    if name == "mnist":
        dl, ds = data_utils.make_mnist_dataloader(
            root=args.data_root,
            batch_size=args.batch,
            train=True,
            normalize_to="neg_one_to_one",
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return dl, 28, 1, len(ds)

    if name == "fashionmnist":
        dl, ds = data_utils.make_fashionmnist_dataloader(
            root=args.data_root,
            batch_size=args.batch,
            train=True,
            normalize_to="neg_one_to_one",
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return dl, 28, 1, len(ds)

    if name == "folder":
        if args.resolution is None:
            raise ValueError("--resolution is required for --dataset folder")
        if args.channels not in (1, 3):
            raise ValueError("--channels must be 1 or 3")

        tfm = data_utils.build_transform(
            resolution=args.resolution,
            hflip=args.hflip,
            channels=args.channels,
            normalize_to="neg_one_to_one",
        )
        ds = data_utils.ImageFolderFlat(root=args.data_root, transform=tfm)
        dl = DataLoader(
            ds,
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return dl, int(args.resolution), int(args.channels), len(ds)

    raise ValueError(f"Unknown dataset: {args.dataset}. Use mnist|fashionmnist|folder")


def build_backbone_from_ckpt_or_args(args, img_res: int, img_ch: int, ckpt: Optional[dict]) -> Tuple[nn.Module, dict]:
    """Build backbone. If resuming and not overriding, prefer ckpt config (incl. legacy DiT)."""
    if ckpt is not None and (not args.resume_override_args):
        if "backbone" in ckpt and ckpt["backbone"] is not None:
            meta = ckpt["backbone"]
            arch = str(meta.get("arch", "dit")).lower()
            cfgd = meta.get("cfg", {})
            if arch == "dit":
                cfg = DiTConfig(**cfgd)
                cfg.img_resolution = img_res
                cfg.img_channels = img_ch
                cfg.print_params = True
                return DiTBackbone(cfg), {"arch": "dit", "cfg": asdict(cfg)}
            if arch == "unet":
                cfg = UNetConfig(**cfgd)
                cfg.img_resolution = img_res
                cfg.img_channels = img_ch
                cfg.print_params = True
                return UNetBackbone(cfg), {"arch": "unet", "cfg": asdict(cfg)}
            if arch == "resnet":
                cfg = ResNetConfig(**cfgd)
                cfg.img_resolution = img_res
                cfg.img_channels = img_ch
                cfg.print_params = True
                return ResNetBackbone(cfg), {"arch": "resnet", "cfg": asdict(cfg)}
            raise ValueError(f"Checkpoint backbone arch={arch} not recognized.")

        if "dit_cfg" in ckpt and ckpt["dit_cfg"] is not None:
            cfgd = ckpt["dit_cfg"]
            cfg = DiTConfig(**cfgd) if isinstance(cfgd, dict) else DiTConfig()
            cfg.img_resolution = img_res
            cfg.img_channels = img_ch
            cfg.print_params = True
            return DiTBackbone(cfg), {"arch": "dit", "cfg": asdict(cfg)}

    # Fresh run or override: build from CLI args
    arch = args.arch.lower()
    if arch == "dit":
        cfg = DiTConfig(
            img_resolution=img_res,
            img_channels=img_ch,
            patch_size=args.patch,
            embed_dim=args.dim,
            depth=args.depth,
            num_heads=args.heads,
            mlp_ratio=args.mlp_ratio,
            print_params=True,
        )
        return DiTBackbone(cfg), {"arch": "dit", "cfg": asdict(cfg)}
    if arch == "unet":
        cfg = UNetConfig(
            img_resolution=img_res,
            img_channels=img_ch,
            base_channels=args.unet_base,
            num_res_blocks=args.unet_resblocks,
            num_mid_blocks=args.unet_midblocks,
            time_dim=args.time_dim,
            gn_groups=args.gn_groups,
            print_params=True,
        )
        return UNetBackbone(cfg), {"arch": "unet", "cfg": asdict(cfg)}
    if arch == "resnet":
        cfg = ResNetConfig(
            img_resolution=img_res,
            img_channels=img_ch,
            width=args.resnet_width,
            num_blocks=args.resnet_blocks,
            time_dim=args.time_dim,
            gn_groups=args.gn_groups,
            print_params=True,
        )
        return ResNetBackbone(cfg), {"arch": "resnet", "cfg": asdict(cfg)}
    raise ValueError("--arch must be one of: dit | unet | resnet")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--dataset", type=str, default="mnist", help="mnist | fashionmnist | folder")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--hflip", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)

    # Backbone
    p.add_argument("--arch", type=str, default="dit", help="dit | unet | resnet")

    # Shuffle
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--shuffle_p", type=int, default=2)
    p.add_argument("--shuffle_fraction", type=float, default=1.0)
    p.add_argument("--shuffle_new", action="store_true")
    p.add_argument(
        "--shuffle_perm_path",
        type=str,
        default=None,
        help="Explicit shared permutation file path. "
            "If set, all runs using this path will share the exact same patch permutation.",
    )

    # Channel mix
    p.add_argument("--channel_mix", action="store_true")
    p.add_argument("--mix_patch", type=int, default=2)
    p.add_argument("--mix_new", action="store_true")
    p.add_argument("--mix_spec_path", type=str, default=None)
    p.add_argument("--mix_mode", type=str, default="two_moons_center_bg", choices=["two_moons_center_bg", "gaussian_bumps", "global_orthogonal_rgb"])
    p.add_argument("--mix_lam", type=float, default=0.45)
    p.add_argument("--mix_tau", type=float, default=1.0)
    p.add_argument("--moon_sigma", type=float, default=0.20)
    p.add_argument("--center_sigma", type=float, default=0.20)
    p.add_argument("--bg_radius", type=float, default=0.85)
    p.add_argument("--bg_sigma", type=float, default=0.18)
    p.add_argument("--moon_span_deg", type=float, default=180.0)
    p.add_argument("--moon_radius", type=float, default=0.42)
    p.add_argument("--moon_shear", type=float, default=0.0)
    p.add_argument("--bump_sigma", type=float, default=0.28)

    # Output
    p.add_argument("--outdir", type=str, default="./runs/exp")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--ckpt_every", type=int, default=5000)
    p.add_argument("--checkpoint_steps", type=int, nargs="*", default=[],
                   help="Additional exact optimizer-step checkpoints, alongside ckpt_every")
    p.add_argument("--sample_every", type=int, default=5000)
    p.add_argument("--sample_n", type=int, default=36)

    # Optimization
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--no_ema", action="store_true")

    # Precision
    p.add_argument("--precision", type=str, default="bf16", help="fp32 | fp16 | bf16")

    # DiT config
    p.add_argument("--patch", type=int, default=2)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=4.0)

    # UNet/ResNet config
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--gn_groups", type=int, default=32)
    p.add_argument("--unet_base", type=int, default=47)
    p.add_argument("--unet_resblocks", type=int, default=3)
    p.add_argument("--unet_midblocks", type=int, default=3)
    p.add_argument("--resnet_width", type=int, default=404)
    p.add_argument("--resnet_blocks", type=int, default=6)

    # Sampling hyperparams
    p.add_argument("--sample_steps", type=int, default=18)
    p.add_argument("--sigma_min", type=float, default=0.002)
    p.add_argument("--sigma_max", type=float, default=80.0)
    p.add_argument("--rho", type=float, default=7.0)
    p.add_argument(
        "--prediction_type", choices=["edm_preconditioned", "x"],
        default="edm_preconditioned",
    )
    p.add_argument(
        "--time_distribution", choices=["uniform_t", "uniform_t2"],
        default="uniform_t",
    )
    p.add_argument("--sigma_data", type=float, default=0.5)
    p.add_argument("--time_min", type=float, default=0.05)
    p.add_argument("--time_max", type=float, default=4.0)
    p.add_argument("--noise_process", choices=["ve", "vp"], default="ve")
    p.add_argument("--vp_logit_mean", type=float, default=-0.8)
    p.add_argument("--vp_logit_std", type=float, default=0.8)
    p.add_argument("--vp_denominator_floor", type=float, default=0.05)

    # Misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--init_from", type=str, default=None,
                   help="Initialize model weights from a checkpoint without restoring optimizer or step state.")
    p.add_argument("--init_weights", choices=["ema", "net"], default="ema")
    p.add_argument(
        "--trainable_groups",
        choices=["all", "block", "block_weights", "attn", "attn_weights", "mlp", "mlp_weights", "qk_weights", "mlp_vo_weights"],
        default=None,
    )
    p.add_argument("--adapt_steps", type=int, default=None,
                   help="Target-domain training steps; an explicit adaptation-oriented alias for --steps.")
    p.add_argument("--resume_override_args", action="store_true",
                   help="Ignore checkpoint backbone+shuffle regime and use CLI args (dangerous).")

    args = p.parse_args()
    if args.resume is not None and args.init_from is not None:
        p.error("--resume and --init_from are mutually exclusive")
    if args.prediction_type == "x" and args.sample_steps < 2:
        p.error("direct-x sampling requires --sample_steps >= 2")
    if args.noise_process == "vp" and args.prediction_type != "x":
        p.error("--noise_process vp requires --prediction_type x")
    if args.adapt_steps is not None:
        if args.adapt_steps <= 0:
            p.error("--adapt_steps must be positive")
        args.steps = args.adapt_steps

    if any(step <= 0 or step > args.steps for step in args.checkpoint_steps):
        p.error("--checkpoint_steps must be positive and no greater than --steps/--adapt_steps")
    checkpoint_steps = frozenset(args.checkpoint_steps)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "checkpoints").mkdir(exist_ok=True)
    (outdir / "samples").mkdir(exist_ok=True)

    # Data
    dl, img_res, img_ch, ds_size = build_dataloader(args)
    print(f"[train] device={device} dataset={args.dataset} size={ds_size} batch={args.batch} res={img_res} ch={img_ch}")

    # Load checkpoint early (if any) to restore regime safely.
    ckpt = None
    init_ckpt = None
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu")
        saved_group = ckpt.get("trainable_manifest", {}).get("group")
        if saved_group is None and isinstance(ckpt.get("args"), dict):
            saved_group = ckpt["args"].get("trainable_groups")
        if args.trainable_groups is None:
            args.trainable_groups = saved_group or "all"
        elif saved_group is not None and args.trainable_groups != saved_group:
            raise ValueError(
                f"--trainable_groups={args.trainable_groups} does not match resumed checkpoint group={saved_group}"
            )

        # Auto-enable shuffle if ckpt had shuffle and user didn't specify it.
        if ("shuffle" in ckpt) and (ckpt["shuffle"] is not None) and (not args.shuffle):
            sd = ckpt["shuffle"]
            args.shuffle = True
            args.shuffle_p = int(sd.get("shuffle_p", args.shuffle_p))
            args.shuffle_fraction = float(sd.get("fraction", 1.0))
            print(f"[shuffle] NOTE: checkpoint contains shuffle; auto-enabling --shuffle --shuffle_p {args.shuffle_p}")

        if ("channel_mix" in ckpt) and (ckpt["channel_mix"] is not None) and (not args.channel_mix):
            md = ckpt["channel_mix"]
            args.channel_mix = True
            args.mix_patch = int(md.get("mix_patch", args.mix_patch))
            args.mix_mode = str(md.get("mode", args.mix_mode))
            args.mix_lam = float(md.get("lam", args.mix_lam))
            args.mix_tau = float(md.get("tau", args.mix_tau))
            args.moon_sigma = float(md.get("moon_sigma", args.moon_sigma))
            args.center_sigma = float(md.get("center_sigma", args.center_sigma))
            args.bg_radius = float(md.get("bg_radius", args.bg_radius))
            args.bg_sigma = float(md.get("bg_sigma", args.bg_sigma))
            args.moon_span_deg = float(md.get("moon_span_deg", args.moon_span_deg))
            args.moon_radius = float(md.get("moon_radius", args.moon_radius))
            args.moon_shear = float(md.get("moon_shear", args.moon_shear))
            args.bump_sigma = float(md.get("bump_sigma", args.bump_sigma))
            print(f"[channel_mix] NOTE: checkpoint contains channel mix; auto-enabling --channel_mix")

        # Refuse accidental regime mismatch (unless override).
        if (not args.resume_override_args):
            ckpt_has_shuffle = ("shuffle" in ckpt) and (ckpt["shuffle"] is not None)
            if ckpt_has_shuffle and (not args.shuffle):
                raise ValueError("Checkpoint was trained with shuffle but --shuffle is disabled (should be auto-enabled).")
            if (not ckpt_has_shuffle) and args.shuffle:
                raise ValueError("You requested --shuffle but checkpoint has no shuffle metadata. Use --resume_override_args if intended.")

            ckpt_has_mix = ("channel_mix" in ckpt) and (ckpt["channel_mix"] is not None)
            if ckpt_has_mix and (not args.channel_mix):
                raise ValueError("Checkpoint was trained with channel_mix but --channel_mix is disabled.")
            if (not ckpt_has_mix) and args.channel_mix:
                raise ValueError("You requested --channel_mix but checkpoint has no channel_mix metadata. Use --resume_override_args if intended.")

    elif args.init_from is not None:
        init_ckpt = torch.load(args.init_from, map_location="cpu")

    if args.trainable_groups is None:
        args.trainable_groups = "all"

    # Restore / create shuffle permutation.
    perm = None
    inv_perm = None
    perm_path = None
    if args.shuffle:
        min_unit = args.patch if args.arch.lower() == "dit" else 1
        if args.shuffle_p < min_unit:
            raise ValueError(
                f"--shuffle_p must be >= model unit. Got shuffle_p={args.shuffle_p} < {min_unit}."
            )

        if args.shuffle_perm_path is not None:
            perm_path = Path(args.shuffle_perm_path)
        else:
            perm_path = outdir / f"shuffle_perm_res{img_res}_p{args.shuffle_p}.pt"

        if ckpt is not None and ("shuffle" in ckpt) and (ckpt["shuffle"] is not None) and (not args.resume_override_args):
            sd = ckpt["shuffle"]
            if int(sd.get("img_res", img_res)) != int(img_res):
                raise ValueError("Checkpoint shuffle img_res mismatch current dataset resolution.")
            if int(sd.get("shuffle_p", args.shuffle_p)) != int(args.shuffle_p):
                raise ValueError("Checkpoint shuffle_p mismatch current --shuffle_p.")

            perm = sd["perm"].long().cpu()
            inv_perm = sd.get("inv_perm", None)
            inv_perm = _invert_perm(perm) if inv_perm is None else inv_perm.long().cpu()

            perm_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                dict(
                    img_res=int(img_res),
                    shuffle_p=int(args.shuffle_p),
                    fraction=float(sd.get("fraction", args.shuffle_fraction)),
                    perm=perm,
                    inv_perm=inv_perm,
                    seed=int(args.seed),
                ),
                perm_path,
            )
            print(f"[shuffle] restored permutation from checkpoint (p={args.shuffle_p}) -> saved {perm_path}")
        else:
            perm, inv_perm, perm_path = load_or_make_shuffle_perm(
                img_res=img_res,
                shuffle_p=int(args.shuffle_p),
                seed=int(args.seed),
                perm_path=perm_path,
                fraction=float(args.shuffle_fraction),
                force_new=bool(args.shuffle_new),
            )
            print(f"[shuffle] enabled: shuffle_p={args.shuffle_p} perm_saved={perm_path}")

    # Restore / create channel-mix spec.
    A_mix = None
    A_inv_mix = None
    mix_w = None
    mix_spec_path = None
    if args.channel_mix:
        if img_ch != 3:
            raise ValueError("--channel_mix currently requires 3-channel images.")
        if img_res % int(args.mix_patch) != 0:
            raise ValueError(f"--mix_patch {args.mix_patch} must divide image resolution {img_res}")

        if args.mix_spec_path is not None:
            mix_spec_path = Path(args.mix_spec_path)
        else:
            mix_spec_path = outdir / f"channel_mix_res{img_res}_p{args.mix_patch}_{args.mix_mode}.pt"

        if ckpt is not None and ("channel_mix" in ckpt) and (ckpt["channel_mix"] is not None) and (not args.resume_override_args):
            md = ckpt["channel_mix"]
            if int(md.get("img_res", img_res)) != int(img_res):
                raise ValueError("Checkpoint channel-mix img_res mismatch current dataset resolution.")
            if int(md.get("mix_patch", args.mix_patch)) != int(args.mix_patch):
                raise ValueError("Checkpoint mix_patch mismatch current --mix_patch.")

            A_mix = md["A"].float().cpu()
            A_inv_mix = md["A_inv"].float().cpu()
            mix_w = md.get("w", None)
            if mix_w is not None:
                mix_w = mix_w.float().cpu()

            mix_meta = {k: md.get(k) for k in ["img_res", "mix_patch", "mode", "lam", "tau", "moon_sigma", "center_sigma", "bg_radius", "bg_sigma", "moon_span_deg", "moon_radius", "moon_shear", "bump_sigma"]}
            mix_meta["A"] = A_mix
            mix_meta["A_inv"] = A_inv_mix
            mix_meta["w"] = mix_w
            mix_spec_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(mix_meta, mix_spec_path)
            print(f"[channel_mix] restored spec from checkpoint -> saved {mix_spec_path}")
        else:
            A_mix, A_inv_mix, mix_w, mix_spec_path = load_or_make_channel_mix_spec(
                img_res=img_res,
                mix_patch=int(args.mix_patch),
                spec_path=mix_spec_path,
                mode=args.mix_mode,
                lam=float(args.mix_lam),
                tau=float(args.mix_tau),
                moon_sigma=float(args.moon_sigma),
                center_sigma=float(args.center_sigma),
                bg_radius=float(args.bg_radius),
                bg_sigma=float(args.bg_sigma),
                moon_span_deg=float(args.moon_span_deg),
                moon_radius=float(args.moon_radius),
                moon_shear=float(args.moon_shear),
                bump_sigma=float(args.bump_sigma),
                force_new=bool(args.mix_new),
            )
            print(f"[channel_mix] enabled: mix_patch={args.mix_patch} spec_saved={mix_spec_path}")

    # Build backbone (prefers resume/init checkpoint cfg unless override)
    architecture_ckpt = ckpt if ckpt is not None else init_ckpt
    backbone, backbone_meta = build_backbone_from_ckpt_or_args(
        args, img_res=img_res, img_ch=img_ch, ckpt=architecture_ckpt,
    )

    if args.prediction_type == "x" and args.noise_process == "vp":
        net = VPXPred(backbone, img_res, img_ch, label_dim=0).to(device)
    elif args.prediction_type == "x":
        net = XPredPrecond(
            backbone, img_res, img_ch, label_dim=0,
            sigma_data=args.sigma_data, t_min=args.time_min, t_max=args.time_max,
        ).to(device)
    else:
        net = EDMPrecond(
            backbone=backbone, img_resolution=img_res, img_channels=img_ch,
            label_dim=0, use_fp16=False,
        ).to(device)

    if ckpt is not None:
        saved_prediction = ckpt.get(
            "prediction_type", ckpt.get("args", {}).get("prediction_type", "edm_preconditioned")
        )
        if saved_prediction != args.prediction_type:
            raise ValueError(
                f"cannot resume prediction_type={saved_prediction!r} checkpoint "
                f"as {args.prediction_type!r}"
            )
        saved_process = ckpt.get("noise_process", ckpt.get("args", {}).get("noise_process", "ve"))
        if args.prediction_type == "x" and saved_process != args.noise_process:
            raise ValueError(
                f"cannot resume noise_process={saved_process!r} checkpoint as {args.noise_process!r}"
            )

    if init_ckpt is not None:
        load_initial_weights(net, init_ckpt, args.init_weights)
        print(f"[train] Initialized {args.init_weights} weights from {args.init_from}; optimizer and step reset")

    trainable, trainable_manifest, frozen_snapshot, gradient_hooks = configure_dit_trainable_groups(
        net, args.trainable_groups,
    )
    del gradient_hooks  # Hooks remain registered on their parameters.
    if args.trainable_groups in {"attn", "mlp", "qk_weights", "mlp_vo_weights"} and args.weight_decay != 0:
        raise ValueError("row-masked fine-tuning requires --weight_decay 0 to preserve frozen coordinates")

    # Save config
    cfg_out = vars(args).copy()
    cfg_out["backbone"] = backbone_meta
    cfg_out["trainable_manifest"] = trainable_manifest
    source_checkpoint = args.init_from or (ckpt.get("source_checkpoint") if ckpt is not None else None)
    source_checkpoint_hash = (
        checkpoint_sha256(args.init_from)
        if args.init_from is not None
        else (ckpt.get("source_checkpoint_sha256") if ckpt is not None else None)
    )
    if source_checkpoint is not None:
        cfg_out["source_checkpoint"] = str(Path(source_checkpoint).resolve())
        cfg_out["source_checkpoint_sha256"] = source_checkpoint_hash
    if perm_path is not None:
        cfg_out["shuffle_perm_path"] = str(perm_path)
    if mix_spec_path is not None:
        cfg_out["mix_spec_path"] = str(mix_spec_path)
    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_out, f, indent=2)

    if args.prediction_type != "x":
        loss_fn = EDMLoss()
    elif args.noise_process == "vp":
        loss_fn = VPVelocityXPredLoss(
            args.vp_logit_mean, args.vp_logit_std, args.vp_denominator_floor,
        )
    else:
        loss_fn = VelocityXPredLoss(args.time_min, args.time_max, args.time_distribution)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    ema = None
    if not args.no_ema:
        ema = EMA(net, decay=args.ema)

    # Precision
    prec = args.precision.lower()
    if prec not in ("fp32", "fp16", "bf16"):
        raise ValueError("--precision must be fp32|fp16|bf16")

    use_amp = (prec != "fp32") and (device == "cuda")
    amp_dtype = torch.float16 if prec == "fp16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and prec == "fp16"))

    # Resume weights/opt
    start_step = 0
    if ckpt is not None:
        net.load_state_dict(ckpt["net"], strict=True)
        opt.load_state_dict(ckpt["opt"])
        start_step = int(ckpt.get("step", 0))
        if scaler is not None and "scaler" in ckpt:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception:
                pass
        if ema is not None and "ema" in ckpt and ckpt["ema"] is not None:
            ema.shadow = {k: v.clone().to(device) for k, v in ckpt["ema"].items()}
        frozen_snapshot = refresh_frozen_snapshot(net, frozen_snapshot)
        print(f"[train] Resumed from {args.resume} at step={start_step}")

    if args.shuffle:
        assert perm is not None and inv_perm is not None
    if args.channel_mix:
        assert A_mix is not None and A_inv_mix is not None

    perm_dev = perm.to(device) if perm is not None else None
    inv_perm_dev = inv_perm.to(device) if inv_perm is not None else None
    A_mix_dev = A_mix.to(device) if A_mix is not None else None
    A_inv_mix_dev = A_inv_mix.to(device) if A_inv_mix is not None else None

    # Training loop
    net.train()
    t0 = time.time()
    running_loss = 0.0
    running_count = 0

    data_iter = iter(dl)
    for step in range(start_step, args.steps):
        try:
            x, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            x, _ = next(data_iter)

        x = x.to(device, non_blocking=True)

        if args.shuffle:
            x = _apply_patch_perm(x, p=int(args.shuffle_p), perm=perm_dev)
        if args.channel_mix:
            x = _apply_patchwise_channel_mix(x, A=A_mix_dev, patch=int(args.mix_patch))

        if args.prediction_type != "x" and prec == "bf16":
            x = x.to(torch.bfloat16)
        elif args.prediction_type != "x" and prec == "fp16":
            x = x.to(torch.float16)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            loss = loss_fn(net, x).mean()

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    trainable, args.grad_clip, error_if_nonfinite=True,
                )
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    trainable, args.grad_clip, error_if_nonfinite=True,
                )
            opt.step()

        if ema is not None:
            ema.update(net)

        running_loss += float(loss.detach().cpu())
        running_count += 1
        if (step + 1) % args.log_every == 0:
            dt = time.time() - t0
            it_s = (step + 1 - start_step) / max(dt, 1e-9)
            avg = running_loss / max(running_count, 1)
            print(f"[train] step {step+1:>7d}/{args.steps} | loss {avg:.6f} | {it_s:.2f} it/s")
            running_loss = 0.0
            running_count = 0

        # Save checkpoint
        if (step + 1) % args.ckpt_every == 0 or (step + 1) == args.steps or (step + 1) in checkpoint_steps:
            assert_frozen_unchanged(net, frozen_snapshot)
            ckpt_path = outdir / "checkpoints" / f"ckpt_step{step+1}.pt"
            payload = {
                "step": step + 1,
                "net": net.state_dict(),
                "opt": opt.state_dict(),
                "backbone": backbone_meta,
                "dit_cfg": backbone_meta["cfg"] if backbone_meta.get("arch") == "dit" else None,
                "args": vars(args),
                "prediction_type": args.prediction_type,
                "noise_process": args.noise_process if args.prediction_type == "x" else "edm",
                "loss_type": "velocity_mse" if args.prediction_type == "x" else "edm",
                "time_distribution": (
                    "logit_normal" if args.noise_process == "vp" else args.time_distribution
                ) if args.prediction_type == "x" else "log_normal",
                "time_min": args.time_min if args.prediction_type == "x" and args.noise_process == "ve" else None,
                "time_max": args.time_max if args.prediction_type == "x" and args.noise_process == "ve" else None,
                "sigma_data": args.sigma_data if args.prediction_type == "x" and args.noise_process == "ve" else None,
                "vp_logit_mean": args.vp_logit_mean if args.noise_process == "vp" else None,
                "vp_logit_std": args.vp_logit_std if args.noise_process == "vp" else None,
                "vp_denominator_floor": args.vp_denominator_floor if args.noise_process == "vp" else None,
                "trainable_manifest": trainable_manifest,
            }
            if source_checkpoint is not None:
                payload["source_checkpoint"] = str(Path(source_checkpoint).resolve())
                payload["source_checkpoint_sha256"] = source_checkpoint_hash
            if args.shuffle:
                payload["shuffle"] = dict(
                    img_res=int(img_res),
                    shuffle_p=int(args.shuffle_p),
                    fraction=float(args.shuffle_fraction),
                    perm=perm.clone(),
                    inv_perm=inv_perm.clone(),
                )
            if args.channel_mix:
                payload["channel_mix"] = dict(
                    img_res=int(img_res),
                    mix_patch=int(args.mix_patch),
                    mode=str(args.mix_mode),
                    lam=float(args.mix_lam),
                    tau=float(args.mix_tau),
                    moon_sigma=float(args.moon_sigma),
                    center_sigma=float(args.center_sigma),
                    bg_radius=float(args.bg_radius),
                    bg_sigma=float(args.bg_sigma),
                    moon_span_deg=float(args.moon_span_deg),
                    moon_radius=float(args.moon_radius),
                    moon_shear=float(args.moon_shear),
                    bump_sigma=float(args.bump_sigma),
                    A=A_mix.clone(),
                    A_inv=A_inv_mix.clone(),
                    w=(mix_w.clone() if mix_w is not None else None),
                )
            if scaler is not None and scaler.is_enabled():
                payload["scaler"] = scaler.state_dict()
            if ema is not None:
                payload["ema"] = {k: v.clone().cpu() for k, v in ema.shadow.items()}
            torch.save(payload, ckpt_path)
            print(f"[train] saved checkpoint: {ckpt_path}")

        # Samples
        if (step + 1) % args.sample_every == 0 or (step + 1) == args.steps:
            sample_path = outdir / "samples" / f"sample_step{step+1}.png"
            sample_min = args.time_min if args.prediction_type == "x" else args.sigma_min
            sample_max = args.time_max if args.prediction_type == "x" else args.sigma_max
            if ema is not None:
                cur = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                ema.copy_to(net)
                save_samples(
                    net=net,
                    out_path=str(sample_path),
                    n=args.sample_n,
                    img_resolution=img_res,
                    img_channels=img_ch,
                    num_steps=args.sample_steps,
                    sigma_min=sample_min,
                    sigma_max=sample_max,
                    rho=args.rho,
                    device=device,
                    shuffle_p=(int(args.shuffle_p) if args.shuffle else None),
                    inv_perm=(inv_perm_dev if args.shuffle else None),
                    mix_patch=(int(args.mix_patch) if args.channel_mix else None),
                    A_inv_mix=(A_inv_mix_dev if args.channel_mix else None),
                    prediction_type=args.prediction_type,
                    noise_process=args.noise_process,
                    vp_denominator_floor=args.vp_denominator_floor,
                )
                net.load_state_dict(cur, strict=True)
            else:
                save_samples(
                    net=net,
                    out_path=str(sample_path),
                    n=args.sample_n,
                    img_resolution=img_res,
                    img_channels=img_ch,
                    num_steps=args.sample_steps,
                    sigma_min=sample_min,
                    sigma_max=sample_max,
                    rho=args.rho,
                    device=device,
                    shuffle_p=(int(args.shuffle_p) if args.shuffle else None),
                    inv_perm=(inv_perm_dev if args.shuffle else None),
                    mix_patch=(int(args.mix_patch) if args.channel_mix else None),
                    A_inv_mix=(A_inv_mix_dev if args.channel_mix else None),
                    prediction_type=args.prediction_type,
                    noise_process=args.noise_process,
                    vp_denominator_floor=args.vp_denominator_floor,
                )
            print(f"[train] saved samples: {sample_path}")


if __name__ == "__main__":
    main()
