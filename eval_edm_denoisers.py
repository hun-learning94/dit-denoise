"""
eval_edm_denoisers.py

Fixed-noise denoising evaluation for EDM-trained DiT / UNet / ResNet checkpoints.

This is the baseline counterpart to eval_analytic.py.  For each checkpoint and
sigma, it creates one output directory with the same kind of files as the
analytic evaluation:

    <outdir_root>/<dataset_name>_<sigma_tag>_<split_name>_<model_label>/
        args.json
        metrics.json
        denoise_grid_batch00000.png
        denoise_grid_batchxxxxx.png

Protocol:
    1. Load an EDM checkpoint trained by train.py.
    2. Use EMA weights by default if present.
    3. Load clean images from valid/test.
    4. Add fixed Gaussian noise:
           x_sigma = x0 + sigma * eps
    5. Evaluate the EDM denoiser:
           pred_x0 = EDMPrecond(backbone)(x_sigma, sigma)
    6. Report MSE / PSNR and save qualitative grids:
           row 1: clean x0
           row 2: noisy x_sigma
           row 3: denoised pred_x0

The EDM networks trained by train.py are already denoisers: EDMLoss corrupts a
clean image with Gaussian noise and trains net(y + n, sigma) to reconstruct y.
Therefore, at evaluation time we call the checkpoint directly at fixed sigma.

Examples:

python eval_edm_denoisers.py \
  --ckpt unet=runs/celeba32_unet/checkpoints/ckpt_step200000.pt \
  --eval_data_root ./datasets/celeba32/test \
  --dataset_name celeba32 \
  --split_name test \
  --resolution 32 \
  --channels 3 \
  --sigmas 0.25 0.5 1.0 2.0 4.0 \
  --eval_batch 64 \
  --max_eval 1024 \
  --precision bf16 \
  --use_ema \
  --noise_seed 12345 \
  --outdir_root evals

This creates:
    evals/celeba32_sigma05_unet/metrics.json
    evals/celeba32_sigma10_unet/metrics.json
    evals/celeba32_sigma20_unet/metrics.json

Multiple checkpoints are allowed:

python eval_edm_denoisers.py \
  --ckpt dit=runs/celeba32_dit/checkpoints/ckpt_step200000.pt \
  --ckpt unet=runs/celeba32_unet/checkpoints/ckpt_step200000.pt \
  --eval_data_root ./datasets/celeba32/test \
  --dataset_name celeba32 --split_name test \
  --resolution 32 --channels 3 \
  --sigmas 0.25 0.5 1.0 2.0 4.0 \
  --outdir_root evals

Notes:
  - Metrics are computed in the normalized image range [-1, 1], so PSNR uses data_range=2.
  - By default, predictions are clamped to [-1, 1] before computing the main mse/psnr,
    matching the usual known image range and the analytic model's clamp-style output.
    Raw unclipped metrics are also saved as mse_raw/psnr_raw.
  - Noise is generated from a local torch.Generator seeded by --noise_seed, so model
    initialization/loading does not affect the corruption. Every checkpoint sees the
    exact same noisy inputs for a given sigma.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.utils import make_grid, save_image

import data as data_utils
from edm import EDMPrecond
from models import (
    DiTBackbone,
    DiTConfig,
    ResNetBackbone,
    ResNetConfig,
    UNetBackbone,
    UNetConfig,
    count_parameters,
)


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_image_range_01(x: torch.Tensor) -> torch.Tensor:
    return ((x.float() + 1.0) / 2.0).clamp(0.0, 1.0)


def mse_to_psnr(mse: float, data_range: float = 2.0) -> float:
    mse = max(float(mse), 1e-12)
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def get_amp_dtype(precision: str):
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision == "fp32":
        return None
    raise ValueError("--precision must be fp32 | fp16 | bf16")


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32" or device.type != "cuda":
        return torch.autocast(device_type="cuda", enabled=False)
    return torch.autocast(
        device_type="cuda",
        dtype=get_amp_dtype(precision),
        enabled=True,
    )


def normalize_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip torch.compile's _orig_mod prefix from compiled checkpoints."""
    prefix = "_orig_mod."
    if not any(k.startswith(prefix) for k in state):
        return state
    return {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in state.items()
    }


def sanitize_label(label: str) -> str:
    label = label.strip().lower()
    label = re.sub(r"[^a-z0-9_.-]+", "_", label)
    return label.strip("_") or "model"


def infer_label_from_path(path: str) -> str:
    p = Path(path)
    if p.parent.name == "checkpoints":
        run_name = p.parent.parent.name
    else:
        run_name = p.stem

    run_name = sanitize_label(run_name)
    toks = run_name.split("_")
    for arch in ("dit", "unet", "resnet"):
        if arch in toks:
            idx = toks.index(arch)
            return sanitize_label("_".join(toks[idx:]))
    return run_name


def parse_ckpt_spec(spec: str) -> Tuple[str, str]:
    """Parse NAME=PATH or PATH."""
    if "=" in spec:
        label, path = spec.split("=", 1)
        return sanitize_label(label), path
    path = spec
    return infer_label_from_path(path), path


def sigma_to_tag(sigma: float) -> str:
    """Match the sigma05/sigma10/sigma20 style used in analytic eval folders."""
    s = f"{float(sigma):.6g}"
    if "e" in s or "E" in s:
        s = f"{float(sigma):.6f}".rstrip("0").rstrip(".")
    if "." not in s:
        # Make 1 -> 1.0 -> sigma10 and 2 -> 2.0 -> sigma20.
        s = s + ".0"
    s = s.replace("-", "m").replace(".", "")
    return f"sigma{s}"


def make_run_outdir(
    *,
    outdir_root: str,
    dataset_name: str,
    sigma: float,
    split_name: str,
    model_label: str,
) -> Path:
    name = "_".join(
        [
            sanitize_label(dataset_name),
            sigma_to_tag(float(sigma)),
            sanitize_label(split_name),
            sanitize_label(model_label),
        ]
    )
    return Path(outdir_root) / name


def infer_split_name(eval_data_root: str) -> str:
    name = Path(eval_data_root).name
    return sanitize_label(name) if name else "eval"


# -----------------------------------------------------------------------------
# Dataset utilities
# -----------------------------------------------------------------------------


def build_folder_dataset(
    *,
    root: str,
    resolution: int,
    channels: int,
    hflip: bool,
    folder_tensor_cache: bool,
    folder_tensor_cache_path: Optional[str],
    rebuild_folder_tensor_cache: bool,
) -> Dataset:
    """Build folder dataset using the same conventions as train.py/eval_analytic.py."""
    if channels not in (1, 3):
        raise ValueError("--channels must be 1 or 3")

    if folder_tensor_cache:
        ds = data_utils.make_tensor_image_dataset(
            root=root,
            resolution=int(resolution),
            channels=int(channels),
            hflip=hflip,
            cache_path=folder_tensor_cache_path,
            rebuild_cache=rebuild_folder_tensor_cache,
        )
    else:
        tfm = data_utils.build_transform(
            resolution=int(resolution),
            hflip=hflip,
            channels=int(channels),
            normalize_to="neg_one_to_one",
        )
        ds = data_utils.ImageFolderFlat(root=root, transform=tfm)

    return ds


def build_eval_loader(args, eval_ds: Dataset) -> DataLoader:
    if args.max_eval is not None and args.max_eval > 0:
        n = min(int(args.max_eval), len(eval_ds))
        eval_ds = Subset(eval_ds, list(range(n)))
        print(f"[eval] Using first {n} eval images.")

    return DataLoader(
        eval_ds,
        batch_size=args.eval_batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )


# -----------------------------------------------------------------------------
# Optional train-time input transforms from train.py checkpoints
# -----------------------------------------------------------------------------


def _apply_patch_perm(x: torch.Tensor, p: int, perm: torch.Tensor) -> torch.Tensor:
    """Apply a fixed permutation over non-overlapping p x p patches. x: [B,C,H,W]."""
    b, c, h, w = x.shape
    if h % p != 0 or w % p != 0:
        raise ValueError(f"shuffle_p={p} must divide H={h} and W={w}")
    gh, gw = h // p, w // p
    num = gh * gw
    if perm.numel() != num:
        raise ValueError(f"perm has {perm.numel()} elems but expected {num} for H=W={h}, p={p}")

    blocks = x.view(b, c, gh, p, gw, p).permute(0, 2, 4, 1, 3, 5).contiguous()
    blocks = blocks.view(b, num, c, p, p)
    blocks = blocks[:, perm, :, :, :]
    blocks = blocks.view(b, gh, gw, c, p, p).permute(0, 3, 1, 4, 2, 5).contiguous()
    return blocks.view(b, c, h, w)


def _apply_patchwise_channel_mix(x: torch.Tensor, A: torch.Tensor, patch: int) -> torch.Tensor:
    """Apply spatially varying 3x3 channel mixing per patch."""
    bsz, c, h, w = x.shape
    if c != 3:
        raise ValueError("Channel mix requires 3-channel images.")
    if h % patch != 0 or w % patch != 0:
        raise ValueError(f"mix_patch={patch} must divide H={h} and W={w}")

    gh, gw = h // patch, w // patch
    if tuple(A.shape[:2]) != (gh, gw):
        raise ValueError(f"A has spatial shape {tuple(A.shape[:2])} but expected {(gh, gw)}")

    A = A.to(device=x.device, dtype=x.dtype)
    x_blocks = x.view(bsz, c, gh, patch, gw, patch).permute(0, 2, 4, 3, 5, 1)
    y_blocks = torch.einsum("hwij,nhwabj->nhwabi", A, x_blocks)
    y = y_blocks.permute(0, 5, 1, 3, 2, 4).contiguous().view(bsz, c, h, w)
    return y


class CheckpointInputTransform:
    """Forward/inverse transforms matching train.py's optional shuffle/channel_mix.

    train.py applies transforms in this order before EDMLoss:
        1. patch shuffle
        2. patchwise channel mix

    For evaluation against original clean images, we apply the same forward transform
    before adding noise, call the denoiser in that transformed domain, then apply the
    inverse transform to the prediction before computing metrics.
    """

    def __init__(self, ckpt: dict, device: torch.device):
        self.shuffle_p: Optional[int] = None
        self.perm: Optional[torch.Tensor] = None
        self.inv_perm: Optional[torch.Tensor] = None

        self.mix_patch: Optional[int] = None
        self.A: Optional[torch.Tensor] = None
        self.A_inv: Optional[torch.Tensor] = None

        if ("shuffle" in ckpt) and (ckpt["shuffle"] is not None):
            sd = ckpt["shuffle"]
            self.shuffle_p = int(sd["shuffle_p"])
            self.perm = sd["perm"].long().to(device)
            self.inv_perm = sd["inv_perm"].long().to(device)
            print(f"[transform] checkpoint uses patch shuffle p={self.shuffle_p}")

        if ("channel_mix" in ckpt) and (ckpt["channel_mix"] is not None):
            md = ckpt["channel_mix"]
            self.mix_patch = int(md["mix_patch"])
            self.A = md["A"].float().to(device)
            self.A_inv = md["A_inv"].float().to(device)
            print(f"[transform] checkpoint uses channel_mix patch={self.mix_patch}")

    @property
    def has_shuffle(self) -> bool:
        return self.shuffle_p is not None

    @property
    def has_channel_mix(self) -> bool:
        return self.mix_patch is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shuffle_p is not None:
            assert self.perm is not None
            x = _apply_patch_perm(x, p=self.shuffle_p, perm=self.perm)
        if self.mix_patch is not None:
            assert self.A is not None
            x = _apply_patchwise_channel_mix(x, A=self.A, patch=self.mix_patch)
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        if self.mix_patch is not None:
            assert self.A_inv is not None
            x = _apply_patchwise_channel_mix(x, A=self.A_inv, patch=self.mix_patch)
        if self.shuffle_p is not None:
            assert self.inv_perm is not None
            x = _apply_patch_perm(x, p=self.shuffle_p, perm=self.inv_perm)
        return x


# -----------------------------------------------------------------------------
# Checkpoint loading
# -----------------------------------------------------------------------------


def _build_backbone_from_meta(meta: dict, *, print_params: bool = True) -> Tuple[nn.Module, dict]:
    arch = str(meta.get("arch", "dit")).lower()
    cfgd = dict(meta.get("cfg", {}))

    if arch == "dit":
        cfg = DiTConfig(**cfgd)
        cfg.print_params = print_params
        return DiTBackbone(cfg), {"arch": "dit", "cfg": asdict(cfg)}

    if arch == "unet":
        cfg = UNetConfig(**cfgd)
        cfg.print_params = print_params
        return UNetBackbone(cfg), {"arch": "unet", "cfg": asdict(cfg)}

    if arch == "resnet":
        cfg = ResNetConfig(**cfgd)
        cfg.print_params = print_params
        return ResNetBackbone(cfg), {"arch": "resnet", "cfg": asdict(cfg)}

    raise ValueError(f"Checkpoint backbone arch={arch!r} is not recognized.")


@torch.no_grad()
def apply_ema_state(model: nn.Module, ema_state: Dict[str, torch.Tensor]) -> int:
    """Copy checkpoint EMA tensors into model where keys match."""
    ema_state = normalize_state_dict(ema_state)
    msd = model.state_dict()

    copied = 0
    for k, v in ema_state.items():
        if k not in msd:
            continue
        if not torch.is_floating_point(msd[k]):
            continue
        msd[k].copy_(v.to(device=msd[k].device, dtype=msd[k].dtype))
        copied += 1

    return copied


def load_edm_model_from_checkpoint(
    *,
    ckpt_path: str,
    device: torch.device,
    use_ema: bool,
    expected_resolution: int,
    expected_channels: int,
) -> Tuple[EDMPrecond, dict, dict, bool, CheckpointInputTransform]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "net" not in ckpt:
        raise KeyError(f"Checkpoint {ckpt_path} is missing key 'net'.")

    if "backbone" in ckpt and ckpt["backbone"] is not None:
        backbone_meta_in = ckpt["backbone"]
    elif "dit_cfg" in ckpt and ckpt["dit_cfg"] is not None:
        backbone_meta_in = {"arch": "dit", "cfg": ckpt["dit_cfg"]}
    else:
        raise KeyError(
            f"Checkpoint {ckpt_path} is missing backbone metadata. "
            "Expected key 'backbone' or legacy key 'dit_cfg'."
        )

    backbone, backbone_meta = _build_backbone_from_meta(backbone_meta_in, print_params=True)
    cfg = backbone_meta["cfg"]

    img_resolution = int(cfg["img_resolution"])
    img_channels = int(cfg["img_channels"])
    if img_resolution != int(expected_resolution) or img_channels != int(expected_channels):
        raise ValueError(
            f"Checkpoint geometry ({img_resolution}, {img_channels}) does not match "
            f"requested ({expected_resolution}, {expected_channels})."
        )

    net = EDMPrecond(
        backbone=backbone,
        img_resolution=img_resolution,
        img_channels=img_channels,
        label_dim=0,
        use_fp16=False,
    ).to(device)

    net.load_state_dict(normalize_state_dict(ckpt["net"]), strict=True)

    used_ema = False
    if use_ema:
        if "ema" in ckpt and ckpt["ema"] is not None:
            copied = apply_ema_state(net, ckpt["ema"])
            used_ema = True
            print(f"[ckpt] applied EMA tensors: {copied}")
        else:
            print("[ckpt] WARNING: --use_ema requested, but checkpoint has no EMA. Using raw weights.")

    transform = CheckpointInputTransform(ckpt, device=device)

    net.eval()
    print(f"[ckpt] loaded {ckpt_path}")
    print(f"[ckpt] step={ckpt.get('step', None)} used_ema={used_ema}")
    print(f"[model] EDMPrecond trainable params={count_parameters(net):,}")

    return net, backbone_meta, ckpt, used_ema, transform


# -----------------------------------------------------------------------------
# Evaluation and visualization
# -----------------------------------------------------------------------------


@torch.no_grad()
def denoise_edm(
    *,
    net: EDMPrecond,
    noisy: torch.Tensor,
    sigma: float,
    precision: str,
    device: torch.device,
) -> torch.Tensor:
    noisy = noisy.to(device=device, non_blocking=True)
    sigma_tensor = torch.full((noisy.shape[0],), float(sigma), device=device, dtype=torch.float32)
    with autocast_context(device, precision):
        pred = net(noisy, sigma_tensor)
    return pred.float()


def save_eval_grid(
    *,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    pred: torch.Tensor,
    out_path: Path,
    nrow: int,
) -> None:
    clean = clean[:nrow]
    noisy = noisy[:nrow]
    pred = pred[:nrow]

    vis = torch.cat(
        [
            to_image_range_01(clean),
            to_image_range_01(noisy),
            to_image_range_01(pred),
        ],
        dim=0,
    )

    grid = make_grid(vis, nrow=nrow, padding=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out_path)
    print(f"[sample] Saved {out_path}")


@torch.no_grad()
def evaluate_one_run(
    *,
    args,
    net: EDMPrecond,
    transform: CheckpointInputTransform,
    eval_ds: Dataset,
    model_label: str,
    ckpt_path: str,
    ckpt: dict,
    used_ema: bool,
    backbone_meta: dict,
    sigma: float,
    sigma_index: int,
    params: int,
    run_outdir: Path,
    device: torch.device,
) -> dict:
    run_outdir.mkdir(parents=True, exist_ok=True)

    run_args = vars(args).copy()
    run_args["ckpt_current"] = ckpt_path
    run_args["model_label"] = model_label
    run_args["sigma_current"] = float(sigma)
    run_args["outdir_current"] = str(run_outdir)
    with open(run_outdir / "args.json", "w", encoding="utf-8") as f:
        json.dump(run_args, f, indent=2)

    eval_loader = build_eval_loader(args, eval_ds)

    # Local generator makes corruption independent of model initialization/loading.
    # Every checkpoint gets the same eps for a given sigma_index.
    noise_seed = int(args.noise_seed) + 1_000_003 * int(sigma_index)
    noise_gen = torch.Generator(device=device)
    noise_gen.manual_seed(noise_seed)

    total_sqerr_raw = 0.0
    total_sqerr_main = 0.0
    total_numel = 0
    total_images = 0
    first_grid_saved = False

    t0 = time.time()

    for batch_idx, batch in enumerate(eval_loader):
        clean_orig = batch[0] if isinstance(batch, (tuple, list)) else batch
        clean_orig = clean_orig.to(device=device, dtype=torch.float32, non_blocking=True)

        clean_model = transform.forward(clean_orig)

        noise = torch.randn(
            clean_model.shape,
            device=device,
            dtype=clean_model.dtype,
            generator=noise_gen,
        )
        noisy_model = clean_model + float(sigma) * noise

        pred_model = denoise_edm(
            net=net,
            noisy=noisy_model,
            sigma=float(sigma),
            precision=args.precision,
            device=device,
        )

        pred_raw = transform.inverse(pred_model)
        noisy_orig = transform.inverse(noisy_model)

        pred_main = pred_raw.clamp(-1.0, 1.0) if args.clip_output else pred_raw

        total_sqerr_raw += ((pred_raw - clean_orig) ** 2).sum().item()
        total_sqerr_main += ((pred_main - clean_orig) ** 2).sum().item()
        total_numel += clean_orig.numel()
        total_images += clean_orig.shape[0]

        if (not first_grid_saved) or (
            args.save_grid_every > 0 and batch_idx % args.save_grid_every == 0
        ):
            save_eval_grid(
                clean=clean_orig.detach().cpu(),
                noisy=noisy_orig.detach().cpu(),
                pred=pred_main.detach().cpu(),
                out_path=run_outdir / f"denoise_grid_batch{batch_idx:05d}.png",
                nrow=min(args.grid_n, clean_orig.shape[0]),
            )
            first_grid_saved = True

        if (batch_idx + 1) % args.log_every == 0:
            mse_so_far = total_sqerr_main / max(total_numel, 1)
            psnr_so_far = mse_to_psnr(mse_so_far, data_range=2.0)
            elapsed = time.time() - t0
            rate = total_images / max(elapsed, 1e-9)
            print(
                f"[eval:{model_label}] sigma={sigma:.4g} batch={batch_idx + 1:05d} "
                f"images={total_images} mse={mse_so_far:.6f} "
                f"psnr={psnr_so_far:.2f}dB imgs/s={rate:.2f}"
            )

    mse = total_sqerr_main / max(total_numel, 1)
    raw_mse = total_sqerr_raw / max(total_numel, 1)
    elapsed = time.time() - t0

    metrics = {
        "ckpt": ckpt_path,
        "checkpoint_step": ckpt.get("step", None),
        "used_ema": bool(used_ema),
        "arch": str(backbone_meta.get("arch", "unknown")),
        "label": str(model_label),
        "sigma": float(sigma),
        "eval_data_root": args.eval_data_root,
        "dataset_name": args.dataset_name,
        "split_name": args.split_name,
        "outdir": str(run_outdir),
        "num_eval_images": int(total_images),
        "num_eval_elements": int(total_numel),
        "mse": float(mse),
        "psnr": float(mse_to_psnr(mse, data_range=2.0)),
        "mse_raw": float(raw_mse),
        "psnr_raw": float(mse_to_psnr(raw_mse, data_range=2.0)),
        "clip_output": bool(args.clip_output),
        "metrics_use_clipped_prediction": bool(args.clip_output),
        "precision": args.precision,
        "noise_seed": int(noise_seed),
        "params": int(params),
        "elapsed_sec": float(elapsed),
        "imgs_per_sec": float(total_images / max(elapsed, 1e-9)),
        "has_shuffle": bool(transform.has_shuffle),
        "has_channel_mix": bool(transform.has_channel_mix),
        "backbone": backbone_meta,
    }

    with open(run_outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[eval] Done.")
    print(json.dumps(metrics, indent=2))

    return metrics


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()

    # One or more checkpoints. Use --ckpt name=path for clean labels.
    p.add_argument(
        "--ckpt",
        type=str,
        action="append",
        required=True,
        help="Checkpoint spec. May be repeated. Format: NAME=PATH or PATH.",
    )

    # Data
    p.add_argument("--eval_data_root", type=str, default="./datasets/celeba32/test")
    p.add_argument("--dataset_name", type=str, default="celeba32")
    p.add_argument("--split_name", type=str, default=None)
    p.add_argument("--resolution", type=int, default=32)
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=4)

    # Dataset cache behavior, matching eval_analytic.py.
    p.add_argument(
        "--folder_tensor_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--eval_folder_tensor_cache_path", type=str, default=None)
    p.add_argument("--rebuild_folder_tensor_cache", action="store_true")

    # Evaluation
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("--eval_batch", type=int, default=64)
    p.add_argument("--max_eval", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--noise_seed",
        type=int,
        default=12345,
        help="Base seed for local Gaussian corruptions. Same sigma/checkpoint => same noisy images.",
    )
    p.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    p.add_argument(
        "--clip_output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clamp denoised predictions to [-1,1] before main mse/psnr and grids.",
    )

    # We use EMA by default.
    p.add_argument(
        "--use_ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use checkpoint EMA weights if present.",
    )

    # Output, matching eval_analytic's one-folder-per-run style.
    p.add_argument(
        "--outdir_root",
        type=str,
        default="./evals",
        help="Root under which folders like celeba32_sigma05_unet are created.",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Backward-compatible alias for --outdir_root.",
    )
    p.add_argument("--grid_n", type=int, default=8)
    p.add_argument("--save_grid_every", type=int, default=10, help="0 means only save the first grid.")
    p.add_argument("--log_every", type=int, default=10)

    args = p.parse_args()

    if args.outdir is not None:
        args.outdir_root = args.outdir
    if args.split_name is None:
        args.split_name = infer_split_name(args.eval_data_root)

    seed_everything(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.outdir_root).mkdir(parents=True, exist_ok=True)

    eval_ds = build_folder_dataset(
        root=args.eval_data_root,
        resolution=args.resolution,
        channels=args.channels,
        hflip=False,
        folder_tensor_cache=args.folder_tensor_cache,
        folder_tensor_cache_path=args.eval_folder_tensor_cache_path,
        rebuild_folder_tensor_cache=args.rebuild_folder_tensor_cache,
    )

    print(f"[data] eval size={len(eval_ds)} root={args.eval_data_root}")
    print(
        f"[eval] dataset={args.dataset_name} split={args.split_name} "
        f"sigmas={args.sigmas} precision={args.precision} device={device}"
    )

    for ckpt_spec in args.ckpt:
        label, ckpt_path = parse_ckpt_spec(ckpt_spec)

        print("=" * 80)
        print(f"[model] label={label} ckpt={ckpt_path}")

        net, backbone_meta, ckpt, used_ema, transform = load_edm_model_from_checkpoint(
            ckpt_path=ckpt_path,
            device=device,
            use_ema=args.use_ema,
            expected_resolution=args.resolution,
            expected_channels=args.channels,
        )

        params = count_parameters(net)

        for sigma_index, sigma in enumerate(args.sigmas):
            run_outdir = make_run_outdir(
                outdir_root=args.outdir_root,
                dataset_name=args.dataset_name,
                sigma=float(sigma),
                split_name=args.split_name,
                model_label=label,
            )
            print(f"[eval] output directory: {run_outdir}")

            evaluate_one_run(
                args=args,
                net=net,
                transform=transform,
                eval_ds=eval_ds,
                model_label=label,
                ckpt_path=ckpt_path,
                ckpt=ckpt,
                used_ema=used_ema,
                backbone_meta=backbone_meta,
                sigma=float(sigma),
                sigma_index=sigma_index,
                params=int(params),
                run_outdir=run_outdir,
                device=device,
            )

        # Free memory before next checkpoint.
        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("[eval] all done.")


if __name__ == "__main__":
    main()
