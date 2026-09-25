"""Compare DPD3--DPD6, grouped DPD6, and EDM baselines.

Checkpoint specs accept ``LABEL=PATH`` or ``LABEL=RUN_DIR``. A DPD6 run
directory is pinned to step 200,000; other run directories resolve to their
numerically latest checkpoint, preferring ``*_final.pt`` at the same step.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader, Subset
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.utils import make_grid, save_image

from edm import EDMPrecond, VPXPred, XPredPrecond, edm_sampler, geometric_heun_sampler
from eval_edm_denoisers import CheckpointInputTransform, build_folder_dataset
from model_dpd3 import DPD3Config, DPD3EDMBackbone, DeepPatchwiseDenoiser3
from model_dpd4 import DPD4Backbone, DPD4Config
from model_dpd5 import DPD5Backbone, DPD5Config
from model_dpd6 import DPD6Backbone, DPD6Config
from model_grouped_dpd6 import GroupedDPD6Backbone, GroupedDPD6Config
from models import DiTBackbone, DiTConfig, ResNetBackbone, ResNetConfig, UNetBackbone, UNetConfig


STEP_RE = re.compile(r"ckpt_step0*(\d+)(?:(_final))?\.pt$")


def latest_checkpoint(path: str) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    ckpt_dir = p / "checkpoints" if (p / "checkpoints").is_dir() else p
    run_dir = ckpt_dir.parent if ckpt_dir.name == "checkpoints" else ckpt_dir
    if run_dir.name.startswith("celeba32_dpd6_"):
        pinned = ckpt_dir / "ckpt_step0200000.pt"
        if not pinned.is_file():
            raise FileNotFoundError(f"missing required step-200000 DPD6 checkpoint: {pinned}")
        return pinned
    candidates = []
    for q in ckpt_dir.glob("ckpt_step*.pt"):
        match = STEP_RE.search(q.name)
        if match:
            candidates.append((int(match.group(1)), bool(match.group(2)), q))
    if not candidates:
        raise FileNotFoundError(f"no ckpt_step*.pt checkpoints under {ckpt_dir}")
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def parse_spec(spec: str) -> Tuple[str, Path]:
    # A caller-provided prefix is accepted for backward compatibility, but the
    # displayed label always comes from the actual run directory containing the
    # resolved checkpoint.
    path = spec.split("=", 1)[1] if "=" in spec else spec
    checkpoint = latest_checkpoint(path)
    if checkpoint.parent.name == "checkpoints":
        label = checkpoint.parent.parent.name
    else:
        label = checkpoint.parent.name
    return label, checkpoint


def config_from_dict(cls, raw: dict):
    valid = {f.name for f in fields(cls)}
    values = {k: v for k, v in raw.items() if k in valid}
    if cls is DPD4Config:
        values["print_params"] = False
        return DPD4Config.from_checkpoint(values)
    if cls is DPD3Config and "hidden_dim" not in values:
        values["hidden_dim"] = None
    values["print_params"] = False
    return cls(**values)


def apply_ema(module: nn.Module, state: Dict[str, torch.Tensor]) -> int:
    target = module.state_dict()
    copied = 0
    for raw_key, value in state.items():
        key = raw_key.removeprefix("_orig_mod.")
        alternatives = (
            key,
            key.removeprefix("model."),
            f"model.{key}",
            f"model.denoiser.{key}",  # raw DPD3 inside DPD3EDMBackbone + EDMPrecond
        )
        for candidate in alternatives:
            if candidate in target and target[candidate].shape == value.shape:
                target[candidate].copy_(value.to(dtype=target[candidate].dtype))
                copied += 1
                break
    return copied


class DirectDenoiserAdapter(nn.Module):
    """Give direct-regression DPD3 the minimal EDM sampler interface."""

    sigma_min = 0.0
    sigma_max = float("inf")

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def forward(self, x, sigma, class_labels=None):
        del class_labels
        if not torch.is_tensor(sigma):
            sigma = torch.as_tensor(sigma, device=x.device)
        sigma = sigma.reshape(-1)
        if sigma.numel() == 1:
            # DPD3's direct path intentionally conditions on one shared sigma.
            return self.model(x.float(), sigma=float(sigma.item()))
        if not torch.allclose(sigma, sigma[:1].expand_as(sigma)):
            raise ValueError("direct DPD3 requires an equal sigma within each batch")
        return self.model(x.float(), sigma=float(sigma[0].item()))


def build_baseline(ckpt: dict) -> nn.Module:
    meta = ckpt.get("backbone")
    if meta is None and ckpt.get("dit_cfg") is not None:
        meta = {"arch": "dit", "cfg": ckpt["dit_cfg"]}
    if meta is None:
        raise KeyError("baseline checkpoint lacks backbone metadata")
    arch = meta["arch"].lower()
    classes = {
        "dit": (DiTConfig, DiTBackbone),
        "resnet": (ResNetConfig, ResNetBackbone),
        "unet": (UNetConfig, UNetBackbone),
    }
    cfg_cls, model_cls = classes[arch]
    cfg = config_from_dict(cfg_cls, meta["cfg"])
    backbone = model_cls(cfg)
    prediction_type = ckpt.get(
        "prediction_type", ckpt.get("args", {}).get("prediction_type", "edm_preconditioned")
    )
    if prediction_type == "x":
        noise_process = ckpt.get("noise_process", ckpt.get("args", {}).get("noise_process", "ve"))
        if noise_process == "vp":
            return VPXPred(backbone, cfg.img_resolution, cfg.img_channels)
        return XPredPrecond(
            backbone, cfg.img_resolution, cfg.img_channels,
            sigma_data=float(ckpt.get("sigma_data", 0.5)),
            t_min=float(ckpt.get("time_min", 0.05)),
            t_max=float(ckpt.get("time_max", 4.0)),
        )
    return EDMPrecond(backbone, cfg.img_resolution, cfg.img_channels)


def load_model(path: Path, device: torch.device, use_ema: bool):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model_type = ckpt.get("model_type")
    is_edm = True
    if model_type == "dpd3":
        cfg = config_from_dict(DPD3Config, ckpt["model_cfg"])
        raw = DeepPatchwiseDenoiser3(cfg)
        raw.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}, strict=True)
        is_edm = bool(ckpt.get("args", {}).get("edm", False))
        model = EDMPrecond(
            DPD3EDMBackbone(raw), cfg.img_resolution, cfg.img_channels,
            sigma_data=float(ckpt.get("args", {}).get("sigma_data", 0.5)),
        ) if is_edm else DirectDenoiserAdapter(raw)
    elif model_type in {"dpd4", "dpd5", "dpd6", "grouped_dpd6"}:
        classes = {
            "dpd4": (DPD4Config, DPD4Backbone),
            "dpd5": (DPD5Config, DPD5Backbone),
            "dpd6": (DPD6Config, DPD6Backbone),
            "grouped_dpd6": (GroupedDPD6Config, GroupedDPD6Backbone),
        }
        cfg_cls, backbone_cls = classes[model_type]
        cfg = config_from_dict(cfg_cls, ckpt["model_cfg"])
        # The fused grouped-GELU Triton kernel is a training optimization.  It
        # has produced architecture-dependent compilation failures during
        # checkpoint evaluation on several DCC GPU classes.  The ordinary
        # einsum/GELU path computes the same forward function and is the safe
        # inference implementation for portable, reproducible metrics.
        if model_type == "grouped_dpd6":
            cfg.fused_expansion = False
        backbone = backbone_cls(cfg)
        prediction_type = ckpt.get("prediction_type", ckpt.get("args", {}).get("prediction_type", "edm_preconditioned"))
        noise_process = ckpt.get("noise_process", ckpt.get("args", {}).get("noise_process", "ve"))
        if prediction_type == "x" and noise_process == "vp":
            model = VPXPred(backbone, cfg.img_resolution, cfg.img_channels)
        elif prediction_type == "x":
            model = XPredPrecond(
                backbone, cfg.img_resolution, cfg.img_channels,
                sigma_data=float(ckpt.get("sigma_data", 0.583087513)),
                t_min=float(ckpt.get("time_min", 0.05)),
                t_max=float(ckpt.get("time_max", 4.0)),
            )
        else:
            model = EDMPrecond(backbone, cfg.img_resolution, cfg.img_channels)
        is_edm = prediction_type != "x"
        model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ckpt["net"].items()}, strict=True)
    else:
        model_type = str(ckpt.get("backbone", {}).get("arch", "dit"))
        model = build_baseline(ckpt)
        model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ckpt["net"].items()}, strict=True)
        is_edm = ckpt.get(
            "prediction_type", ckpt.get("args", {}).get("prediction_type", "edm_preconditioned")
        ) != "x"
    used_ema = False
    if use_ema and ckpt.get("ema") is not None:
        copied = apply_ema(model, ckpt["ema"])
        used_ema = copied > 0
        print(f"[load] EMA tensors copied={copied}")
    model = model.to(device).eval()
    if model_type == "grouped_dpd6":
        backbone.prepare_projected_cache()
    transform = CheckpointInputTransform(ckpt, device)
    return model, ckpt, model_type, is_edm, used_ema, transform


def model_call(model, noisy, sigma: float, precision: str):
    enabled = noisy.device.type == "cuda" and precision != "fp32"
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    sigma_batch = torch.full((noisy.shape[0],), sigma, device=noisy.device)
    with torch.autocast(device_type=noisy.device.type, dtype=dtype, enabled=enabled):
        return model(noisy, sigma_batch).float()


@torch.no_grad()
def denoising_rmse(model, transform, loader, sigmas, device, precision, noise_seed):
    rows = []
    for sigma_index, sigma in enumerate(sigmas):
        generator = torch.Generator(device=device).manual_seed(noise_seed + 1_000_003 * sigma_index)
        squared_error = squared_error_raw = 0.0
        elements = images = 0
        for batch in loader:
            clean = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device).float()
            clean_model = transform.forward(clean)
            noise = torch.randn(clean_model.shape, generator=generator, device=device)
            noisy = clean_model + sigma * noise
            pred_raw = transform.inverse(model_call(model, noisy, sigma, precision))
            pred = pred_raw.clamp(-1, 1)
            squared_error += (pred - clean).square().sum().item()
            squared_error_raw += (pred_raw - clean).square().sum().item()
            elements += clean.numel()
            images += clean.shape[0]
        mse = squared_error / elements
        mse_raw = squared_error_raw / elements
        rows.append({"sigma": sigma, "rmse": math.sqrt(mse), "mse": mse,
                     "rmse_raw": math.sqrt(mse_raw), "mse_raw": mse_raw, "images": images})
    return rows


class InceptionFeatures(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.net = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).to(device).eval()
        self.net.fc = nn.Identity()
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images):
        images = images.repeat(1, 3, 1, 1) if images.shape[1] == 1 else images
        images = nn.functional.interpolate(images, (299, 299), mode="bilinear", align_corners=False)
        mean = images.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = images.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        return self.net((images - mean) / std)


def stats(features: np.ndarray):
    return features.mean(0), np.cov(features, rowvar=False)


def fid_score(real, generated):
    mu_r, cov_r = stats(real)
    mu_g, cov_g = stats(generated)
    covmean = sqrtm(cov_r @ cov_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_r - mu_g
    return max(float(diff @ diff + np.trace(cov_r + cov_g - 2 * covmean)), 0.0)


@torch.no_grad()
def real_features(feature_net, loader, device, limit):
    output, seen = [], 0
    for batch in loader:
        images = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device).float()
        images = images[: max(0, limit - seen)]
        output.append(feature_net(((images + 1) / 2).clamp(0, 1)).cpu())
        seen += images.shape[0]
        if seen >= limit:
            break
    return torch.cat(output).numpy().astype(np.float64)


@torch.no_grad()
def sample_features(model, transform, feature_net, latents, args, device):
    features, preview = [], []
    for start in range(0, len(latents), args.sample_batch):
        z = latents[start:start + args.sample_batch].to(device)
        generated = edm_sampler(
            model, z, num_steps=args.sampler_steps, sigma_min=args.sigma_min,
            sigma_max=args.sigma_max, rho=args.rho, solver="euler",
        ).float()
        generated = transform.inverse(generated).clamp(-1, 1)
        images01 = (generated + 1) / 2
        features.append(feature_net(images01).cpu())
        if sum(x.shape[0] for x in preview) < args.grid_n:
            preview.append(images01.cpu())
    return torch.cat(features).numpy().astype(np.float64), torch.cat(preview)[:args.grid_n]


def make_loader(args, max_images):
    dataset = build_folder_dataset(
        root=args.eval_data_root, resolution=args.resolution, channels=args.channels,
        hflip=False, folder_tensor_cache=args.folder_tensor_cache,
        folder_tensor_cache_path=args.eval_folder_tensor_cache_path,
        rebuild_folder_tensor_cache=False,
    )
    if max_images and max_images < len(dataset):
        dataset = Subset(dataset, range(max_images))
    return DataLoader(dataset, batch_size=args.eval_batch, shuffle=False, num_workers=args.num_workers,
                      pin_memory=True, persistent_workers=args.num_workers > 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="CHECKPOINT_OR_RUN_DIR (an optional ignored LABEL= prefix is accepted)",
    )
    parser.add_argument("--eval_data_root", default="./datasets/celeba32/test")
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.25, 0.5, 1, 2, 4])
    parser.add_argument("--max_denoise", type=int, default=1024)
    parser.add_argument("--fid_n", type=int, default=2000)
    parser.add_argument("--fid_real_n", type=int, default=10000)
    parser.add_argument("--eval_batch", type=int, default=32)
    parser.add_argument("--sample_batch", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sampler_steps", type=int, default=18)
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--noise_seed", type=int, default=12345)
    parser.add_argument("--sample_seed", type=int, default=54321)
    parser.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--folder_tensor_cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval_folder_tensor_cache_path", default=None)
    parser.add_argument("--outdir", default="evals/model_comparison_euler")
    parser.add_argument("--summary_csv", default="evals/celeba32_eval_summary.csv")
    parser.add_argument("--summary_md", default="evals/celeba32_eval_summary.md")
    parser.add_argument(
        "--train_num_images", type=int, default=162771,
        help="Training-set size used to compute effective training epochs.",
    )
    parser.add_argument("--grid_n", type=int, default=64)
    parser.add_argument(
        "--skip_fid", action="store_true",
        help="Evaluate fixed-noise denoising only; do not initialize Inception or sample images.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Keep completed model rows from OUTDIR/metrics.json and skip those labels.",
    )
    args = parser.parse_args()
    if args.sampler_steps < 2 or not (0 < args.sigma_min < args.sigma_max):
        parser.error("require sampler_steps >= 2 and 0 < sigma_min < sigma_max")
    device = torch.device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    denoise_loader = make_loader(args, args.max_denoise)
    feature_net = real = latents = None
    if not args.skip_fid:
        real_loader = make_loader(args, args.fid_real_n)
        feature_net = InceptionFeatures(device)
        real = real_features(feature_net, real_loader, device, args.fid_real_n)
        generator = torch.Generator().manual_seed(args.sample_seed)
        latents = torch.randn(args.fid_n, args.channels, args.resolution, args.resolution, generator=generator)
    metrics_path = outdir / "metrics.json"
    results = []
    if args.resume and metrics_path.exists():
        results = json.loads(metrics_path.read_text(encoding="utf-8")).get("results", [])
        print(f"[resume] loaded {len(results)} completed model rows from {metrics_path}")
    completed_labels = {row["label"] for row in results}
    for spec in args.model:
        label, path = parse_spec(spec)
        if label in completed_labels:
            print(f"[resume] skipping completed model {label}")
            continue
        print(f"[model] {label}: {path}")
        started = time.time()
        model, ckpt, model_type, is_edm, used_ema, transform = load_model(path, device, args.use_ema)
        rmse = denoising_rmse(model, transform, denoise_loader, args.sigmas, device, args.precision, args.noise_seed)
        fid = None
        if is_edm and not args.skip_fid:
            generated, preview = sample_features(model, transform, feature_net, latents, args, device)
            fid = fid_score(real, generated)
            save_image(
                make_grid(preview, nrow=max(1, int(math.sqrt(args.grid_n)))),
                outdir / f"samples_{label}.png",
            )
        elif not is_edm:
            print(f"[fid] skipped {label}: checkpoint was trained without EDM preconditioning")
        step = ckpt.get("step")
        checkpoint_args = ckpt.get("args", {})
        train_batch = checkpoint_args.get("batch") if isinstance(checkpoint_args, dict) else None
        training_epoch = (
            float(step) * int(train_batch) / int(args.train_num_images)
            if step is not None and train_batch is not None and args.train_num_images > 0
            else None
        )
        row = {"label": label, "checkpoint": str(path), "step": step,
               "train_batch": train_batch, "training_epoch": training_epoch,
               "model_type": model_type,
               "edm_preconditioned": is_edm, "used_ema": used_ema, "rmse": rmse, "fid": fid,
               "sampler": "euler", "sampler_steps": args.sampler_steps, "sigma_min": args.sigma_min,
               "sigma_max": args.sigma_max, "rho": args.rho, "fid_n": args.fid_n,
               "elapsed_sec": time.time() - started}
        results.append(row)
        metrics_path.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary_csv = Path(args.summary_csv)
    summary_md = Path(args.summary_md)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    columns = ["label", "step", "training_epoch", *[f"rmse_sigma_{s:g}" for s in args.sigmas], "fid"]
    table_rows = [
        [row["label"], row["step"], row["training_epoch"], *[x["rmse"] for x in row["rmse"]], row["fid"]]
        for row in results
    ]
    with summary_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(table_rows)
    md_lines = [
        "# CelebA32 evaluation summary",
        "",
        (
            f"Denoising uses common Gaussian corruptions. Sampling FID uses {args.fid_n} "
            f"common initial latents and an Euler EDM schedule with {args.sampler_steps} steps, "
            f"sigma_max={args.sigma_max:g}, sigma_min={args.sigma_min:g}, rho={args.rho:g}. "
            "FID is not evaluated for models trained without EDM preconditioning. "
            f"Training epoch is optimizer step × training batch size / {args.train_num_images} "
            "CelebA32 training images."
        ),
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for values in table_rows:
        formatted = []
        for index, value in enumerate(values):
            if value is None:
                formatted.append("N/A")
            elif index == 2:
                formatted.append(f"{float(value):.2f}")
            elif index >= 3:
                formatted.append(f"{float(value):.6f}" if index < len(values) - 1 else f"{float(value):.4f}")
            else:
                formatted.append(str(value))
        md_lines.append("| " + " | ".join(formatted) + " |")
    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[summary] wrote {summary_csv}")
    print(f"[summary] wrote {summary_md}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
