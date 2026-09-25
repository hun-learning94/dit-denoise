"""Resumable FID-50k evaluation with reusable real-image statistics."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import sqrtm

from edm import edm_sampler, geometric_heun_sampler, vp_heun_sampler
from eval_model_comparison import InceptionFeatures, load_model, make_loader


def moments(chunks):
    features = np.concatenate([np.asarray(chunk) for chunk in chunks]).astype(np.float64)
    count = len(features)
    if count < 2:
        raise ValueError("at least two features are required")
    mean = features.mean(axis=0)
    covariance = np.cov(features, rowvar=False)
    return count, mean, covariance


def save_npz_atomic(path: Path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}.npz")
    np.savez(temporary, **values)
    os.replace(temporary, path)


@torch.no_grad()
def image_features(network, images):
    return network(((images + 1) / 2).clamp(0, 1)).cpu().numpy().astype(np.float32)


def compute_real(args, device):
    output = Path(args.real_stats)
    if output.is_file():
        print(f"[resume] real statistics already exist: {output}")
        return
    network = InceptionFeatures(device)
    loader_args = argparse.Namespace(**vars(args))
    loader_args.eval_data_root = args.data_root
    loader_args.eval_batch = args.batch
    loader = make_loader(loader_args, 0)
    chunks = []
    seen = 0
    for batch in loader:
        images = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device).float()
        chunks.append(image_features(network, images))
        seen += len(images)
        print(f"[real] {seen}", flush=True)
    count, mean, covariance = moments(chunks)
    save_npz_atomic(output, n=count, mu=mean, sigma=covariance)
    print(f"[real] saved n={count}: {output}")


@torch.no_grad()
def compute_model(args, device):
    real_path = Path(args.real_stats)
    if not real_path.is_file():
        raise FileNotFoundError(f"missing real statistics: {real_path}")
    label = args.label or Path(args.checkpoint).parent.parent.name
    root = Path(args.outdir) / label
    shard_dir = root / "feature_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint, _, is_edm, used_ema, transform = load_model(Path(args.checkpoint), device, True)
    prediction_type = checkpoint.get(
        "prediction_type", checkpoint.get("args", {}).get("prediction_type", "edm_preconditioned")
    )
    noise_process = checkpoint.get("noise_process", checkpoint.get("args", {}).get("noise_process", "ve"))
    network = InceptionFeatures(device)
    shard_count = math.ceil(args.num_images / args.shard_size)
    for shard_index in range(shard_count):
        start = shard_index * args.shard_size
        size = min(args.shard_size, args.num_images - start)
        path = shard_dir / f"features_{start:06d}_{start + size:06d}.npy"
        if path.is_file():
            print(f"[resume] {path.name}", flush=True)
            continue
        generator = torch.Generator().manual_seed(args.seed + shard_index)
        latent = torch.randn(size, args.channels, args.resolution, args.resolution, generator=generator)
        output = []
        for offset in range(0, size, args.batch):
            z = latent[offset:offset + args.batch].to(device)
            amp_dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype,
                enabled=device.type == "cuda" and args.precision != "fp32",
            ):
                if prediction_type == "x" and noise_process == "vp":
                    generated = vp_heun_sampler(
                        model, z, num_steps=args.sampler_steps,
                        denominator_floor=float(checkpoint.get("vp_denominator_floor", 0.05)),
                    ).float()
                elif prediction_type == "x":
                    generated = geometric_heun_sampler(
                        model, z, num_steps=args.sampler_steps,
                        t_min=args.time_min, t_max=args.time_max,
                    ).float()
                else:
                    generated = edm_sampler(
                        model, z, num_steps=args.sampler_steps, sigma_min=args.sigma_min,
                        sigma_max=args.sigma_max, rho=args.rho, solver="euler",
                    ).float()
            generated = transform.inverse(generated).clamp(-1, 1)
            output.append(image_features(network, generated))
        temporary = path.with_name(path.name + f".tmp-{os.getpid()}.npy")
        np.save(temporary, np.concatenate(output))
        os.replace(temporary, path)
        print(f"[generated] {label} {start + size}/{args.num_images}", flush=True)
    paths = sorted(shard_dir.glob("features_*.npy"))
    loaded = [np.load(path, mmap_mode="r") for path in paths]
    bad = sum(int((~np.isfinite(chunk)).sum()) for chunk in loaded)
    if bad:
        raise RuntimeError(f"found {bad} non-finite generated feature values in {shard_dir}")
    count, mean, covariance = moments(loaded)
    if count != args.num_images:
        raise RuntimeError(f"expected {args.num_images} features, found {count}")
    real = np.load(real_path)
    diff = real["mu"] - mean
    covariance_mean = sqrtm(real["sigma"] @ covariance)
    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real
    fid = max(float(diff @ diff + np.trace(real["sigma"] + covariance - 2 * covariance_mean)), 0.0)
    result = {
        "label": label, "checkpoint": args.checkpoint, "generated_images": count,
        "real_images": int(real["n"]), "fid": fid, "used_ema": used_ema,
        "sampler_steps": args.sampler_steps, "seed": args.seed,
        "prediction_type": prediction_type,
        "noise_process": noise_process if prediction_type == "x" else "edm",
        "time_min": args.time_min if prediction_type == "x" else None,
        "time_max": args.time_max if prediction_type == "x" else None,
    }
    root.mkdir(parents=True, exist_ok=True)
    result_path = root / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    # Keep the compact metric result, but remove the large transient feature
    # shards only after the result has been written successfully. Interrupted
    # jobs retain their shards and can still resume.
    shutil.rmtree(shard_dir, ignore_errors=True)
    if shard_dir.exists():
        print(f"[cleanup] warning: could not fully remove transient FID feature shards: {shard_dir}", flush=True)
    else:
        print(f"[cleanup] removed transient FID feature shards: {shard_dir}", flush=True)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["real", "model"])
    parser.add_argument("--checkpoint")
    parser.add_argument("--label", default=None)
    parser.add_argument("--data_root", default="datasets/celeba32/test")
    parser.add_argument("--real_stats", default="evals/fid50k_test/celeba32_test_stats.npz")
    parser.add_argument("--outdir", default="evals/fid50k_test")
    parser.add_argument("--num_images", type=int, default=50_000)
    parser.add_argument("--shard_size", type=int, default=500)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--sampler_steps", type=int, default=18)
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--time_min", type=float, default=0.05)
    parser.add_argument("--time_max", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=54321)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--folder_tensor_cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval_folder_tensor_cache_path", default=None)
    args = parser.parse_args()
    if args.mode == "model" and not args.checkpoint:
        parser.error("model mode requires --checkpoint")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (compute_real if args.mode == "real" else compute_model)(args, device)


if __name__ == "__main__":
    main()
