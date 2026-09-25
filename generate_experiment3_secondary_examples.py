#!/usr/bin/env python3
"""Generate matched-latent examples for the all eight cube states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from edm import edm_sampler, geometric_heun_sampler, vp_heun_sampler
from eval_model_comparison import load_model


CODES = ("000", "100", "010", "110", "001", "101", "011", "111")
TITLES = ("Source", "+QK", "+VO", "+QK+VO", "+MLP", "+QK+MLP", "+VO+MLP", "Full adapted")


@torch.no_grad()
def sample(checkpoint_path: Path, latents: torch.Tensor, steps: int) -> torch.Tensor:
    device = latents.device
    model, checkpoint, _, _, _, transform = load_model(checkpoint_path, device, True)
    error = (transform.inverse(transform.forward(latents)) - latents).abs().max().item()
    if error > 1e-5:
        raise RuntimeError(f'Transform inverse round-trip failed: {error}')
    prediction_type = checkpoint.get(
        "prediction_type", checkpoint.get("args", {}).get("prediction_type", "edm_preconditioned")
    )
    noise_process = checkpoint.get(
        "noise_process", checkpoint.get("args", {}).get("noise_process", "ve")
    )
    with torch.autocast(device_type="cuda", enabled=False):
        if prediction_type == "x" and noise_process == "vp":
            images = vp_heun_sampler(
                model, latents, num_steps=steps,
                denominator_floor=float(checkpoint.get("vp_denominator_floor", 0.05)),
            )
        elif prediction_type == "x":
            images = geometric_heun_sampler(model, latents, num_steps=steps)
        else:
            images = edm_sampler(model, latents, num_steps=steps, solver="euler")
    images = transform.inverse(images.float())
    if not torch.isfinite(images).all():
        raise RuntimeError(f'Nonfinite generated images: {checkpoint_path}')
    images = images.clamp(-1, 1)
    del model
    torch.cuda.empty_cache()
    return ((images + 1) / 2).cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cube_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--images", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--qk_pairs", action="store_true", help="Place QK-off/on pairs in adjacent columns")
    args = parser.parse_args()
    codes = ("000", "100", "010", "110", "001", "101", "011", "111") if args.qk_pairs else CODES
    titles = tuple(dict(zip(CODES, TITLES))[code] for code in codes)

    assert torch.cuda.is_available(), "GPU unavailable; refusing CPU fallback"
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(args.seed)
    latents = torch.randn(
        args.images, 3, args.resolution, args.resolution, generator=generator
    ).to(device)
    columns = []
    for code in codes:
        path = args.cube_dir / f"ckpt_{code}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        columns.append(sample(path, latents, args.steps).numpy())

    # Rows share latent noise; columns differ only in transplanted parameter groups.
    fig, axes = plt.subplots(args.images, len(CODES), figsize=(12, 1.48 * args.images))
    if args.images == 1:
        axes = axes[None, :]
    for row in range(args.images):
        for col, (title, batch) in enumerate(zip(titles, columns)):
            axes[row, col].imshow(np.transpose(batch[row], (1, 2, 0)))
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].text(-.06, .5, str(row+1), transform=axes[row, col].transAxes, ha="right", va="center")
            if row == 0:
                axes[row, col].set_title(title, fontsize=11, pad=5)
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.96, wspace=0.025, hspace=0.025)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    np.savez_compressed(
        args.output.with_suffix(".npz"), codes=np.asarray(codes), titles=np.asarray(titles),
        images=np.stack(columns), latent_seed=args.seed, sampler_steps=args.steps, latents=latents.cpu().numpy(),
        precision="fp32",
    )
    args.output.with_suffix(".json").write_text(json.dumps(dict(
        cube_dir=str(args.cube_dir), codes=codes, latent_seed=args.seed, images=args.images,
        display_space="original", inverse_before_clipping=True, inverse_roundtrip_checked=True,
        sampler_steps=args.steps, precision="fp32", checkpoint_sha256={
            code: hashlib.sha256((args.cube_dir / f"ckpt_{code}.pt").read_bytes()).hexdigest()
            for code in codes}), indent=2)+"\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
