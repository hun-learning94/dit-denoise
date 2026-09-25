#!/usr/bin/env python3
"""Run summary models at every denoiser level of an EDM sampling schedule."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np


def edm_sigmas(num_steps: int, sigma_min: float, sigma_max: float, rho: float):
    indices = np.arange(num_steps, dtype=np.float64)
    return (
        sigma_max ** (1 / rho)
        + indices / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", default="evals/celeba32_eval_summary.csv")
    parser.add_argument("--outdir", default="evals/sampler_schedule_denoising")
    parser.add_argument("--python", default="python")
    parser.add_argument("--data_root", default="./datasets/celeba32/test")
    parser.add_argument("--num_steps", type=int, default=18)
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument(
        "--max_eval",
        type=int,
        default=5000,
        help="Number of CelebA32 test images used for denoising RMSE (default: 5000).",
    )
    parser.add_argument("--eval_batch", type=int, default=32)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--noise_seed", type=int, default=12345)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.num_steps < 2:
        parser.error("--num_steps must be at least 2")

    with Path(args.summary_csv).open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    model_specs = []
    for row in summary_rows:
        label, step = row["label"], int(row["step"])
        candidates = list((Path("runs") / label / "checkpoints").glob("ckpt_step*.pt"))
        matches = [path for path in candidates if int(path.stem.removeprefix("ckpt_step").removesuffix("_final")) == step]
        if not matches:
            raise FileNotFoundError(f"no step-{step} checkpoint for {label}")
        model_specs.append(max(matches, key=lambda path: (path.stem.endswith("_final"), path.name)))

    sigmas = edm_sigmas(args.num_steps, args.sigma_min, args.sigma_max, args.rho)
    command = [
        args.python, "eval_model_comparison.py",
        "--eval_data_root", args.data_root,
        "--resolution", "32", "--channels", "3",
        "--sigmas", *[f"{sigma:.17g}" for sigma in sigmas],
        "--max_denoise", str(args.max_eval),
        "--eval_batch", str(args.eval_batch),
        "--precision", args.precision, "--device", args.device,
        "--noise_seed", str(args.noise_seed),
        "--sampler_steps", str(args.num_steps),
        "--sigma_min", str(args.sigma_min), "--sigma_max", str(args.sigma_max),
        "--rho", str(args.rho), "--use_ema", "--skip_fid",
        "--outdir", args.outdir,
        "--summary_csv", str(Path(args.outdir) / "raw_summary.csv"),
        "--summary_md", str(Path(args.outdir) / "raw_summary.md"),
    ]
    if args.resume:
        command.append("--resume")
    for checkpoint in model_specs:
        command.extend(["--model", str(checkpoint)])
    print("Schedule sigmas:", ", ".join(f"{sigma:.8g}" for sigma in sigmas), flush=True)
    subprocess.run(command, check=True)
    metrics_path = Path(args.outdir) / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    completed = {row["label"] for row in metrics.get("results", [])}
    requested = {path.parent.parent.name for path in model_specs}
    missing = sorted(requested - completed)
    if missing:
        raise RuntimeError(
            f"evaluation returned before completing {len(missing)} requested models; "
            f"rerun with --resume. Missing: {missing}"
        )
    subprocess.run([args.python, "rebuild_celeba32_eval_summary.py"], check=True)


if __name__ == "__main__":
    main()
