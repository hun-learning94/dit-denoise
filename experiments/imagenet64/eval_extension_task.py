import sys
"""Evaluate the newly available ImageNet64 H200 extension checkpoints."""
import os
from pathlib import Path
import subprocess
import torch


ENDPOINTS = {
    500: 2502280,
    600: 3002736,
}
METRICS = ("vp", "denoising", "fid")


def main():
    task = int(os.environ["SLURM_ARRAY_TASK_ID"])
    endpoint_index, metric_index = divmod(task, len(METRICS))
    # Default preserves task IDs in the already-submitted September 16 array.
    epochs = tuple(int(e) for e in os.environ.get('IMAGENET_EPOCH_ORDER', '500,600').split(','))
    epoch = epochs[endpoint_index]
    metric = METRICS[metric_index]
    checkpoint = Path(
        "runs/imagenet64_20m/seed0/h200_b256/checkpoints"
    ) / f"ckpt_step{ENDPOINTS[epoch]}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires CUDA")

    label = f"h200_epoch{epoch:03d}"
    output_dir = Path("evals/imagenet64_20m") / {
        "vp": "vp_velocity_loss",
        "denoising": "denoising",
        "fid": "fid50k",
    }[metric]
    result = output_dir / label / ("result.json" if metric == "fid" else "summary.json")
    if result.is_file():
        print("Already complete:", result, flush=True)
        return

    command = [
        sys.executable, "-u",
        "-m", "experiments.imagenet64.evaluate", metric,
        "--checkpoint", str(checkpoint), "--label", label,
        "--data_root", "datasets/imagenet64/prepared",
        "--outdir", str(output_dir), "--batch", "32", "--num_workers", "0",
        "--precision", "fp32", "--resolution", "64", "--channels", "3",
    ]
    if metric == "fid":
        command += ["model"]
        command += [
            "--num_images", "50000", "--shard_size", "500", "--sampler_steps", "50",
            "--real_stats", "evals/imagenet64_20m/real_stats/validation_stats.npz",
        ]
    elif metric == "denoising":
        command += ["--time_min", ".05", "--time_max", "4", "--auc_steps", "32"]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
