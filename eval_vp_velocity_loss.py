"""Held-out evaluation of the exact VP velocity training objective."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval_denoising_paired import interval
from eval_model_comparison import load_model, make_loader


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp32")
    p.add_argument("--eval_seed", type=int, default=271828)
    p.add_argument("--resolution", type=int, default=32)
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--folder_tensor_cache", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval_folder_tensor_cache_path", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.outdir) / args.label
    root.mkdir(parents=True, exist_ok=True)
    output = root / "summary.json"
    if output.is_file():
        print(f"Already complete: {output}")
        return

    model, ckpt, _, _, used_ema, transform = load_model(Path(args.checkpoint), device, True)
    saved = ckpt.get("args", {})
    prediction_type = ckpt.get("prediction_type", saved.get("prediction_type"))
    noise_process = ckpt.get("noise_process", saved.get("noise_process"))
    if prediction_type != "x" or noise_process != "vp":
        raise ValueError(f"VP velocity loss requires x/vp checkpoint, got {prediction_type}/{noise_process}")
    logit_mean = float(saved.get("vp_logit_mean", -0.8))
    logit_std = float(saved.get("vp_logit_std", 0.8))
    denominator_floor = float(saved.get("vp_denominator_floor", 0.05))

    loader_args = argparse.Namespace(**vars(args))
    loader_args.eval_data_root = args.data_root
    loader_args.eval_batch = args.batch
    loader = make_loader(loader_args, 0)
    generator = torch.Generator(device=device).manual_seed(args.eval_seed)
    losses = []
    time_sum = 0.0
    time_count = 0
    for batch_index, batch in enumerate(loader):
        clean = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device).float()
        clean_model = transform.forward(clean)
        batch_size = clean_model.shape[0]
        logits = torch.randn((batch_size,), generator=generator, device=device, dtype=torch.float32)
        times = torch.sigmoid(logit_mean + logit_std * logits)
        shape = (batch_size,) + (1,) * (clean_model.ndim - 1)
        times_broadcast = times.view(shape)
        noise = torch.randn(clean_model.shape, generator=generator, device=device, dtype=torch.float32)
        noisy = times_broadcast * clean_model + (1.0 - times_broadcast) * noise
        use_amp = device.type == "cuda" and args.precision != "fp32"
        amp_dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            prediction = model(noisy, times).float()
        denominator = (1.0 - times_broadcast).clamp_min(denominator_floor)
        predicted_velocity = (prediction - noisy) / denominator
        target_velocity = (clean_model - noisy) / denominator
        dims = tuple(range(1, prediction.ndim))
        losses.append((predicted_velocity - target_velocity).square().mean(dim=dims).cpu().numpy())
        time_sum += float(times.sum().cpu())
        time_count += batch_size
        if (batch_index + 1) % 100 == 0:
            print(f"[velocity] images={(batch_index + 1) * args.batch}", flush=True)

    values = np.concatenate(losses).astype(np.float64)
    result = {
        "label": args.label,
        "checkpoint": args.checkpoint,
        "used_ema": used_ema,
        "num_images": int(len(values)),
        "eval_seed": args.eval_seed,
        "draws_per_image": 1,
        "vp_logit_mean": logit_mean,
        "vp_logit_std": logit_std,
        "vp_denominator_floor": denominator_floor,
        "mean_sampled_time": time_sum / time_count,
        "vp_velocity_loss": float(values.mean()),
        "vp_velocity_loss_ci95": interval(values, args.eval_seed + 17),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
