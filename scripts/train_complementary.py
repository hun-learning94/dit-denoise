#!/usr/bin/env python3
"""Matched MLP/SMLP scale training with allocation-local crops and exact RNG resume."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
PROJECT = Path(os.environ.get("PIXEL_DIT_PROJECT_ROOT", CODE))
from data import ImageFolderFlat, build_transform
from edm import VPXPred, VPVelocityXPredLoss, vp_heun_sampler
from experiments.complementary_benchmarks.celeba128_data import publish
from experiments.complementary_training.common import (
    DATASETS, EMA, build_model, restore_rng, rng_state, seed_training,
)
from experiments.complementary_training.input import EpochBatches, build_cache


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


class Training:
    def __init__(self, args, dataset, cache):
        self.args, self.dataset, self.cache = args, dataset, cache
        self.resolution = DATASETS[args.dataset]["resolution"]
        self.steps_per_epoch = len(dataset)//args.batch
        assert self.steps_per_epoch > 0
        self.target_steps = self.steps_per_epoch*args.epochs
        self.step = 0
        raw, cfg, self.model_info = build_model(args.arch, self.resolution, args.seed)
        self.raw, self.model_cfg = raw.cuda().train(), cfg
        self.canonical = VPXPred(self.raw, self.resolution, 3)
        self.compiled = torch.compile(self.raw, backend="inductor", mode="default", dynamic=False)
        self.net = VPXPred(self.compiled, self.resolution, 3)
        learning_rate = float(getattr(args, "lr", 2e-4))
        assert np.isfinite(learning_rate) and learning_rate > 0
        self.optimizer = torch.optim.AdamW(self.canonical.parameters(), lr=learning_rate,
            betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        self.ema = EMA(self.canonical)
        self.loss_fn = VPVelocityXPredLoss(-0.8, 0.8, 0.05)
        self.recipe = dict(arch=args.arch, dataset=args.dataset, seed=args.seed,
            epochs=args.epochs, batch=args.batch, hflip=args.hflip,
            dataset_count=len(dataset), dataset_identity=cache["identity"],
            model_cfg=cfg, objective="VP direct-x velocity loss", precision="bf16",
            lr=learning_rate, betas=[0.9, 0.999], eps=1e-8, weight_decay=0,
            grad_clip=1.0, ema_decay=0.999, scheduler="constant; no warmup",
            drop_last=True, steps_per_epoch=self.steps_per_epoch, total_steps=self.target_steps,
            diagnostic_every=args.diagnostic_every, sample_steps=50, sample_batch=64)
        if getattr(args, "projection_fix", None):
            self.recipe["projection_fix"] = args.projection_fix
        self.recipe_hash = hashlib.sha256(json.dumps(self.recipe, sort_keys=True).encode()).hexdigest()
        seed_training(20261019+args.seed)

    def loader(self, step=None):
        step = self.step if step is None else step
        epoch, offset = divmod(step, self.steps_per_epoch)
        return DataLoader(self.dataset,
            batch_sampler=EpochBatches(len(self.dataset), self.args.batch, self.args.seed, epoch, offset),
            num_workers=self.args.workers, pin_memory=True,
            persistent_workers=self.args.workers > 0,
            **({"prefetch_factor": 2, "multiprocessing_context": "spawn"} if self.args.workers else {}),
            generator=torch.Generator().manual_seed(4000001+self.args.seed+epoch))

    def update(self, images):
        self.net.train()
        images = images.to("cuda", non_blocking=True)
        if self.args.hflip:
            mask = torch.rand(len(images), device="cuda") < 0.5
            images = torch.where(mask[:, None, None, None], images.flip(-1), images)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = self.loss_fn(self.net, images).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.canonical.parameters(), 1.0, error_if_nonfinite=True)
        self.optimizer.step()
        self.ema.update(self.canonical)
        self.step += 1
        return loss.detach()

    def metadata(self):
        return dict(format="pixel-dit-complementary-v1", recipe=self.recipe,
            recipe_sha256=self.recipe_hash, step=self.step, args=vars(self.args),
            backbone=dict(arch="dit" if self.args.arch == "mlp" else "grouped_dpd6", cfg=self.model_cfg),
            model_type="dit" if self.args.arch == "mlp" else "grouped_dpd6",
            model_cfg=self.model_cfg, prediction_type="x", noise_process="vp",
            loss_type="velocity_mse", vp_logit_mean=-0.8, vp_logit_std=0.8,
            vp_denominator_floor=0.05, model_info=self.model_info)

    def save(self, path):
        # Canonical names exclude torch.compile's _orig_mod wrapper. Online and
        # EMA weights remain distinct; both belong to this completed update.
        atomic_torch_save(dict(self.metadata(), net=self.canonical.state_dict(),
            opt=self.optimizer.state_dict(), ema=self.ema.shadow, rng=rng_state(),
            data_position=dict(epoch=self.step//self.steps_per_epoch,
                               batch_offset=self.step % self.steps_per_epoch)), path)

    def load(self, path):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        from experiments.complementary_training.epoch_extension import validate_resume
        extended = validate_resume(saved, self.recipe, self.recipe_hash,
                                   getattr(self.args, "allow_epoch_extension", False))
        if extended:
            print(f"EXTENDING CHECKPOINT step={saved['step']} epochs=200->400; optimizer/EMA/RNG/data position preserved", flush=True)
        assert "opt" in saved and "rng" in saved, "Cannot resume online training from EMA-only weights"
        self.canonical.load_state_dict(saved["net"], strict=True)
        self.optimizer.load_state_dict(saved["opt"])
        self.ema.shadow = {k: v.to("cuda", dtype=torch.float32) for k, v in saved["ema"].items()}
        self.step = int(saved["step"])
        assert saved["data_position"] == dict(epoch=self.step//self.steps_per_epoch,
            batch_offset=self.step % self.steps_per_epoch)
        restore_rng(saved["rng"])
        del saved

    @torch.no_grad()
    def evaluate(self, panel, output, *, sample=True):
        """EMA diagnostics do not consume the training RNG or alter online state."""
        online = {k: v.detach().clone() for k, v in self.canonical.state_dict().items()}
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(73001)
                self.net.eval()
                self.canonical.load_state_dict(self.ema.shadow, strict=True)
                if self.args.arch == "smlp":
                    self.raw.clear_projected_cache()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        self.raw.prepare_projected_cache()
                    assert self.raw._projected_cache[0][0].dtype == torch.bfloat16
                total_v, total_x, count = 0.0, 0.0, 0
                # Uncompiled evaluation shares the trained backbone and cache;
                # it avoids a new compile for a non-full final diagnostic batch.
                for images, _ in DataLoader(panel, batch_size=64, num_workers=0):
                    clean = images.cuda()
                    t = self.loss_fn.sample_t(len(clean), clean.device)
                    epsilon = torch.randn_like(clean) * self.loss_fn.noise_scale
                    z = t*clean+(1-t)*epsilon
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        predicted = self.canonical(z, t.flatten()).float()
                    mse = (predicted-clean).square()
                    total_x += mse.mean((1, 2, 3)).sum().item()
                    total_v += (mse/(1-t).clamp_min(.05).square()).mean((1, 2, 3)).sum().item()
                    count += len(clean)
                if sample:
                    torch.manual_seed(54321)
                    latents = torch.randn(64, 3, self.resolution, self.resolution, device="cuda")
                    latents = latents * self.loss_fn.noise_scale
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        generated = vp_heun_sampler(self.net, latents, num_steps=50)
                    assert torch.isfinite(generated).all()
                    path = Path(output)/"samples"/f"step{self.step:09d}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    save_image(((generated[:36].float()+1)/2).clamp(0, 1), path, nrow=6)
        finally:
            if self.args.arch == "smlp":
                self.raw.clear_projected_cache()
            self.canonical.load_state_dict(online, strict=True)
            self.net.train()
        return dict(step=self.step, count=count, velocity_mse=total_v/count,
                    denoising_mse=total_x/count, weights="EMA", model_precision="bf16")


def diagnostic_panel(args, dataset, cache):
    if args.dataset == "celeba128":
        source = ImageFolderFlat(str(PROJECT/"datasets/celeba128/valid"), build_transform(128))
        indices = np.random.default_rng(73001).choice(len(source), 1024, replace=False).tolist()
        kind = "official validation"
        membership = [str(source.paths[i].relative_to(PROJECT)) for i in indices]
    else:
        # The prepared sorted index retains the original class directory names.
        names = (PROJECT/"datasets/imagenet_source/train_paths.txt").read_text().splitlines()
        first = {}
        for i, name in enumerate(names):
            first.setdefault(name.split('/')[0], i)
        assert len(first) == 1000
        indices = list(first.values())
        remaining = sorted(set(np.linspace(0, len(names)-1, 1024, dtype=int).tolist())-set(indices))
        indices = sorted(indices+remaining[:24])
        assert len(indices) == 1024
        source, kind = dataset, "class-stratified training diagnostic; not held out"
        membership = [names[i] for i in indices]
    record = dict(kind=kind, seed=73001, count=1024, indices=indices, membership=membership)
    publish(Path(args.outdir)/"diagnostic_manifest.json", record)
    return Subset(source, indices)


def run(args):
    assert args.batch == getattr(args, "expected_batch", 256), "Unexpected physical/global batch"
    assert torch.cuda.is_available(), "CUDA required; no CPU fallback"
    torch.ones(1, device="cuda").sum().item()
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
    assert torch.cuda.is_bf16_supported()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch._dynamo.config.cache_size_limit = 64
    output = Path(args.outdir)
    output.mkdir(parents=True, exist_ok=True)
    stop = {"requested": False}
    def request_stop(*_):
        stop["requested"] = True
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    # Avoid rebuilding a 235 GiB cache for a finished array task after requeue.
    if (output/"DONE.json").exists():
        completed = json.loads((output/"DONE.json").read_text())
        if completed["epochs"] == args.epochs:
            print(f"ALREADY COMPLETE {output}", flush=True)
            return
        assert getattr(args, "allow_epoch_extension", False)
        assert completed["epochs"] == 200 and args.epochs == 400
    dataset, cache = build_cache(PROJECT, args.dataset, args.cache_workers)
    assert len(dataset) == DATASETS[args.dataset]["count"]
    publish(output/"cache.json", cache)
    trainer = Training(args, dataset, cache)
    latest = output/"latest.json"
    if latest.exists():
        trainer.load(output/json.loads(latest.read_text())["checkpoint"])
    elif getattr(args, 'resume_checkpoint', None):
        trainer.load(args.resume_checkpoint)
    publish(output/"config.json", dict(recipe=trainer.recipe, model_info=trainer.model_info,
        data_loader_start_method="spawn; transforms and mmap recreated in each worker",
        source_snapshot=str(CODE), host=socket.gethostname(), job_id=os.environ.get("SLURM_JOB_ID"),
        precision="BF16 train/sample; FP32 weights/optimizer/EMA; unchanged Heun solver",
        source_sha256={name: hashlib.sha256((CODE/name).read_bytes()).hexdigest()
            for name in ("models.py", "model_grouped_dpd6.py", "fused_grouped_gelu_wide.py", "scripts/train_complementary.py")}))
    panel = diagnostic_panel(args, dataset, cache)

    def checkpoint():
        relative = Path("checkpoints")/f"step{trainer.step:09d}.pt"
        trainer.save(output/relative)
        publish(latest, dict(checkpoint=str(relative), step=trainer.step,
                            recipe_sha256=trainer.recipe_hash))
        old = sorted((output/"checkpoints").glob("step*.pt"))
        for path in old[:-2]:
            path.unlink()
        print(f"CHECKPOINT {trainer.step} {relative}", flush=True)

    def diagnostics():
        measurement = trainer.evaluate(panel, output)
        with (output/"diagnostics.jsonl").open("a") as stream:
            stream.write(json.dumps(measurement)+"\n")
        print(f"DIAGNOSTIC {json.dumps(measurement)}", flush=True)

    if trainer.step == 0:
        diagnostics()
    started, previous, losses, logged_step = time.monotonic(), time.monotonic(), [], trainer.step
    while trainer.step < trainer.target_steps:
        loader = trainer.loader()
        for images, _ in loader:
            if stop["requested"]:
                checkpoint()
                return 75
            loss = trainer.update(images)
            losses.append(loss)
            if trainer.step % 100 == 0:
                mean_loss = torch.stack(losses).mean().item()
                assert np.isfinite(mean_loss)
                now = time.monotonic()
                record = dict(status="training", step=trainer.step, target_steps=trainer.target_steps,
                    epoch_equivalent=trainer.step/trainer.steps_per_epoch, loss=mean_loss,
                    seconds_per_update=(now-previous)/(trainer.step-logged_step),
                    session_seconds=now-started, job_id=os.environ.get("SLURM_JOB_ID"),
                    learning_rate=trainer.optimizer.param_groups[0]['lr'])
                publish(output/"progress.json", record)
                print(f"PROGRESS {json.dumps(record)}", flush=True)
                losses, previous, logged_step = [], now, trainer.step
            if trainer.step % args.diagnostic_every == 0 or trainer.step == trainer.target_steps:
                diagnostics()
            if trainer.step % args.checkpoint_every == 0 or trainer.step == trainer.target_steps:
                checkpoint()
            if stop["requested"]:
                checkpoint()
                return 75
        del loader
    atomic_torch_save(dict(trainer.metadata(), ema=trainer.ema.shadow, net=trainer.ema.shadow,
        weights="EMA only; not an online resume checkpoint"), output/"ema_final.pt")
    publish(output/"DONE.json", dict(epochs=args.epochs, step=trainer.step,
        recipe_sha256=trainer.recipe_hash, final_ema="ema_final.pt", fid_status="not yet evaluated"))
    publish(output/"progress.json", dict(status="completed", step=trainer.step,
                                        target_steps=trainer.target_steps, epochs=args.epochs))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--arch", choices=("mlp", "smlp"), required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-workers", type=int, default=16)
    parser.add_argument("--hflip", action="store_true")
    parser.add_argument("--diagnostic-every", type=int, default=5000)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(arguments()))
