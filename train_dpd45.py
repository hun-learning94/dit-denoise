"""Shared training implementation for DPD4, DPD5, DPD6, and grouped DPD6."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from edm import (
    EDMLoss, EDMPrecond, VelocityXPredLoss, VPVelocityXPredLoss,
    VPXPred, XPredPrecond,
)
from model_dpd4 import DPD4Backbone, DPD4Config
from model_dpd5 import DPD5Backbone, DPD5Config
from model_dpd6 import DPD6Backbone, DPD6Config
from model_grouped_dpd6 import GroupedDPD6Backbone, GroupedDPD6Config
from train import EMA, build_dataloader, save_samples, seed_everything


class GroupedEDMLoss(EDMLoss):
    """EDM loss with G independently sampled sigmas assigned evenly to B images."""

    def __init__(self, noise_groups: int = 8, **kwargs):
        super().__init__(**kwargs)
        if noise_groups < 1:
            raise ValueError("noise_groups must be positive")
        self.noise_groups = int(noise_groups)

    def __call__(self, net, images, labels=None, augment_pipe=None):
        batch_size = images.shape[0]
        groups = min(self.noise_groups, batch_size)
        rnd_normal = torch.randn([groups, 1, 1, 1], device=images.device)
        group_sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        group_ids = torch.arange(batch_size, device=images.device) % groups
        group_ids = group_ids[torch.randperm(batch_size, device=images.device)]
        sigma = group_sigma.index_select(0, group_ids)
        weight = (sigma.square() + self.sigma_data**2) / (sigma * self.sigma_data).square()
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        noisy = y + torch.randn_like(y) * sigma
        denoised = net(
            noisy,
            sigma,
            labels,
            augment_labels=augment_labels,
            noise_group_ids=group_ids,
        )
        return weight * (denoised - y).square()


ModelType = Literal["dpd4", "dpd5", "dpd6", "grouped_dpd6"]


class GroupedVPVelocityXPredLoss(VPVelocityXPredLoss):
    """Same per-image VP objective, sharing sampled times across noise groups."""

    def __init__(self, noise_groups, *args):
        super().__init__(*args)
        if noise_groups < 1:
            raise ValueError("noise_groups must be positive")
        self.noise_groups = noise_groups

    def sample_t(self, batch_size, device):
        groups = min(self.noise_groups, batch_size)
        times = super().sample_t(groups, device)
        ids = torch.arange(batch_size, device=device) % groups
        ids = ids[torch.randperm(batch_size, device=device)]
        return times[ids]


def build_parser(model_type: ModelType) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Train {model_type.upper()} with EDM")
    parser.add_argument("--dataset", default="folder", choices=["folder", "mnist", "fashionmnist"])
    parser.add_argument("--data_root", default="./datasets/celeba32/train")
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--hflip", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--outdir", default=f"runs/celeba32_{model_type}")

    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    if model_type == "dpd4":
        parser.add_argument("--memory_attention_heads", type=int, default=8)
        parser.add_argument("--memory_size", type=int, default=32)
        parser.add_argument("--activation_checkpointing", action="store_true",
                            help="Recompute block activations during backward to reduce GPU memory.")
        parser.add_argument("--memory_token_dim", type=int, default=None,
                            help="Direct memory-token width (default: backbone width). "
                                 "Nonmatching widths use one depth-shared key-only adapter; "
                                 "values project directly from the full-width bank.")
        # Accept old direct-only launch scripts (including queued Slurm copies),
        # but do not advertise retired factorization controls.
        parser.add_argument("--memory_mode", choices=["direct"], default="direct", help=argparse.SUPPRESS)
        parser.add_argument("--memory_code_dim", type=int, default=None, help=argparse.SUPPRESS)
        parser.add_argument("--memory_qk_head_dim", type=int, default=32)
        parser.add_argument("--memory_value_head_dim", type=int, default=32)
        parser.add_argument(
            "--memory_value_source", choices=["raw", "contextualized"], default=None,
            help="New DPD4 runs default to raw same-position values. On resume, "
                 "preserve checkpoint semantics; an explicit conflicting choice is rejected.",
        )
        parser.add_argument(
            "--memory_gain_mode",
            choices=["none", "per_head"],
            default="none",
            help="Optional tokenwise gain applied independently to each memory-attention head.",
        )
        parser.add_argument(
            "--noise_groups",
            type=int,
            default=8,
            help="Distinct EDM noise levels sampled per training minibatch.",
        )
        parser.add_argument(
            "--compile_blocks", action="store_true",
            help="Compile DPD4 blocks while retaining checkpoint-compatible state names.",
        )
        parser.add_argument(
            "--dpd4_explicit_memory_attention", action="store_true",
            help="Use the explicit FP32 softmax/einsum memory-attention path instead of fused SDPA.",
        )
    elif model_type == "dpd5":
        parser.add_argument("--prototype_attention_heads", type=int, default=8)
        parser.add_argument("--prototype_count", type=int, default=1024)
        parser.add_argument("--prototype_key_width", type=int, default=128)
        parser.add_argument("--prototype_value_width", type=int, default=160)
        parser.add_argument("--prototype_qk_head_dim", type=int, default=16)
        parser.add_argument("--prototype_value_head_dim", type=int, default=24)
        parser.add_argument(
            "--prototype_gain_mode", choices=["scalar", "per_head"], default="per_head"
        )
    elif model_type == "dpd6":
        parser.add_argument("--mlp_ratio", type=float, default=4.0)
        parser.add_argument("--key_library_dim", type=int, default=256)
        parser.add_argument("--value_library_dim", type=int, default=256)
        parser.add_argument(
            "--sharing_mode",
            choices=["both", "key", "value"],
            default="both",
            help="Share both MLP sides, the detector/key side only, or the output/value side only.",
        )
        parser.add_argument("--mlp_drop", type=float, default=0.0)
    else:
        parser.add_argument("--prototype_count", type=int, default=1024)
        parser.add_argument("--coefficient_groups", type=int, default=8)
        parser.add_argument("--group_dim", type=int, default=32)
        parser.add_argument("--prototype_key_width", type=int, default=256)
        parser.add_argument("--prototype_value_width", type=int, default=256)
        parser.add_argument(
            "--fused_grouped_gelu",
            action="store_true",
            help="Fuse detector contraction, bias, GELU, and value aggregation with Triton.",
        )
    parser.add_argument("--attn_drop", type=float, default=0.0)
    parser.add_argument("--proj_drop", type=float, default=0.0)

    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--override_resume_lr",
        action="store_true",
        help="After restoring optimizer state, replace every parameter-group LR with --lr.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--diagnose_nonfinite_grads", action="store_true",
        help="Report every parameter with a non-finite gradient before clipping.",
    )
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--no_ema", action="store_true")

    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=5000)
    parser.add_argument("--sample_every", type=int, default=5000)
    parser.add_argument("--sample_n", type=int, default=36)
    parser.add_argument("--sample_steps", type=int, default=18)
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--prediction_type", choices=["edm_preconditioned", "x"], default="edm_preconditioned")
    parser.add_argument("--noise_process", choices=["ve", "vp"], default="ve")
    parser.add_argument("--time_distribution", choices=["uniform_t", "uniform_t2"], default="uniform_t")
    parser.add_argument("--sigma_data", type=float, default=0.583087513)
    parser.add_argument("--time_min", type=float, default=0.05)
    parser.add_argument("--time_max", type=float, default=4.0)
    parser.add_argument("--vp_logit_mean", type=float, default=-0.8)
    parser.add_argument("--vp_logit_std", type=float, default=0.8)
    parser.add_argument("--vp_denominator_floor", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", default=None)
    return parser


def _build_model(model_type: str, args, img_res: int, img_ch: int, ckpt: Optional[dict]):
    if ckpt is not None:
        ckpt_type = ckpt.get("model_type")
        if ckpt_type != model_type:
            raise ValueError(f"checkpoint model_type={ckpt_type!r}, expected {model_type!r}")
        cfg_dict = ckpt.get("model_cfg")
        if not isinstance(cfg_dict, dict):
            raise ValueError("checkpoint is missing model_cfg")
        cfg_cls = {
            "dpd4": DPD4Config,
            "dpd5": DPD5Config,
            "dpd6": DPD6Config,
            "grouped_dpd6": GroupedDPD6Config,
        }[model_type]
        cfg = DPD4Config.from_checkpoint(cfg_dict) if model_type == "dpd4" else cfg_cls(**cfg_dict)
        if model_type == "dpd4":
            if cfg.memory_mode == "factorized":
                raise ValueError("Factorized-memory training is retired. Old checkpoints are evaluation-only; "
                                 "start a fresh direct-memory run instead.")
            checkpoint_width = cfg.memory_token_dim if cfg.memory_token_dim is not None else cfg.embed_dim
            if args.memory_token_dim is not None and args.memory_token_dim != checkpoint_width:
                raise ValueError("Cannot change memory_token_dim on resume; start a new run.")
            requested = args.memory_value_source
            if requested is not None and requested != cfg.memory_value_source:
                raise ValueError(
                    f"Cannot resume DPD4 with memory_value_source={requested!r}; "
                    f"checkpoint uses {cfg.memory_value_source!r}. Start a new run."
                )
        if cfg.img_resolution != img_res or cfg.img_channels != img_ch:
            raise ValueError("checkpoint model shape does not match the selected dataset")
    elif model_type == "dpd4":
        cfg = DPD4Config(
            img_resolution=img_res,
            img_channels=img_ch,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            memory_size=args.memory_size,
            memory_token_dim=args.memory_token_dim,
            memory_attention_heads=args.memory_attention_heads,
            memory_mode=args.memory_mode,
            memory_code_dim=args.memory_code_dim,
            memory_qk_head_dim=args.memory_qk_head_dim,
            memory_value_head_dim=args.memory_value_head_dim,
            memory_gain_mode=args.memory_gain_mode,
            memory_value_source=args.memory_value_source or "raw",
            attn_drop=args.attn_drop,
            proj_drop=args.proj_drop,
        )
    elif model_type == "dpd5":
        cfg = DPD5Config(
            img_resolution=img_res,
            img_channels=img_ch,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            prototype_count=args.prototype_count,
            prototype_attention_heads=args.prototype_attention_heads,
            prototype_key_width=args.prototype_key_width,
            prototype_value_width=args.prototype_value_width,
            prototype_qk_head_dim=args.prototype_qk_head_dim,
            prototype_value_head_dim=args.prototype_value_head_dim,
            prototype_gain_mode=args.prototype_gain_mode,
            attn_drop=args.attn_drop,
            proj_drop=args.proj_drop,
        )
    elif model_type == "dpd6":
        cfg = DPD6Config(
            img_resolution=img_res,
            img_channels=img_ch,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            key_library_dim=args.key_library_dim,
            value_library_dim=args.value_library_dim,
            sharing_mode=args.sharing_mode,
            attn_drop=args.attn_drop,
            proj_drop=args.proj_drop,
            mlp_drop=args.mlp_drop,
        )
    else:
        cfg = GroupedDPD6Config(
            img_resolution=img_res, img_channels=img_ch, patch_size=args.patch_size,
            embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
            prototype_count=args.prototype_count,
            coefficient_groups=args.coefficient_groups, group_dim=args.group_dim,
            prototype_key_width=args.prototype_key_width,
            prototype_value_width=args.prototype_value_width,
            fused_expansion=args.fused_grouped_gelu,
            attn_drop=args.attn_drop, proj_drop=args.proj_drop,
        )
    model_cls = {
        "dpd4": DPD4Backbone,
        "dpd5": DPD5Backbone,
        "dpd6": DPD6Backbone,
        "grouped_dpd6": GroupedDPD6Backbone,
    }[model_type]
    return model_cls(cfg), cfg


def _save_checkpoint(path: Path, net, cfg, model_type: str, opt, scaler, ema, step: int, args):
    payload = {
        "step": int(step),
        "model_type": model_type,
        "model_cfg": asdict(cfg),
        "net": net.state_dict(),
        "opt": opt.state_dict(),
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
    }
    if scaler.is_enabled():
        payload["scaler"] = scaler.state_dict()
    if ema is not None:
        payload["ema"] = {key: value.detach().cpu() for key, value in ema.shadow.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _report_nonfinite_gradients(net: torch.nn.Module, step: int) -> None:
    bad = []
    for name, parameter in net.named_parameters():
        grad = parameter.grad
        if grad is None or torch.isfinite(grad).all().item():
            continue
        finite = grad[torch.isfinite(grad)]
        finite_max = float(finite.abs().max()) if finite.numel() else float("nan")
        bad.append(
            f"{name}: nan={int(torch.isnan(grad).sum())} "
            f"inf={int(torch.isinf(grad).sum())} finite_max={finite_max:.6g}"
        )
    if bad:
        print(f"[diagnostic] non-finite gradients at step {step}:", flush=True)
        for item in bad:
            print(f"[diagnostic]   {item}", flush=True)


def run(model_type: ModelType) -> None:
    args = build_parser(model_type).parse_args()
    if args.noise_process == "vp" and args.prediction_type != "x":
        raise ValueError("noise_process=vp requires prediction_type=x")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed DPD4 training requires CUDA on every rank")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    seed_everything(args.seed + rank)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0
    # A single-process run honors --noise_groups.  In distributed training,
    # each rank owns one local noise group so that a global batch of W*B
    # contains W shared-noise groups (one per GPU/rank).  This also keeps the
    # grouped memory branch as small as possible on each rank.
    requested_noise_groups = args.noise_groups
    effective_noise_groups = 1 if distributed and model_type == "dpd4" else requested_noise_groups
    outdir = Path(args.outdir)
    if model_type == "dpd4" and not args.resume and (
        (outdir / "config.json").exists()
        or any((outdir / "checkpoints").glob("*.pt"))
    ):
        raise ValueError("Fresh DPD4 training requires a new output directory; do not overwrite historical runs.")
    (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (outdir / "samples").mkdir(parents=True, exist_ok=True)

    loader, img_res, img_ch, dataset_size = build_dataloader(args)
    if distributed:
        sampler = DistributedSampler(
            loader.dataset, num_replicas=world_size, rank=rank,
            shuffle=True, drop_last=True,
        )
        loader = DataLoader(
            loader.dataset,
            batch_size=args.batch,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    backbone, cfg = _build_model(model_type, args, img_res, img_ch, checkpoint)
    if model_type == "dpd4":
        args.memory_value_source = cfg.memory_value_source
        backbone.activation_checkpointing = args.activation_checkpointing
        if is_main:
            print(f"[dpd4] memory_value_source={cfg.memory_value_source} "
                  f"memory_token_dim={cfg.memory_token_dim} "
                  f"shared_key_adapter={cfg.memory_token_dim != cfg.embed_dim}", flush=True)
    if model_type == "dpd4" and args.dpd4_explicit_memory_attention:
        for block in backbone.blocks:
            block.memory_attn.use_explicit_attention = True
        if is_main:
            print("[diagnostic] using explicit FP32 DPD4 memory attention")
    if model_type == "dpd4" and args.compile_blocks:
        backbone.compile_blocks()
    if args.prediction_type == "x" and args.noise_process == "vp":
        net = VPXPred(backbone, img_res, img_ch, label_dim=0).to(device)
        vp_parameters = (args.vp_logit_mean, args.vp_logit_std, args.vp_denominator_floor)
        if model_type == "dpd4":
            # Earlier VP runs ignored noise_groups. Preserve their sampling
            # convention on resume instead of silently changing it mid-run.
            if distributed:
                args.effective_vp_noise_groups = 1
            else:
                args.effective_vp_noise_groups = (
                    checkpoint.get("args", {}).get("effective_vp_noise_groups", args.batch)
                    if checkpoint is not None else args.noise_groups
                )
            loss_fn = GroupedVPVelocityXPredLoss(args.effective_vp_noise_groups, *vp_parameters)
            if is_main:
                print(f"[dpd4] effective_vp_noise_groups={args.effective_vp_noise_groups} "
                      f"activation_checkpointing={args.activation_checkpointing}", flush=True)
        else:
            loss_fn = VPVelocityXPredLoss(*vp_parameters)
    elif args.prediction_type == "x":
        net = XPredPrecond(
            backbone, img_res, img_ch, label_dim=0,
            sigma_data=args.sigma_data, t_min=args.time_min, t_max=args.time_max,
        ).to(device)
        loss_fn = VelocityXPredLoss(args.time_min, args.time_max, args.time_distribution)
    else:
        net = EDMPrecond(backbone, img_res, img_ch, label_dim=0, use_fp16=False).to(device)
        loss_fn = GroupedEDMLoss(effective_noise_groups) if model_type == "dpd4" else EDMLoss()
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    ema = None if args.no_ema else EMA(net, decay=args.ema)

    use_amp = args.precision != "fp32" and device.type == "cuda"
    amp_dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and args.precision == "fp16")
    start_step = 0
    if checkpoint is not None:
        saved_prediction = checkpoint.get("prediction_type", checkpoint.get("args", {}).get("prediction_type", "edm_preconditioned"))
        saved_process = checkpoint.get("noise_process", checkpoint.get("args", {}).get("noise_process", "ve"))
        if saved_prediction != args.prediction_type or (args.prediction_type == "x" and saved_process != args.noise_process):
            raise ValueError("checkpoint prediction/noise-process semantics do not match this run")
        net.load_state_dict(checkpoint["net"], strict=True)
        optimizer.load_state_dict(checkpoint["opt"])
        if args.override_resume_lr:
            for group in optimizer.param_groups:
                group["lr"] = args.lr
            print(f"[resume] optimizer learning rate overridden to {args.lr:g}")
        if scaler.is_enabled() and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        if ema is not None and checkpoint.get("ema") is not None:
            ema.shadow = {
                key: value.to(device=device, dtype=torch.float32)
                for key, value in checkpoint["ema"].items()
            }
        start_step = int(checkpoint.get("step", 0))
        if is_main:
            print(f"[resume] loaded {args.resume} at step={start_step}")

    raw_net = net
    if distributed:
        net = DDP(raw_net, device_ids=[local_rank], output_device=local_rank,
                  broadcast_buffers=False, find_unused_parameters=False)

    if is_main:
        with (outdir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump({
                **vars(args), "model_type": model_type, "model_cfg": asdict(cfg),
                "distributed": distributed, "world_size": world_size,
                "effective_batch": args.batch * world_size,
                "requested_noise_groups": requested_noise_groups,
                "effective_noise_groups": effective_noise_groups,
            }, handle, indent=2)
    if distributed:
        dist.barrier()

    group_text = (
        f" noise_groups={requested_noise_groups} effective_local_noise_groups={effective_noise_groups}"
        if model_type == "dpd4" else ""
    )
    if is_main:
        print(
            f"[train] model={model_type} device={device} dataset={args.dataset} "
            f"size={dataset_size} batch={args.batch} world_size={world_size} "
            f"effective_batch={args.batch * world_size} res={img_res} ch={img_ch}{group_text}"
        )
    net.train()
    iterator = iter(loader)
    sampler_epoch = 0
    running_loss = 0.0
    running_count = 0
    started = time.time()
    for step in range(start_step, args.steps):
        try:
            images, _ = next(iterator)
        except StopIteration:
            if distributed:
                sampler_epoch += 1
                sampler.set_epoch(sampler_epoch)
            iterator = iter(loader)
            images, _ = next(iterator)
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            loss = loss_fn(net, images).mean()
        if not torch.isfinite(loss.detach()).item():
            raise FloatingPointError(
                f"non-finite loss at step {step + 1}; refusing optimizer and EMA updates"
            )
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                try:
                    torch.nn.utils.clip_grad_norm_(
                        net.parameters(), args.grad_clip, error_if_nonfinite=True
                    )
                except RuntimeError:
                    if args.diagnose_nonfinite_grads:
                        _report_nonfinite_gradients(net, step + 1)
                    raise
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                try:
                    torch.nn.utils.clip_grad_norm_(
                        net.parameters(), args.grad_clip, error_if_nonfinite=True
                    )
                except RuntimeError:
                    if args.diagnose_nonfinite_grads:
                        _report_nonfinite_gradients(net, step + 1)
                    raise
            optimizer.step()
        if ema is not None:
            ema.update(raw_net)

        running_loss += float(loss.detach())
        running_count += 1
        step1 = step + 1
        if is_main and step1 % args.log_every == 0:
            elapsed = max(time.time() - started, 1e-9)
            print(
                f"[train] step {step1:>7d}/{args.steps} loss="
                f"{running_loss / running_count:.6f} {(step1-start_step)/elapsed:.2f} it/s"
            )
            running_loss = 0.0
            running_count = 0

        if step1 % args.ckpt_every == 0 or step1 == args.steps:
            path = outdir / "checkpoints" / f"ckpt_step{step1:07d}.pt"
            if is_main:
                _save_checkpoint(path, raw_net, cfg, model_type, optimizer, scaler, ema, step1, args)
                print(f"[ckpt] saved {path}")
            if distributed:
                dist.barrier()

        if is_main and (step1 % args.sample_every == 0 or step1 == args.steps):
            current = None
            if ema is not None:
                current = {key: value.detach().clone() for key, value in raw_net.state_dict().items()}
                ema.copy_to(raw_net)
            save_samples(
                raw_net,
                str(outdir / "samples" / f"sample_step{step1:07d}.png"),
                args.sample_n,
                img_res,
                img_ch,
                num_steps=args.sample_steps,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                rho=args.rho,
                device=str(device),
                prediction_type=args.prediction_type,
                noise_process=args.noise_process,
                vp_denominator_floor=args.vp_denominator_floor,
            )
            if current is not None:
                raw_net.load_state_dict(current, strict=True)

    if distributed:
        dist.destroy_process_group()


__all__ = ["run", "GroupedEDMLoss"]
