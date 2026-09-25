"""Unconditional VP DiT: torchrun DDP, per-rank batches, gradient accumulation.

Steps always count optimizer updates. Legacy train.py and existing runs are untouched.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler

from edm import VPXPred, VPVelocityXPredLoss
from models import DiTBackbone, DiTConfig
from train import EMA, save_samples
from .data import ImageNet64


class RemainingBatches(BatchSampler):
    """Skip already-consumed indices on resume, without decoding skipped images."""
    def __init__(self, sampler, batch_size, offset):
        super().__init__(sampler, batch_size, drop_last=True)
        self.offset = offset

    def __iter__(self):
        for i, batch in enumerate(super().__iter__()):
            if i >= self.offset:
                yield batch

    def __len__(self):
        return max(0, super().__len__() - self.offset)


def accumulated_backward(model, batches, loss_fn, accumulation, autocast_factory=nullcontext):
    """DDP averages ranks; scale microbatch means once, synchronizing only the last."""
    total = None
    for micro in range(accumulation):
        sync = model.no_sync() if isinstance(model, DDP) and micro + 1 < accumulation else nullcontext()
        with sync:
            with autocast_factory():
                loss = loss_fn(model, next(batches)).mean()
            (loss / accumulation).backward()
        total = loss.detach() / accumulation if total is None else total + loss.detach() / accumulation
    return total


def atomic_save(payload, path):
    temporary = path.with_suffix('.pt.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_root', type=Path, default=Path('datasets/imagenet64/prepared'))
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--steps', type=int, required=True)
    p.add_argument('--batch', type=int, default=32, help='Microbatch per rank')
    p.add_argument('--accumulation', type=int, default=2)
    p.add_argument('--num_workers', type=int, default=1, help='Workers per rank')
    p.add_argument('--dim', type=int, default=320)
    p.add_argument('--depth', type=int, default=10)
    p.add_argument('--heads', type=int, default=10)
    p.add_argument('--patch', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--ema', type=float, default=.999)
    p.add_argument('--grad_clip', type=float, default=1.)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--precision', choices=['auto', 'fp32', 'bf16'], default='fp32')
    p.add_argument('--hflip', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--log_every', type=int, default=100)
    p.add_argument('--ckpt_every', type=int, default=5000)
    p.add_argument('--sample_every', type=int, default=5000)
    p.add_argument('--sample_n', type=int, default=16)
    p.add_argument('--sample_steps', type=int, default=50)
    p.add_argument('--resume', type=Path)
    p.add_argument('--checkpoint_steps', type=int, nargs='*', default=[])
    p.add_argument('--rolling_recovery', action='store_true')
    p.add_argument('--eval_campaign', type=Path)
    p.add_argument('--stop_file', type=Path)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='CPU only for small smoke tests')
    args = p.parse_args()
    for key in ['steps', 'batch', 'accumulation', 'log_every', 'ckpt_every']:
        if getattr(args, key) <= 0:
            p.error(f'{key} must be positive')
    if args.sample_every < 0 or args.num_workers < 0:
        p.error('sample_every and num_workers must be nonnegative')
    if args.sample_steps < 2:
        p.error('sample_steps must be at least 2')
    if any(v <= 0 or v > args.steps for v in args.checkpoint_steps):
        p.error('checkpoint_steps must lie within the training horizon')
    special_steps = set(args.checkpoint_steps)
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local = int(os.environ.get('LOCAL_RANK', '0'))
    torch.set_num_threads(1)
    automatic_precision = args.precision == 'auto'
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; refusing implicit CPU training')
        torch.cuda.set_device(local)
        if automatic_precision:
            name = torch.cuda.get_device_name(local).lower()
            args.precision = 'bf16' if '2080' not in name and torch.cuda.is_bf16_supported() else 'fp32'
        if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
            raise ValueError('BF16 unavailable on this GPU; use FP32 on 2080')
    if args.device == 'cpu' and automatic_precision:
        args.precision = 'fp32'
    device = torch.device(args.device, local) if args.device == 'cuda' else torch.device('cpu')
    if world > 1:
        dist.init_process_group('nccl' if args.device == 'cuda' else 'gloo')
    torch.manual_seed(args.seed)
    dataset = ImageNet64(args.data_root)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True,
                                 seed=args.seed, drop_last=True)
    batches_per_pass = len(sampler) // args.batch
    if not batches_per_pass:
        raise ValueError('Dataset too small for a complete distributed microbatch')
    effective_batch = args.batch * world * args.accumulation
    fingerprint = hashlib.sha256((args.data_root / 'manifest.json').read_bytes()).hexdigest()
    cfg = DiTConfig(img_resolution=64, img_channels=3, patch_size=args.patch,
                    embed_dim=args.dim, depth=args.depth, num_heads=args.heads, mlp_ratio=4,
                    print_params=False)
    net = VPXPred(DiTBackbone(cfg), 64, 3).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0)
    ema = EMA(net, args.ema)
    start = 0
    restored_rng = None
    previous_seconds = 0.0
    protocol = dict(world_size=world, batch_per_rank=args.batch, accumulation=args.accumulation,
                    effective_batch=effective_batch, train_images=len(dataset), manifest_sha256=fingerprint,
                    seed=args.seed, precision=args.precision, hflip=args.hflip,
                    lr=args.lr, ema=args.ema, grad_clip=args.grad_clip)
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        old_protocol = dict(ckpt['distributed_protocol'])
        current_protocol = dict(protocol)
        if automatic_precision:
            old_protocol.pop('precision')
            current_protocol.pop('precision')
        if old_protocol != current_protocol or ckpt['backbone']['cfg'] != asdict(cfg):
            raise ValueError('Resume requires identical data, model, batch, seed, precision and optimization protocol')
        net.load_state_dict(ckpt['net'], strict=True)
        opt.load_state_dict(ckpt['opt'])
        ema.shadow = {k: v.to(device) for k, v in ckpt['ema'].items()}
        start = ckpt['step']
        previous_seconds = ckpt.get('training_seconds', 0.0)
        restored_rng = ckpt['rank_rng'][rank]
        del ckpt
    model = DDP(net, device_ids=[local] if args.device == 'cuda' else None) if world > 1 else net
    # Separate rank streams for diffusion noise and augmentation, same initialized model.
    torch.manual_seed(args.seed + 100003 * (rank + 1))
    if restored_rng:
        torch.set_rng_state(restored_rng['cpu'])
        if args.device == 'cuda':
            torch.cuda.set_rng_state(restored_rng['cuda'], device)
    loss_fn = VPVelocityXPredLoss(-.8, .8, .05)
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / 'checkpoints').mkdir(exist_ok=True)
    (args.outdir / 'samples').mkdir(exist_ok=True)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(prediction_type='x', noise_process='vp', vp_logit_mean=-.8,
                    vp_logit_std=.8, vp_denominator_floor=.05, resolution=64, channels=3,
                    trainable_groups='all', arch='dit')
    if rank == 0:
        (args.outdir / 'config.json').write_text(json.dumps(dict(metadata, distributed_protocol=protocol,
            backbone=dict(arch='dit', cfg=asdict(cfg))), indent=2) + '\n')
        print('Protocol:', json.dumps(protocol), 'start_step=', start,
              'parameters=', sum(p.numel() for p in net.parameters()), flush=True)
    stop = False
    def request_stop(signum, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def batch_stream():
        pass_index, offset = divmod(start * args.accumulation, batches_per_pass)
        while True:
            sampler.set_epoch(pass_index)
            # Dedicated loader RNG prevents iterator creation affecting diffusion RNG.
            loader = DataLoader(dataset, batch_sampler=RemainingBatches(sampler, args.batch, offset),
                num_workers=args.num_workers, pin_memory=args.device == 'cuda',
                generator=torch.Generator().manual_seed(args.seed + pass_index))
            for raw in loader:
                images = raw.to(device, non_blocking=True).float().div_(127.5).sub_(1.)
                if args.hflip:
                    flip = torch.rand(len(images), device=device) < .5
                    images[flip] = images[flip].flip(-1)
                yield images
            pass_index += 1
            offset = 0

    stream = iter(batch_stream())
    amp = lambda: torch.autocast(device_type=args.device, dtype=torch.bfloat16,
                                 enabled=args.precision == 'bf16')
    running = torch.zeros((), device=device)
    count = 0
    beginning = time.monotonic()
    model.train()
    for step in range(start + 1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        loss = accumulated_backward(model, stream, loss_fn, args.accumulation, amp)
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip, error_if_nonfinite=True)
        opt.step()
        ema.update(net)
        running += loss
        count += 1
        if step % args.log_every == 0:
            average = running / count
            if world > 1:
                dist.all_reduce(average)
                average /= world
            if rank == 0:
                rate = (step - start) / (time.monotonic() - beginning)
                print(f'step={step}/{args.steps} loss={average.item():.6f} updates/s={rate:.3f} '
                      f'images/s={rate*effective_batch:.1f} image_epochs={step*effective_batch/len(dataset):.4f}', flush=True)
            running.zero_(); count = 0
        if args.stop_file and args.stop_file.exists():
            stop = True
        stop_flag = torch.tensor(int(stop), device=device)
        if world > 1:
            dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
        stopping = bool(stop_flag.item())
        if step % args.ckpt_every == 0 or step == args.steps or step in special_steps or stopping:
            state = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device) if args.device == 'cuda' else None)
            states = [None] * world
            if world > 1:
                dist.all_gather_object(states, state)
            else:
                states[0] = state
            if rank == 0:
                filename = f'ckpt_step{step}.pt' if (not args.rolling_recovery or step in special_steps or step == args.steps) else 'latest.pt'
                path = args.outdir / 'checkpoints' / filename
                atomic_save(dict(step=step, net=net.state_dict(), opt=opt.state_dict(),
                    ema={k: v.cpu() for k, v in ema.shadow.items()}, args=metadata,
                    backbone=dict(arch='dit', cfg=asdict(cfg)), dit_cfg=asdict(cfg),
                    prediction_type='x', noise_process='vp', loss_type='velocity_mse',
                    vp_logit_mean=-.8, vp_logit_std=.8, vp_denominator_floor=.05,
                    distributed_protocol=protocol, images_seen=step*effective_batch, rank_rng=states,
                    training_seconds=previous_seconds + time.monotonic() - beginning), path)
                pointer = args.outdir / 'resume.json.tmp'
                pointer.write_text(json.dumps(dict(step=step, checkpoint=str(path))) + '\n')
                pointer.replace(args.outdir / 'resume.json')
                print('Saved', path, flush=True)
                if args.eval_campaign:
                    from .release_evaluations import release
                    try:
                        release(args.eval_campaign)
                    except Exception as error:
                        print('Evaluation release will retry at next checkpoint:', repr(error), flush=True)
            if world > 1:
                dist.barrier()
        if ((args.sample_every and (step % args.sample_every == 0 or step == args.steps)) or step in special_steps) and not stopping:
            if rank == 0:
                current = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                ema.copy_to(net)
                # Sampling never perturbs training RNG; identical latent seed across snapshots.
                with torch.random.fork_rng(devices=[local] if args.device == 'cuda' else []):
                    torch.manual_seed(20260904)
                    save_samples(net=net, out_path=str(args.outdir / 'samples' / f'sample_step{step}.png'),
                        n=args.sample_n, img_resolution=64, img_channels=3, num_steps=args.sample_steps,
                        sigma_min=.05, sigma_max=4., rho=7., device=args.device,
                        prediction_type='x', noise_process='vp', vp_denominator_floor=.05)
                net.load_state_dict(current)
            if world > 1:
                dist.barrier()
        if stopping:
            break
    telemetry = dict(rank=rank, device=str(device), start_step=start, final_step=step if start < args.steps else start,
                     elapsed_seconds=time.monotonic()-beginning, job_id=os.environ.get('SLURM_JOB_ID'),
                     gpu_name=torch.cuda.get_device_name(device) if args.device=='cuda' else None,
                     peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if args.device=='cuda' else 0,
                     peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if args.device=='cuda' else 0)
    (args.outdir / f'telemetry_rank{rank}.json').write_text(json.dumps(telemetry, indent=2)+'\n')
    if world > 1:
        dist.destroy_process_group()
    if stop:
        raise SystemExit(75)


if __name__ == '__main__':
    main()
