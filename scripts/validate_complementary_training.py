#!/usr/bin/env python3
"""H200 production-path checks before long complementary training jobs."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import Subset

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
PROJECT = Path(os.environ.get("PIXEL_DIT_PROJECT_ROOT", CODE))
from experiments.complementary_benchmarks.celeba128_data import publish
from experiments.complementary_benchmarks.checks import compiled_model_check
from experiments.complementary_training.common import DATASETS, rng_state
from experiments.complementary_training.input import EpochBatches, build_cache
from scripts.train_complementary import Training


class Stream:
    def __init__(self, trainer):
        self.trainer, self.loader, self.iterator = trainer, None, None

    def next(self):
        if self.iterator is not None:
            try:
                return next(self.iterator)[0]
            except StopIteration:
                self.close()
        self.loader = self.trainer.loader()
        self.iterator = iter(self.loader)
        return next(self.iterator)[0]

    def close(self):
        if self.iterator is not None and hasattr(self.iterator, "_shutdown_workers"):
            # A preflight deliberately stops mid-epoch. Drain its small subset
            # before shutting down pinned-memory workers; interrupting pending
            # transfers previously aborted the ImageNet256 worker process.
            for _ in self.iterator:
                pass
            self.iterator._shutdown_workers()
        self.iterator, self.loader = None, None


def compare(left, right):
    square, norm, maximum = 0., 0., 0.
    for key, expected in right.items():
        actual = left[key]
        delta = actual.detach().float()-expected.detach().float()
        square += delta.square().sum().item()
        norm += expected.detach().float().square().sum().item()
        maximum = max(maximum, delta.abs().max().item())
    return dict(relative_l2=(square/max(norm, 1e-30))**.5, max_abs=maximum)


def check_case(args, name, report):
    opts = argparse.Namespace(dataset=name, arch=args.arch, outdir=str(args.output/name),
        seed=0, epochs=200, batch=256, workers=8, cache_workers=16, hflip=False,
        diagnostic_every=5000, checkpoint_every=5000,
        lr=getattr(args, 'lr', 2e-4), projection_fix=getattr(args, 'projection_fix', None))
    dataset, cache = build_cache(PROJECT, name, 16, limit=4096)
    training = Training(opts, dataset, cache)
    stream = Stream(training)
    durations = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(60):
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss = training.update(stream.next())
        torch.cuda.synchronize()
        if i >= 20:
            durations.append(time.perf_counter()-start)
        if i in (0, 19, 59):
            assert torch.isfinite(loss).item()
            print(f"VALIDATE {name} {args.arch} update{i+1}: {time.perf_counter()-start:.4f}s", flush=True)
    case = dict(cache=cache, model_info=training.model_info, config=training.model_cfg,
        complete_update=dict(mean_seconds=float(np.mean(durations)), measured_updates=40,
            warmup_updates=20, batch=256, precision="bf16", sample_scope="4096-image local-cache preflight subset",
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
    if args.arch == "smlp":
        assert training.raw.key_prototypes.grad.float().norm().item()>0
        assert training.raw.value_prototypes.grad.float().norm().item()>0
        if getattr(args, 'projection_fix', None):
            from experiments.complementary_training.gradient_check import compare_compiled_gradients
            full_batch = torch.stack([dataset[i][0] for i in range(256)])
            case['full_batch_compiled_gradients'] = compare_compiled_gradients(training, full_batch)
            del full_batch
    resident = torch.stack([dataset[i][0] for i in range(2)]).cuda()
    if 'full_batch_compiled_gradients' in case:
        case['compiled_forward_backward'] = case['full_batch_compiled_gradients']
    else:
        case["compiled_forward_backward"] = compiled_model_check(training.raw, training.compiled, resident)
    training.net.train()
    # Place a checkpoint one batch before an epoch boundary; replay crosses it.
    while training.step % training.steps_per_epoch != training.steps_per_epoch-1:
        training.update(stream.next())
    stream.close()
    saved_step = training.step
    checkpoint = Path(opts.outdir)/"resume_probe.pt"
    training.save(checkpoint)
    before = rng_state()
    expected_inputs, expected_losses = [], []
    for _ in range(3):
        images = stream.next()
        expected_inputs.append(images.clone())
        expected_losses.append(training.update(images).item())
    stream.close()
    expected = {k: v.clone() for k, v in training.canonical.state_dict().items()}
    expected_ema = {k: v.clone() for k, v in training.ema.shadow.items()}
    expected_optimizer = copy.deepcopy(training.optimizer.state_dict())
    after = rng_state()
    training.load(checkpoint)
    loaded = rng_state()
    assert torch.equal(before["cpu"], loaded["cpu"])
    assert torch.equal(before["cuda"], loaded["cuda"])
    assert training.step == saved_step
    online_before = {k: v.clone() for k, v in training.canonical.state_dict().items()}
    case["bf16_ema_sampling"] = training.evaluate(Subset(dataset, range(64)), opts.outdir)
    assert all(torch.equal(value, training.canonical.state_dict()[key]) for key,value in online_before.items())
    assert torch.equal(loaded["cuda"], rng_state()["cuda"])
    assert torch.equal(loaded["cpu"], rng_state()["cpu"])
    if args.arch == "smlp":
        assert training.raw._projected_cache is None
    del online_before
    actual_losses = []
    for images in expected_inputs:
        replay = stream.next()
        assert torch.equal(replay, images), "Resume changed data order or pixels"
        actual_losses.append(training.update(replay).item())
    stream.close()
    resumed = rng_state()
    assert torch.equal(after["cpu"], resumed["cpu"])
    assert torch.equal(after["cuda"], resumed["cuda"])
    online_error = compare(training.canonical.state_dict(), expected)
    ema_error = compare(training.ema.shadow, expected_ema)
    optimizer_actual, optimizer_expected = {}, {}
    actual_state = training.optimizer.state_dict()
    for key, state in expected_optimizer["state"].items():
        for state_name, value in state.items():
            optimizer_actual[f"{key}/{state_name}"] = actual_state["state"][key][state_name]
            optimizer_expected[f"{key}/{state_name}"] = value
    optimizer_error = compare(optimizer_actual, optimizer_expected)
    # Flash-attention backward can have small floating-point replay differences;
    # data and RNG replay are exact, and complete model/EMA/optimizer states agree.
    assert online_error["relative_l2"] < 1e-5 and online_error["max_abs"] < 2e-5, online_error
    assert ema_error["relative_l2"] < 1e-5 and ema_error["max_abs"] < 2e-5, ema_error
    assert optimizer_error["relative_l2"] < .002, optimizer_error
    assert np.allclose(expected_losses, actual_losses, rtol=1e-4, atol=1e-6)
    case["resume"] = dict(checkpoint_step=saved_step, replay_updates=3,
        crossed_epoch_boundary=True, data_order_and_pixels_exact=True, rng_exact=True,
        online=online_error, ema=ema_error, optimizer=optimizer_error,
        uninterrupted_losses=expected_losses, resumed_losses=actual_losses)
    del expected, expected_ema, expected_optimizer, optimizer_actual, optimizer_expected
    case["diagnostic_preserves_online_weights_and_rng"] = True
    case["resume_after_ema_diagnostic_matches_uninterrupted_training"] = True
    if args.arch == "mlp" and name == "imagenet256":
        from scripts.evaluate_complementary import Features
        features = Features()(resident.add(1).div(2).clamp(0, 1))
        assert features.shape == (2, 2048) and features.dtype == torch.float32
        assert torch.isfinite(features).all()
        case["fid_feature_preflight"] = dict(shape=[2, 2048], dtype="fp32", passed=True)
    checkpoint.unlink()
    case["passed"] = True
    report["cases"][name] = case
    publish(args.output/f"case_{name}.json", case)
    del training, dataset, stream
    gc.collect()
    torch.cuda.empty_cache()
    torch._dynamo.reset()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", choices=("mlp", "smlp"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", choices=DATASETS)
    args = p.parse_args()
    assert torch.cuda.is_available(), "CUDA required"
    torch.ones(1, device="cuda").sum().item()
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch._dynamo.config.cache_size_limit = 64
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(arch=args.arch, gpu=torch.cuda.get_device_name(), host=socket.gethostname(),
        job_id=os.environ.get("SLURM_JOB_ID"), source_snapshot=str(CODE), cases={}, passed=False)
    publish(args.output/"results.json", report)
    try:
        if args.dataset:
            check_case(args, args.dataset, report)
            return
        # Production runs use separate processes/jobs. Match that lifecycle,
        # keeping each model's compiler/autograd cache isolated.
        for name in DATASETS:
            subprocess.run([sys.executable, str(Path(__file__).resolve()),
                "--arch", args.arch, "--output", str(args.output), "--dataset", name], check=True)
            report["cases"][name] = json.loads((args.output/f"case_{name}.json").read_text())
            publish(args.output/"results.json", report)
        report["passed"] = True
        publish(args.output/"results.json", report)
    except Exception:
        import traceback
        report["error"] = traceback.format_exc()
        publish(args.output/"results.json", report)
        raise


if __name__ == "__main__":
    main()
