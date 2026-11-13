#!/usr/bin/env python3
"""One resumable final FID-50k per scale run, with explicit reference provenance."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader, Subset
from torchvision.models import inception_v3, Inception_V3_Weights

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
PROJECT = Path(os.environ.get("PIXEL_DIT_PROJECT_ROOT", CODE))
from data import ImageFolderFlat, build_transform
from edm import VPXPred, vp_heun_sampler
from experiments.complementary_benchmarks.celeba128_data import publish
from experiments.complementary_training.common import build_model
from experiments.complementary_training.input import SpawnImageFolder


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8*2**20):
            digest.update(block)
    return digest.hexdigest()


def save_npz(path, **values):
    path = Path(path)
    temporary = path.with_name(path.name+f".tmp-{os.getpid()}.npz")
    np.savez(temporary, **values)
    os.replace(temporary, path)


class Features(torch.nn.Module):
    """Same FP32 torchvision Inception protocol as eval_model_comparison.py."""
    def __init__(self):
        super().__init__()
        self.net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                                transform_input=False).cuda().eval()
        self.net.fc = torch.nn.Identity()
        self.net.requires_grad_(False)

    def forward(self, images):
        images = torch.nn.functional.interpolate(images.float(), (299, 299),
                                                mode="bilinear", align_corners=False)
        mean = images.new_tensor([.485, .456, .406])[None, :, None, None]
        std = images.new_tensor([.229, .224, .225])[None, :, None, None]
        return self.net((images-mean)/std)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--sampling-precision", choices=("bf16", "fp32"), default=None)
    args = p.parse_args()
    assert torch.cuda.is_available(), "CUDA required; no CPU fallback"
    torch.ones(1, device="cuda").sum().item()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = args.run_dir/"ema_final.pt"
    assert (args.run_dir/"DONE.json").exists() and checkpoint.exists()
    reference = json.loads(args.reference.read_text())
    for relative, digest in reference["source_fingerprints"].items():
        assert file_hash(PROJECT/relative) == digest, f"Reference source changed: {relative}"
    reference_hash = hashlib.sha256(json.dumps(reference, sort_keys=True).encode()).hexdigest()
    args.outdir.mkdir(parents=True, exist_ok=True)
    feature_net = Features()
    weights_path = Path(torch.hub.get_dir())/"checkpoints/inception_v3_google-0cc3c7bd.pth"
    extractor = dict(protocol="project torchvision Inception pooled features; not pytorch-fid weights",
        weights="Inception_V3_Weights.IMAGENET1K_V1", weights_sha256=file_hash(weights_path),
        resize="bilinear 299x299 align_corners=False", normalize="ImageNet mean/std",
        precision="fp32", torch_version=torch.__version__)
    real_identity = dict(reference_sha256=reference_hash, extractor=extractor)
    statistics = args.outdir.parent/f"{reference['dataset']}_real_stats.npz"
    with statistics.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if statistics.exists():
            real = np.load(statistics)
            assert json.loads(str(real["identity"])) == real_identity, "Real-statistics provenance changed"
        else:
            source = SpawnImageFolder(PROJECT/reference["data_root"], reference["resolution"])
            assert len(source) == reference["source_count"]
            # Preserve exactly the reference order while sending only the
            # selected paths to each spawned worker.
            source.paths = [source.paths[i] for i in reference["indices"]]
            assert len(source) == reference["count"]
            loader = DataLoader(source, batch_size=64,
                                num_workers=8, pin_memory=True, multiprocessing_context="spawn")
            chunks = []
            for images, _ in loader:
                chunks.append(feature_net(((images.cuda()+1)/2).clamp(0, 1)).cpu().numpy())
            features = np.concatenate(chunks).astype(np.float64)
            assert len(features) == reference["count"]
            save_npz(statistics, n=len(features), mu=features.mean(0), sigma=np.cov(features, rowvar=False),
                     identity=json.dumps(real_identity, sort_keys=True))
            del chunks, features, loader, source
            real = np.load(statistics)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    head_fp32 = bool(saved['model_cfg'].get('output_head_fp32', False))
    noise_scale = float(saved['recipe'].get('noise_scale', 1.0))
    sampling_precision = args.sampling_precision or (
        'fp32' if saved['recipe'].get('precision') == 'fp32' else 'bf16')
    provenance = dict(checkpoint_sha256=file_hash(checkpoint), reference_sha256=reference_hash,
        extractor=extractor, generated_images=50000, seed=54321, shard_size=512,
        sampling=dict(steps=50, calls=99, precision=sampling_precision, batch=64, solver="existing VP Heun float64 state"))
    if head_fp32:
        provenance['sampling']['output_head_precision'] = 'fp32'
    if 'noise_scale' in saved['recipe']:
        provenance['sampling']['noise_scale'] = noise_scale
    provenance_path = args.outdir/"provenance.json"
    if provenance_path.exists():
        assert json.loads(provenance_path.read_text()) == provenance, "FID resume provenance changed"
    else:
        publish(provenance_path, provenance)
    if (args.outdir/"result.json").exists():
        result = json.loads((args.outdir/"result.json").read_text())
        assert result["provenance"] == provenance
        done = json.loads((args.run_dir/"DONE.json").read_text())
        done.update(fid_status="completed", fid=result["fid"], fid_result=str(args.outdir/"result.json"))
        publish(args.run_dir/"DONE.json", done)
        print(f"FID ALREADY COMPLETE {args.outdir}", flush=True)
        return
    recipe = saved["recipe"]
    raw, config, _ = build_model(recipe["arch"], reference["resolution"], recipe["seed"])
    raw.cfg.output_head_fp32 = head_fp32
    config['output_head_fp32'] = head_fp32
    expected_config = dict(saved['model_cfg'])
    expected_config.setdefault('output_head_fp32', False)
    assert config == expected_config
    raw = raw.cuda().eval()
    canonical = VPXPred(raw, reference["resolution"], 3)
    canonical.load_state_dict(saved["ema"], strict=True)
    del saved
    if recipe["arch"] == "smlp":
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=sampling_precision == 'bf16'):
            raw.prepare_projected_cache()
        assert raw._projected_cache[0][0].dtype == (torch.bfloat16 if sampling_precision == 'bf16' else torch.float32)
    model = VPXPred(torch.compile(raw, dynamic=False), reference["resolution"], 3).eval()
    shard_dir = args.outdir/"features"
    shard_dir.mkdir(exist_ok=True)
    for shard in range(math.ceil(50000/512)):
        path = shard_dir/f"{shard:04d}.npy"
        size = min(512, 50000-shard*512)
        if path.exists():
            array = np.load(path, mmap_mode="r")
            assert array.shape == (size, 2048) and array.dtype == np.float32
            continue
        latents = torch.randn(512, 3, reference["resolution"], reference["resolution"],
            generator=torch.Generator().manual_seed(54321+shard))
        latents = latents * noise_scale
        output = []
        for offset in range(0, size, 64):
            # Keep model batch 64 even for the final sixteen retained samples.
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=sampling_precision == 'bf16'):
                generated = vp_heun_sampler(model, latents[offset:offset+64].cuda(), num_steps=50)
            assert torch.isfinite(generated).all()
            count = min(64, size-offset)
            output.append(feature_net(((generated[:count].float()+1)/2).clamp(0, 1)).cpu().numpy())
        temporary = path.with_suffix(".npy.tmp")
        with temporary.open("wb") as stream:
            np.save(stream, np.concatenate(output).astype(np.float32))
        temporary.replace(path)
        print(f"FID SHARD {shard+1}/98", flush=True)
    features = np.concatenate([np.load(shard_dir/f"{i:04d}.npy") for i in range(98)]).astype(np.float64)
    assert features.shape == (50000, 2048)
    mean, covariance = features.mean(0), np.cov(features, rowvar=False)
    root_covariance = sqrtm(real["sigma"]@covariance)
    if np.iscomplexobj(root_covariance):
        root_covariance = root_covariance.real
    delta = real["mu"]-mean
    score = max(float(delta@delta+np.trace(real["sigma"]+covariance-2*root_covariance)), 0.)
    assert np.isfinite(score)
    result = dict(fid=score, real_images=int(real["n"]), generated_images=50000,
                  checkpoint_step=recipe["total_steps"], provenance=provenance)
    publish(args.outdir/"result.json", result)
    done = json.loads((args.run_dir/"DONE.json").read_text())
    done.update(fid_status="completed", fid=score, fid_result=str(args.outdir/"result.json"))
    publish(args.run_dir/"DONE.json", done)
    print(f"FID {score:.6f}", flush=True)


if __name__ == "__main__":
    main()
