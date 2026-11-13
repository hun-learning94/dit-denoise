"""Resumable native Heun50 banks, paired held-out losses and CPU KID/FID."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import signal
import time
import numpy as np
from PIL import Image
import torch
from edm import vp_heun_sampler
from experiments.component_transfer_scale.model import load_source, plain_model, digest, write, seed_all
from experiments.uvit_metfaces.evaluate import Features, metrics, array_save
from .train import setup

STOP = False


def stop(*_):
    global STOP
    STOP = True


def source_model(m, state=None):
    net = plain_model(m['source'], state, m['source']['model_cfg'])
    if state is None:
        saved = torch.load(m['source']['checkpoint'], map_location='cpu', weights_only=False)
        net.load_state_dict(saved['ema'], strict=True)
    return net.cuda().eval().requires_grad_(False)


@torch.no_grad()
def sample(m, net, seed, shard, count):
    generator = torch.Generator().manual_seed(seed + 1000003 * shard)
    initial = torch.randn(count, 3, 128, 128, generator=generator).cuda() * .5
    class Counter(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.calls = 0
        def forward(self, x, t, labels=None):
            self.calls += 1
            return net(x, t, labels)
    counter = Counter()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        result = vp_heun_sampler(counter, initial, num_steps=50, denominator_floor=.05)
    assert counter.calls == 99 and result.dtype == torch.float64
    assert torch.isfinite(result).all()
    return ((result.float() + 1) * 127.5).clamp(0, 255).round().byte()


@torch.no_grad()
def real_features(m):
    base = Path(m['eval_output']); base.mkdir(parents=True, exist_ok=True)
    ready = base / 'REAL_READY.json'
    if ready.exists():
        r = json.loads(ready.read_text())
        assert r['data_sha256'] == m['data_sha256']
        for split, h in r['files'].items():
            assert digest(base / f'real_{split}.npy') == h
        return
    data = json.loads(Path(m['data']).read_text())
    pixels = np.load(data['images'], mmap_mode='r'); feature = Features(m)
    for split in ['train', 'test']:
        ids = np.load(data['splits'][split]['path']); parts = []
        for first in range(0, len(ids), 32):
            batch = torch.from_numpy(np.array(pixels[ids[first:first + 32]], copy=True)).permute(0, 3, 1, 2).cuda().float() / 255
            parts.append(feature(batch).cpu().numpy())
        array_save(base / f'real_{split}.npy', np.concatenate(parts))
    write(ready, dict(data_sha256=m['data_sha256'], weights_sha256=m['feature_weights_sha256'],
                     files={s: digest(base / f'real_{s}.npy') for s in ['train', 'test']}))


@torch.no_grad()
def denoise(m, net, output):
    if (output / 'denoising.json').exists():
        return
    data = json.loads(Path(m['data']).read_text())
    target = np.load(data['images'], mmap_mode='r')[np.load(data['splits']['test']['path'])]
    source = np.load(data['source_test'], mmap_mode='r')
    stats = {}
    for domain, pixels in [('source', source), ('target', target)]:
        seed_all(2026111001)
        native, clean = [], []
        for first in range(0, len(pixels), 8):
            x = torch.from_numpy(np.array(pixels[first:first + 8], copy=True)).permute(0, 3, 1, 2).cuda().float() / 127.5 - 1
            losses, mses = [], []
            for _ in range(4):
                t = (torch.randn(len(x), 1, 1, 1, device='cuda') * .8 - .8).sigmoid()
                z = t * x + (1 - t) * torch.randn_like(x) * .5
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    prediction = net(z, t.flatten())
                error = (prediction.float() - x).square()
                losses.append((error / (1 - t).clamp_min(.05).square()).flatten(1).mean(1).cpu().numpy())
                mses.append(error.flatten(1).mean(1).cpu().numpy())
            native.extend(np.stack(losses, axis=1).tolist()); clean.extend(np.stack(mses, axis=1).tolist())
            if STOP:
                raise SystemExit(75)
        stats[domain] = dict(images=len(pixels), draws=4, native_mean=float(np.mean(native)),
                             clean_mse_mean=float(np.mean(clean)), per_image_native=native,
                             per_image_clean_mse=clean)
    write(output / 'denoising.json', dict(seed=2026111001, objective=m['source']['objective'], **stats))


@torch.no_grad()
def evaluate(m, task):
    row = m['evaluation_tasks'][task]
    base = Path(m['eval_output']); out = base / row['label']; out.mkdir(parents=True, exist_ok=True)
    if (out / 'SAMPLING_READY.json').exists():
        return
    if row['arm'] == 'source':
        checkpoint = m['source']['checkpoint']; epoch = 0
        net = source_model(m)
    else:
        run = Path(m['runs'][row['arm']]); assert (run / 'COMPLETE.json').exists()
        checkpoint = str(run / (row['endpoint'] + '.pt'))
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        identity = payload['identity']
        assert identity['arm'] == row['arm'] and identity['data'] == m['data_sha256']
        assert identity['source'] == m['source']['checkpoint_sha256'] and identity['code_sha256'] == m['code_sha256']
        epoch = payload['epoch']; net = source_model(m, payload['ema']); del payload
    identity = dict(task=row, checkpoint_sha256=digest(checkpoint), data_sha256=m['data_sha256'],
                    code_sha256=m['code_sha256'], feature_weights_sha256=m['feature_weights_sha256'],
                    sampler='native VP Heun50,99 calls,noise0.5,FP64 state,BF16 transformer,FP32 head',
                    batch=m['evaluation']['batch'])
    if (out / 'identity.json').exists():
        assert json.loads((out / 'identity.json').read_text()) == identity
    else:
        write(out / 'identity.json', identity)
    feature = Features(m)
    batch = m['evaluation']['batch']; n = m['evaluation']['generated_images']
    start = time.monotonic()
    for shard, first in enumerate(range(0, n, batch)):
        f = out / f'features_{shard:04d}.npy'; count = min(batch, n - first)
        if f.exists():
            a = np.load(f); assert a.shape == (count, 2048) and np.isfinite(a).all()
        else:
            images = sample(m, net, row['seed'], shard, count)
            values = feature(images.float() / 255).cpu().numpy()
            assert values.shape == (count, 2048) and np.isfinite(values).all()
            array_save(f, values)
            if shard == 0:
                for i, im in enumerate(images[:32].permute(0, 2, 3, 1).cpu().numpy()):
                    Image.fromarray(im).save(out / f'example_{i:02d}.png')
            write(out / 'progress.json', dict(images=first + count, total=n, epoch=epoch,
                        seconds_this_attempt=time.monotonic() - start, cloud_run_id=os.getenv('CLOUD_RUN_ID')))
        if STOP:
            raise SystemExit(75)
    del feature; gc.collect(); torch.cuda.empty_cache()
    # One denoising diagnostic per trained endpoint, shared across its sampling seeds.
    diagnostic = base / 'denoising' / f"{row['arm']}_{row['endpoint']}"
    diagnostic.mkdir(parents=True, exist_ok=True)
    denoise(m, net, diagnostic)
    write(out / 'SAMPLING_READY.json', dict(identity=identity, task=row, epoch=epoch, checkpoint=checkpoint))


def score(m, task):
    row = m['evaluation_tasks'][task]; base = Path(m['eval_output']); out = base / row['label']
    if (out / 'result.json').exists():
        return
    ready = json.loads((out / 'SAMPLING_READY.json').read_text())
    assert ready['identity'] == json.loads((out / 'identity.json').read_text())
    n, batch = m['evaluation']['generated_images'], m['evaluation']['batch']
    arrays = [np.load(out / f'features_{i:04d}.npy') for i in range(math.ceil(n / batch))]
    gen = np.concatenate(arrays); assert gen.shape == (n, 2048) and np.isfinite(gen).all()
    real_ready = json.loads((base / 'REAL_READY.json').read_text())
    assert real_ready['data_sha256'] == m['data_sha256']
    assert digest(base / 'real_test.npy') == real_ready['files']['test']
    result = metrics(np.load(base / 'real_test.npy'), gen)
    assert all(math.isfinite(result[k]) for k in ['fid', 'kid'])
    write(out / 'result.json', dict(**row, **result, epoch=ready['epoch'], checkpoint=ready['checkpoint'],
                                  identity=ready['identity'], extractor=m['evaluation']['extractor']))


@torch.no_grad()
def smoke(m):
    torch.cuda.reset_peak_memory_stats()
    net = source_model(m)
    first = sample(m, net, 202611091, 0, 2)
    assert torch.equal(first, sample(m, net, 202611091, 0, 2)), 'Sampler replay changed'
    assert not torch.equal(first, sample(m, net, 202611092, 0, 2)), 'Seed did not affect generation'
    feature = Features(m)
    torch.cuda.synchronize(); start = time.monotonic()
    images = sample(m, net, 202611091, 1, m['evaluation']['batch'])
    feats = feature(images.float() / 255); torch.cuda.synchronize()
    seconds = time.monotonic() - start
    assert feats.shape == (m['evaluation']['batch'], 2048) and torch.isfinite(feats).all()
    control = np.array([[0., 1.], [2., 3.], [1., 4.]])
    k = (control @ control.T / 2 + 1) ** 3
    expected_kid = 2 * (k.sum() - np.trace(k)) / 6 - 2 * k.mean()
    result = metrics(control, control)
    assert abs(result['fid']) < 1e-10 and abs(result['kid'] - expected_kid) < 1e-10
    write(Path(m['report']) / 'EVAL_SMOKE_PASS.json', dict(native_sampler=True, calls=99,
        initial_noise_scale=.5, solver_dtype='float64', replay_exact=True, distinct_sampling_seeds=True,
        batch=m['evaluation']['batch'], images_per_second=m['evaluation']['batch'] / seconds,
        feature_shape=list(feats.shape), analytic_kid_and_fid_passed=True,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30, gpu=torch.cuda.get_device_name()))


def main():
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['smoke', 'real', 'sample', 'score'])
    p.add_argument('--manifest', required=True); p.add_argument('--task', type=int)
    a = p.parse_args(); m = json.loads(Path(a.manifest).read_text())
    if a.stage == 'score':
        score(m, a.task); return
    setup(m)
    for sig in [signal.SIGUSR1, signal.SIGTERM]: signal.signal(sig, stop)
    if a.stage == 'smoke': smoke(m)
    elif a.stage == 'real': real_features(m)
    else: evaluate(m, a.task)


if __name__ == '__main__':
    main()
