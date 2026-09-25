"""Dataset-first spatial structure analysis inspired by Singh et al. (iREPA).

LDS/CDS/RMSC characterize each image. Dataset similarity instead compares
distributions of spatially normalized patch self-similarity matrices using MMD.
No generator samples or transfer scores enter descriptor calibration.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'reports/experiment3'))
from hybrid_analysis import digest, read, resolve

BASE = ROOT/'reports/2026-09-20/experiment3_dataset_geometry'
DOMAINS = ['celeba64', 'aahq64', 'ffhq64', 'afhq_v2_64', 'stl10_64', 'imagenet64']
SEED = 20260920


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


def prepare():
    import hashlib
    from PIL import Image
    protocol = read(ROOT/'reports/2026-09-18/imagenet600_hybrid_metrics/protocol.json')
    old = read(ROOT/'reports/experiment3/archive/legacy_reports/primary_metrics_1024/real_manifest.json')['domains']
    newer = read(ROOT/'reports/2026-09-18/imagenet600_hybrid_metrics/real_manifest.json')['domains']
    banks = {}
    for domain in DOMAINS:
        roles = newer.get(domain, old.get(domain))
        calibration, evaluation = roles['test'][:128], roles['reference']
        assert len(evaluation) == 1000
        for item in calibration + evaluation:
            if 'decoded_sha256' not in item:
                with Image.open(resolve(item['path'])) as im:
                    array = np.asarray(im.convert('RGB'))
                item['decoded_sha256'] = hashlib.sha256(str(array.shape).encode()+array.tobytes()).hexdigest()
        hashes = [r['decoded_sha256'] for r in calibration + evaluation]
        assert len(set(hashes)) == len(hashes), domain
        banks[domain] = dict(calibration=calibration, evaluation=evaluation)
    manifest = dict(domains=banks, encoder={k: v for k, v in protocol.items() if k.startswith('feature_')},
                    code_sha256=digest(__file__), seed=SEED, paper='https://arxiv.org/abs/2512.10794',
                    sample_counts=dict(calibration_per_domain=128, evaluation_per_domain=1000),
                    descriptor='Final DINOv2 patch tokens; channelwise spatial mean subtraction and sample-standard-deviation division (epsilon=1e-6), then patch L2 normalization; upper off-diagonal cosine self-similarity, raster order',
                    paper_statistics='Raw patch-token LDS (0 < Manhattan distance < 8 versus >= 8), CDS (distances 1..30), RMSC',
                    dataset_metric='Unbiased Gaussian-kernel MMD squared; bandwidth squared is median squared descriptor distance among pooled calibration images; independent-image bootstrap, 2000 draws',
                    validation='Identical images and 4/8-pixel reflected horizontal translations; brightness=0.8 and saturation=0.5 appearance controls, measured on calibration images only',
                    validation_gate='Every domain: identity max < 1e-8; median shift8 > max(median brightness, median saturation); median shift8 > median shift4',
                    limitation='A fixed-grid, encoder-dependent structural proxy, not pure geometry or evidence that a particular generator component controls geometry')
    path = BASE/'manifest.json'
    if path.exists():
        assert read(path) == manifest, 'Frozen geometry manifest differs'
    else:
        write(path, manifest)
    print(path, flush=True)


def setup():
    import torch
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
    manifest = read(BASE/'manifest.json')
    assert manifest['code_sha256'] == digest(__file__), 'Geometry implementation changed'
    assert torch.cuda.is_available(), 'CUDA preflight failed; no CPU fallback'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = manifest['encoder']
    assert subprocess.check_output(['git', '-C', cfg['feature_repo'], 'rev-parse', 'HEAD'], text=True).strip() == cfg['feature_repo_commit']
    assert not subprocess.check_output(['git', '-C', cfg['feature_repo'], 'status', '--porcelain'], text=True).strip()
    assert digest(cfg['feature_weights']) == cfg['feature_weights_sha256']
    model = torch.hub.load(cfg['feature_repo'], cfg['feature_model'], source='local', pretrained=False)
    model.load_state_dict(torch.load(cfg['feature_weights'], map_location='cpu', weights_only=True), strict=True)
    model.eval().requires_grad_(False).cuda()
    transform = Compose([Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True),
                         ToTensor(), Normalize([.485, .456, .406], [.229, .224, .225])])
    return manifest, model, transform


def descriptors(model, tensor):
    import torch
    import torch.nn.functional as F
    tokens = model.forward_features(tensor)['x_norm_patchtokens'].float()
    assert tokens.shape[1:] == (256, 768)
    raw = F.normalize(tokens, dim=-1)
    kernel = raw @ raw.transpose(1, 2)
    y, x = torch.meshgrid(torch.arange(16, device=tensor.device), torch.arange(16, device=tensor.device), indexing='ij')
    pos = torch.stack((y.flatten(), x.flatten()), 1)
    distance = (pos[:, None] - pos[None]).abs().sum(-1)
    lds = kernel[:, (distance > 0) & (distance < 8)].mean(1) - kernel[:, distance >= 8].mean(1)
    correlogram = torch.stack([kernel[:, distance == d].mean(1) for d in range(1, 31)], 1)
    ds = torch.arange(1, 31, device=tensor.device, dtype=torch.float32); ds = ds - ds.mean()
    cds = -(correlogram * ds).sum(1) / ds.square().sum()
    rmsc = (1 - raw.mean(1).square().sum(1)).clamp_min(0).sqrt()
    spatial = (tokens - tokens.mean(1, keepdim=True)) / (tokens.std(1, keepdim=True) + 1e-6)
    spatial = F.normalize(spatial, dim=-1)
    spatial_kernel = spatial @ spatial.transpose(1, 2)
    triangle = torch.triu_indices(256, 256, offset=1, device=tensor.device)
    return (spatial_kernel[:, triangle[0], triangle[1]],
            kernel[:, triangle[0], triangle[1]], torch.stack((lds, cds, rmsc), 1), correlogram)


def extract(task):
    import torch
    from PIL import Image, ImageEnhance
    manifest, model, transform = setup()
    domain = DOMAINS[task]
    def images(items):
        result = []
        for item in items:
            path = resolve(item['path'])
            assert digest(path) == item['sha256']
            with Image.open(path) as im:
                result.append(im.convert('RGB').copy())
        return result
    def compute(imgs):
        outputs = [[], [], [], []]
        with torch.inference_mode():
            for start in range(0, len(imgs), 16):
                batch = torch.stack([transform(im) for im in imgs[start:start+16]]).cuda()
                for out, value in zip(outputs, descriptors(model, batch)):
                    out.append(value.cpu().numpy())
        return [np.concatenate(out) for out in outputs]
    bank = manifest['domains'][domain]
    calibration = images(bank['calibration'])
    cal, cal_raw, _, _ = compute(calibration)
    def shift(im, pixels):
        array = np.asarray(im)
        return Image.fromarray(np.pad(array, ((0, 0), (pixels, 0), (0, 0)), mode='reflect')[:, :array.shape[1]])
    transformations = dict(identity=calibration,
                           brightness=[ImageEnhance.Brightness(im).enhance(.8) for im in calibration],
                           saturation=[ImageEnhance.Color(im).enhance(.5) for im in calibration],
                           shift4=[shift(im, 4) for im in calibration],
                           shift8=[shift(im, 8) for im in calibration])
    validation = {}
    for name, imgs in transformations.items():
        transformed = compute(imgs)[0]
        validation[name] = np.mean((cal - transformed).astype(np.float64)**2, axis=1)
    structure, raw, statistics, correlogram = compute(images(bank['evaluation']))
    dest = BASE/'cache'/f'{domain}.npz'; dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, structure=structure, raw_structure=raw, calibration=cal,
                        raw_calibration=cal_raw, statistics=statistics, correlogram=correlogram,
                        manifest_sha256=digest(BASE/'manifest.json'),
                        **{f'validation_{k}': v for k, v in validation.items()})
    os.replace(tmp, dest)
    print('Complete', domain, {k: float(np.median(v)) for k, v in validation.items()}, flush=True)


def merge():
    import torch
    assert torch.cuda.is_available(), 'CUDA preflight failed; no CPU fallback'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = read(BASE/'manifest.json'); assert manifest['code_sha256'] == digest(__file__)
    arrays = {}
    for domain in DOMAINS:
        with np.load(BASE/'cache'/f'{domain}.npz') as a:
            assert str(a['manifest_sha256']) == digest(BASE/'manifest.json')
            arrays[domain] = {k: a[k] for k in a.files if k != 'manifest_sha256'}
    validations = {}
    for domain, a in arrays.items():
        vals = {k.removeprefix('validation_'): v for k, v in a.items() if k.startswith('validation_')}
        med = {k: float(np.median(v)) for k, v in vals.items()}
        passed = bool(vals['identity'].max() < 1e-8 and med['shift8'] > max(med['brightness'], med['saturation']) and med['shift8'] > med['shift4'])
        validations[domain] = dict(medians=med, passed=passed,
            shift8_exceeds_brightness_fraction=float(np.mean(vals['shift8'] > vals['brightness'])),
            shift8_exceeds_saturation_fraction=float(np.mean(vals['shift8'] > vals['saturation'])))
    rows, bandwidths = [], {}
    rng = np.random.default_rng(SEED)
    def kernel(x, y, bandwidth):
        # Mean squared difference, preserving the fixed raster-coordinate ordering.
        d = ((x*x).sum(1)[:, None] + (y*y).sum(1)[None] - 2*x@y.T).clamp_min(0) / x.shape[1]
        return torch.exp(-d / (2*bandwidth))
    def estimate(kxx, kyy, kxy):
        n, m = len(kxx), len(kyy)
        return ((kxx.sum()-kxx.diag().sum())/(n*(n-1))
                + (kyy.sum()-kyy.diag().sum())/(m*(m-1)) - 2*kxy.mean()).item()
    def ci(kxx, kyy, kxy):
        n, m = len(kxx), len(kyy)
        draws = []
        # Ordinary independent-image bootstrap; same sampled image indices
        # are used in each within-domain and cross-domain kernel term.
        for start in range(0, 2000, 100):
            cx = torch.tensor(rng.multinomial(n, np.full(n, 1/n), size=100), device='cuda', dtype=torch.float32)
            cy = torch.tensor(rng.multinomial(m, np.full(m, 1/m), size=100), device='cuda', dtype=torch.float32)
            v = ((cx@kxx*cx).sum(1)-cx@kxx.diag())/(n*(n-1))
            v += ((cy@kyy*cy).sum(1)-cy@kyy.diag())/(m*(m-1))
            v -= 2*(cx@kxy*cy).sum(1)/(n*m)
            draws.extend(v.cpu().tolist())
        # Center bootstrap estimation error at the observed statistic; an
        # uncentered percentile interval is biased upward near the MMD null.
        point = estimate(kxx, kyy, kxy)
        centered = np.asarray(draws) - np.mean(draws)
        lo, hi = np.quantile(centered, [.025, .975])
        return float(point-hi), float(point-lo)
    for variant, key, calkey in [('spatial_normalized', 'structure', 'calibration'), ('raw_tokens', 'raw_structure', 'raw_calibration')]:
        calibration = np.concatenate([a[calkey] for a in arrays.values()])
        cx = torch.tensor(calibration, device='cuda')
        d = ((cx*cx).sum(1)[:, None] + (cx*cx).sum(1)[None] - 2*cx@cx.T).clamp_min(0)/cx.shape[1]
        bandwidth = float(d[torch.triu(torch.ones_like(d, dtype=torch.bool), diagonal=1)].median().item())
        assert bandwidth > 1e-8
        bandwidths[variant] = bandwidth
        del cx, d
        values = {domain: torch.tensor(a[key], device='cuda') for domain, a in arrays.items()}
        within = {domain: kernel(x, x, bandwidth) for domain, x in values.items()}
        for i, source in enumerate(DOMAINS):
            x = values[source]
            k = within[source]
            for target in DOMAINS[i:]:
                if source == target:
                    # Independent, disjoint real-image halves, never self-vs-self.
                    kxx, kyy, kxy = k[:500, :500], k[500:, 500:], k[:500, 500:]
                    n = 500
                else:
                    kxx, kyy = k, within[target]
                    kxy = kernel(x, values[target], bandwidth); n = 1000
                value = estimate(kxx, kyy, kxy)
                lo, hi = ci(kxx, kyy, kxy)
                rows.append(dict(source=source, target=target, variant=variant, mmd2=value,
                                 ci95_low=lo, ci95_high=hi, sample_count_per_domain=n,
                                 mean_structure_mse=float(np.mean((arrays[source][key].mean(0)-arrays[target][key].mean(0))**2)) if source != target else None))
                print(variant, source, target, value, flush=True)
        del values, within
    summaries = {}
    for domain, a in arrays.items():
        summaries[domain] = dict(statistics=dict(zip(('LDS', 'CDS', 'RMSC'), a['statistics'].mean(0).astype(float))),
                                 correlogram=a['correlogram'].mean(0).astype(float).tolist())
    result = dict(protocol=manifest, manifest_sha256=digest(BASE/'manifest.json'),
                  bandwidth_squared=bandwidths, validation=validations,
                  validated=all(v['passed'] for v in validations.values()),
                  rows=rows, domain_summaries=summaries,
                  uncertainty='2000 independent-image bootstrap draws with recentered errors; descriptive fixed-encoder finite-bank intervals, not a calibrated equality test near MMD=0')
    # Keep raw image selections in the campaign manifest, not the report export.
    result['protocol'] = {k: v for k, v in manifest.items() if k != 'domains'}
    write(BASE/'results.json', result)
    destination = ROOT/'reports/experiment3/metrics/dataset_geometry.json'
    write(destination, result)
    print('Dataset geometry exported; validation:', result['validated'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'extract', 'merge'])
    parser.add_argument('--task', type=int, default=0)
    args = parser.parse_args()
    if args.stage == 'prepare': prepare()
    elif args.stage == 'extract': extract(args.task)
    else: merge()
