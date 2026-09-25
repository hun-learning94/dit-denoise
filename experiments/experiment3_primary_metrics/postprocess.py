"""Pinned feature/analysis continuation of the immutable September 11 generation run.

Generation provenance is retained verbatim; its unused placeholder encoder fields
are superseded only for postprocessing. Shared feature/probe banks use file locks.
"""
import argparse
import fcntl
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from . import pipeline as p

BASE = p.ROOT / 'reports/experiment3/primary_metrics_1024'
OUT = BASE / 'evaluation'

def prepare():
    generation = json.loads((BASE / 'manifest.json').read_text())
    config = dict(generation['protocol'])
    repo = p.ROOT / 'tools/dinov2'
    weights = p.ROOT / 'dinov2_vitb14_pretrain.pth'
    config.update(feature_repo=str(repo), feature_repo_commit=subprocess.check_output(
        ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        feature_weights=str(weights), feature_weights_sha256=p.digest(weights))
    seen, domains, skipped = set(), {}, {}
    for domain in sorted({r[d] for r in generation['rows'] for d in ['source', 'target']}):
        root = p.ROOT / 'datasets' / domain
        def ordered(split):
            return sorted(root.joinpath(split).glob('*.png'), key=lambda x: p.signature([config['latent_seed'], domain, x.name]))
        train, valid, test = ordered('train'), ordered('valid'), ordered('test')
        if not valid:
            valid, train = train[:config['real_validation_per_domain']], train[config['real_validation_per_domain']:]
        domains[domain] = {}
        # Reserve held-out observations before selecting training observations.
        for role, pool, count in [('test', test, config['real_test_per_domain']),
                ('reference', test, config['real_reference_per_domain']),
                ('validation', valid, config['real_validation_per_domain']),
                ('train', train, config['real_train_per_domain'])]:
            selected = []
            skipped[f'{domain}/{role}'] = 0
            for path in pool:
                with Image.open(path) as im:
                    arr = np.asarray(im.convert('RGB'))
                key = hashlib.sha256(str(arr.shape).encode() + arr.tobytes()).hexdigest()
                if key in seen:
                    skipped[f'{domain}/{role}'] += 1
                    continue
                seen.add(key)
                selected.append(dict(path=str(path.resolve()), sha256=p.digest(path), decoded_sha256=key))
                if len(selected) == count:
                    break
            if len(selected) != count:
                raise ValueError(f'Insufficient unique images: {domain}/{role}')
            domains[domain][role] = selected
    OUT.mkdir(exist_ok=True)
    if (OUT / 'manifest.json').exists():
        raise FileExistsError('Evaluation manifest is immutable')
    p.write_json(OUT / 'protocol.json', config)
    p.write_json(OUT / 'real_manifest.json', dict(domains=domains, protocol_hash=p.signature(config), skipped_duplicates=skipped))
    p.write_json(OUT / 'manifest.json', dict(protocol=config, generation_manifest_sha256=p.digest(BASE / 'manifest.json'),
        generation_protocol_hash=generation['protocol_hash'], implementation=p.implementation_stamp(),
        runner_sha256=p.digest(__file__), real_manifest_sha256=p.digest(OUT / 'real_manifest.json'), rows=generation['rows']))
    print('Frozen evaluation protocol and content-disjoint real banks', flush=True)

def run(task):
    import torch
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
    torch.set_num_threads(1)
    manifest = json.loads((OUT / 'manifest.json').read_text())
    config = manifest['protocol']
    assert p.digest(BASE / 'manifest.json') == manifest['generation_manifest_sha256']
    assert p.implementation_stamp() == manifest['implementation']
    assert p.digest(__file__) == manifest['runner_sha256']
    assert p.digest(OUT / 'real_manifest.json') == manifest['real_manifest_sha256']
    repo = config['feature_repo']
    assert subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip() == config['feature_repo_commit']
    assert not subprocess.check_output(['git','-C',repo,'status','--porcelain'],text=True).strip()
    assert p.digest(config['feature_weights']) == config['feature_weights_sha256']
    row = manifest['rows'][task]
    cube = json.loads((BASE / 'cache/cubes' / (row['id'] + '.json')).read_text())
    assert cube['row'] == row and cube['protocol_hash'] == manifest['generation_protocol_hash']
    model = torch.hub.load(repo, config['feature_model'], source='local', pretrained=False)
    model.load_state_dict(torch.load(config['feature_weights'], map_location='cpu', weights_only=True), strict=True)
    model.eval().requires_grad_(False).cuda()
    transform = Compose([Resize((224,224), interpolation=InterpolationMode.BICUBIC, antialias=True),
                         ToTensor(), Normalize([.485,.456,.406],[.229,.224,.225])])
    feature_protocol = dict(model=config['feature_model'], commit=config['feature_repo_commit'],
        weights=config['feature_weights_sha256'], preprocess='RGB-full224-bicubic-ImageNet-L2-CLS-fp32',
        runner=manifest['runner_sha256'], packages=manifest['implementation']['packages'])
    def compute(key, count, load):
        dest = OUT / 'shared_features' / (p.signature([key, feature_protocol]) + '.npz')
        dest.parent.mkdir(exist_ok=True)
        with dest.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if dest.exists():
                with np.load(dest) as a:
                    assert a['features'].shape == (count, 768) and np.isfinite(a['features']).all()
                return str(dest)
            features = []
            with torch.inference_mode():
                for start in range(0, count, config['feature_batch']):
                    x = torch.stack([transform(load(i)) for i in range(start, min(count, start + config['feature_batch']))]).cuda()
                    features.append(model(x).float().cpu().numpy())
            p.save_npz(dest, features=p.unit(np.concatenate(features)).astype('float32'), encoder_sha256=config['feature_weights_sha256'])
            print('Cached', dest.name, count, flush=True)
        return str(dest)
    real = json.loads((OUT / 'real_manifest.json').read_text())
    index = dict(real={}, generated={}, protocol_hash=p.signature(config))
    for domain in [row['source'], row['target']]:
        index['real'][domain] = {}
        for role, items in real['domains'][domain].items():
            def load(i):
                item = items[i]
                assert p.digest(item['path']) == item['sha256']
                with Image.open(item['path']) as im:
                    return im.convert('RGB').copy()
            index['real'][domain][role] = compute(items, len(items), load)
    for code, path in cube['images'].items():
        with np.load(path) as a:
            images = a['images']
            assert len(images) == config['images'] and str(a['latent_hash']) == cube['latent_hash']
            assert str(a['checkpoint_sha256']) == row['checkpoints'][code]['sha256']
        index['generated'][path] = compute(p.digest(path), len(images), lambda i: Image.fromarray(images[i].transpose(1,2,0)))
    del model
    torch.cuda.empty_cache()
    cache = OUT / 'tasks' / str(task)
    cube['generation_protocol_hash'] = cube['protocol_hash']
    cube['protocol_hash'] = p.signature(config)
    p.write_json(cache / 'cubes' / (row['id'] + '.json'), cube)
    p.write_json(cache / 'features/index.json', index)
    p.analyze(config, cache)
    print('COMPLETE', row['id'], flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'run'])
    parser.add_argument('--task', type=int, default=0)
    args = parser.parse_args()
    prepare() if args.stage == 'prepare' else run(args.task)
