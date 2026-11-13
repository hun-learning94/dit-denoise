"""Freeze native-resolution, disjoint adaptation memberships for the scale study."""
from __future__ import annotations
from revision.runtime import path as _release_path
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(_release_path('.'))
WORK = Path(_release_path('assets/datasets/component_transfer_matched_20261106'))

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def publish(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(path)

def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))

def face_records(name, resolution):
    folder = ROOT / 'datasets' / f'{name}{resolution}'
    ready = folder / 'READY.json'
    assert ready.exists(), f'Unvalidated dataset: {folder}'
    inputs = [ready]
    manifest = folder / 'manifest.csv'
    if manifest.exists():
        records = rows(manifest); inputs.append(manifest)
    else:
        records = []
        for split in ('train', 'valid', 'test'):
            p = folder / f'{split}_manifest.csv'
            if p.exists():
                records += rows(p); inputs.append(p)
    assert records
    if name == 'aahq' and resolution == 128:
        # The later 256 preparation audited the original sides and effective
        # artwork alignment crop, including excluded and duplicate records.
        audit = ROOT / 'datasets/aahq256'
        native = {}
        for p in audit.glob('*_manifest.csv'):
            inputs.append(p)
            native.update({r['source_file']: r for r in rows(p)})
        for n in ('excluded_native_resolution.json', 'excluded_duplicates.json'):
            p = audit / n; inputs.append(p)
            native.update({r['source_file']: r for r in json.loads(p.read_text())})
        for r in records:
            a = native[r['source_file']]
            r.update({k: a[k] for k in ('original_width', 'original_height', 'native_crop_extent')})
    excluded = []
    good = []
    seen = {}
    for r in sorted(records, key=lambda r: ({'test':0, 'valid':1, 'train':2}[r['split']], r['output_file'])):
        if name == 'ffhq' and resolution == 128:
            native_ok = True  # Official native 128 thumbnails, verified by preparation.
        elif name == 'afhq_v2_':
            native_ok = True  # Official 512 originals; preparation asserts shape.
        else:
            native_ok = min(int(r['original_width']), int(r['original_height'])) >= resolution
            if name == 'aahq': native_ok &= float(r['native_crop_extent']) >= resolution
        h = r['pixel_sha256']
        if not native_ok:
            excluded.append(dict(r, reason='insufficient_native_resolution')); continue
        if h in seen and r['split'] != 'test':
            excluded.append(dict(r, reason='pixel_duplicate', matches=seen[h])); continue
        seen.setdefault(h, r['split'] + '/' + r['output_file'])
        r = dict(r, path=str(folder / r['split'] / r['output_file']))
        good.append(r)
    if name == 'afhq_v2_':
        groups = defaultdict(list)
        for r in good:
            if r['split'] == 'train': groups[r['category']].append(r)
        for group in groups.values():
            ordered = sorted(group, key=lambda r: hashlib.sha256(('val-seed0:' + r['source_file']).encode()).hexdigest())
            for r in ordered[:max(1, round(len(group) * .05))]: r['split'] = 'valid'
    split_records = {s: sorted([r for r in good if r['split'] == s], key=lambda r:r['path']) for s in ('train','valid','test')}
    assert all(split_records.values())
    hashes = {s: {r['pixel_sha256'] for r in rs} for s, rs in split_records.items()}
    assert not (hashes['train'] & hashes['valid'] or hashes['train'] & hashes['test'] or hashes['valid'] & hashes['test'])
    return split_records, excluded, inputs

def native_size(item):
    index, rel = item
    with Image.open(_release_path('assets/external/train') + rel) as im:
        w, h = im.size
    return index, rel.split('/')[0], w, h

def prepare(name, resolution, output, workers):
    destination = WORK / f'{name}{resolution}'
    destination.mkdir(parents=True, exist_ok=True)
    if name != 'imagenet':
        records, excluded, inputs = face_records(name, resolution)
        files = {}
        for split, rr in records.items():
            p = destination / f'{split}.txt'
            p.write_text(''.join(r['path'] + '\n' for r in rr))
            files[split] = dict(path=str(p), sha256=digest(p), count=len(rr))
        publish(destination / 'excluded.json', excluded)
        result = dict(kind='images', resolution=resolution, splits=files,
                      excluded_count=len(excluded), native_resolution_verified=True,
                      cross_split_pixel_duplicates=0,
                      split_policy='Inherited splits; AFHQ reserves deterministic stratified 5% of training for validation; pixel duplicate holdout precedence',
                      inputs={str(p):digest(p) for p in inputs})
    else:
        assert resolution == 128
        index = ROOT / 'datasets/imagenet_source/train_paths.txt'
        assert digest(index) == 'f69e51dbbcd0c65ef3257ceb0f621d8068adfa5754089a72a9e4f0329771856c'
        cache = ROOT / 'datasets/complementary_pixel_cache/imagenet128-28e494915e45f595'
        paths = index.read_text().splitlines()
        shape = np.load(cache / 'images.npy', mmap_mode='r').shape
        assert shape == (len(paths), 128, 128, 3)
        groups, excluded = defaultdict(list), []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for done, (i, cls, w, h) in enumerate(pool.map(native_size, enumerate(paths), chunksize=256), 1):
                if min(w,h) < 128: excluded.append(dict(index=i,path=paths[i],width=w,height=h))
                else: groups[cls].append(i)
                if done % 100000 == 0: print(json.dumps(dict(native_audited=done,total=len(paths))),flush=True)
        memberships = dict(train=[], valid=[], test=[])
        for cls, indices in sorted(groups.items()):
            ordered = sorted(indices, key=lambda i: hashlib.sha256(('split-seed0:' + paths[i]).encode()).hexdigest())
            assert len(ordered) > 10
            memberships['valid'] += ordered[:5]
            memberships['test'] += ordered[5:10]
            memberships['train'] += ordered[10:]
        files = {}
        for split, indices in memberships.items():
            p = destination / f'{split}_indices.npy'
            np.save(p, np.array(sorted(indices), dtype=np.int64))
            files[split] = dict(path=str(p),sha256=digest(p),count=len(indices))
        publish(destination / 'excluded_native_resolution.json', excluded)
        result = dict(kind='mmap', resolution=128, splits=files, images=str(cache/'images.npy'),
                      excluded_count=len(excluded), native_resolution_verified=True,
                      split_policy='Deterministic 5 validation and 5 test images per class held out from official training; adaptation uses remainder. Official validation is not available locally.',
                      inputs={str(index):digest(index),str(cache/'READY.json'):digest(cache/'READY.json')})
    result.update(dataset=f'{name}{resolution}', seed=0, preparation_sha256=digest(__file__),job_id=os.getenv('SLURM_JOB_ID'))
    publish(destination / 'READY.json',result)
    publish(output,result)
    print(json.dumps(result),flush=True)

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--resolution',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();prepare(a.dataset,a.resolution,a.output,a.workers)
