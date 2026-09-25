"""Stream verified NPZ archives into reusable NPY memory maps; keep official splits."""
import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', type=Path, default=Path('datasets/imagenet64/raw'))
    p.add_argument('--out', type=Path, default=Path('datasets/imagenet64/prepared'))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = dict(variant='Chrabaszcz ImageNet64 NPZ', layout='uint8 N x (3*64*64), channel-first',
                    splits={'train': [], 'validation': []}, archives=[])
    for suffix, split in [('train_part1', 'train'), ('train_part2', 'train'), ('val', 'validation')]:
        archive = args.raw / f'Imagenet64_{suffix}_npz.zip'
        verified = json.loads(archive.with_suffix('.verified.json').read_text())
        assert archive.stat().st_size == verified['bytes']
        assert sha256(archive) == verified['sha256'], f'Archive changed: {archive}'
        manifest['archives'].append(verified)
        with zipfile.ZipFile(archive) as outer:
            members = sorted(n for n in outer.namelist() if n.endswith('.npz'))
            assert members
            for member in members:
                dest = args.out / f'{suffix}_{Path(member).stem}.npy'
                if not dest.exists():
                    # Stream nested ZIP then its data.npy; no large ndarray allocation.
                    with tempfile.TemporaryDirectory(dir=args.out) as tmp:
                        inner_path = Path(tmp) / 'batch.npz'
                        with outer.open(member) as src, inner_path.open('wb') as dst:
                            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
                        with zipfile.ZipFile(inner_path) as inner:
                            assert 'data.npy' in inner.namelist()
                            staged = Path(tmp) / 'data.npy'
                            with inner.open('data.npy') as src, staged.open('wb') as dst:
                                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
                            a = np.load(staged, mmap_mode='r', allow_pickle=False)
                            assert a.dtype == np.uint8 and a.ndim == 2 and a.shape[1] == 12288
                            del a
                            staged.replace(dest)
                a = np.load(dest, mmap_mode='r', allow_pickle=False)
                assert a.dtype == np.uint8 and a.ndim == 2 and a.shape[1] == 12288
                manifest['splits'][split].append(dict(path=dest.name, count=len(a), sha256=sha256(dest)))
                print(split, dest.name, len(a), flush=True)
    assert sum(s['count'] for s in manifest['splits']['train']) == 1281167
    assert sum(s['count'] for s in manifest['splits']['validation']) == 50000
    staged = args.out / 'manifest.json.tmp'
    staged.write_text(json.dumps(manifest, indent=2) + '\n')
    staged.replace(args.out / 'manifest.json')


if __name__ == '__main__':
    main()
