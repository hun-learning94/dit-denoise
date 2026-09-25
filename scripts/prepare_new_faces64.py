"""Prepare official face archives without extracting large originals to disk."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile
from collections import Counter
from PIL import Image

RESOLUTION = 64

def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()

def save_json(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    temp.replace(path)

def prepare_afhq():
    raw = Path('datasets/raw/afhq_v2/afhq_v2.zip')
    out = Path('datasets/afhq_v2_64')
    out.mkdir(parents=True, exist_ok=True)
    counts = Counter(); original = Counter(); seen = set(); records = []
    with zipfile.ZipFile(raw) as z:
        entries = []
        for name in z.namelist():
            parts = Path(name).parts
            if name.startswith('__MACOSX') or Path(name).suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                continue
            split = next((s for s in ('train', 'val', 'test') if s in parts), None)
            if split is None:
                raise ValueError(f'Unrecognized image layout: {name}')
            category = parts[parts.index(split)+1]
            assert category in ('cat','dog','wild'), name
            original[f'{split}/{category}'] += 1
            # Keep official holdout as test; reserve deterministic 5% by category.
            output_split = 'test' if split in ('val', 'test') else 'train'
            entries.append((output_split, name, category))
        assert len(entries) == 15803, (len(entries), original)
        # Test takes precedence if duplicate pixels occur across partitions.
        entries.sort(key=lambda x: ({'test':0,'valid':1,'train':2}[x[0]],x[1]))
        for split, name, category in entries:
            with Image.open(io.BytesIO(z.read(name))) as im:
                assert im.size == (512,512), (name,im.size)
                im = im.convert('RGB').resize((64,64), Image.Resampling.LANCZOS)
            pixel_hash=hashlib.sha256(im.tobytes()).hexdigest()
            if pixel_hash in seen:
                records.append(dict(source=name, excluded='duplicate_64px_pixels',pixel_sha256=pixel_hash));continue
            seen.add(pixel_hash)
            dest=out/split/(category+'_'+digest(name)[:24]+'.png');dest.parent.mkdir(exist_ok=True)
            im.save(dest.with_suffix('.tmp'),format='PNG');dest.with_suffix('.tmp').replace(dest)
            records.append(dict(source=name,split=split,category=category,output=str(dest),pixel_sha256=pixel_hash))
            counts[split]+=1
            if len(records)%1000==0:print('processed',len(records),dict(counts),flush=True)
    for split in ('train','test'):
        assert counts[split]>0
        assert sum(1 for _ in (out/split).glob('*.png'))==counts[split]
    save_json(out/'manifest.json',records)
    save_json(out/'provenance.json',dict(dataset='AFHQ-v2',source='https://github.com/clovaai/stargan-v2',raw_images=sum(original.values()),original_counts=dict(original),counts=dict(counts),resolution=64,resize='Pillow Lanczos direct 512->64',split_policy='Official test (val in older releases) retained as test; No validation split; exact 64px duplicates removed with test precedence.',primary_target='pooled cat/dog/wild',archive_bytes=raw.stat().st_size))
    print('Prepared',out,dict(counts),flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('dataset',choices=['afhq_v2','icartoonface']);args=parser.parse_args()
    if args.dataset=='afhq_v2':prepare_afhq()
    else:
        from prepare_icartoonface64 import prepare
        prepare()
