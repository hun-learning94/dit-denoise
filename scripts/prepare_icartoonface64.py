"""Canonical annotation-based iCartoonFace recognition crops, with source provenance."""
import hashlib
import io
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import Counter
import zipfile
import numpy as np
from PIL import Image
from prepare_new_faces64 import digest, save_json


def crop_face(im, box):
    x1,y1,x2,y2=box
    w,h=im.size
    x1=max(0,x1); y1=max(0,y1); x2=min(w,x2); y2=min(h,y2)
    if not (x1 < x2 and y1 < y2):
        raise ValueError(f'Invalid bounding box {box} for {im.size}')
    side=math.ceil(1.2*max(x2-x1,y2-y1));cx=(x1+x2)/2;cy=(y1+y2)/2
    left=round(cx-side/2);top=round(cy-side/2)
    # Shift a square into bounds where possible; edge-pad only if unavoidable.
    if side<=w:left=max(0,min(left,w-side))
    if side<=h:top=max(0,min(top,h-side))
    right=left+side;bottom=top+side
    pads=(max(0,-left),max(0,-top),max(0,right-w),max(0,bottom-h))
    if any(pads):
        arr=np.asarray(im)
        im=Image.fromarray(np.pad(arr,((pads[1],pads[3]),(pads[0],pads[2]),(0,0)),mode='edge'))
        left+=pads[0];top+=pads[1]
    return im.crop((left,top,left+side,top+side)).resize((64,64),Image.Resampling.LANCZOS)


def annotations(text):
    result={}
    for line in text.splitlines():
        fields=line.strip().split()
        if not fields:continue
        assert len(fields) in (5,6), fields
        name=Path(fields[0]).name
        assert name not in result, name
        result[name]=tuple(map(int,fields[1:5]))
    return result


def prepare():
    raw=Path('datasets/raw/icartoonface');out=Path('datasets/icartoonface64');out.mkdir(parents=True,exist_ok=True)
    counts=Counter();excluded=Counter();original=Counter();seen=set();records=[]
    workers=max(1,int(os.environ.get("SLURM_CPUS_PER_TASK","4"))); reused=0; started=time.monotonic()
    print("Preparation workers",workers,flush=True)
    for official in ('test','train'):
        archive=raw/f'personai_icartoonface_rec{official}.zip'
        with zipfile.ZipFile(archive) as z:
            names=[n for n in z.namelist() if Path(n).suffix.lower() in ('.jpg','.jpeg','.png') and not n.startswith('__MACOSX')]
            assert len(names)==(389678 if official=='train' else 22500), (official,len(names))
            meta=[n for n in z.namelist() if n.endswith(f'icartoonface_rec{official}_det.txt')]
            assert len(meta)==1,meta
            boxes=annotations(z.read(meta[0]).decode())
            original[official]=len(names)
            entries=[]
            for name in names:
                ident=Path(name).parent.name
                split=official
                entries.append((split,name,ident))
            entries.sort(key=lambda x:({'test':0,'valid':1,'train':2}[x[0]],x[1]))
            def decode(entry):
                split,name,ident=entry
                base=dict(source=name,archive=archive.name,official_split=official,identity=ident if official=='train' else None)
                if Path(name).name not in boxes:raise ValueError(f'Missing bbox: {name}')
                box=boxes[Path(name).name];content=z.read(name)
                try:
                    with Image.open(io.BytesIO(content)) as image:
                        size=image.size;im=crop_face(image.convert('RGB'),box)
                    sha=hashlib.sha256(im.tobytes()).hexdigest()
                    return base,split,im,sha,box,size,None
                except (OSError,ValueError) as error:
                    return base,split,None,None,box,None,str(error)
            def write_image(im,dest,sha):
                if dest.exists():
                    try:
                        with Image.open(dest) as previous:
                            if previous.mode=='RGB' and previous.size==(64,64) and hashlib.sha256(previous.tobytes()).hexdigest()==sha:
                                return 1
                    except OSError:
                        pass
                im.save(dest.with_suffix('.tmp'),format='PNG');dest.with_suffix('.tmp').replace(dest)
                return 0
            for split in ('train','test'):(out/split).mkdir(exist_ok=True)
            # Bounded batches keep memory modest; read/crop and PNG writes overlap
            # across the allocated workers. Deduplication remains ordered and deterministic.
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for start in range(0,len(entries),128):
                    decoded=list(pool.map(decode,entries[start:start+128]));writes=[]
                    for base,split,im,sha,box,size,error in decoded:
                        if error:
                            excluded['invalid_image_or_bbox']+=1
                            records.append(dict(base,excluded='invalid_image_or_bbox',error=error,bbox=box));continue
                        if sha in seen:
                            excluded['duplicate_64px_pixels']+=1
                            records.append(dict(base,excluded='duplicate_64px_pixels',pixel_sha256=sha));continue
                        seen.add(sha);dest=out/split/(digest(base['source'])[:32]+'.png')
                        writes.append(pool.submit(write_image,im,dest,sha));counts[split]+=1
                        records.append(dict(base,split=split,output=str(dest),bbox=box,source_size=size,pixel_sha256=sha))
                    for future in writes:reused+=future.result()
                    if start%5120==0:print('processed',len(records),dict(counts),dict(excluded),'reused',reused,'elapsed_s',round(time.monotonic()-started,1),flush=True)
    assert excluded['invalid_image_or_bbox']<0.01*sum(original.values()),excluded
    assert set(counts)=={'train','test'},counts
    save_json(out/'manifest.json',records)
    save_json(out/'provenance.json',dict(dataset='iCartoonFace recognition',source='https://github.com/luxiangju-PersonAI/iCartoonFace',original_counts=dict(original),counts=dict(counts),excluded=dict(excluded),resolution=64,resize='Pillow Lanczos',crop='Official bbox clipped to image bounds; square centered on clipped bbox, side 1.2*max(width,height); shift to fit, edge-pad only when necessary; no landmark alignment',split_policy='Official test retained. No validation split. Exact duplicate 64px pixels removed with test precedence. Official test includes identities also present in training; not an identity-disjoint test.',near_duplicate_audit='Exact pixel deduplication only; visually similar frames can remain.',archives={p.name:p.stat().st_size for p in raw.glob('*.zip')}))
    print('Prepared',out,dict(counts),dict(excluded),flush=True)
