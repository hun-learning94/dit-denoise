"""Atomic task artifacts and frozen CPU image membership."""
import hashlib,json,os
from pathlib import Path
import numpy as np
from PIL import Image

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.tmp-{os.getpid()}')
    t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def save_npz(p,**x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.tmp-{os.getpid()}')
    with t.open('wb') as f:np.savez(f,**x)
    t.replace(p)
def images(view,rows):
    result=[]
    for path in rows:
        with Image.open(path) as im:
            im=im.convert('RGB')
            if view['transform']=='uvit_celeba':
                assert im.size==(178,218);im=im.crop((25,57,153,185)).resize((64,64),Image.Resampling.BILINEAR)
            else:assert im.size==(view['resolution'],)*2,(path,im.size)
            result.append(np.asarray(im,dtype=np.uint8).copy())
    return np.stack(result)
def view_rows(view,split):
    r=view[split];assert digest(r['path'])==r['sha256'];rows=Path(r['path']).read_text().splitlines();assert len(rows)==r['count'];return rows
