"""Resumable FP32 VAE posterior cache; never cache one fixed sampled latent."""
import argparse,gc,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from experiments.component_transfer_scale.runner import vae_for
from . import common as c
def read_images(paths):
 a=[]
 for p in paths:
  with Image.open(p) as im:
   assert im.mode=='RGB' and im.size==(256,256),(p,im.mode,im.size)
   a.append(np.array(im,copy=True))
 return torch.from_numpy(np.stack(a)).permute(0,3,1,2).cuda().float()/127.5-1
def prepare(m):
 base=Path(m['cache']);base.mkdir(parents=True,exist_ok=True)
 if (base/'READY.json').exists():verify(m);return
 vae=vae_for(m['source']);data=json.loads(Path(m['data']).read_text())
 groups={};offset=0;splits={}
 for split in ['train','valid','test']:
  rec=data['splits'][split];assert c.digest(rec['path'])==rec['sha256'];rows=Path(rec['path']).read_text().splitlines();assert len(rows)==rec['count']
  splits[split]=list(range(offset,offset+len(rows)));offset+=len(rows);groups.setdefault('target',[]).extend(rows)
 groups['source']=Path(m['source_view']['path']).read_text().splitlines();assert c.digest(m['source_view']['path'])==m['source_view']['sha256']
 records={};checks=[]
 with torch.inference_mode():
  for group,paths in groups.items():
   orient=2 if group=='target' else 1;folder=base/group;folder.mkdir(exist_ok=True)
   for first in range(0,len(paths),32):
    path=folder/f'posterior_{first:04d}.npz'
    if not path.exists():
     x=read_images(paths[first:first+32]);x=torch.stack([x,x.flip(-1)],1).flatten(0,1) if orient==2 else x
     means=[];logs=[]
     for part in x.split(2):
      post=vae.encode(part).latent_dist;means.append(post.mean.float().cpu().numpy());logs.append(post.logvar.float().cpu().numpy())
     mean=np.concatenate(means).reshape(-1,orient,4,32,32);logvar=np.concatenate(logs).reshape(mean.shape)
     assert np.isfinite(mean).all() and np.isfinite(logvar).all();c.npz(path,mean=mean,logvar=logvar,paths=np.array(paths[first:first+32]),identity=json.dumps(m['cache_identity'],sort_keys=True))
     print('POSTERIOR_SHARD',group,first,flush=True)
    if c.STOP:raise SystemExit(75)
   means=[];logs=[]
   for first in range(0,len(paths),32):
    with np.load(folder/f'posterior_{first:04d}.npz') as z:
     assert json.loads(str(z['identity']))==m['cache_identity'];assert z['paths'].tolist()==paths[first:first+32]
     means.append(z['mean']);logs.append(z['logvar'])
   mean=np.concatenate(means);logvar=np.concatenate(logs);assert mean.shape==(len(paths),orient,4,32,32)
   # Re-encode both original/flipped inputs and compare to the saved posterior.
   x=read_images(paths[:2]);x=torch.stack([x,x.flip(-1)],1).flatten(0,1) if orient==2 else x
   pos=0
   for part in x.split(2):
    post=vae.encode(part).latent_dist
    expected=torch.from_numpy(mean[:2].reshape(-1,4,32,32)[pos:pos+len(part)]).cuda();torch.testing.assert_close(post.mean,expected,rtol=1e-6,atol=1e-6)
    expected_log=torch.from_numpy(logvar[:2].reshape(-1,4,32,32)[pos:pos+len(part)]).cuda();torch.testing.assert_close(post.logvar,expected_log,rtol=1e-6,atol=1e-6);pos+=len(part)
   checks.append(dict(group=group,posterior_reencode=True,orientations=orient))
   for name,value in [('mean',mean),('logvar',logvar)]:
    p=base/f'{group}_{name}.npy';c.npy(p,value);records[p.name]=dict(sha256=c.digest(p),shape=list(value.shape),bytes=p.stat().st_size)
 c.write(base/'READY.json',dict(identity=m['cache_identity'],files=records,splits=splits,checks=checks));del vae;gc.collect();torch.cuda.empty_cache()
def verify(m):
 base=Path(m['cache']);r=json.loads((base/'READY.json').read_text());assert r['identity']==m['cache_identity']
 for name,rec in r['files'].items():assert c.digest(base/name)==rec['sha256'],name
 return r
class Cache:
 def __init__(self,m,source=False):
  self.ready=verify(m);self.splits=self.ready['splits'];base=Path(m['cache']);groups=['target','source'] if source else ['target'];self.values={}
  for group in groups:
   self.values[group]=[torch.from_numpy(np.load(base/f'{group}_{key}.npy')).cuda() for key in ['mean','logvar']]
 def draw(self,indices,group='target',flips=None,generator=None):
  idx=torch.as_tensor(indices,device='cuda');orientation=torch.zeros(len(idx),dtype=torch.long,device='cuda') if flips is None else flips.long()
  mean,logvar=[a[idx,orientation] for a in self.values[group]]
  noise=torch.randn(mean.shape,device='cuda',generator=generator)
  return (mean+torch.exp(.5*logvar)*noise)*.18215
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);a=p.parse_args();prepare(c.setup(a.manifest))
