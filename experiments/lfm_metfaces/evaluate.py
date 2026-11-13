"""Native LFM Heun sampling, durable 64-image features, CPU KID/FID scoring."""
import argparse,gc,json,time,os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from experiments.component_transfer_scale.model import load_source,plain_model
from experiments.component_transfer_scale.runner import vae_for
from experiments.uvit_metfaces.evaluate import Features,metrics
from . import common as c
from .cache import Cache,read_images
from .train import denoise

@torch.no_grad()
def sample(net,vae,seed,shard,count):
 x=torch.randn(count,4,32,32,generator=torch.Generator().manual_seed(seed+1000003*shard)).cuda()
 times=torch.linspace(1,0,51,device='cuda')
 def predict(x,t):
  with torch.autocast('cuda',dtype=torch.bfloat16):v=net(t.expand(len(x)),x)
  return v.float()
 for a,b in zip(times[:-1],times[1:]):
  dt=b-a;v=predict(x,a);v2=predict(x+dt*v,b);x=x+dt*(v+v2)/2
  assert torch.isfinite(x).all(),'Nonfinite trajectory'
 out=[]
 for part in x.split(2):out.append(vae.decode(part/.18215).sample)
 result=torch.cat(out);assert torch.isfinite(result).all()
 return ((result.clamp(-1,1)+1)*127.5).round().byte()

def checkpoint(m,arm,endpoint):
 net,source,_=load_source(m['source']);step=0;sha=m['source']['checkpoint_sha256'];path=m['source']['checkpoint']
 if arm!='source':
  folder=Path(m['runs'][arm]);assert (folder/'COMPLETE.json').exists()
  path=str(folder/f'{endpoint}.pt');p=torch.load(path,map_location='cpu',weights_only=False)
  assert p['identity']==c.identity(m,arm) and p['source_sha256']==m['source']['checkpoint_sha256']
  state=dict(source) if p['ema_is_partial'] else {};state.update(p['ema']);assert set(state)==set(source)
  net.load_state_dict(state,strict=True);step=p['step'];sha=c.digest(path)
 return net.cuda().eval().requires_grad_(False),dict(checkpoint=path,checkpoint_sha256=sha,epoch=step/33,arm=arm,endpoint=endpoint)

@torch.no_grad()
def real(m):
 base=Path(m['eval_output']);base.mkdir(parents=True,exist_ok=True);ready=base/'REAL_READY.json'
 if ready.exists():
  r=json.loads(ready.read_text());assert r['data_sha256']==m['data_sha256'] and r['feature_weights_sha256']==m['feature_weights_sha256']
  for name,h in r['files'].items():assert c.digest(base/name)==h
  return
 feature=Features(m);data=json.loads(Path(m['data']).read_text());files={}
 for split in ['train','test']:
  rec=data['splits'][split];assert c.digest(rec['path'])==rec['sha256'];paths=Path(rec['path']).read_text().splitlines();parts=[]
  folder=base/'real_shards'/split;folder.mkdir(parents=True,exist_ok=True)
  for first in range(0,len(paths),32):
   p=folder/f'{first:04d}.npy'
   if not p.exists():c.npy(p,feature((read_images(paths[first:first+32])+1)/2).cpu().numpy())
   value=np.load(p);assert value.shape==(min(32,len(paths)-first),2048) and np.isfinite(value).all();parts.append(value)
   if c.STOP:raise SystemExit(75)
  p=base/f'real_{split}.npy';c.npy(p,np.concatenate(parts));files[p.name]=c.digest(p)
 c.write(ready,dict(data_sha256=m['data_sha256'],feature_weights_sha256=m['feature_weights_sha256'],files=files))

@torch.no_grad()
def generate(m,task):
 c.check_pass(m);row=m['evaluation_tasks'][task];out=Path(m['eval_output'])/row['label'];out.mkdir(parents=True,exist_ok=True)
 if (out/'FEATURES_READY.json').exists():return
 if row['endpoint']=='final' and json.loads((Path(m['runs'][row['arm']])/'COMPLETE.json').read_text())['best_epoch']==1000:
  other=Path(m['eval_output'])/f"{row['arm']}_best_seed{row['seed']}";old=json.loads((other/'FEATURES_READY.json').read_text());folder=Path(m['runs'][row['arm']])
  a=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);b=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
  assert a['step']==b['step']==33000 and a['identity']==b['identity'] and set(a['ema'])==set(b['ema'])
  assert all(torch.equal(a['ema'][k],b['ema'][k]) for k in a['ema']), 'Endpoint alias requires exact EMA tensor equality'
  for name,h in old['files'].items():
   assert c.digest(other/name)==h
   if not (out/name).exists():os.link(other/name,out/name)
  if not (out/'examples.npy').exists():os.link(other/'examples.npy',out/'examples.npy')
  old.update(checkpoint=str(folder/'final.pt'),checkpoint_sha256=c.digest(folder/'final.pt'),endpoint='final',reused_after_exact_ema_equality=str(other))
  c.write(out/'FEATURES_READY.json',old);return
 net,provenance=checkpoint(m,row['arm'],row['endpoint']);vae=vae_for(m['source']);feature=Features(m)
 ident=dict(**provenance,manifest_sha256=c.digest(m['_manifest']),seed=row['seed'],sampler=m['evaluation'],feature_weights_sha256=m['feature_weights_sha256'])
 meta=out/'identity.json'
 if meta.exists():assert json.loads(meta.read_text())==ident
 else:c.write(meta,ident)
 start=time.monotonic();files={}
 for shard,first in enumerate(range(0,5000,64)):
  p=out/f'features_{shard:03d}.npy';count=min(64,5000-first)
  if not p.exists():
   x=sample(net,vae,row['seed'],shard,count);f=feature(x.float()/255).cpu().numpy();assert f.shape==(count,2048) and np.isfinite(f).all()
   if shard==0:c.npy(out/'examples.npy',x[:32].cpu().numpy())
   c.npy(p,f);print('FEATURE_SHARD',row['label'],shard,'elapsed',time.monotonic()-start,flush=True)
  f=np.load(p);assert f.shape==(count,2048) and np.isfinite(f).all();files[p.name]=c.digest(p)
  if c.STOP:raise SystemExit(75)
 c.write(out/'FEATURES_READY.json',dict(**ident,files=files,images=5000,seconds_this_invocation=time.monotonic()-start))

@torch.no_grad()
def losses(m,task):
 row=m['loss_tasks'][task];base=Path(m['eval_output'])/'losses';base.mkdir(parents=True,exist_ok=True);out=base/f"{row['arm']}_{row['endpoint']}.json"
 if out.exists():return
 net,provenance=checkpoint(m,row['arm'],row['endpoint']);cache=Cache(m,source=True)
 records={}
 for group,indices,seed in [('source',list(range(1024)),2026111003),('target',cache.splits['test'],2026111004)]:
  shard=base/f"{row['arm']}_{row['endpoint']}_{group}.json"
  if not shard.exists():c.write(shard,dict(**provenance,group=group,metric=denoise(net,cache,indices,seed,4,group)))
  records[group]=json.loads(shard.read_text())
  if c.STOP:raise SystemExit(75)
 c.write(out,dict(**provenance,losses=records))

@torch.no_grad()
def smoke(m):
 net,_,_=load_source(m['source']);net.cuda().eval().requires_grad_(False);vae=vae_for(m['source']);feature=Features(m)
 torch.cuda.reset_peak_memory_stats();start=time.monotonic();a=sample(net,vae,202611091,0,64);f=feature(a.float()/255)
 assert f.shape==(64,2048) and torch.isfinite(f).all();torch.cuda.synchronize();seconds=time.monotonic()-start
 # Repeat a complete production shard: tests native sampler, VAE and feature extraction.
 b=sample(net,vae,202611091,0,64);assert torch.equal(a,b)
 c.write(Path(m['report'])/'preflight'/'evaluation.json',dict(passed=True,manifest_sha256=c.digest(m['_manifest']),production_shard_images=64,seconds=seconds,peak_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),repeat_equal=True,finite_features=True,gpu=torch.cuda.get_device_name()))
 print('SAMPLING_PREFLIGHT_PASS',seconds,flush=True)

def score(m,task):
 row=m['evaluation_tasks'][task];base=Path(m['eval_output']);out=base/row['label'];r=json.loads((out/'FEATURES_READY.json').read_text());assert r['manifest_sha256']==c.digest(m['_manifest'])
 real_info=json.loads((base/'REAL_READY.json').read_text());assert real_info['data_sha256']==m['data_sha256']
 for name,h in real_info['files'].items():assert c.digest(base/name)==h
 arrays=[]
 for name,h in sorted(r['files'].items()):assert c.digest(out/name)==h;arrays.append(np.load(out/name))
 gen=np.concatenate(arrays);assert len(gen)==5000;real=np.load(base/'real_test.npy');assert len(real)==143
 result=dict(task=row,provenance=r,**metrics(real,gen))
 # Feature-space nearest training image for each of the first 32 fixed samples.
 train=np.load(base/'real_train.npy');d=((gen[:32,None,:]-train[None,:,:])**2).mean(-1);result['nearest_train']=[dict(generated=i,train_index=int(j),feature_mse=float(d[i,j])) for i,j in enumerate(d.argmin(1))]
 c.write(out/'scores.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('provenance','nearest_train')}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--mode',choices=['real','generate','loss','smoke','score'],required=True);p.add_argument('--task',type=int,default=0);a=p.parse_args()
 m=c.setup(a.manifest,gpu=a.mode!='score');m['_manifest']=a.manifest
 if a.mode=='real':real(m)
 elif a.mode=='smoke':smoke(m)
 elif a.mode=='generate':generate(m,a.task)
 elif a.mode=='loss':losses(m,a.task)
 else:score(m,a.task)
