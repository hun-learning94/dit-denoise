"""Post-hoc density calibration; leaves the frozen primary campaign unchanged."""
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
from . import pipeline as p
BASE=p.ROOT/'reports/experiment3/primary_metrics_1024/evaluation'
OUT=BASE.parent/'density_checks'
DOMAINS=['aahq64','ffhq64','afhq_v2_64','stl10_64']
KS=[5,10,20,50]
def readfeatures(path):
 with np.load(path) as z:return p.unit(z['features'])
def prepare():
 real=json.loads((BASE/'real_manifest.json').read_text());m=json.loads((BASE/'manifest.json').read_text());out={}
 for domain in DOMAINS:
  roles=real['domains'][domain];seen={i['decoded_sha256'] for r in ['test','reference'] for i in roles[r]};extra=[]
  paths=sorted((p.ROOT/'datasets'/domain/'test').glob('*.png'),key=lambda x:p.signature([m['protocol']['latent_seed'],domain,x.name]))
  from PIL import Image
  for path in paths:
   with Image.open(path) as im:a=np.asarray(im.convert('RGB'))
   sha=hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()
   if sha in seen:continue
   seen.add(sha);extra.append(dict(path=str(path.resolve()),sha256=p.digest(path),decoded_sha256=sha))
   if len(extra)==2000:break
  n=1000+len(extra);sizes=sorted({1000,*[k for k in [1500,2000,3000] if k<=n],n})
  out[domain]=dict(extra=extra,reference_sizes=sizes,real_query_count=len(roles['test']))
 OUT.mkdir(exist_ok=True)
 dest=OUT/'manifest.json'
 if dest.exists():raise FileExistsError(dest)
 p.write_json(dest,dict(parent_manifest_sha256=p.digest(BASE/'manifest.json'),domains=out,ks=KS,code_sha256=p.digest(__file__)))
 print({d:dict(extra=len(v['extra']),sizes=v['reference_sizes']) for d,v in out.items()},flush=True)
def run(task):
 import torch
 from PIL import Image
 from torchvision.transforms import Compose,Resize,ToTensor,Normalize,InterpolationMode
 torch.set_num_threads(1)
 m=json.loads((BASE/'manifest.json').read_text());config=m['protocol'];checks=json.loads((OUT/'manifest.json').read_text())
 assert p.digest(BASE/'manifest.json')==checks['parent_manifest_sha256'] and p.digest(__file__)==checks['code_sha256']
 assert p.implementation_stamp()==m['implementation']
 domain=DOMAINS[task];spec=checks['domains'][domain]
 selected=[(i,r) for i,r in enumerate(m['rows']) if r['target']==domain]
 index=json.loads((BASE/'tasks'/str(selected[0][0])/'features/index.json').read_text())
 query=readfeatures(index['real'][domain]['test']);original=readfeatures(index['real'][domain]['reference'])
 extra_path=OUT/f'{domain}_extra_features.npz'
 if not extra_path.exists():
  import subprocess
  assert subprocess.check_output(['git','-C',config['feature_repo'],'rev-parse','HEAD'],text=True).strip()==config['feature_repo_commit']
  assert not subprocess.check_output(['git','-C',config['feature_repo'],'status','--porcelain'],text=True).strip()
  assert p.digest(config['feature_weights'])==config['feature_weights_sha256']
  model=torch.hub.load(config['feature_repo'],config['feature_model'],source='local',pretrained=False)
  model.load_state_dict(torch.load(config['feature_weights'],map_location='cpu',weights_only=True),strict=True)
  model.eval().requires_grad_(False).cuda()
  transform=Compose([Resize((224,224),interpolation=InterpolationMode.BICUBIC,antialias=True),ToTensor(),Normalize([.485,.456,.406],[.229,.224,.225])]);fs=[]
  with torch.inference_mode():
   for i in range(0,len(spec['extra']),32):
    imgs=[]
    for item in spec['extra'][i:i+32]:
     assert p.digest(item['path'])==item['sha256']
     with Image.open(item['path']) as im:imgs.append(transform(im.convert('RGB')))
    fs.append(model(torch.stack(imgs).cuda()).float().cpu().numpy())
  p.save_npz(extra_path,features=p.unit(np.concatenate(fs)).astype('float32'),manifest_sha256=p.digest(OUT/'manifest.json'))
  del model;torch.cuda.empty_cache()
 reference=np.concatenate([original,readfeatures(extra_path)])
 # Squared Euclidean distances in unit feature space, computed in float64.
 def distances(x,y):return np.maximum(0,2-2*x@y.T)
 fullrr=distances(reference,reference);np.fill_diagonal(fullrr,np.inf)
 radii={n:np.partition(fullrr[:n,:n],np.array(KS)-1,axis=1)[:,np.array(KS)-1] for n in spec['reference_sizes']}
 def density(x):
  sums={(n,k):[] for n in spec['reference_sizes'] for k in KS}
  for start in range(0,len(x),128):
   dist=distances(x[start:start+128],reference)
   for n in spec['reference_sizes']:
    for j,k in enumerate(KS):sums[n,k].append((dist[:,:n]<=radii[n][:,j]).sum(1)/k)
  return {key:np.concatenate(v) for key,v in sums.items()}
 rng=np.random.default_rng(20260911);idx=rng.integers(0,len(query),(2000,len(query)));records=[]
 realvals=density(query)
 for (n,k),v in realvals.items():
  ci=np.quantile(v[idx].mean(1),[.025,.975]);records.append(dict(kind='real',domain=domain,n=n,k=k,value=float(v.mean()),ci95=ci.tolist(),std=float(v.std(ddof=1))))
 for task,row in selected:
  cache=BASE/'tasks'/str(task);cube=json.loads((cache/'cubes'/f"{row['id']}.json").read_text());ind=json.loads((cache/'features/index.json').read_text());vals={}
  for code in p.CODES:
   f=readfeatures(ind['generated'][cube['images'][code]]);d=density(f);vals[code]=d
   with np.load(cache/'measurements'/f"{row['id']}.npz") as old:assert np.allclose(d[1000,5],old['densities'][list(p.CODES).index(code)])
   for (n,k),v in d.items():records.append(dict(kind='generated',domain=domain,n=n,k=k,value=float(v.mean()),task=task,id=row['id'],scale=row['scale'],source=row['source'],seed=row['adaptation_seed'],code=code))
  for n in spec['reference_sizes']:
   for k in KS:
    p.save_npz(OUT/f"task{task}_n{n}_k{k}.npz",density=np.stack([vals[c][n,k] for c in p.CODES]),codes=np.array(p.CODES),latent_hash=cube['latent_hash'])
  print('Checked',row['id'],flush=True)
 p.write_json(OUT/f'{domain}_results.json',dict(domain=domain,records=records,manifest_sha256=p.digest(OUT/'manifest.json'),note='Real CI resamples 400 query images conditional on reference bank; reference subsets are nested and generated images fixed.'))
 print('COMPLETE',domain,flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','run']);a.add_argument('--task',type=int,default=0);args=a.parse_args();prepare() if args.stage=='prepare' else run(args.task)
