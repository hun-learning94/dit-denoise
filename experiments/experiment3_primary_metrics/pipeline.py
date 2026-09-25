"""Staged, approval-gated fixed-endpoint evaluation. Never submits Slurm jobs."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,tempfile
import numpy as np
from .core import CODES,unit,fit_probe,probe_scores,discrimination,density_per_image,bootstrap
ROOT=Path(__file__).resolve().parents[2]

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def implementation_stamp():
 from importlib.metadata import version
 files=['edm.py','models.py','eval_model_comparison.py','experiments/experiment3_primary_metrics/core.py','experiments/experiment3_primary_metrics/pipeline.py']
 return dict(files={f:digest(ROOT/f) for f in files},packages={p:version(p) for p in ['numpy','scipy','torch','torchvision','Pillow']})
def signature(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()
def write_json(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False) as f:json.dump(obj,f,indent=2,allow_nan=False);tmp=f.name
 os.replace(tmp,path)
def save_npz(path,**arrays):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(suffix='.npz',dir=path.parent,delete=False) as f:tmp=f.name
 np.savez_compressed(tmp,**arrays);os.replace(tmp,path)
def approved(config):
 if config['status']!='approved':raise RuntimeError('Awaiting user go: production execution is disabled')
 if not all(config.get(k) for k in ['feature_repo','feature_repo_commit','feature_weights','feature_weights_sha256']):raise RuntimeError('Pin local DINOv2 code and weights before execution')

def preview(choices):
 ready=[];deferred=[]
 for r in choices:
  reason=None
  if r['epoch'] is None:reason='No selected joint endpoint'
  elif not r.get('budget_fid_complete'):reason='In-budget paired FID coverage incomplete'
  elif len(r['selection_paired_epochs'])!=8:reason='Await aligned eight-checkpoint grid'
  (deferred if reason else ready).append(dict(r,reason=reason) if reason else r)
 return dict(ready=ready,deferred=deferred,note='Preview only; no checkpoint is frozen and no hybrid ranking is computed')

def freeze(config,choices,out):
 approved(config);out=Path(out)
 if out.exists():raise FileExistsError('Manifest is immutable; use a new campaign directory')
 sys.path.insert(0,str(ROOT/'reports/experiment3'))
 from build_primary_source_report import artifact_stem,source_checkpoint
 scan=preview(choices);rows=[];hashes={}
 for r in scan['ready']:
  for seed in [0,1]:
   stem=ROOT/'reports/experiment3/figures'/artifact_stem(r['scale'],r['source'],r['target'],r['epoch'],seed)
   meta=json.loads(stem.with_suffix('.json').read_text());cube=ROOT/meta['cube_dir'];paths={}
   for code in CODES:
    p=(ROOT/source_checkpoint(r['scale'],r['source'])) if code=='000' else cube/f'ckpt_{code}.pt'
    key=str(p.resolve());hashes.setdefault(key,None)
    if hashes[key] is None:hashes[key]=digest(p)
    if code!='000' and hashes[key]!=meta['checkpoint_sha256'][code]:raise ValueError('Cube changed since generation')
    paths[code]=dict(path=key,sha256=hashes[key])
   rows.append(dict(id=f"{r['scale']}_{r['source']}_to_{r['target']}_seed{seed}_epoch{r['epoch']}",source=r['source'],target=r['target'],scale=r['scale'],adaptation_seed=seed,selected_epoch=r['epoch'],selection=r,checkpoints=paths))
 write_json(out,dict(protocol=config,protocol_hash=signature(config),rows=rows,deferred=scan['deferred'],implementation=implementation_stamp()))

def image_key(config,checkpoint):
 return signature(dict(checkpoint=checkpoint['sha256'],images=config['images'],latent_seed=config['latent_seed'],steps=config['sampler_steps'],precision=config['precision'],batch=config['generation_batch'],implementation=digest(Path(__file__))))

def generate(config,row,cache):
 approved(config)
 import torch
 from eval_model_comparison import load_model
 from edm import vp_heun_sampler
 if not torch.cuda.is_available():raise RuntimeError('GPU required')
 torch.set_num_threads(1);latents=torch.randn(config['images'],3,64,64,generator=torch.Generator().manual_seed(config['latent_seed']))
 latent_hash=hashlib.sha256(latents.numpy().tobytes()).hexdigest();outputs={}
 for code in CODES:
  ck=row['checkpoints'][code]
  if digest(ck['path'])!=ck['sha256']:raise ValueError('Frozen checkpoint changed')
  dest=Path(cache)/'images'/f'{image_key(config,ck)}.npz';outputs[code]=str(dest)
  import fcntl
  dest.parent.mkdir(parents=True,exist_ok=True)
  with dest.with_suffix('.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   if dest.exists():
    with np.load(dest) as a:assert str(a['latent_hash'])==latent_hash and len(a['images'])==config['images']
    continue
   model,checkpoint,_,_,_,transform=load_model(Path(ck['path']),torch.device('cuda'),True);model.eval()
   assert checkpoint.get('noise_process',checkpoint.get('args',{}).get('noise_process'))=='vp'
   generated=[]
   with torch.inference_mode():
    for start in range(0,len(latents),config['generation_batch']):
     z=latents[start:start+config['generation_batch']].cuda()
     x=vp_heun_sampler(model,z,num_steps=config['sampler_steps'],denominator_floor=float(checkpoint.get('vp_denominator_floor',.05)))
     x=transform.inverse(x.float())
     if not torch.isfinite(x).all():raise ValueError('Nonfinite generated image')
     generated.append(((x.clamp(-1,1)+1)*127.5).round().byte().cpu().numpy())
   save_npz(dest,images=np.concatenate(generated),latent_hash=latent_hash,checkpoint_sha256=ck['sha256'])
   del model,checkpoint;torch.cuda.empty_cache()
 write_json(Path(cache)/'cubes'/f"{row['id']}.json",dict(row=row,images=outputs,latent_hash=latent_hash,protocol_hash=signature(config)))

def real_manifest(config,domains,out):
 approved(config);out=Path(out)
 if out.exists():raise FileExistsError(out)
 rows={}
 for domain in sorted(domains):
  root=ROOT/'datasets'/domain
  def ordered(split):return sorted(root.joinpath(split).glob('*.png'),key=lambda p:signature([config['latent_seed'],domain,p.name]))
  train=ordered('train');valid=ordered('valid');test=ordered('test')
  nt,nv,ne,nr=[config[k] for k in ['real_train_per_domain','real_validation_per_domain','real_test_per_domain','real_reference_per_domain']]
  if not valid:valid,train=train[:nv],train[nv:]
  if len(train)<nt or len(valid)<nv or len(test)<ne+nr:raise ValueError(f'Insufficient real split: {domain}')
  roles={'train':train[:nt],'validation':valid[:nv],'test':test[:ne],'reference':test[ne:ne+nr]}
  rows[domain]={role:[dict(path=str(p.resolve()),sha256=digest(p)) for p in files] for role,files in roles.items()}
 # Reject exact duplicate file paths across selected roles/domains.
 # Image content duplicate checks are disabled to avoid dataset-side duplicate artifacts
 # (e.g., near-identical train/validation copies) while preserving split-level disjointness.
 seen={}
 for domain,roles in rows.items():
  for role,items in roles.items():
   for item in items:
    path=item['path']
    if path in seen: raise ValueError(f'Duplicate real image path: {seen[path]} and {domain}/{role}; revise real manifest before fitting')
    seen[path]=(domain,role)
 write_json(out,dict(protocol_hash=signature(config),domains=rows))

def extract(config,cache,real):
 approved(config)
 import torch
 from PIL import Image
 from torchvision.transforms import Compose,Resize,ToTensor,Normalize,InterpolationMode
 commit=subprocess.check_output(['git','-C',config['feature_repo'],'rev-parse','HEAD'],text=True).strip()
 if commit!=config['feature_repo_commit'] or digest(config['feature_weights'])!=config['feature_weights_sha256']:raise ValueError('Encoder provenance mismatch')
 if subprocess.check_output(['git','-C',config['feature_repo'],'status','--porcelain'],text=True).strip():raise ValueError('Encoder checkout must be clean')
 model=torch.hub.load(config['feature_repo'],config['feature_model'],source='local',pretrained=False)
 model.load_state_dict(torch.load(config['feature_weights'],map_location='cpu',weights_only=True),strict=True)
 model.eval().requires_grad_(False).cuda()
 transform=Compose([Resize((224,224),interpolation=InterpolationMode.BICUBIC,antialias=True),ToTensor(),Normalize([.485,.456,.406],[.229,.224,.225])])
 def compute(key,images):
  path=Path(cache)/'features'/f'{key}.npz'
  if path.exists():return str(path)
  fs=[]
  with torch.inference_mode():
   for i in range(0,len(images),config['feature_batch']):fs.append(model(torch.stack([transform(x) for x in images[i:i+config['feature_batch']]]).cuda()).float().cpu().numpy())
  save_npz(path,features=unit(np.concatenate(fs)).astype('float32'),encoder_sha256=config['feature_weights_sha256'],protocol_hash=signature(config))
  return str(path)
 index={'real':{},'generated':{},'protocol_hash':signature(config)}
 for domain,roles in real['domains'].items():
  index['real'][domain]={}
  for role,items in roles.items():
   images=[]
   for item in items:
    if digest(item['path'])!=item['sha256']:raise ValueError('Real image changed')
    with Image.open(item['path']) as im:images.append(im.convert('RGB').copy())
   index['real'][domain][role]=compute(signature([items,signature(config)]),images)
 for cube in sorted((Path(cache)/'cubes').glob('*.json')):
  row=json.loads(cube.read_text());assert row['protocol_hash']==signature(config)
  for code,path in row['images'].items():
   if path not in index['generated']:
    with np.load(path) as a:images=[Image.fromarray(x.transpose(1,2,0)) for x in a['images']]
    index['generated'][path]=compute(signature([digest(path),signature(config)]),images)
 write_json(Path(cache)/'features/index.json',index)

def analyze(config,cache):
 approved(config);cache=Path(cache);index=json.loads((cache/'features/index.json').read_text());assert index['protocol_hash']==signature(config)
 def feats(path):
  with np.load(path) as a:return a['features']
 probes={}
 for path in sorted((cache/'cubes').glob('*.json')):
  cube=json.loads(path.read_text());assert cube['protocol_hash']==signature(config)
  r=cube['row'];pair=(r['source'],r['target'])
  if pair not in probes:
   sets={role:[feats(index['real'][domain][role]) for domain in pair] for role in ['train','validation','test']}
   def xy(role):return np.concatenate(sets[role]),np.concatenate([np.zeros(len(sets[role][0])),np.ones(len(sets[role][1]))])
   x,y=xy('train');vx,vy=xy('validation');candidates=[]
   for lam in config['regularizations']:
    w=fit_probe(x,y,lam);candidates.append((discrimination(vy,probe_scores(vx,w))['log_loss'],lam,w))
   _,lam,w=min(candidates,key=lambda c:(c[0],-c[1]));tx,ty=xy('test')
   quality=discrimination(ty,probe_scores(tx,w));probes[pair]=(w,lam,quality)
   save_npz(cache/'probes'/f'{pair[0]}_to_{pair[1]}.npz',weights=w,regularization=lam,protocol_hash=signature(config))
  w,lam,quality=probes[pair];f={c:feats(index['generated'][cube['images'][c]]) for c in CODES}
  if any(len(x)!=config['images'] for x in f.values()):raise ValueError('Feature count mismatch')
  reference=feats(index['real'][r['target']]['reference'])
  scores={c:probe_scores(f[c],w) for c in CODES};densities={c:density_per_image(f[c],reference,config['density_k']) for c in CODES};distances={c:np.sum((unit(f[c])-unit(f['111']))**2,axis=1) for c in CODES}
  save_npz(cache/'measurements'/f"{r['id']}.npz",scores=np.stack([scores[c] for c in CODES]),densities=np.stack([densities[c] for c in CODES]),distances=np.stack([distances[c] for c in CODES]),codes=np.array(CODES),latent_hash=cube['latent_hash'],protocol_hash=signature(config))
  result=bootstrap(scores,densities,distances,repeats=config['bootstrap_repeats'],seed=config['latent_seed'],min_score_gap=config['min_score_gap'],min_endpoint_distance=config['min_endpoint_distance'])
  write_json(cache/'results'/f"{r['id']}.json",dict(row=r,probe=quality,regularization=lam,domain_score_informative=quality['auroc']>=config['min_heldout_auroc'],summary=result,uncertainty='Paired latent bootstrap conditional on fixed probe, reference set and trained endpoint; not training-seed uncertainty'))

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['preview','freeze','real-manifest','generate','features','analyze']);p.add_argument('--config',default='experiments/experiment3_primary_metrics/protocol.json');p.add_argument('--choices',default='reports/2026-09-08/primary_source_endpoint_choices.json');p.add_argument('--manifest',default='reports/experiment3/primary_metrics/manifest.json');p.add_argument('--real',default='reports/experiment3/primary_metrics/real_manifest.json');p.add_argument('--cache',default='reports/experiment3/primary_metrics/cache');p.add_argument('--task',type=int,default=0);a=p.parse_args();config=json.loads(Path(a.config).read_text())
 if a.stage=='preview':print(json.dumps(preview(json.loads(Path(a.choices).read_text())),indent=2));return
 approved(config)
 if a.stage=='freeze':freeze(config,json.loads(Path(a.choices).read_text()),a.manifest);return
 manifest=json.loads(Path(a.manifest).read_text());assert manifest['protocol_hash']==signature(config)
 if manifest['implementation']!=implementation_stamp():raise ValueError('Code/dependency versions changed after freeze')
 if a.stage=='real-manifest':real_manifest(config,{d for r in manifest['rows'] for d in [r['source'],r['target']]},a.real)
 elif a.stage=='generate':generate(config,manifest['rows'][a.task],a.cache)
 elif a.stage=='features':
  real=json.loads(Path(a.real).read_text());assert real['protocol_hash']==signature(config);extract(config,a.cache,real)
 elif a.stage=='analyze':analyze(config,a.cache)
if __name__=='__main__':main()
