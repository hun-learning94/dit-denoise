"""Blinded export and strict import of adjudicated human ratings."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
from .core import CODES,CONTRASTS
from .pipeline import approved,signature,write_json

def export(config,cache,out):
 approved(config)
 from PIL import Image
 out=Path(out)
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);(out/'images').mkdir();rng=np.random.default_rng(config['latent_seed'])
 selected=np.sort(rng.choice(config['images'],config['rating_latents_per_cube'],replace=False));private=[]
 for file in sorted((Path(cache)/'cubes').glob('*.json')):
  cube=json.loads(file.read_text());assert cube['protocol_hash']==signature(config)
  for code in CODES:
   with np.load(cube['images'][code]) as a:
    for i in selected:
     opaque=hashlib.sha256(f"{rng.integers(2**63)}:{cube['row']['id']}:{code}:{i}".encode()).hexdigest()[:24]
     Image.fromarray(a['images'][i].transpose(1,2,0)).save(out/'images'/f'{opaque}.png')
     private.append(dict(item_id=opaque,cube=cube['row']['id'],code=code,latent=int(i),source=cube['row']['source'],target=cube['row']['target']))
 rng.shuffle(private)
 with (out/'ratings.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['item_id','image','source_domain','target_domain','coherent','domain','rater']);w.writeheader()
  for r in private:w.writerow(dict(item_id=r['item_id'],image=f"images/{r['item_id']}.png",source_domain=r['source'],target_domain=r['target']))
 write_json(out/'PRIVATE_key.json',private)
 (out/'README.txt').write_text('Give raters images/ and ratings.csv only; keep PRIVATE_key.json hidden. Show source/target real-image reference sheets separately, identical for every state. coherent: yes/no/uncertain. domain: source/target/mixed/uncertain. Two independent raters; preserve their original files. Enter adjudicated answers into a new CSV for import. Do not delete failures or uncertain images.\n')

def summarize(key,ratings,repeats=2000,seed=20260911):
 private=json.loads(Path(key).read_text())
 with Path(ratings).open() as f:rows=list(csv.DictReader(f))
 byid={r['item_id']:r for r in rows}
 if len(byid)!=len(rows) or set(byid)!={r['item_id'] for r in private}:raise ValueError('Require exactly one adjudicated rating for every sampled image')
 cubes={}
 for p in private:
  r=byid[p['item_id']]
  if r['coherent'] not in ['yes','no','uncertain'] or r['domain'] not in ['source','target','mixed','uncertain']:raise ValueError('Missing/invalid rating')
  if not r['rater'].strip():raise ValueError('Identify adjudicator')
  cubes.setdefault(p['cube'],{}).setdefault(p['code'],{})[p['latent']]=(r['coherent']=='yes',r['coherent']=='yes' and r['domain']=='target',r['coherent']=='uncertain',r['domain']=='uncertain')
 output={};rng=np.random.default_rng(seed)
 for cube,states in cubes.items():
  assert set(states)==set(CODES)
  latents=sorted(states['000']);assert all(sorted(states[c])==latents for c in CODES)
  values={c:np.array([states[c][i] for i in latents],float) for c in CODES}
  def stats(idx):
   metrics={name:{c:float(values[c][idx,j].mean()) for c in CODES} for j,name in enumerate(['C','J','coherence_uncertain','domain_uncertain'])}
   return dict(metrics=metrics,contrasts={m:{name:sum(v*metrics[m][c] for c,v in weights.items()) for name,weights in CONTRASTS.items()} for m in ['C','J']})
  point=stats(np.arange(len(latents)));reps=[stats(rng.integers(0,len(latents),len(latents))) for _ in range(repeats)]
  point['ci95']={group:{metric:{name:list(map(float,np.quantile([r[group][metric][name] for r in reps],[.025,.975]))) for name in point[group][metric]} for metric in point[group]} for group in ['metrics','contrasts']}
  output[cube]=point
 return output

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['export','summarize']);p.add_argument('--config',default='experiments/experiment3_primary_metrics/protocol.json');p.add_argument('--cache');p.add_argument('--out',required=True);p.add_argument('--key');p.add_argument('--ratings');a=p.parse_args();config=json.loads(Path(a.config).read_text());approved(config)
 if a.stage=='export':export(config,a.cache,a.out)
 else:write_json(a.out,summarize(a.key,a.ratings,config['bootstrap_repeats'],config['latent_seed']))
if __name__=='__main__':main()
