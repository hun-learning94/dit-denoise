"""CPU-only paired summaries, real spatial MMD, and dual-reference FID."""
import argparse,csv,math,os
from pathlib import Path
import numpy as np
from .common import read,write,digest,save_npz
STATES={'000':'Source','100':'QK','001':'MLP','011':'VO+MLP','111':'Joint'}
def ci(x,seed=41,reps=2000):
 x=np.asarray(x,dtype=np.float64);rng=np.random.default_rng(seed);means=[]
 for _ in range(reps//100):means.extend(x[rng.integers(len(x),size=(100,len(x)))].mean(1))
 return float(x.mean()),*map(float,np.quantile(means,[.025,.975]))
def d2(a,b):
 a=np.asarray(a,np.float64);b=np.asarray(b,np.float64)
 return np.maximum(((a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*a@b.T)/a.shape[1],0)
def mmd(kxx,kyy,kxy):
 n=len(kxx);m=len(kyy)
 return float((kxx.sum()-np.trace(kxx))/(n*(n-1))+(kyy.sum()-np.trace(kyy))/(m*(m-1))-2*kxy.mean())
def frechet(mu,cov,rmu,rcov):
 vals,vecs=np.linalg.eigh(rcov);assert vals.min()>-1e-7
 root=(vecs*np.sqrt(vals.clip(0)))@vecs.T;cross=root@cov@root;ev=np.linalg.eigvalsh((cross+cross.T)/2)
 assert ev.min()>-1e-6
 value=float(((mu-rmu)**2).sum()+np.trace(cov)+np.trace(rcov)-2*np.sqrt(ev.clip(0)).sum())
 assert value>-1e-5,value
 return max(value,0.)
def csvwrite(path,rows):
 if not rows:return
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def table(rows,columns):
 return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+['| '+' | '.join(str(r.get(k,'pending')) for k in columns)+' |' for r in rows])
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--stage',choices=['paired','final'],required=True);a=p.parse_args();m=read(a.manifest);cache=Path(m['cache']);report=Path(m['report'])/('paired_report' if a.stage=='paired' else 'final_report');report.mkdir(exist_ok=True)
 for rel,h in m['driver_sha256'].items():assert digest(Path(m['code'])/rel)==h
 sha=digest(a.manifest);paired=[];lossrows=[];fidrows=[];mmdrows=[];pending=[]
 def key(row,bits):return row['model']+'_source' if bits=='000' else row['id']+'_'+bits
 def load_bank(k,n=4,field=None):
  path=cache/'generated'/k
  if not (path/'identity.json').exists():return None
  assert read(path/'identity.json')['manifest_sha256']==sha
  files=[path/f'shard{i:03d}.npz' for i in range(n)]
  if not all(p.exists() for p in files):return None
  fields=[field] if field else ['global_features','spatial','images'];parts={f:[] for f in fields}
  for p in files:
   with np.load(p) as z:
    for f in fields:parts[f].append(z[f])
  return {f:np.concatenate(v) for f,v in parts.items()}
 # One set of bootstrap indices per sample count/seed is reused implicitly by ci.
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(3,3,figsize=(15,13));figs,axess=plt.subplots(3,3,figsize=(15,13));families=list(m['models']);targets=['aahq','afhq','ffhq'];colors=['gray','royalblue','darkorange','seagreen','black']
 for row in m['transfers']:
  banks={bits:load_bank(key(row,bits)) for bits in STATES}
  if any(v is None for v in banks.values()):pending.append(row['id']+': paired banks incomplete');continue
  assert all(len(v['global_features'])==m['paired_size'] for v in banks.values())
  source,joint=banks['000'],banks['111'];perimage={};local=[]
  for descriptor,field,denom in [('global','global_features',1),('spatial','spatial',32640)]:
   delta=((source[field].astype(np.float64)-joint[field])**2).sum(1)/denom;separation=float(delta.mean());threshold=.01 if descriptor=='global' else 1e-6
   axis=(axes if descriptor=='global' else axess)[families.index(row['model']),targets.index(row['target'])]
   for (bits,label),color in zip(STATES.items(),colors):
    bank=banks[bits][field].astype(np.float64);ds=((bank-source[field])**2).sum(1)/denom;dj=((bank-joint[field])**2).sum(1)/denom
    if bits=='000':assert np.max(ds)==0
    if bits=='111':assert np.max(dj)==0
    sm,sl,sh=ci(ds,reps=m['bootstrap']);jm,jl,jh=ci(dj,reps=m['bootstrap'])
    r=dict(transfer=row['id'],state=label,bits=bits,descriptor=descriptor,n=len(ds),source_distance=sm,source_low=sl,source_high=sh,joint_distance=jm,joint_low=jl,joint_high=jh,separation=separation,source_normalized=sm/separation if separation>=threshold else '',joint_normalized=jm/separation if separation>=threshold else '')
    paired.append(r);local.append(r);perimage[descriptor+'_'+bits+'_source']=ds;perimage[descriptor+'_'+bits+'_joint']=dj
    axis.errorbar(sm,jm,xerr=[[sm-sl],[sh-sm]],yerr=[[jm-jl],[jh-jm]],marker='o',color=color,label=label,capsize=2)
   axis.set(title=row['id'],xlabel='Distance to source',ylabel='Distance to joint');axis.legend(fontsize=7);axis.grid(alpha=.2)
  save_npz(report/(row['id']+'_paired_distances.npz'),**perimage)
  # Fixed first eight fresh trajectories, all states; no favorable-example selection.
  from PIL import Image,ImageDraw
  res=m['models'][row['model']]['resolution'];canvas=Image.new('RGB',(5*res,8*res+24),'white');draw=ImageDraw.Draw(canvas)
  for col,(bits,label) in enumerate(STATES.items()):
   draw.text((col*res+2,4),label,fill='black')
   for i in range(8):canvas.paste(Image.fromarray(banks[bits]['images'][i]),(col*res,i*res+24))
  canvas.save(report/(row['id']+'_examples.png'))
  del banks
 for f,name in [(fig,'global'),(figs,'spatial')]:f.tight_layout();f.savefig(report/(name+'_endpoint_distances.png'),dpi=160);f.savefig(report/(name+'_endpoint_distances.pdf'));plt.close(f)
 for row in m['transfers']:
  for domain in ['source',row['target']]:
   for bits,label in STATES.items():
    path=cache/'loss'/row['id']/f'{domain}_{bits}.npz'
    if not path.exists():pending.append(f'{row["id"]} {domain} {bits}: loss pending');continue
    with np.load(path) as z:
     values=z['losses'];strata=z['strata']
    assert values.shape[2]==m['loss_replicates']
    for stratum,v in zip(strata,values):
     mean,low,high=ci(v.mean(1),reps=m['bootstrap']);lossrows.append(dict(transfer=row['id'],domain=domain,state=label,bits=bits,stratum=str(stratum),n=len(v),replicates=v.shape[1],mean=mean,low=low,high=high))
 # A single bandwidth, calibrated exclusively on held-out real calibration images.
 ready=all((cache/'real'/k/'READY.json').exists() for k in m['views'])
 if ready:
  for k,v in m['views'].items():assert read(cache/'real'/k/'READY.json')['view']==v
  calibration=np.concatenate([np.load(cache/'real'/k/'calibration.npz')['spatial'] for k in m['views']]);dist=d2(calibration,calibration);tau2=float(np.median(dist[np.triu_indices(len(dist),1)]));assert tau2>0;del calibration,dist
  write(report/'spatial_kernel.json',dict(tau_squared=tau2,kernel='exp(-mean_squared_descriptor_distance/(2*tau_squared))',calibration_per_view=128,views=list(m['views']),estimator='unbiased MMD squared'))
  spatial={};selfkern={}
  for k in m['views']:
   x=np.load(cache/'real'/k/'spatial.npz')['spatial'];spatial[k]=x;selfkern[k]=np.exp(-d2(x,x)/(2*tau2));n=len(x)//2;kk=selfkern[k]
   mmdrows.append(dict(comparison=k+'_split_half',source=k,target=k,n_source=n,n_target=len(x)-n,mmd2=mmd(kk[:n,:n],kk[n:,n:],kk[:n,n:])))
  for row in m['transfers']:
   s=row['model']+'_source';t=row['model']+'_'+row['target'];cross=np.exp(-d2(spatial[s],spatial[t])/(2*tau2))
   mmdrows.append(dict(comparison=row['id'],source=s,target=t,n_source=len(spatial[s]),n_target=len(spatial[t]),mmd2=mmd(selfkern[s],selfkern[t],cross)))
  del spatial,selfkern
 else:pending.append('Real spatial features incomplete')
 if a.stage=='final':
  statdir=cache/'fid_statistics';statdir.mkdir(exist_ok=True)
  def stats(k,real=False):
   path=statdir/(('real_' if real else 'generated_')+k+'.npz')
   if path.exists():
    with np.load(path) as z:return z['mean'],z['cov'],int(z['n'])
   if real:
    if not (cache/'real'/k/'READY.json').exists():return None
    with np.load(cache/'real'/k/'test.npz') as z:x=z['features'].astype(np.float64)
   else:
    bank=load_bank(k,math.ceil(m['bank_size']/m['shard_size']),field='features')
    if bank is None:return None
    x=bank['features'].astype(np.float64);assert len(x)==m['bank_size']
   mean=x.mean(0);cov=np.cov(x,rowvar=False);n=len(x);save_npz(path,mean=mean,cov=cov,n=n);return mean,cov,n
  for row in m['transfers']:
   for bits,label in STATES.items():
    k=key(row,bits);g=stats(k)
    if g is None:pending.append(k+': FID-10k incomplete');continue
    for domain in ('source',row['target']):
     ref=row['model']+'_'+domain;r=stats(ref,True)
     if r is None:continue
     fp=statdir/(k+'__'+ref+'.json')
     if fp.exists():value=read(fp)['fid']
     else:value=frechet(g[0],g[1],r[0],r[1]);write(fp,dict(fid=value,n_generated=g[2],n_real=r[2],extractor_sha256=m['inception_sha256']))
     fidrows.append(dict(transfer=row['id'],state=label,bits=bits,reference=domain,fid10k=value,n_generated=g[2],n_real=r[2]))
 csvwrite(report/'paired_distances.csv',paired);csvwrite(report/'denoising.csv',lossrows);csvwrite(report/'real_spatial_mmd.csv',mmdrows);csvwrite(report/'fid10k.csv',fidrows)
 write(report/'summary.json',dict(manifest_sha256=sha,stage=a.stage,paired=paired,losses=lossrows,real_mmd=mmdrows,fid=fidrows,pending=sorted(set(pending))))
 lines=['# Epoch-100 component transplantation quantitative results','','Adaptation seed0; fixed epoch100; QK/VO/MLP bit order. DiT64 is the rank64 MLP budget control. BF16 model evaluation; FP32 feature extraction. Bootstrap intervals quantify sampling uncertainty for fixed checkpoints, not training variability. No cross-direction correlations.','','FID uses the project torchvision Inception weights and preprocessing, not standard pytorch-fid. Source and target real-reference columns must be compared within a column. U-ViT uses its released CelebA crop and matched target crops; the DiTs use their original prepared views. AFHQ64 has no validation split: its spatial-kernel calibration subset is disjoint from its spatial evaluation subset.','','[Global endpoint-distance plot](global_endpoint_distances.png) · [Spatial endpoint-distance plot](spatial_endpoint_distances.png)','','## Real spatial distance','',table(mmdrows,['comparison','n_source','n_target','mmd2'])]
 for row in m['transfers']:
  lines+=['','## '+row['id'],'',f'Epoch100, optimizer step {row["step"]}. Source and joint below refer to generated endpoints.',''];table_rows=[]
  for bits,label in STATES.items():
   r={'State':label}
   for descriptor,short in [('global','DINO'),('spatial','Spatial')]:
    v=next((x for x in paired if x['transfer']==row['id'] and x['bits']==bits and x['descriptor']==descriptor),None)
    if v:
     for endpoint in ('source','joint'):r[short+'→'+endpoint]=f'{v[endpoint+"_distance"]:.5g} [{v[endpoint+"_low"]:.5g}, {v[endpoint+"_high"]:.5g}]'
   for domain,short in [('source','Source'),(row['target'],'Target')]:
    v=next((x for x in fidrows if x['transfer']==row['id'] and x['bits']==bits and x['reference']==domain),None)
    if v:r[short+' FID10k']=f'{v["fid10k"]:.4f}'
    v=next((x for x in lossrows if x['transfer']==row['id'] and x['bits']==bits and x['domain']==domain and x['stratum']=='native'),None)
    if v:r[short+' loss']=f'{v["mean"]:.5g} [{v["low"]:.5g}, {v["high"]:.5g}]'
   table_rows.append(r)
  lines+=[table(table_rows,['State','DINO→source','DINO→joint','Spatial→source','Spatial→joint','Source FID10k','Target FID10k','Source loss','Target loss']),'',f'[First eight trajectories]({row["id"]}_examples.png)']
 lines+=['','## Interpretation','','Assess both distances together. Small distance to source with large distance to joint supports source-associated appearance; the reverse supports joint-associated appearance. A state far from both endpoints remains evidence of incompatibility, not evidence of a clean functional separation. Real spatial MMD and paired source–joint separation measure different quantities. Metrics do not show that fixed QK produces fixed attention maps, or that restricted fine-tuning will succeed; the MetFaces case study tests that practical implication.','','Noise-stratified losses and all numerical intervals are in the CSV files. No outcome has been filtered for agreement with the qualitative hypothesis.','', '## Pending','', '\n'.join('- '+x for x in sorted(set(pending))) or 'None.']
 (report/'REPORT.md').write_text('\n'.join(lines)+'\n');print(str(report/'REPORT.md'),flush=True)
if __name__=='__main__':main()
