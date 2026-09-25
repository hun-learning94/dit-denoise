"""Build report comparisons from the completed main plotting summaries."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.experiment0.plot import save
ROOT=Path(__file__).resolve().parents[2]
out=ROOT/'reports/experiment0/figures/main'
manifest=json.loads((ROOT/'reports/experiment0/main_manifest.json').read_text());cfg=manifest['config'];qi=cfg['qs'].index(.9)
# This report displays only completed 20M checkpoints; historical evaluations
# remain in their original manifests and model directories.
models=[m for m in manifest['models'] if m['status']=='ready' and m['label'].endswith('_20m')]
directories={m['label']:out/m['label'] for m in models}
datasets=list(dict.fromkeys(m['dataset'] for m in models))
results={m['label']:dict(np.load(directories[m['label']]/'summary.npz')) for m in models}
fig,axes=plt.subplots(len(datasets),2,figsize=(11,3*len(datasets)),squeeze=False,layout='constrained')
stats={}
for row,ds in enumerate(datasets):
 for scale in ['20m']:
  label=ds+'_'+scale
  if label not in results:
   for ax in axes[row]:
    ax.text(.02,.95,scale+' not evaluated',transform=ax.transAxes,va='top',fontsize=8)
   continue
  r=results[label];s=r['sigmas'];color={'10m':'C0','20m':'C1'}[scale]
  for col,key in enumerate(['radius','count']):
   y=r[key][:,:,qi].mean(1);axes[row,col].plot(s,y,label=scale,color=color)
   axes[row,col].fill_between(s,r['ci'][0,:,col,qi],r['ci'][1,:,col,qi],alpha=.2,color=color)
  refresh=[]
  for si in range(len(s)):
   chunks=[]
   for p in sorted((ROOT/'evals/experiment0/main'/label).glob(f'sigma{si:02d}_images*.npz')):
    with np.load(p) as z:chunks.append(z['refresh'])
   chunks=np.concatenate(chunks,axis=0);refresh.append(chunks.mean((0,4)))
  refresh=np.stack(refresh);tail=r['tail'][:,:,cfg['radii']]
  ratio=np.divide(refresh,tail[...,None],out=np.full_like(refresh,np.nan),where=tail[...,None]>cfg['floor'])
  stats[label]={
   'sigmas':s.tolist(),'radius90':r['radius'][:,:,qi].mean(1).tolist(),'count90':r['count'][:,:,qi].mean(1).tolist(),
   'total_energy':r['total'].mean(1).tolist(),'self_fraction':r['self_fraction'].mean(1).tolist(),
   'tail_r2':r['tail'][:,:,2].mean(1).tolist(),
   'mean_attention_count90':r['mean_attention_count'][:,:,qi].mean(1).tolist(),
   'mean_attention_radius90':r['mean_attention_radius'][:,:,qi].mean(1).tolist(),
   'nonself_radius90':r['nonself_radius'][:,:,qi].mean(1).tolist(),
   'nonself_count90':r['nonself_count'][:,:,qi].mean(1).tolist(),
   'per_image_count90':r['image_count'][:,:,:,qi].mean((1,2)).tolist(),
   'half_probe_max_mean_radius_difference':float(np.max(np.abs((r['nested_radius'][:,:,qi]-r['radius'][:,:,qi]).mean(1)))),
   'half_probe_max_mean_count_difference':float(np.max(np.abs((r['nested_count'][:,:,qi]-r['count'][:,:,qi]).mean(1)))),
   'refresh_tail_median_ratio_by_sigma_c':np.nanmedian(ratio,axis=(1,2)).tolist(),
   'refresh_mean_by_sigma_radius_c':refresh.mean(1).tolist(),
   'invalid_targets':int((~r['valid']).sum())}
 for col in range(2):
  axes[row,col].set(xscale='log',xlabel='Effective noise sigma',title=ds,ylabel='90% radius (patch hops)' if col==0 else '90% patch count');axes[row,col].legend()
save(fig,out/'sensitivity_coverage')
(out/'analysis_summary.json').write_text(json.dumps(stats,indent=2)+'\n')
status={}
for m in models:
 directory=directories[m['label']]
 names=['controls','attention_layers','robustness_refresh']+[f'matched_target{t:03d}' for t in cfg['targets']]
 assert all((directory/(name+'.'+ext)).is_file() for name in names for ext in ['png','pdf']),m['label']
 status[m['label']]='Complete sweep; figures generated'
(out/'coverage.json').write_text(json.dumps(status,indent=2)+'\n')
(out/'summary_sources.json').write_text(json.dumps({label:str(path.relative_to(ROOT)) for label,path in directories.items()},indent=2)+'\n')
print('Wrote comparisons and analysis_summary.json')
