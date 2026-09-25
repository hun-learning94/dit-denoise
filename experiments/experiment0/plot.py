"""Aggregate complete paired shards and produce reviewable figures, PNG/PDF and NPZ."""
import argparse
import json
from pathlib import Path
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from experiments.experiment0.core import coverage, distances


def save(fig,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf']: fig.savefig(path.with_suffix('.'+ext),dpi=160,bbox_inches='tight')
    plt.close(fig)


def aggregate(energy,attention,cfg):
    """Energy: S,N,T,P,M; attention: S,N,L,T,M. Resample images, never probes."""
    d=distances(16,cfg['targets']); E=energy.mean(3); avg=E.mean(1)
    main=coverage(avg,d,cfg['qs'],cfg['floor'])
    per_image=coverage(E,d,cfg['qs'],cfg['floor'])
    no_self=avg.copy()
    for ti,t in enumerate(cfg['targets']): no_self[:,ti,t]=0
    other=coverage(no_self,d,cfg['qs'],cfg['floor'])
    attn=coverage(attention.mean(1),d,cfg['qs'],cfg['floor'])
    # Average maps across layers before thresholding, not layerwise radii/counts.
    pooled_attn=coverage(attention.mean((1,2)),d,cfg['qs'],cfg['floor'])
    n=E.shape[1]
    rng=np.random.default_rng(cfg['bootstrap_seed'])
    boot=np.empty((cfg['bootstrap'],len(avg),2,len(cfg['qs'])))
    for b in range(cfg['bootstrap']):
        idx=rng.integers(0,n,n)
        c=coverage(E[:,idx].mean(1),d,cfg['qs'],cfg['floor'])
        boot[b,:,0]=c['radius'].mean(1); boot[b,:,1]=c['count'].mean(1)
    ci=np.quantile(boot,[.025,.975],axis=0)
    self_energy=np.stack([avg[:,ti,t] for ti,t in enumerate(cfg['targets'])],1)
    self_fraction=np.divide(self_energy,main['total'],out=np.full_like(self_energy,np.nan),where=main['valid'])
    nested=coverage(energy[:,:,:,:max(1,energy.shape[3]//2)].mean(3).mean(1),d,cfg['qs'],cfg['floor'])
    return {**main, 'ci':ci, 'self_fraction':self_fraction,
            'nonself_radius':other['radius'],'nonself_count':other['count'],
            'image_radius':per_image['radius'],'image_count':per_image['count'],
            'attention_radius':attn['radius'],'attention_count':attn['count'],
            'mean_attention_radius':pooled_attn['radius'],'mean_attention_count':pooled_attn['count'],
            'nested_radius':nested['radius'],'nested_count':nested['count']}


def load_complete(manifest,spec,input_root):
    cfg=manifest['config']; data=[]; provenance=[]; reference_ids=None
    checkpoint_hash=None
    for si in range(len(cfg['sigmas'])):
        parts=[]
        tasks=sorted([t for t in manifest['tasks'] if t['model']==spec['label'] and t['sigma_index']==si],key=lambda t:t['start'])
        if not tasks: return None,'No ready tasks'
        for task in tasks:
            path=input_root/cfg['stage']/spec['label']/f'sigma{si:02d}_images{task["start"]:04d}.npz'
            if not path.exists(): return None,f'Missing {path.name}'
            with np.load(path,allow_pickle=False) as z:
                part={k:z[k].copy() for k in ['influence','attention','refresh','refresh_raw','images','indices']}
                meta=json.loads(str(z['metadata']))
                if meta['config']!=cfg or meta['task']!=task or meta['model']!=spec:
                    raise ValueError(f'Manifest mismatch: {path}')
                if checkpoint_hash is None: checkpoint_hash=meta['checkpoint_sha256']
                if checkpoint_hash!=meta['checkpoint_sha256']: raise ValueError('Mixed checkpoint weights')
                if provenance and meta.get('source_sha256')!=provenance[0].get('source_sha256'):
                    raise ValueError('Mixed computation code versions')
                if float(z['sigma'])!=cfg['sigmas'][si]: raise ValueError('Noise mismatch')
                if len(part['indices'])!=task['stop']-task['start']: raise ValueError('Incomplete shard')
                if any(not np.isfinite(v).all() for v in part.values()): raise ValueError('Nonfinite artifact')
                provenance.append(meta)
                parts.append(part)
        joined={k:np.concatenate([x[k] for x in parts]) for k in parts[0]}
        ids=joined['indices']
        if len(ids)!=cfg['num_images'] or len(np.unique(ids))!=len(ids): raise ValueError('Image coverage mismatch')
        if reference_ids is None: reference_ids=ids
        if not np.array_equal(reference_ids,ids): raise ValueError('Unpaired noise sweep')
        data.append(joined)
    return ({k:np.stack([x[k] for x in data]) for k in data[0]},provenance),None


def model_figures(data,result,cfg,label,out):
    s=np.array(cfg['sigmas']); q90=cfg['qs'].index(.9)
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for qi,q in enumerate(cfg['qs']):
        axes[0,0].plot(s,result['radius'][:,:,qi].mean(1),label=f'q={q:g}')
        axes[0,1].plot(s,result['count'][:,:,qi].mean(1),label=f'q={q:g}')
    for ax,title in zip(axes[0,:2],['Sensitivity radius (patch hops)','Sensitivity patch count']): ax.set_title(title); ax.legend(fontsize=8)
    axes[0,2].plot(s,result['total'].mean(1)); axes[0,2].set(title='Total absolute sensitivity',yscale='log')
    axes[1,0].plot(s,result['self_fraction'].mean(1)); axes[1,0].set_title('Self-patch fraction')
    for r in cfg['radii']:
        axes[1,1].plot(s,result['tail'][:,:,r].mean(1),label=f'r={r}')
        axes[1,2].plot(s,result['relative_tail'][:,:,r].mean(1),label=f'r={r}')
    axes[1,1].set(title='Absolute outside sensitivity',yscale='log'); axes[1,2].set_title('Relative outside sensitivity')
    axes[1,1].legend(fontsize=7); axes[1,2].legend(fontsize=7)
    for ax in axes.flat: ax.set_xscale('log'); ax.set_xlabel('Effective noise')
    fig.suptitle(label); save(fig,out/label/'controls')
    attention_figures(result,cfg,label,out)
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for ax,key,title in zip(axes[0,:2],['radius','count'],['90% radius','90% patch count']):
        ax.plot(s,result[key][:,:,q90].mean(1),label='Population map')
        ax.plot(s,result['image_'+key][:,:,:,q90].mean((1,2)),label='Mean per-image coverage')
        ax.plot(s,result['nonself_'+key][:,:,q90].mean(1),label='Self excluded')
        ax.plot(s,result['nested_'+key][:,:,q90].mean(1),ls=':',label='Half the probes')
        ax.set_title(title); ax.legend(fontsize=7)
    boundary=np.array([t//16 in [0,15] or t%16 in [0,15] for t in cfg['targets']])
    for mask,label_region in [(boundary,'Boundary'),(~boundary,'Interior')]:
        if mask.any(): axes[0,2].plot(s,result['radius'][:,mask,q90].mean(1),label=label_region)
    axes[0,2].set_title('90% radius by location'); axes[0,2].legend(fontsize=8)
    norm=data['refresh'].mean((1,2,5)); raw=data['refresh_raw'].mean((1,2,5))
    for ri,r in enumerate(cfg['radii']):
        if r==15: continue
        axes[1,0].plot(s,result['tail'][:,:,r].mean(1),label=f'Jacobian r={r}')
        for ci,c in enumerate(cfg['cs']):
            axes[1,1].plot(s,norm[:,ri,ci],label=f'r={r}, c={c}')
            axes[1,2].plot(s,raw[:,ri,ci],label=f'r={r}, c={c}')
    for ax,title in zip(axes[1],['Jacobian absolute tails','Normalized finite refresh','Raw response per output coordinate']):
        ax.set(title=title,yscale='log'); ax.legend(fontsize=6)
    for ax in axes.flat: ax.set_xscale('log'); ax.set_xlabel('Effective noise')
    save(fig,out/label/'robustness_refresh')
    matched_figures(data,cfg,label,out)


def attention_figures(result,cfg,label,out):
    s=np.array(cfg['sigmas']); q90=cfg['qs'].index(.9)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for li in range(result['attention_radius'].shape[1]):
        for ax,key in zip(axes,['attention_radius','attention_count']):
            ax.plot(s,result[key][:,li,:,q90].mean(1),label=f'Layer {li+1}')
    for ax,key in zip(axes,['mean_attention_radius','mean_attention_count']):
        ax.plot(s,result[key][:,:,q90].mean(1),color='black',linewidth=2,label='Layer-averaged map')
    for ax,title in zip(axes,['Attention 90% radius','Attention 90% patch count']):
        ax.set(xscale='log',xlabel='Effective noise',title=title); ax.legend(fontsize=7,ncol=2)
    save(fig,out/label/'attention_layers')


def matched_figures(data,cfg,label,out):
    if 'noisy_inputs' not in data:
        with np.load(out/label/'noisy_previews.npz') as z:
            np.testing.assert_array_equal(z['indices'],data['indices'][:,0])
            np.testing.assert_allclose(z['sigmas'],cfg['sigmas'])
            data={**data,'noisy_inputs':z['noisy_inputs'][:,None]}
    s=np.array(cfg['sigmas'])
    # Same image, same target, same noise; all normalized maps share one color scale.
    wanted=[cfg['sigmas'].index(x) for x in [.05,1,16] if x in cfg['sigmas']]
    for ti,target in enumerate(cfg['targets']):
        fig,axes=plt.subplots(len(wanted),3,figsize=(9,3*len(wanted)),squeeze=False,layout='constrained')
        for row,si in enumerate(wanted):
            noisy=data['noisy_inputs'][si,0].transpose(1,2,0)
            axes[row,0].imshow(np.clip((noisy+1)/2,0,1)); axes[row,0].set_title(f'Noisy input X_t, sigma={s[si]:g}')
            maps=[data['influence'][si,0,ti].mean(0),data['attention'][si,0,:,ti].mean(0)]
            for col,(m,title) in enumerate(zip(maps,['Full-denoiser sensitivity','Layer-averaged attention']),1):
                total=m.sum(); w=m/total if total>cfg['floor'] else np.full_like(m,np.nan)
                im=axes[row,col].imshow(w.reshape(16,16),norm=LogNorm(vmin=1e-6,vmax=1),cmap='magma')
                axes[row,col].plot(target%16,target//16,'s',mfc='none',mec='cyan',ms=9)
                axes[row,col].set_title(f'{title}\nraw sum={total:.3g}')
            for ax in axes[row]: ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im,ax=axes[:,1:].ravel().tolist(),label='Normalized mass (shared scale)')
        save(fig,out/label/f'matched_target{target:03d}')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True)
    p.add_argument('--models',nargs='+',help='Render only the named model checkpoints')
    p.add_argument('--input-root',default='evals/experiment0'); p.add_argument('--output-root',default='reports/experiment0/figures')
    a=p.parse_args(); manifest=json.loads(Path(a.manifest).read_text()); cfg=manifest['config']
    out=ROOT/a.output_root/cfg['stage']; out.mkdir(parents=True,exist_ok=True)
    results={}; status={}; paired_ids={}
    for spec in manifest['models']:
        if a.models and spec['label'] not in a.models: continue
        if spec['status']!='ready': status[spec['label']]='Missing terminal checkpoint'; continue
        loaded,reason=load_complete(manifest,spec,ROOT/a.input_root)
        if reason: status[spec['label']]=reason; continue
        data,provenance=loaded
        if spec['dataset'] in paired_ids and not np.array_equal(paired_ids[spec['dataset']],data['indices']):
            raise ValueError('Model scales use different images')
        paired_ids[spec['dataset']]=data['indices']
        r=aggregate(data['influence'],data['attention'],cfg); results[spec['label']]=r
        path=out/spec['label']; path.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path/'summary.npz',**r,sigmas=cfg['sigmas'],qs=cfg['qs'],targets=cfg['targets'])
        (path/'numerics.json').write_text(json.dumps(dict(
            sigmas=cfg['sigmas'],invalid_target_fraction=(~r['valid']).mean(1).tolist(),
            total_energy_min=r['total'].min(1).tolist(),
            invalid_image_target_fraction=np.isnan(r['image_radius'][...,0]).mean((1,2)).tolist(),
            energy_floor=cfg['floor'],bootstrap_unit='image',
            note='Coverage means and confidence bands are undefined if required targets have insufficient energy.'
        ),indent=2)+'\n')
        (path/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        model_figures(data,r,cfg,spec['label'],out)
        status[spec['label']]='Complete sweep; figures generated'
    (out/'coverage.json').write_text(json.dumps(status,indent=2)+'\n')
    if results:
        datasets=list(dict.fromkeys(m['dataset'] for m in manifest['models']))
        fig,axes=plt.subplots(len(datasets),2,figsize=(11,3*len(datasets)),squeeze=False,layout='constrained')
        qi=cfg['qs'].index(.9)
        for row,ds in enumerate(datasets):
            for col,key in enumerate(['radius','count']):
                ax=axes[row,col]
                for scale in ['10m','20m']:
                    label=ds+'_'+scale
                    if label not in results: continue
                    r=results[label]; y=r[key][:,:,qi].mean(1)
                    line,=ax.plot(cfg['sigmas'],y,label=scale)
                    ax.fill_between(cfg['sigmas'],r['ci'][0,:,col,qi],r['ci'][1,:,col,qi],alpha=.2,color=line.get_color())
                ax.set(xscale='log',xlabel='Effective noise',title=ds,ylabel='90% radius (patch hops)' if col==0 else '90% patch count')
                if ax.lines: ax.legend()
                else: ax.text(.5,.5,'No complete model sweep',ha='center',transform=ax.transAxes)
        save(fig,out/'sensitivity_coverage')
    print(json.dumps(status,indent=2))


if __name__=='__main__': main()
