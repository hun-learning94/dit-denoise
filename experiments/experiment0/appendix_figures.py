"""Appendix population curves and matched 95% context maps from completed evaluations."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from experiments.experiment0.core import gaussian
from experiments.experiment0.consolidated import OUT as SAMPLE_DIR, context_mask, outline_context, clean_reference, sample_path
from experiments.experiment0.reporting import REPORT

OUT=REPORT/'figures/appendix'
NAMES={'celeba64':'CelebA','ffhq64':'FFHQ','aahq64':'AAHQ','afhq_v2_64':'AFHQ-v2',
       'icartoonface64':'iCartoonFace','stl10_64':'STL-10','imagenet64':'ImageNet64'}


def save(fig,name,pad_inches=.04):
    for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=200,bbox_inches='tight',pad_inches=pad_inches)
    plt.close(fig)


def samples(spec,summary):
    saved=sample_path(spec['label'])
    if spec['dataset'] in ['celeba64','imagenet64','stl10_64'] and not saved.exists():
        raise FileNotFoundError(f'Run refresh_qualitative first: {saved}')
    if saved.exists():
        with np.load(saved) as z:data={k:z[k].copy() for k in z.files}
        meta=json.loads(str(data['metadata']))
        assert meta['checkpoint_sha256']==summary['checkpoint_sha256']
        assert meta['model']['epoch']==spec['epoch']
        if 'clean_input' not in data:
            data['clean_input']=clean_reference(data,meta)
            np.savez_compressed(saved,**data)
        return data,meta,str(saved)
    # Select by a fixed rule, independently of heatmap values.
    target=136
    slot=4 if spec['dataset']=='icartoonface64' else 0
    inputs=[];energies=[];attentions=[];sources=[];reference=None
    for si in [5,11]:
        path=Path('evals/experiment0/main')/spec['label']/f'sigma{si:02d}_images0000.npz'
        with np.load(path) as z:
            meta=json.loads(str(z['metadata']));cfg=meta['config'];idx=int(z['indices'][slot])
            assert meta['checkpoint_sha256']==summary['checkpoint_sha256']
            if reference is None:reference=idx
            assert idx==reference
            ti=cfg['targets'].index(target)
            energy=z['influence'][slot,ti].mean(0,dtype=np.float64)
            attn=z['attention'][slot,:,ti].mean(0,dtype=np.float64)
            sigma=float(z['sigma']);assert sigma==[.75,49.][len(inputs)]
            clean=torch.from_numpy(z['images'][slot:slot+1].copy())
            # gaussian() draws on CPU before transferring, matching the original RNG.
            eps=gaussian(clean.shape,cfg['noise_seed']+idx,'cpu')
            inputs.append(((clean+sigma*eps)/(1+sigma))[0].numpy())
            energies.append(energy/energy.sum());attentions.append(attn/attn.sum());sources.append(str(path))
    meta=dict(model=spec,checkpoint_sha256=summary['checkpoint_sha256'],image_index=idx,
              target=target,target_row=target//16,target_column=target%16,sources=sources,
              selection=('Brighter, more colorful image chosen by appearance from the first 16 evaluated iCartoonFace test images; slot 4, fixed central patch (8,8), without inspecting heatmap response.' if slot==4 else 'First prespecified evaluation image and fixed central patch (8,8); independent of heatmap response.'),
              noise_seed=cfg['noise_seed']+idx,probes=cfg['probes'])
    data=dict(noisy_inputs=np.array(inputs),sensitivity=np.array(energies),attention=np.array(attentions),
              sigmas=np.array([.75,49.]),reference_input=inputs[0],clean_input=clean[0].numpy(),metadata=json.dumps(meta))
    saved=OUT/(spec['label']+'_samples_sigma075.npz');np.savez_compressed(saved,**data)
    return data,meta,str(saved)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((REPORT/'allpatch_manifest.json').read_text())
    specs=[next(s for s in manifest['models'] if s['dataset']==dataset)
           for dataset in ['icartoonface64','celeba64','imagenet64','stl10_64']]
    summaries=[];rows=[]
    for spec in specs:
        d=json.loads((REPORT/'figures/allpatch'/(spec['label']+'.json')).read_text())
        assert d['model']==spec and d['config']==manifest['config']
        assert spec['num_images']==1024 and len(d['config']['targets'])==256
        summaries.append(d)
    colors=plt.get_cmap('tab10').colors
    with plt.rc_context({'font.size':11,'pdf.fonttype':42}):
        fig,axes=plt.subplots(1,4,figsize=(18,5.2))
        metrics=[('sensitivity_radius','Sensitivity radius','Patch hops'),('attention_radius','Attention radius','Patch hops'),
                 ('sensitivity_count','Sensitivity count','Patches'),('attention_count','Attention count','Patches')]
        for i,(spec,d) in enumerate(zip(specs,summaries)):
            for ax,(key,title,unit) in zip(axes,metrics):
                for q,style in [(.9,'-'),(.95,'--')]:
                    qi=d['config']['qs'].index(q);values=np.asarray(d[key])[:,qi]
                    assert np.isfinite(values).all()
                    ax.plot(d['config']['sigmas'],values,color=colors[i],ls=style,lw=2.2)
                    rows.extend(dict(model=spec['label'],epoch=spec['epoch'],num_images=1024,targets=256,
                                     metric=key,q=q,sigma=sigma,value=float(v)) for sigma,v in zip(d['config']['sigmas'],values))
                ax.set_xscale('log');ax.set_title(title,fontsize=20,pad=5)
                ax.set_xlabel(r'Effective noise $\sigma_t$',fontsize=18,labelpad=2)
                ax.set_ylabel(unit,fontsize=18,labelpad=2)
                ax.tick_params(axis='both',labelsize=17)
                ax.grid(alpha=.2)
        handles=[Line2D([],[],color=colors[i],label=f"{NAMES[s['dataset']]} (e{s['epoch']})") for i,s in enumerate(specs)]
        handles += [Line2D([],[],color='black',ls=style,label=f'{int(q*100)}%') for q,style in [(.9,'-'),(.95,'--')]]
        fig.legend(handles=handles,loc='lower center',ncol=6,frameon=False,fontsize=18,handlelength=2.2,columnspacing=1.3,handletextpad=.5)
        fig.subplots_adjust(left=.055,right=.997,top=.91,bottom=.24,wspace=.30)
        save(fig,'context_metrics',pad_inches=.01)
        map_specs=[s for s in specs if s['dataset'] in ['icartoonface64','stl10_64']]
        panels=[samples(s,d) for s,d in zip(specs,summaries) if s in map_specs]
        attention=np.concatenate([z['attention'].ravel() for z,_,_ in panels]);positive=attention[attention>0]
        norms={'sensitivity':LogNorm(1e-6,1),'attention':LogNorm(10.**np.floor(np.log10(positive.min())),1)}
        cmaps={'sensitivity':'magma','attention':'viridis'}
        fig,axes=plt.subplots(len(map_specs),5,figsize=(13,2.55*len(map_specs)+.8),layout='compressed')
        mappables={};contexts=[];provenance=[]
        for row,(spec,(z,meta,source)) in enumerate(zip(map_specs,panels)):
            tr,tc=meta['target_row'],meta['target_column']
            axes[row,0].imshow(np.clip((z['clean_input'].transpose(1,2,0)+1)/2,0,1))
            axes[row,0].scatter(4*tc+1.5,4*tr+1.5,s=22,c='cyan',edgecolors='black',linewidths=.5,zorder=4)
            axes[row,0].set_ylabel(f"{NAMES[spec['dataset']]}\nepoch {spec['epoch']}",fontweight='bold')
            for col,(key,si) in enumerate([('sensitivity',0),('sensitivity',1),('attention',0),('attention',1)],1):
                mass=z[key][si];np.testing.assert_allclose(mass.sum(),1,atol=1e-5)
                mappables[key]=axes[row,col].imshow(mass.reshape(16,16),cmap=cmaps[key],norm=norms[key])
                for q,color in [(.95,'white'),(.5,'#FFD700')]:
                    mask,record=context_mask(mass,q)
                    outline_context(axes[row,col],mask,color)
                    contexts.append(dict(model=spec['label'],method=key,sigma=float(z['sigmas'][si]),q=q,**record))
                axes[row,col].scatter(tc,tr,s=22,c='cyan',edgecolors='black',linewidths=.5,zorder=4)
            for ax in axes[row]:ax.set_xticks([]);ax.set_yticks([]);ax.set_box_aspect(1)
            provenance.append(dict(model=spec,source=source,selection=meta['selection'],image_index=meta['image_index'],target=meta['target']))
        for ax,title in zip(axes[0],['Clean reference','Sensitivity\nσ = 0.75','Sensitivity\nσ = 49','Attention\nσ = 0.75','Attention\nσ = 49']):ax.set_title(title)
        for key,columns,label in [('sensitivity',slice(1,3),'Normalized derivative energy'),('attention',slice(3,5),'Normalized attention mass')]:
            bar=fig.colorbar(mappables[key],ax=axes[:,columns].ravel().tolist(),orientation='horizontal',fraction=.01,pad=.008,aspect=35)
            bar.set_label(label+' (log scale)')
        fig.suptitle('Contours: gold 50% · white 95% · Ticks point inside · Cyan dots: target patches',fontsize=12)
        save(fig,'context_maps')
    with (OUT/'context_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'provenance.json').write_text(json.dumps(dict(
        population='1024 held-out images per dataset; population maps first, coverage per target, equal average over all 256 targets.',
        quantiles=[.9,.95],context_quantile=.95,map_sigmas=[.75,49.],reference_sigma=0.,reference_type='clean held-out image',context_display='Full log heatmaps; gold 50% and white 95% contours with inward ticks.',context_contours=[.5,.95],qualitative_samples=provenance,context_sets=contexts,
        heatmaps='Individual noisy-image maps, not population maps; sensitivity averages probes and attention averages heads/layers.',
        color_scales={k:dict(vmin=v.vmin,vmax=v.vmax,cmap=cmaps[k]) for k,v in norms.items()}),indent=2)+'\n')
    print(f'Wrote two appendix figures: {len(specs)} datasets, {len(rows)} metric values, {len(contexts)} context sets.')


if __name__=='__main__':main()
