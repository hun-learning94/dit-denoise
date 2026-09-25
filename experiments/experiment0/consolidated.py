"""Paper figure: two datasets, four normalized maps, and population coverage."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe
from experiments.experiment0.reporting import REPORT

OUT=REPORT/'figures/consolidated'


def sample_path(label):
    suffix='_samples_sigma075_helmet.npz' if label.startswith('imagenet64_20m_epoch') else '_samples_sigma075.npz'
    return OUT/(label+suffix)


def population(manifest,label):
    spec=next(m for m in manifest['models'] if m['label']==label)
    path=REPORT/'figures/allpatch'/(label+'.json')
    record=json.loads(path.read_text())
    assert record['model']==spec and record['config']==manifest['config']
    assert spec['num_images']==1024 and record['config']['targets']==list(range(256))
    qi=record['config']['qs'].index(.95)
    result={}
    for source,target in [('sensitivity_radius','radius'),('sensitivity_count','count'),
                          ('attention_radius','attention_radius'),('attention_count','attention_count')]:
        values=np.asarray(record[source],dtype=float)
        assert values.shape==(len(record['config']['sigmas']),len(record['config']['qs']))
        assert np.isfinite(values).all()
        result[target]=values[:,[qi]]
    return dict(**result,sigmas=record['config']['sigmas'],targets=record['config']['targets'],
                epoch=spec['epoch'],num_images=spec['num_images'],summary_source=str(path),
                checkpoint_sha256=record['checkpoint_sha256'])


def context_mask(mass,q=.95):
    values=np.asarray(mass,dtype=float).ravel()
    assert np.isfinite(values).all() and (values>=0).all() and values.sum()>0
    values=values/values.sum()
    order=np.argsort(-values,kind='stable')
    count=int(np.searchsorted(np.cumsum(values[order]),q,side='left'))+1
    chosen=order[:count]
    assert values[chosen].sum()>=q-1e-12
    assert count==1 or values[order[:count-1]].sum()<q
    mask=np.zeros(values.size,dtype=bool);mask[chosen]=True
    return mask.reshape(16,16),dict(patches=chosen.tolist(),count=count,mass=float(values[chosen].sum()))


def outline_context(ax,mask,color='white'):
    # Draw only exposed edges, preserving disconnected components and holes.
    edges=[];ticks=[]
    for r,c in np.argwhere(mask):
        for dr,dc,edge in [(-1,0,[(c-.5,r-.5),(c+.5,r-.5)]),
                           (1,0,[(c-.5,r+.5),(c+.5,r+.5)]),
                           (0,-1,[(c-.5,r-.5),(c-.5,r+.5)]),
                           (0,1,[(c+.5,r-.5),(c+.5,r+.5)])]:
            rr,cc=r+dr,c+dc
            if not (0<=rr<16 and 0<=cc<16) or not mask[rr,cc]:
                edges.append(edge)
                midpoint=np.mean(edge,axis=0)
                ticks.append([midpoint,midpoint-.22*np.array([dc,dr])])
    outline=LineCollection(edges+ticks,colors=color,linewidths=1.25,zorder=3)
    outline.set_path_effects([pe.Stroke(linewidth=2.1,foreground='black',alpha=.65),pe.Normal()])
    ax.add_collection(outline)


def clean_reference(data,meta):
    if 'clean_input' in data:return data['clean_input']
    from diagnose_dpd import build_folder_dataset,get_images
    dataset=build_folder_dataset(meta['model']['data_root'],64,3)
    clean=get_images(dataset,[meta['image_index']])[0].numpy()
    assert clean.shape==(3,64,64) and np.isfinite(clean).all()
    return clean


def main():
    p=argparse.ArgumentParser();p.add_argument('--require-imagenet600',action='store_true')
    p.add_argument('--require-imagenet-epoch',type=int);a=p.parse_args()
    manifest=json.loads((REPORT/'allpatch_manifest.json').read_text())
    image_spec,=[s for s in manifest['models'] if s['dataset']=='imagenet64']
    image_label=image_spec['label']
    required=600 if a.require_imagenet600 else a.require_imagenet_epoch
    if required is not None:assert image_spec['epoch']==required
    selected=[('CelebA','celeba64_20m',manifest),('ImageNet64',image_label,manifest)]
    curves=[population(manifest,label) for _,label,_ in selected]
    ready=True
    contexts=[]
    samples=[]
    for (_,label,_),curve in zip(selected,curves):
        with np.load(sample_path(label)) as z:samples.append(dict(z))
        meta=json.loads(str(samples[-1]['metadata']))
        manifest=next(m for _,lab,m in selected if lab==label)
        spec=next(m for m in manifest['models'] if m['label']==label)
        assert all(meta['model'][k]==spec[k] for k in ['label','checkpoint','epoch'])
        if curve is not None:assert meta['checkpoint_sha256']==curve['checkpoint_sha256']
        np.testing.assert_array_equal(samples[-1]['sigmas'],[.75,49.])
        if 'clean_input' not in samples[-1]:
            samples[-1]['clean_input']=clean_reference(samples[-1],meta)
            np.savez_compressed(sample_path(label),**samples[-1])
    attention_values=np.concatenate([z['attention'].ravel() for z in samples])
    positive=attention_values[attention_values>0]
    attention_min=10.**np.floor(np.log10(positive.min()))
    attention_max=10.**np.ceil(np.log10(positive.max()))
    if attention_max<=attention_min:attention_max=10*attention_min
    norms={'sensitivity':LogNorm(vmin=1e-6,vmax=1),
           'attention':LogNorm(vmin=attention_min,vmax=attention_max)}
    cmaps={'sensitivity':'magma','attention':'viridis'}
    mappables={}
    with plt.rc_context({'font.size':14,'axes.titlesize':15,'axes.labelsize':14,
                         'xtick.labelsize':12,'ytick.labelsize':12,'pdf.fonttype':42}):
        fig,axes=plt.subplots(2,6,figsize=(20,6.3),layout='compressed',gridspec_kw={'width_ratios':[1,1,1,1,1,1.3]})
        fig.get_layout_engine().set(w_pad=.025,h_pad=.01,wspace=.035,hspace=.005)
        for row,((name,label,_),curve,z) in enumerate(zip(selected,curves,samples)):
            meta=json.loads(str(z['metadata']))
            tr,tc=meta['target_row'],meta['target_column']
            axes[row,0].imshow(np.clip((z['clean_input'].transpose(1,2,0)+1)/2,0,1))
            axes[row,0].scatter(4*tc+1.5,4*tr+1.5,s=22,c='cyan',edgecolors='black',linewidths=.5,zorder=4)
            axes[row,0].set_ylabel(f"{name}\nepoch {meta['model']['epoch']}",fontweight='bold')
            for col,(key,si) in enumerate([('sensitivity',0),('sensitivity',1),('attention',0),('attention',1)],1):
                mass=z[key][si];np.testing.assert_allclose(mass.sum(),1,atol=1e-5)
                mappables[key]=axes[row,col].imshow(mass.reshape(16,16),cmap=cmaps[key],norm=norms[key])
                for q,color in [(.95,'white'),(.5,'#FFD700')]:
                    mask,context=context_mask(mass,q)
                    outline_context(axes[row,col],mask,color)
                    contexts.append(dict(model=label,method=key,sigma=float(z['sigmas'][si]),q=q,**context))
                axes[row,col].scatter(tc,tr,s=22,c='cyan',edgecolors='black',linewidths=.5,zorder=4)
            for ax in axes[row,:5]:ax.set_xticks([]);ax.set_yticks([]);ax.set_box_aspect(1)
        titles=['Clean reference','Sensitivity\nσ = 0.75','Sensitivity\nσ = 49','Attention\nσ = 0.75','Attention\nσ = 49']
        for ax,title in zip(axes[0,:5],titles):ax.set_title(title)
        for dataset,(name,_,_) in enumerate(selected):
            c=curves[dataset]
            if c is None:continue
            color=['#0072B2','#D55E00'][dataset]
            for row,key in enumerate(['radius','attention_radius']):
                for qi,(q,style) in enumerate([(95,'-')]):
                    axes[row,5].plot(c['sigmas'],c[key][:,qi],color=color,linestyle=style,
                                     linewidth=1.8,label=name)
        for row,ax in enumerate(axes[:,5]):
            ax.set(xscale='log',
                   title='95% sensitivity radius' if row==0 else '95% attention radius')
            ax.set_box_aspect(1);ax.grid(alpha=.15);ax.legend(fontsize=10,handlelength=2.8,loc='upper left')
        if not ready:
            for ax in axes[:,5]:
                ax.text(.97,.04,'ImageNet64 radius pending',transform=ax.transAxes,
                        ha='right',va='bottom',fontsize=9,color='#D55E00')
        axes[0,5].sharex(axes[1,5])
        axes[0,5].tick_params(axis='x',which='both',bottom=False,labelbottom=False)
        colorbars=[]
        for key,columns,label in [('sensitivity',slice(1,3),'Normalized derivative energy'),
                                  ('attention',slice(3,5),'Normalized attention mass')]:
            bar=fig.colorbar(mappables[key],ax=axes[:,columns].ravel().tolist(),
                             orientation='horizontal',fraction=.025,pad=.008,shrink=.9,aspect=35)
            bar.ax.tick_params(pad=1)
            bar.set_label(label+' (log scale)',fontsize=13,labelpad=1)
            colorbars.append((bar,columns))
        axes[1,5].set_xlabel(r'Effective noise $\sigma_t$')
        fig.suptitle('Contours: gold 50% · white 95% · Ticks point inside · Cyan: target · Curves: 1,024 held-out images, all 256 targets',fontsize=13)
        # Anchor bars to the heatmaps rather than the adjacent curve's x-label.
        fig.canvas.draw()
        fig.set_layout_engine('none')
        for bar,columns in colorbars:
            position=bar.ax.get_position()
            bottom=min(ax.get_position().y0 for ax in axes[1,columns])
            bar.ax.set_position([position.x0,bottom-.012-position.height,
                                 position.width,position.height])
        for ext in ['png','pdf']:fig.savefig(OUT/f'learned_denoisers.{ext}',dpi=240,bbox_inches='tight',pad_inches=.025)
        plt.close(fig)
    record=dict(models=[dict(label=label,epoch=c['epoch'],num_images=c['num_images'],summary_source=c['summary_source'],target_patches=c['targets'],radius_status='complete' if c is not None else 'pending') for (_,label,manifest),c in zip(selected,curves)],
                qualitative_targets=[{k:json.loads(str(z['metadata']))[k] for k in ['target','target_row','target_column','image_index','selection']} for z in samples],sigmas=[.75,49.],reference_sigma=0.,reference_type='clean held-out image',context_display='Full log heatmaps; gold 50% and white 95% contours with inward ticks.',context_contours=[.5,.95],
                heatmap_order=['sensitivity_low','sensitivity_high','attention_low','attention_high'],
                color_scales={key:dict(vmin=norm.vmin,vmax=norm.vmax,cmap=cmaps[key]) for key,norm in norms.items()},
                summary_quantiles=[.95],summary_metric='patch_radius',summary_methods=['sensitivity','layer_averaged_attention'],
                coverage='Population maps over 1024 held-out images; coverage per target, then equal average over all 256 target patches.',
                context_sets=contexts,context_quantile=.95,context_selection='Descending normalized mass; shortest prefix reaching each contour threshold (50% or 95%); ties by row-major index. Heatmap contexts use the displayed single-image maps, not population maps.',
                imagenet_epoch=image_spec['epoch'],imagenet_complete=ready)
    (OUT/'learned_denoisers.json').write_text(json.dumps(record,indent=2)+'\n')
    report=REPORT/'experiment0.md'
    text=report.read_text()
    start='<!-- consolidated-status:start -->';end='<!-- consolidated-status:end -->'
    if start in text and end in text:
        status=f"- **Consolidated figure checkpoints:** CelebA 20M epoch 400; ImageNet64 20M epoch {image_spec['epoch']}. "
        status+='Both curves use 1,024 held-out images and all 256 target patches. Gold and white contours mark minimum-count 50% and 95% contexts on the displayed single-image sensitivity and attention maps; ticks point into each selected set.'
        text=text[:text.index(start)+len(start)]+'\n'+status+'\n'+text[text.index(end):]
        report.write_text(text)
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
