"""Donor-only paired, image-level reports; hierarchical ImageNet uncertainty."""
from pathlib import Path
import argparse,csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import Formatter, MaxNLocator
from .prepare import atomic_json,sha256
from .imagenet_fields import save
from .donor_full import checked

LABELS=['Distance','Sensitivity','Wiener','Attention (average)','Attention','Self attention (average)','Self attention']
COLORS=['#333333','#E69F00','#009E73']
MODEL_NAMES={'mlp':'MLP','smlp':'SMLP','jit_b':'JiT-B','jit_l':'JiT-L'}
MODEL_COLORS={'mlp':'#0072B2','smlp':'#AA4499','jit_b':'#0072B2','jit_l':'#AA4499'}

class ScientificTicks(Formatter):
    """One decimal per tick, with a shared multiplicative scale above the axis."""
    def __init__(self, exponent):
        self.exponent=exponent

    def __call__(self, value, pos=None):
        scaled=value/10.**self.exponent
        if abs(scaled)<.05:scaled=0.
        return self.fix_minus(f'{scaled:.1f}')

    def get_offset(self):
        return rf'$\times 10^{{{self.exponent}}}$'

def curve_style(d,recipient,mi):
    if d['methods'][mi]=='wiener_pooled':return 'Wiener (pooled)','#D55E00','-'
    if mi==2 and 'wiener_pooled' in d['methods']:return 'Wiener (class-specific)',COLORS[mi],'-'
    if mi<3:return LABELS[mi],COLORS[mi],'-'
    source=d['donor_for'][recipient] if mi<5 else recipient
    average=mi in (3,5)
    return MODEL_NAMES[source]+' attention'+(' (average)' if average else ''),MODEL_COLORS[source],'--' if average else '-'

def estimates(values,indices):
    draws=np.stack([values[ii].mean(0) for ii in indices]);return values.mean(0),np.quantile(draws,[.025,.975],axis=0)

def bootstrap(d):
    rng=np.random.default_rng(20261108);N=len(d['evaluation']);out=[]
    for _ in range(d['bootstrap_replicates']):
        if d['dataset']=='celeba128':ix=rng.integers(N,size=N)
        else:
            cls=rng.integers(32,size=32);ix=np.concatenate([c*8+rng.integers(8,size=8) for c in cls])
        out.append(ix)
    return out

def plot_curves(root,d,outputs):
    limits=[]
    plt.rcParams.update({'font.size':12,'axes.labelsize':13,'axes.titlesize':13,'legend.fontsize':11,'pdf.fonttype':42,'ps.fonttype':42})
    for zoom in [False,True]:
        for all_times in [False,True]:
            times=d['times'] if all_times else d['primary_times'];ks=np.array(d['budgets']);mask=(ks>=12)&(ks<=48) if zoom else np.ones(len(ks),bool)
            fig,axes=plt.subplots(2,len(times),figsize=(4*len(times),6.2),squeeze=False,layout='constrained')
            for ri,model in enumerate(d['models']):
                data=outputs[model]
                for col,t in enumerate(times):
                    ti=d['times'].index(t);ax=axes[ri,col]
                    for mi in range(len(d['methods'])):
                        label,color,style=curve_style(d,model,mi)
                        attention=3<=mi<=6
                        ax.plot(ks[mask],data['excess_mean'][ti,mask,mi],color=color,linestyle=style,linewidth=2.5 if attention else 1.5,label=label,zorder=5 if attention else 2)
                    # Fit only the displayed means; a zero reference must
                    # not enlarge the panel's data range.
                    visible=data['excess_mean'][ti,mask,:].ravel()
                    lo,hi=float(visible.min()),float(visible.max());pad=max((hi-lo)*.06,max(abs(lo),abs(hi))*1e-6,1e-12)
                    ax.set_ylim(lo-pad,hi+pad)
                    locator=MaxNLocator(nbins=5,steps=[1,2,5,10],min_n_ticks=3)
                    ax.yaxis.set_major_locator(locator)
                    exponent=int(np.floor(np.log10(max(abs(lo),abs(hi),1e-12))))
                    spacing=float(np.diff(locator.tick_values(lo-pad,hi+pad)).min())
                    while spacing/10.**exponent<.1:exponent-=1
                    ax.yaxis.set_major_formatter(ScientificTicks(exponent))
                    if lo-pad<=0<=hi+pad:ax.axhline(0,color='.55',lw=.7)
                    limits.append(dict(zoom=zoom,all_times=all_times,recipient=model,t=t,lower=lo-pad,upper=hi+pad,exponent=exponent))
                    ax.set_title(f"Denoiser: {MODEL_NAMES[model]}, t={t:g}")
                    ax.set_xticks(ks[mask] if zoom else [9,32,64,128,192,256]);ax.margins(x=.01);ax.grid(alpha=.16)
                    if ri==1:ax.set_xlabel('Retained patches')
                    if col==0:ax.set_ylabel('Excess clean-patch MSE')
            legend={}
            for ax in axes[:,0]:
                handles,labels=ax.get_legend_handles_labels()
                for handle,label in zip(handles,labels):legend.setdefault(label,handle)
            desired=[curve_style(d,d['models'][0],i)[0] for i in range(3)]
            if 'wiener_pooled' in d['methods']:desired+=['Wiener (pooled)']
            desired+=[MODEL_NAMES[m]+' attention'+suffix for m in d['models'] for suffix in [' (average)','']]
            labels=[label for label in desired if label in legend]
            fig.legend([legend[label] for label in labels],labels,loc='outside lower center',ncol=3,frameon=False)
            stem=('preview_12_48' if zoom else 'full_context')+('_all_times' if all_times else '_primary')
            for ext in ['png','pdf']:fig.savefig(root/f'{stem}.{ext}',dpi=200,bbox_inches='tight',pad_inches=.03)
            plt.close(fig)
    atomic_json(root/'plot_presentation.json',dict(yaxis='Per-panel visible mean-curve range, six percent padding; zero only when in range',yaxis_ticks='One decimal with a shared mathtext power-of-ten multiplier per panel; no additive offset',confidence_bands=False,confidence_intervals='Retained in plot_data.npz, curves.csv, and paired_contrasts.csv',baseline_labels=LABELS[:3],attention_labels=[MODEL_NAMES[m]+' attention'+suffix for m in d['models'] for suffix in [' (average)','']],source_colors={m:MODEL_COLORS[m] for m in d['models']},baseline_colors=dict(zip(LABELS[:3],COLORS)),limits=limits,plot_source_sha256=sha256(Path(__file__))))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--report',required=True,type=Path);ap.add_argument('--plots-only',action='store_true');a=ap.parse_args();root=a.report.resolve();d=json.loads((root/'design.json').read_text())
    if a.plots_only:
        outputs={}
        for model in d['models']:
            with np.load(root/model/'plot_data.npz') as z:outputs[model]={k:z[k] for k in ['excess_mean','excess_low','excess_high']}
        plot_curves(root,d,outputs);return
    dh=sha256(root/'design.json');idx=bootstrap(d);outputs={};rows=[];contrasts=[]
    for model in d['models']:
        z=checked(root/model/'EVALUATION_COMPLETE.json',dh);loss=z['losses'];dense=z['independent_dense']
        assert loss.shape==(256,2,len(d['times']),8,11,len(d['methods'])) and np.isfinite(loss).all()
        assert np.max(np.abs(loss[:,:,:,:,-1,:]-dense[...,None]))<1e-5
        image_mse=loss.mean((1,3));image_dense=dense.mean((1,3));excess=image_mse-image_dense[:,:,None,None]
        mean,ci=estimates(excess,idx);mse=image_mse.mean(0)
        outputs[model]=dict(image_mse=image_mse,image_dense=image_dense,excess_mean=mean,excess_low=ci[0],excess_high=ci[1])
        save(root/model/'plot_data.npz',**outputs[model],times=d['times'],budgets=d['budgets'],methods=d['methods'],image_ids=z['image_ids'])
        for ti,t in enumerate(d['times']):
            for ki,k in enumerate(d['budgets']):
                for mi,m in enumerate(d['methods']):rows.append(dict(donor=d['donor_for'][model],recipient=model,t=t,budget=k,method=m,masked_mse=mse[ti,ki,mi],dense_mse=image_dense[:,ti].mean(),excess=mean[ti,ki,mi],low=ci[0,ti,ki,mi],high=ci[1,ti,ki,mi]))
        pairs=[(4,3),(4,0),(3,0),(4,1),(4,2)]+([(6,5),(6,4),(5,3),(6,0),(6,2)] if len(d['methods'])>=7 else [])
        if 'wiener_pooled' in d['methods']:pairs+=[(7,2),(7,0),(4,7),(6,7)]
        for lhs,rhs in pairs:
            contrast,cci=estimates(excess[:,:,:,lhs]-excess[:,:,:,rhs],idx)
            for ti,t in enumerate(d['times']):
                for ki,k in enumerate(d['budgets']):contrasts.append(dict(donor=d['donor_for'][model],recipient=model,t=t,budget=k,comparison=d['methods'][lhs]+' minus '+d['methods'][rhs],mean=contrast[ti,ki],low=cci[0,ti,ki],high=cci[1,ti,ki]))
        if d['dataset']=='imagenet256':
            cls=image_mse.reshape(32,8,*image_mse.shape[1:]).mean(1);save(root/model/'class_data.npz',image_mse=cls,class_ids=[i['class_id'] for i in d['classes']])
    for name,records in [('curves.csv',rows),('paired_contrasts.csv',contrasts)]:
        with (root/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    plot_curves(root,d,outputs)
    text=f'''# {d['dataset']} attention context full results

All evaluations and dense-endpoint checks passed. Both directions use 256 images, eight balanced positions/image, two noise draws, {len(d['methods'])} selectors, and the same 11 context budgets. All attention is averaged over heads and layers. Each recipient is scored against its own dense baseline.

[Compact 12–48 preview](preview_12_48_primary.png) · [PDF](preview_12_48_primary.pdf) · [All times](preview_12_48_all_times.png) · [Full count range](full_context_primary.png) · [CSV](curves.csv) · [Paired contrasts](paired_contrasts.csv).

Intervals are paired, pointwise 95% bootstrap intervals over {'images' if d['dataset']=='celeba128' else 'classes, then images within sampled class'}, after averaging targets and noise draws within image. They do not establish equivalence. Smaller excess loss means less damage from restriction; negative values are possible for a frozen network.

{d['evaluation_history']}

Current-image selectors read the full noisy input; recipient closure is conditional on frozen selected indices. These results test useful transferable context selection, not an exact population decomposition term or interchangeability of learned QK weights.
'''
    (root/'RESULTS.md').write_text(text)
    atomic_json(root/'COMPLETE.json',dict(passed=True,design_sha256=dh,directions=d['donor_for'],curves_sha256=sha256(root/'curves.csv'),contrasts_sha256=sha256(root/'paired_contrasts.csv'),report_sha256=sha256(root/'RESULTS.md')))
if __name__=='__main__':main()
