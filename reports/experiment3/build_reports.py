"""Build the active 20M design, images, and metrics reports from durable results.

The STL-10-source block-transfer replacement uses the completed epoch-800
source campaign and is selected by the same paired mean-FID rule. Existing
hybrid cubes remain tied to their own frozen checkpoints.
"""
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SPE={'imagenet64':40036,'celeba64':5086,'ffhq64':1875,'aahq64':480,'stl10_64':3265,'afhq_v2_64':447,'icartoonface64':12147}
NAMES={'imagenet64':'ImageNet64','celeba64':'CelebA','ffhq64':'FFHQ','aahq64':'AAHQ','stl10_64':'STL-10','afhq_v2_64':'AFHQ-v2','icartoonface64':'iCartoonFace'}
PAIRS=[('celeba64',t) for t in ['aahq64','ffhq64','stl10_64','afhq_v2_64','icartoonface64']]+[('stl10_64',t) for t in ['celeba64','aahq64','ffhq64','afhq_v2_64']]+[(s,'celeba64') for s in ['ffhq64','aahq64','afhq_v2_64']]
PAIRS += [('imagenet64',t) for t in ['celeba64','aahq64','ffhq64','afhq_v2_64']]
DRAFT_TRAJECTORY_TARGETS={
    'celeba64':['aahq64','ffhq64','afhq_v2_64','stl10_64'],
    'stl10_64':['aahq64','ffhq64','afhq_v2_64','celeba64'],
    'imagenet64':['aahq64','ffhq64','afhq_v2_64','celeba64'],
}
KINDS={'denoising':('summary.json','log_sigma_auc_mse_raw','Denoising AUC'),
       'vp_velocity_loss':('summary.json','vp_velocity_loss','VP velocity loss'),
       'fid50k':('result.json','fid','FID-50k')}
SERIES={'joint':('#1768ac','Joint QK+VO+MLP'),'qk_weights':('#a05b19','Restricted QK'),
        'attn_weights':('#28936d','Restricted QK+VO'),'scratch':('#666666','Scratch (seed 0)')}
CODES=['000','100','010','110','001','101','011','111']
IMAGENET_SOURCE_EPOCH=800 if (HERE/'inputs/imagenet800_endpoints.json').exists() else 600
IMAGENET_REPLACEMENT_NOTE='The current ImageNet64 results use the 20M epoch-600 source; we plan to replace them with epoch-800 results after the full transfer and evaluation pipeline completes, retaining epoch 600 as the fallback if time is limited.'

def imagenet_report_text(text):
    if IMAGENET_SOURCE_EPOCH==600:
        if IMAGENET_REPLACEMENT_NOTE not in text:
            parts=text.split('\n\n',2)
            text='\n\n'.join(parts[:2])+ '\n\n'+IMAGENET_REPLACEMENT_NOTE+'\n\n'+parts[2]
        return text
    text=text.replace(IMAGENET_REPLACEMENT_NOTE+'\n\n','')
    text=text.replace('ImageNet64 uses source600 until the validated replacement is published.',
                      'ImageNet64 uses the validated source800 replacement.')
    # Replace checkpoint-specific prose with evidence from the new selection,
    # before updating the general source-epoch references.
    import re
    endpoints=load(HERE/'inputs/imagenet800_endpoints.json')
    description='; '.join(f"{NAMES[r['target']]}: epoch {r['reference_epoch']:g}, mean FID {r['mean_fid']:.3f}" for r in endpoints)
    text=re.sub(r'The ImageNet64 source600 endpoints are frozen[^\n]*',
                lambda _: 'The ImageNet64 source800 endpoints are recorded in [the selection record](inputs/imagenet800_endpoints.json), using all eight paired FID checkpoints: '+description+'.',text)
    text=re.sub(r'The ImageNet64 source600 extension uses \[AAHQ[^\n]*',
                'The ImageNet64 source800 campaign is recorded in [its manifest](../2026-09-19/imagenet800_replacement/selected_manifest.json). Both adaptation seeds are complete at every selected endpoint.',text)
    text=text.replace('; the ongoing source extension to epoch 800 does not change these experiments','')
    return text.replace('imagenet600_endpoints.json','imagenet800_endpoints.json').replace('source600','source800').replace('epoch-600','epoch-800').replace('epoch 600','epoch 800')

def load(path):return json.loads(path.read_text())
def dump(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')
def title(pair):return ' → '.join(NAMES[x] for x in pair)
def ref(pair):return min(pair,key=SPE.get)
def read_value(row,metric):
    file,key,_=KINDS[metric]
    path=ROOT/row['output_root']/metric/row['label']/file
    if not path.is_file():return None
    d=load(path);v=float(d[key]);assert math.isfinite(v),path
    if metric=='fid50k':assert d['generated_images']==50000,path
    return v,str(path.relative_to(ROOT))

def collect():
    historical=load(HERE/'inputs/historical_joint_rows_20m.json')
    joint={}
    # Historical target-reference grids are already exact; source-reference
    # directions use their corrected manifests below.
    for r in historical:
        pair=r['source'],r['target'];step=r['epoch']*SPE[r['target']]
        if ref(pair)==r['target'] and step<=200*SPE[ref(pair)]:
            joint[(*pair,r['seed'],step)]=r
    aligned=load(ROOT/'reports/2026-09-11/aligned200_evaluation_manifest.json')
    reverse=load(ROOT/'reports/2026-09-11/reverse_budget200_evaluation_manifest.json')
    for r in reverse+aligned:
        if r['scale']!='20m' or r.get('mode','transfer')!='transfer':continue
        source=r.get('source') or Path(r['checkpoint']).parts[2].split('_to_')[0]
        if source=='stl10_64':
            continue
        joint[source,r['target'],r['seed'],r['adaptation_updates']]=r
    # Replace the old STL-10-source epoch-400 block trajectories with the
    # measured epoch-800-source campaign. These are the same block_weights
    # adaptation protocol, with a newer source checkpoint.
    for target in ['celeba64','aahq64','ffhq64','afhq_v2_64']:
        for seed in [0,1]:
            reference_spe=min(SPE['stl10_64'],SPE[target])
            for epoch in range(25,201,25):
                joint['stl10_64',target,seed,epoch*reference_spe]=dict(
                    output_root='evals/experiment3_stl10_source800',
                    label=f'stl10_64_to_{target}_seed{seed}_block_weights_epoch{epoch:03d}')
    for target in ['celeba64','aahq64','ffhq64','afhq_v2_64']:
        for seed in [0,1]:
            for epoch in range(25,201,25):
                joint['imagenet64',target,seed,epoch*SPE[target]]=dict(
                    output_root=f'evals/experiment3_transfer_20m_imagenet{IMAGENET_SOURCE_EPOCH}',
                    label=f'imagenet{IMAGENET_SOURCE_EPOCH}_to_{target}_seed{seed}_block_weights_epoch{epoch:03d}')
    points=[]
    def add(pair,series,seed,step,row,metrics=KINDS):
        if step>200*SPE[ref(pair)]:return
        for metric in metrics:
            v=read_value(row,metric)
            if v is not None:points.append(dict(source=pair[0],target=pair[1],series=series,seed=seed,
                reference=ref(pair),reference_epoch=step/SPE[ref(pair)],step=step,metric=metric,value=v[0],artifact=v[1]))
    for (s,t,seed,step),row in joint.items():add((s,t),'joint',seed,step,row)
    restricted=load(ROOT/'reports/2026-09-13/restricted20m_aligned200/comparison_manifest.json')
    for r in restricted['rows']:add((r['source'],r['target']),r['scope'],r['seed'],r['step'],r,[r['metric']])
    # The STL-10 epoch-800 restricted evaluations use the same source and
    # reference grid as the active joint trajectories.
    stl_restricted=ROOT/'reports/2026-09-14/stl10_source_epoch800'
    for scope in ('qk_weights','attn_weights'):
        for r in load(stl_restricted/f'eval_{scope}.json')['rows']:
            assert r['source']=='stl10_64' and r['series']==scope
            assert r['output_root']=='evals/experiment3_stl10_source800'
            add((r['source'],r['target']),scope,r['seed'],r['step'],r)
    recovery=load(ROOT/'reports/2026-09-24/stl10_restricted_metrics_recovery/manifest.json')
    for r in recovery['rows']:
        assert r['source']=='stl10_64' and r['target']=='celeba64'
        assert r['series'] in ('qk_weights','attn_weights')
        assert r['step'] in (175*SPE['stl10_64'],200*SPE['stl10_64'])
        add((r['source'],r['target']),r['series'],r['seed'],r['step'],r)
    completion=load(ROOT/'reports/2026-09-25/stl10_restricted_trajectory_completion/manifest.json')
    for r in completion['rows']:
        assert r['source']=='stl10_64' and r['target']=='celeba64'
        assert r['series'] in ('qk_weights','attn_weights')
        assert r['step'] in (125*SPE['stl10_64'],150*SPE['stl10_64'])
        assert r['output_root']=='evals/experiment3_stl10_source800'
        assert all(read_value(r,metric) is not None for metric in KINDS), r['label']
        add((r['source'],r['target']),r['series'],r['seed'],r['step'],r)
    for source,target in PAIRS:
        if source!='imagenet64':continue
        for scope in ['qk_weights','attn_weights']:
            for seed in [0,1]:
                for epoch in range(25,201,25):
                    add((source,target),scope,seed,epoch*SPE[target],dict(
                        output_root=f'evals/experiment3_restricted20m_imagenet{IMAGENET_SOURCE_EPOCH}',
                        label=f'imagenet{IMAGENET_SOURCE_EPOCH}_to_{target}_seed{seed}_{scope}_epoch{epoch:03d}'))
    # Scratch is a separate seed-0 random-initialization trajectory. Exact
    # source-reference checkpoints come from the aligned scratch union grid.
    for pair in PAIRS:
        source,target=pair
        if ref(pair)==target:
            for epoch in range(25,201,25):
                add(pair,'scratch',0,epoch*SPE[target],dict(output_root='evals/experiment3_20m',label=f'{target}_seed0_scratch_epoch{epoch:03d}'))
        else:
            for r in aligned:
                if r['scale']=='20m' and r['mode']=='scratch' and r['target']==target and r['adaptation_updates'] in [e*SPE[ref(pair)] for e in range(25,201,25)]:
                    add(pair,'scratch',0,r['adaptation_updates'],r)
    keys=[tuple(r[k] for k in ['source','target','series','seed','step','metric']) for r in points]
    assert len(keys)==len(set(keys))
    return points,historical,joint

def select(points,historical):
    frozen={(r['source'],r['target']):r for r in load(HERE/'inputs/frozen_endpoints.json')}
    hybrid_results={
        tuple(r['row'][k] for k in ('source','target'))
        for r in (load(p) for p in sorted((ROOT/'reports/2026-09-16/experiment3_hybrid_metrics/cache/results').glob('*.json')))
        if 'row' in r
    }
    imagenet={(r['source'],r['target']):r for r in load(HERE/f'inputs/imagenet{IMAGENET_SOURCE_EPOCH}_endpoints.json')}
    stl_path=HERE/'inputs/stl800_endpoints.json'
    stl800={(r['source'],r['target']):r for r in load(stl_path)} if stl_path.exists() else {}
    selections=[]
    for pair in PAIRS:
        fid={(r['seed'],r['step']):r['value'] for r in points if (r['source'],r['target'])==pair and r['series']=='joint' and r['metric']=='fid50k'}
        paired=[e*SPE[ref(pair)] for e in range(25,201,25) if all((seed,e*SPE[ref(pair)]) in fid for seed in [0,1])]
        best=min(paired,key=lambda step:((fid[0,step]+fid[1,step])/2,step)) if paired else None
        # STL-10-source trajectories now use the paired-FID selection from
        # the epoch-800-source replacement campaign; their old hybrid cubes
        # are retained separately at the old endpoints.
        step=frozen[pair]['target_epoch']*SPE[pair[1]] if pair in frozen and pair[0]!='stl10_64' else best
        r=dict(source=pair[0],target=pair[1],reference=ref(pair),step=step,
               reference_epoch=step/SPE[ref(pair)] if step else None,
               target_epoch=step/SPE[pair[1]] if step else None,
               status='Frozen measured endpoint' if pair in frozen else 'Provisional FID candidate' if len(paired)<8 else 'FID-selected; images/metrics not yet generated',
               paired_fid_coverage=len(paired),current_grid_candidate_step=best,
               current_grid_candidate_reference_epoch=best/SPE[ref(pair)] if best else None,
               mean_fid=None)
        if pair[0]=='stl10_64':
            r['status']='FID-selected epoch-800-source replacement; hybrid metrics remain at legacy endpoint'
        if pair not in frozen:
            if step is None:
                r['status']='No eligible paired-FID endpoint'
            else:
                stem=f'primary_20m_{pair[0]}_to_{pair[1]}_step{step}_generation_seed0'
                if all((HERE/'figures'/f'{stem}.{ext}').is_file() for ext in ['png','pdf','npz','json']):
                    r['status']=('Provisional FID selection' if len(paired)<8 else 'FID-selected')+'; seed-0 images available; hybrid metrics pending'
        if pair in hybrid_results:
            r['status']='FID-selected; paired hybrid metrics measured at this endpoint'
        if pair in stl800:
            endpoint=stl800[pair]
            assert len(paired)==8 and best==endpoint['step'] and step==endpoint['step'], ('STL800 endpoint differs from complete FID grid',pair)
            r.update(source_training_epoch=800,selection_evidence=endpoint['selection_evidence'],
                     source_checkpoint=endpoint['source_checkpoint'],adapted_checkpoints=endpoint['adapted_checkpoints'])
            available=[load(p) for p in (HERE/'metrics/stl800_results').glob('*.json')]
            completed={x['row']['adaptation_seed'] for x in available
                       if (x['row']['source'],x['row']['target'])==pair
                       and x['row']['adaptation_updates']==step and x['row']['source_training_epoch']==800}
            assert completed=={0,1}, ('Incomplete STL800 endpoint metrics',pair)
            r['status']='Final source800 endpoint; paired hybrid metrics measured'
        if pair in imagenet:
            endpoint=imagenet[pair]
            assert len(paired)==8 and best==endpoint['step'], ('Frozen ImageNet endpoint differs from complete FID grid',pair)
            assert step==endpoint['step']
            r.update(source_training_epoch=IMAGENET_SOURCE_EPOCH,selection_evidence=endpoint['selection_evidence'],
                     source_checkpoint=endpoint['source_checkpoint'],adapted_checkpoints=endpoint['adapted_checkpoints'])
            available=[load(p) for p in (HERE/f'metrics/imagenet{IMAGENET_SOURCE_EPOCH}_results').glob('*.json')]
            completed={x['row']['adaptation_seed'] for x in available
                       if (x['row']['source'],x['row']['target'])==pair
                       and x['row']['adaptation_updates']==step and x['row']['source_training_epoch']==IMAGENET_SOURCE_EPOCH}
            r['status']=f'Frozen source{IMAGENET_SOURCE_EPOCH} endpoint; '+('paired hybrid metrics measured' if completed=={0,1} else 'images/metrics pipeline pending')
        vals=[]
        if step:
            for seed in [0,1]:
                if (seed,step) in fid:vals.append(fid[seed,step])
                else:
                    old=next((x for x in historical if (x['source'],x['target'],x['seed'],x['epoch']*SPE[x['target']])==(*pair,seed,step)),None)
                    value=read_value(old,'fid50k') if old else None
                    if value:vals.append(value[0])
        if len(vals)==2:r['mean_fid']=sum(vals)/2
        selections.append(r)
    return selections

def plot_trajectories(points,selections):
    selected={(r['source'],r['target']):r for r in selections}
    indexed={}
    for row in points:
        key=(row['source'],row['target'],row['metric'],row['series'],row['seed'])
        indexed.setdefault(key,{})[row['reference_epoch']]=row['value']
    metric_names={'denoising':'Denoising AUC','vp_velocity_loss':'VP loss','fid50k':'FID-50k'}
    short_labels={'joint':'Joint (111)','qk_weights':'QK only',
                  'attn_weights':'QK+VO only','scratch':'Scratch'}
    legend=[Line2D([],[],color=color,label=short_labels[series])
            for series,(color,_) in SERIES.items()]
    legend += [Line2D([],[],color='black',ls='-',label='Seed 0'),
               Line2D([],[],color='black',ls='--',label='Seed 1'),
               Line2D([],[],color='#9a2b64',ls='-.',label='Selected 111 endpoint')]

    def panel(ax,source,target,metric,compact=False):
        pair=(source,target)
        for series,(color,_) in SERIES.items():
            for seed in ([0] if series=='scratch' else [0,1]):
                vals=indexed.get((source,target,metric,series,seed),{})
                xs=list(range(25,201,25))
                ys=[vals.get(x,np.nan) for x in xs]
                ax.plot(xs,ys,color=color,ls=':' if series=='scratch' else '-' if seed==0 else '--',
                        marker='^' if series=='scratch' else 'o' if seed==0 else 's',
                        ms=2.0 if compact else 2.6,lw=.9 if compact else 1.1,clip_on=False)
        endpoint=selected[pair]['reference_epoch']
        if endpoint is not None:
            ax.axvline(endpoint,color='#9a2b64',ls='-.',lw=.65 if compact else .9,alpha=.8)
        ax.set_xlim(25,200)
        ax.set_xticks([25,100,200])
        ax.grid(alpha=.18,lw=.4)
        ax.spines[['top','right']].set_visible(False)
        ax.ticklabel_format(axis='y',style='plain',useOffset=False)
        ax.tick_params(axis='both',labelsize=7.3 if compact else 8,length=2,pad=1)

    figures=[]
    for source,targets in DRAFT_TRAJECTORY_TARGETS.items():
        fig,axes=plt.subplots(3,4,figsize=(12.4,6.7),squeeze=False)
        for i,metric in enumerate(KINDS):
            for j,target in enumerate(targets):
                ax=axes[i,j]
                panel(ax,source,target,metric)
                if i==0:ax.set_title(f'{NAMES[source]} → {NAMES[target]}',fontsize=10)
                if j==0:ax.set_ylabel(metric_names[metric],fontsize=9)
                if i==2:ax.set_xlabel('Reference epochs',fontsize=8)
        fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.005),ncol=4,frameon=False,fontsize=8)
        fig.subplots_adjust(left=.072,right=.965,top=.958,bottom=.14,hspace=.30,wspace=.30)
        name=f'trajectories_{source}_20m'
        for ext in ['png','pdf']:fig.savefig(HERE/'figures'/f'{name}.{ext}',dpi=150)
        plt.close(fig);figures.append((source,name))

    fig,axes=plt.subplots(9,4,figsize=(7.2,7.8),squeeze=False)
    for block,(source,targets) in enumerate(DRAFT_TRAJECTORY_TARGETS.items()):
        for i,metric in enumerate(KINDS):
            for j,target in enumerate(targets):
                ax=axes[3*block+i,j]
                panel(ax,source,target,metric,compact=True)
                if i==0:ax.set_title(f'{NAMES[source]} → {NAMES[target]}',fontsize=8.3,pad=2)
                if j==0:ax.set_ylabel(metric_names[metric],fontsize=7.5,labelpad=2)
                if block==2 and i==2:ax.set_xlabel('Reference epochs',fontsize=7,labelpad=1)
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.005),ncol=4,frameon=False,fontsize=7,
               handlelength=1.7,columnspacing=.5,labelspacing=.1)
    fig.subplots_adjust(left=.085,right=.958,top=.98,bottom=.09,hspace=.48,wspace=.32)
    combined=HERE/'figures'/'trajectories_draft_20m_combined'
    for ext in ('pdf','png'):fig.savefig(combined.with_suffix(f'.{ext}'),dpi=220)
    plt.close(fig)
    return figures

def write_design(points,selections,figures):
    text='''# Experiment 3: design and adaptation trajectories

[Primary endpoint images](experiment3_images.md) · [Hybrid metrics](experiment3_metrics.md) · [Selected findings](experiment3_summary.md)

## Motivation and hypotheses

The practical question is which parts of a pretrained denoiser should adapt to a new image distribution. In the aggregator–predictor interpretation, QK selects contextual evidence, VO transports that evidence, and the MLP maps contextualized features to patch corrections. These roles interact through the residual stream; none is assumed to store geometry or semantics exclusively.

1. **Compatible context, changed appearance:** source QK may remain useful when the target can be modeled with similar contextual organization, while adapted MLP/VO carry substantial visible change. CelebA→AAHQ is a motivating example.
2. **Coordinated adaptation:** a larger distribution shift may require jointly adapted QK, VO, and MLP; successful joint generation can coexist with failed partial transplants.
3. **Directional reuse:** source-QK compatibility may differ between opposite transfer directions. Dataset labels only motivate this hypothesis; they do not isolate geometry from appearance.

Two experiments answer distinct questions. The **joint-model transplant cube** identifies compatibility among jointly learned updates. **Independent restricted fine-tuning** measures what QK-only or QK+VO-only optimization can learn under the same adaptation-update budget. A poor transplant does not imply that independently training that subset must fail.

## Generation quality and scope of claims

This is a controlled study of component-specific adaptation, not a benchmark of state-of-the-art unconditional generation. The fixed 20M architecture produces imperfect samples, particularly on broad natural-image domains such as STL-10; absolute fidelity and object coherence should not be confused with the relative component effects studied here. We limit qualitative claims to effects that remain interpretable in matched-latent comparisons and consider them alongside quantitative measurements. Illustrative samples do not establish typical quality or success frequency; complete grids and selection rules are provided in the companion reports.

Extended scratch training is a check on sensitivity to training duration, not evidence of convergence by itself. Its results must be measured before claiming that additional training leaves quality or adaptation conclusions unchanged. Hardware constraints and epoch counts alone do not establish adequate training, and the present results do not demonstrate that 20M capacity is intrinsically insufficient. The STL-10-source block-transfer replacement is the separately documented exception: it uses the epoch-800 source and its own paired-FID endpoint record.

## Datasets and model

All active models are 20M-class DiTs (width 320, depth 10, 10 heads, MLP ratio 4), operating on 64×64 RGB images with 4×4 patches. Dataset counts refer to prepared files after filtering, rather than nominal download sizes.

| Dataset | Training images | Test images | Content |
|---|---:|---:|---|
'''
    descriptions={'celeba64':'Aligned photographic celebrity faces','ffhq64':'Photographic faces','aahq64':'Artistic portraits','stl10_64':'Natural objects; retained labeled + unlabeled training images','afhq_v2_64':'Pooled cat/dog/wild faces; pixel duplicates removed','icartoonface64':'Recognition face crops; test identities may overlap training'}
    for r in load(ROOT/'reports/2026-09-11/dataset_split_counts.json'):
        text+=f"| {NAMES[r['dataset']]} | {r['train']:,} | {r['test']:,} | {descriptions[r['dataset']]} |\n"
    text+='| ImageNet64 | 1,281,167 | 50,000 | Downsampled natural images; official validation used for evaluation |\n'
    text+=r'''
Joint transfers cover CelebA→AAHQ/FFHQ/STL-10/AFHQ-v2/iCartoonFace, STL-10→CelebA/AAHQ/FFHQ/AFHQ-v2, FFHQ/AAHQ/AFHQ-v2→CelebA, and ImageNet64→CelebA/AAHQ/FFHQ/AFHQ-v2. Independent QK-only and QK+VO-only training covers these directions. Scratch comparisons use target-domain seed 0 only.

## Training and equal-budget convention

Each original transfer starts from its source's **epoch-400 seed-0 scratch EMA**, resets optimizer and adaptation step, and uses adaptation seeds 0 and 1. The STL-10-source replacement starts from the epoch-800 STL-10 scratch EMA. ImageNet64 transfers start from its epoch-600 seed-0 EMA; the ongoing source extension to epoch 800 does not change these experiments. Joint adaptation trains QK, VO, and MLP block weights; restricted adaptation trains QK or QK+VO. AdaLN, patch/time/position machinery, output head, and all unselected weights stay frozen. Scratch starts from random initialization and trains all parameters.

Training uses batch 32, AdamW at learning rate 0.0002, zero weight decay, gradient clipping 1, and EMA decay 0.999 under the VP direct-image-prediction/velocity-loss objective. Precision follows the established hardware-dependent launcher setting. Equal update budgets are not equal wall-clock compute budgets, and source pretraining is excluded from adaptation axes.

The **reference dataset is the smaller source/target training dataset**. One reference epoch is $s_R=\min(\lfloor N_S/32\rfloor,\lfloor N_T/32\rfloor)$ optimizer updates. Every primary comparison is capped at $200s_R$ updates, with checkpoints at $25s_R,50s_R,\ldots,200s_R$. Joint, independently restricted, and scratch curves use the same denominator. Earlier source-reference restricted trajectories are superseded; only the corrected run directories enter these plots.

| Direction | Reference dataset | Updates per 25 reference epochs | Maximum adaptation updates |
|---|---|---:|---:|
'''
    for p in PAIRS:text+=f'| {title(p)} | {NAMES[ref(p)]} | {25*SPE[ref(p)]:,} | {200*SPE[ref(p)]:,} |\n'
    text+='''
## Primary endpoint selection

The selection rule minimizes **two-seed mean joint-model FID-50k** within the reference-epoch cap, uses the same checkpoint for both seeds, and breaks exact ties toward the earlier checkpoint. Denoising and VP losses do not rank endpoints. Missing paired FIDs make a new selection provisional. Selection uses the monitored test split and is not an untouched-test estimate.

The previously reviewed hybrid endpoints remain **frozen** during this report refresh. Their images and measurements stay tied to their original checkpoints. STL-10-source transfer trajectories now use the completed epoch-800-source block campaign and the paired-FID selections recorded in the replacement manifest; existing STL-10 hybrid cubes remain legacy measurements until recomputed at those new endpoints.

The ImageNet64 source600 endpoints are frozen in [their selection record](inputs/imagenet600_endpoints.json). CelebA selects epoch 175 (890,050 updates), mean FID 14.849, after completing all eight paired checkpoints; epoch 200 has mean FID 15.013. AAHQ, FFHQ, and AFHQ-v2 retain their selected epochs 150, 100, and 175.

| Direction | Selected reference epoch | Equivalent target epoch | Joint mean FID | Paired FID coverage on current grid | Endpoint status |
|---|---:|---:|---:|---:|---|
'''
    def fmt(x):return 'Pending' if x is None else f'{x:.3f}'.rstrip('0').rstrip('.')
    for r in selections:
        text+=f"| {title((r['source'],r['target']))} | {fmt(r['reference_epoch'])} | {fmt(r['target_epoch'])} | {fmt(r['mean_fid'])} | {r['paired_fid_coverage']}/8 | {r['status']} |\n"
    differences=[r for r in selections if r['current_grid_candidate_step'] is not None and r['step']!=r['current_grid_candidate_step']]
    if differences:
        text+='\nCurrent-grid candidates that differ from the frozen endpoint: '+ '; '.join(f"{title((r['source'],r['target']))}: {fmt(r['current_grid_candidate_reference_epoch'])} reference epochs" for r in differences)+'. These have not replaced the measured endpoint.\n'
    text+='''
[Exact endpoint steps and selection record](tables/endpoint_selections.json) · [Metric values and artifact provenance](tables/trajectories_20m.csv)

## Adaptation trajectories

The panels show the twelve transfer directions included in the draft. Each source figure has three metric rows and four target columns: held-out denoising AUC, VP velocity loss, and FID-50k (all lower is better). Colors distinguish joint, QK-only, QK+VO-only, and scratch training. Solid/dashed lines distinguish adaptation seeds; dotted scratch curves use seed 0. Missing evaluations remain gaps. The STL-10→CelebA restricted trajectories now have both seeds and all three metrics at every reference epoch from 25 to 200. Vertical markers identify the selected primary endpoint; STL-10-source curves and markers use the epoch-800 source.

'''
    for source,name in figures:text+=f'### {NAMES[source]} source\n\n![20M {NAMES[source]} adaptation trajectories](figures/{name}.png)\n\n'
    text+=r'''## Source reversion and generation protocol

At a fixed jointly trained endpoint, form $\theta_{rvm}=\theta_S+r\Delta_{QK}+v\Delta_{VO}+m\Delta_{MLP}$ for bits $(r,v,m)\in\{0,1\}^3$. Zero means **restore source parameters**, never zero weights or activations. All eight hybrids share the same frozen AdaLN and global machinery.

| State | Adapted components retained |
|---|---|
| 000 | Source only |
| 100 | QK |
| 010 | VO |
| 110 | QK+VO |
| 001 | MLP |
| 101 | QK+MLP |
| 011 | VO+MLP |
| 111 | Joint QK+VO+MLP |

For the ImageNet64 source600 extension, displayed grids use the first 16 predetermined quantitative latents (seed 20260911); they are not an independent visual bank. The original full visual bank has 16 predetermined Gaussian latents (seed 20260904), shared across states, directions, and adaptation seeds, with EMA weights, FP32, and 50-step VP Heun sampling. Retain all rows, including failed hybrids; common noise does not guarantee common object identity. The quantitative bank has 1,024 latents (seed 20260911) per state and adaptation seed. Visual hypotheses and fixed-endpoint feature measurements are distinct from independent restricted-training performance.

The [images report](experiment3_images.md) displays the complete available endpoint grids; the [metrics report](experiment3_metrics.md) defines the quantitative questions and results. The [summary](experiment3_summary.md) retains the selected illustrative rows and interpretation. Rebuild the first three reports with `python reports/experiment3/build_reports.py`; the summary remains separately edited.
'''
    if (HERE/'inputs/stl800_endpoints.json').exists():
        text=text.replace('STL-10-source transfer trajectories now use the completed epoch-800-source block campaign and the paired-FID selections recorded in the replacement manifest; existing STL-10 hybrid cubes remain legacy measurements until recomputed at those new endpoints.',
                          'STL-10-source trajectories, endpoint images, and hybrid metrics use the epoch-800 source and the complete paired-FID selections in [the source800 selection record](inputs/stl800_endpoints.json). Historical source400 measurements remain archived.')
    (HERE/'experiment3.md').write_text(imagenet_report_text(text))

def write_images(selections):
    text='''# Experiment 3: primary endpoint images

[Design and authoritative endpoint choices](experiment3.md) · [Hybrid metrics](experiment3_metrics.md) · [Selected findings](experiment3_summary.md)

These are the twelve selected 20M joint-model source-reversion cubes, ordered CelebA → AAHQ, FFHQ, AFHQ-v2, STL-10; STL-10 → AAHQ, FFHQ, AFHQ-v2, CelebA; ImageNet64 → AAHQ, FFHQ, AFHQ-v2, CelebA. The [metrics report](experiment3_metrics.md) begins with real-dataset spatial similarity before considering generated-image geometry. Endpoints retain their recorded provenance. Only adaptation seed 0 is displayed, with all 16 predetermined latents and all eight states in display order `000, 100, 010, 110, 001, 101, 011, 111` (QK/VO/MLP bits). Sampling uses EMA, FP32, and 50-step VP Heun. Original visual grids use latent seed 20260904; ImageNet64 source600 grids show the first 16 quantitative latents, seed 20260911. Zero bits restore source parameters. These are not samples from independently restricted fine-tuning. Seed-1 artifacts and the two-seed quantitative analysis are retained separately.

Previously measured hybrid directions retain their frozen checkpoints. STL-10-source endpoint images use the new epoch-800-source selections; their legacy quantitative hybrid measurements still refer to the older source and endpoints. NPZ files retain image arrays and latent information (indices for ImageNet64 source600); JSON sidecars retain sampling provenance and component-checkpoint hashes.

'''
    import sys
    sys.path.insert(0, str(HERE))
    from hybrid_analysis import PAIRS as image_pairs
    selected = {(r['source'], r['target']): r for r in selections}
    for pair in image_pairs:
        r = selected[pair]
        text+=f'## {title(pair)}\n\n'
        if r['step'] is None:text+='No eligible paired-FID endpoint yet.\n\n';continue
        text+=f"Selected endpoint: **{r['reference_epoch']:.3f} {NAMES[r['reference']]} reference epochs**, {r['step']:,} updates ({r['target_epoch']:.3f} target epochs). {r['status']}.\n\n"
        if pair[0]=='imagenet64':text+='Source: ImageNet64 epoch 600. Displayed latents: first 16 of the quantitative bank (seed 20260911).\n\n'
        for seed in [0]:
            stem=f"primary_20m_{pair[0]}_to_{pair[1]}_step{r['step']}_generation_seed{seed}"
            if not (HERE/'figures'/f'{stem}.png').exists():
                stem=f"primary_20m_{pair[0]}_to_{pair[1]}_epoch{r['target_epoch']:g}_generation_seed{seed}"
            if (HERE/'figures'/f'{stem}.png').exists():
                text+=f'### Adaptation seed {seed}\n\n![{title(pair)}, seed {seed}, all 16 latents](figures/{stem}.png)\n\n[PDF](figures/{stem}.pdf) · [Arrays and latents](figures/{stem}.npz) · [Provenance](figures/{stem}.json)\n\n'
            else:text+=f'Seed {seed}: generation at this exact checkpoint is not yet available.\n\n'
    if (HERE/'inputs/stl800_endpoints.json').exists():
        text=text.replace('STL-10-source endpoint images use the new epoch-800-source selections; their legacy quantitative hybrid measurements still refer to the older source and endpoints.',
                          'STL-10-source endpoint images and quantitative hybrid measurements use the same finalized epoch-800-source selections. Both adaptation seeds have complete visual banks; seed 0 is displayed here.')
    (HERE/'experiment3_images.md').write_text(imagenet_report_text(text))

def write_metrics():
    import sys
    sys.path.insert(0, str(HERE))
    from build_hybrid_metrics import build
    build()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-metrics-only', action='store_true',
                        help='Use saved endpoints; preserve design, summary, and their assets')
    parser.add_argument('--trajectories-only', action='store_true',
                        help='Refresh trajectory data, plots, and design report without recomputing hybrid metrics')
    parser.add_argument('--plots-only', action='store_true',
                        help='Rebuild trajectory figures from the saved metric table and endpoint selections')
    args = parser.parse_args()
    assert sum((args.images_metrics_only,args.trajectories_only,args.plots_only))<=1
    if args.plots_only:
        with (HERE/'tables/trajectories_20m.csv').open() as f:
            points=list(csv.DictReader(f))
        for row in points:
            row['seed']=int(row['seed'])
            row['reference_epoch']=float(row['reference_epoch'])
            row['value']=float(row['value'])
        plot_trajectories(points,load(HERE/'tables/endpoint_selections.json'))
        print('Rebuilt 3x4 source trajectories and the combined appendix figure from saved metrics.')
        return
    if args.images_metrics_only:
        selections = load(HERE/'tables/endpoint_selections.json')
        write_images(selections)
        write_metrics()
        return
    points,historical,_=collect();selections=select(points,historical)
    with (HERE/'tables/trajectories_20m.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(points[0]),lineterminator='\n');writer.writeheader();writer.writerows(points)
    dump(HERE/'tables/endpoint_selections.json',selections)
    figures=plot_trajectories(points,selections)
    write_design(points,selections,figures)
    if not args.trajectories_only:
        write_images(selections);write_metrics()
    dump(HERE/'tables/build_audit.json',dict(timestamp=datetime.now(timezone.utc).isoformat(),
        trajectory_metric_count=len(points),direction_count=len(PAIRS),frozen_endpoint_count=8,
        imagenet_source_epoch=IMAGENET_SOURCE_EPOCH,imagenet_frozen_endpoint_count=4,
        restricted_source='reports/2026-09-13/restricted20m_aligned200/comparison_manifest.json',
        stl800_restricted_sources=[f'reports/2026-09-14/stl10_source_epoch800/eval_{scope}.json'
                                   for scope in ('qk_weights','attn_weights')]
                                  +['reports/2026-09-24/stl10_restricted_metrics_recovery/manifest.json',
                                    'reports/2026-09-25/stl10_restricted_trajectory_completion/manifest.json'],
        draft_trajectory_direction_count=sum(map(len,DRAFT_TRAJECTORY_TARGETS.values()))))
    report_kind='trajectory report' if args.trajectories_only else 'three 20M reports'
    print(f'Built {report_kind}; {len(points)} trajectory metrics; frozen hybrid endpoints unchanged.')

if __name__=='__main__':main()
