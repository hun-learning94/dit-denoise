"""Paired heldout-image token-score gallery with donor and recipient attention."""
import argparse,gc,json,os
from pathlib import Path
import numpy as np
import torch
from .prepare import atomic_json,sha256
from .protocol import seed_for
from .imagenet_fields import clean,save
from .donor_models import Model
from .donor_full import read,checked
from .donor_covariance import tokens


def setup(parent):
    root=parent/'score_comparison';root.mkdir(exist_ok=True)
    assert not (root/'design.json').exists()
    d={'parent_report':str(parent),'input_hashes':{},'primary_times':[.1,.5,.9],'gallery':{},'selection':'First four heldout records for CelebA; first two heldout images in each of the first two frozen ImageNet classes. Fixed positions nearest tokens85,136,187; selection does not use scores/losses.','noise':'Same Gaussian realization across each image pair at each time; all models receive identical physical corruption.','score_display':'Native16x16 token maps; Distance uses Chebyshev token distance (lower is nearer); other scores sum-normalized to mean1, common LogNorm(.03,30), saturated ends shown by colorbar.'}
    for ds in ['celeba128','imagenet256']:
        old=read(parent/ds/'design.json');recs=[];pairs=[]
        for pair in range(2):
            if ds=='celeba128':rr=old['evaluation'][pair*2:pair*2+2];qs=[85,136,187]
            else:
                info=old['classes'][pair];rr=[r for r in old['evaluation'] if r['wnid']==info['wnid']][:2];qs=[]
                for q in [85,136,187]:
                    available=[v for v in info['queries'] if v not in qs];qs.append(min(available,key=lambda v:(abs(v//16-q//16)+abs(v%16-q%16),v)))
            for r in rr:recs.append(dict(**r,gallery_pair=pair,gallery_queries=qs))
            pairs.append(dict(pair=pair,images=[r['image_id'] for r in rr],queries=qs))
        d['gallery'][ds]=dict(records=recs,pairs=pairs,models=old['models'],donor=old['donor_for'][old['models'][0]],recipient=old['models'][0])
        d['input_hashes'][str(parent/ds/'design.json')]=sha256(parent/ds/'design.json')
    atomic_json(root/'design.json',d)


def noisy(rec,ds,t,d):
    x=clean(rec);z=torch.randn(x.shape,generator=torch.Generator().manual_seed(seed_for('self_gallery_shared_pair_noise_v1',ds,rec['gallery_pair'],t)))
    return t*x+d['noise_scale']*(1-t)*z


def extract(parent):
    root=parent/'score_comparison';g=read(root/'design.json');work=Path(read(parent/'celeba128/design.json')['work_root']).parent/'score_comparison';work.mkdir(exist_ok=True)
    for p,h in g['input_hashes'].items():assert sha256(p)==h
    for ds,spec in g['gallery'].items():
        d=read(parent/ds/'design.json');cases=[(r,t) for r in spec['records'] for t in g['primary_times']]
        x=torch.stack([noisy(r,ds,t,d) for r,t in cases]);tt=torch.tensor([t for r,t in cases],device='cuda');ll=torch.tensor([r.get('class_id',0) for r,t in cases],device='cuda');qs=np.array([r['gallery_queries'] for r,t in cases])
        for name in spec['models']:
            p=work/f'{ds}_{name}.npz';rp=root/f'{ds}_{name}.json'
            if rp.exists():assert sha256(p)==read(rp)['sha256'];continue
            model=Model(name,d)
            with torch.inference_mode():
                with model.readout(qs) as got:model(x.cuda(),tt,ll)
                score=got['score'].cpu().numpy();mass=got['class_mass'].cpu().numpy()
            assert np.isfinite(score).all() and np.allclose(score.sum(-1)+mass,1,atol=2e-6)
            save(p,score=score,class_mass=mass,queries=qs,noisy=x.numpy())
            atomic_json(rp,dict(passed=True,path=str(p),sha256=sha256(p),gallery_design_sha256=sha256(root/'design.json'),model=model.receipt))
            del model,got;gc.collect();torch.cuda.empty_cache()
    atomic_json(root/'EXTRACT_COMPLETE.json',dict(passed=True,design_sha256=sha256(root/'design.json')))


def draw(parent,covariance_override=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm,Normalize
    from matplotlib.patches import Rectangle
    root=parent/'score_comparison';g=read(root/'design.json');assert read(root/'EXTRACT_COMPLETE.json')['passed'];images=[];plt.rcParams.update({'font.size':13,'axes.titlesize':13,'pdf.fonttype':42})
    for ds,spec in g['gallery'].items():
        d=read(parent/ds/'design.json');dh=sha256(parent/ds/'design.json');donor=spec['donor'];recipient=spec['recipient'];readouts={};fields={};grad={};cov={}
        for name in spec['models']:
            r=read(root/f'{ds}_{name}.json');assert sha256(r['path'])==r['sha256']
            with np.load(r['path']) as z:readouts[name]={k:z[k] for k in z.files}
            for wn in sorted({'face' if ds=='celeba128' else rec['wnid'] for rec in spec['records']}):
                fields[name,wn]=checked(parent/ds/name/'fields'/f'{wn}.json',dh)
        if ds=='celeba128':
            assert sha256(d['reused_fields'])==d['reused_fields_sha256']
            with np.load(d['reused_fields']) as z:grad['face']=tokens(z['gradient']);cov['face']=tokens(z['covariance'])
        else:
            for wn in sorted({rec['wnid'] for rec in spec['records']}):
                grad[wn]=checked(parent/ds/donor/'gradient'/f'{wn}.json',dh)['score'];cov[wn]=checked(parent/ds/'covariance'/f'{wn}.json',dh)['score']
        if covariance_override is not None:
            for wn in cov:
                replacement=covariance_override[ds][wn]
                lookup={int(q):j for j,q in enumerate(replacement['queries'])}
                qs=[lookup[int(q)] for q in fields[donor,wn]['queries']]
                ts=[int(np.flatnonzero(np.isclose(replacement['times'],t,rtol=0,atol=1e-12))[0]) for t in d['times']]
                cov[wn]=replacement['score'][ts][:,qs]
        for ii,rec in enumerate(spec['records']):
            wn='face' if ds=='celeba128' else rec['wnid'];fq=fields[donor,wn]['queries'];lookup={int(q):j for j,q in enumerate(fq)}
            for qi,q in enumerate(rec['gallery_queries']):
                rr,cc=divmod(q,16);r,c=np.divmod(np.arange(256),16);distance=np.maximum(abs(r-rr),abs(c-cc)).reshape(16,16)
                fig,axes=plt.subplots(3,8,figsize=(21,8.4),layout='constrained');titles=['Noisy input','Distance',f'Sensitivity\n{donor.upper()}','Wiener',f'Attention (average)\n{donor.upper()}',f'Attention\n{donor.upper()}',f'Self attention (average)\n{recipient.upper()}',f'Self attention\n{recipient.upper()}']
                raw_scores=[]
                for row,t in enumerate(g['primary_times']):
                    ti=d['times'].index(t);frame=ii*3+row;j=lookup[q]
                    scores=[grad[wn][ti,j],cov[wn][ti,j],fields[donor,wn]['calibration'][ti,j],readouts[donor]['score'][frame,qi],fields[recipient,wn]['calibration'][ti,j],readouts[recipient]['score'][frame,qi]]
                    raw_scores.append(np.stack(scores))
                    arr=readouts[donor]['noisy'][frame];assert np.array_equal(arr,readouts[recipient]['noisy'][frame]);axes[row,0].imshow(np.clip(arr.transpose(1,2,0)/2+.5,0,1));p=arr.shape[-1]//16;axes[row,0].add_patch(Rectangle((cc*p-.5,rr*p-.5),p,p,fill=False,ec='#ef3038',lw=2))
                    dim=axes[row,1].imshow(distance,cmap='viridis_r',norm=Normalize(0,15),interpolation='nearest')
                    for col,values in enumerate(scores,2):
                        assert np.isfinite(values).all() and values.min()>=0 and values.sum()>0
                        display=values*256/values.sum();heat=axes[row,col].imshow(display.reshape(16,16),cmap='magma',norm=LogNorm(.03,30),interpolation='nearest')
                    for col in range(1,8):axes[row,col].add_patch(Rectangle((cc-.5,rr-.5),1,1,fill=False,ec='#24dfd0',lw=1.5))
                    axes[row,0].set_ylabel(f't={t:g}',fontsize=15)
                    for col in range(8):
                        axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
                        if row==0:axes[row,col].set_title(titles[col])
                fig.colorbar(dim,ax=axes[:,1].tolist(),location='bottom',shrink=.8,pad=.025,label='Token distance (nearer is brighter)')
                fig.colorbar(heat,ax=axes[:,2:].ravel().tolist(),location='bottom',shrink=.75,pad=.025,label='Score / spatial mean (display only)',extend='both')
                fig.suptitle(f'{ds} · {rec.get("class_name","")} · {rec["image_id"]} · target token {q} (row {rr}, col {cc}; zero-based)',fontsize=16)
                stem=f'{ds}_pair{rec["gallery_pair"]}_image{ii%2}_q{q:03d}';out=root/'figures';out.mkdir(exist_ok=True)
                for ext in ['png','pdf']:fig.savefig(out/f'{stem}.{ext}',dpi=150,bbox_inches='tight',pad_inches=.04)
                plt.close(fig);save(out/f'{stem}_raw.npz',scores=np.stack(raw_scores),times=g['primary_times'],query=q,distance=distance)
                images.append(dict(dataset=ds,pair=rec['gallery_pair'],image=ii%2,image_id=rec['image_id'],query=q,path=f'score_comparison/figures/{stem}.png'))
    text='''## Score-map comparison

Each figure fixes one heldout image and target patch. Rows are **t=0.1,0.5,0.9**, from noisier to cleaner. Columns show noisy input, Distance, Sensitivity, Wiener, donor average/current attention, and self-model average/current attention. Each image is immediately followed by its paired image at the **same target location**, using the **same Gaussian noise tensor at each time**. The target is outlined; these are score fields, not thresholded masks.

Distance displays Chebyshev token distance with a separate reversed scale. Other fields are displayed on the common16×16 token grid, divided by their spatial mean, using the same logarithmic color scale (0.03–30; saturated endpoints indicated). This display normalization does not change rankings or evaluation losses and does not make different raw score units numerically equivalent. Sensitivity and Wiener use the validated token-maximum construction. Average attention and sensitivity are calibrated on disjoint images (within the same class for ImageNet). They remain fixed across a same-class pair; an object–scene pair uses separate class-specific templates, so changes include class conditioning as well as image content. All attention maps already average heads and layers; “average” additionally averages calibration observations. Current attention is recomputed from the displayed noisy observation. Labels identify the model supplying each field.

The displayed Sensitivity reference is SMLP for CelebA and JiT-L for ImageNet, matching the donor of the first recipient direction. Wiener is dataset/class-specific. The four attention columns expose both models; this gallery does not switch the Sensitivity reference between columns. Images were selected by clean-image content/framing before inspecting their attention scores; all selected examples are included. These are illustrative gallery choices and do not change the full evaluation sets or numerical conclusions.
'''
    for ds in ['celeba128','imagenet256']:
        text+=f'\n### {ds}\n'
        for pair in [0,1]:
            qs=sorted({i['query'] for i in images if i['dataset']==ds and i['pair']==pair})
            for q in qs:
                text+=f'\n#### Image pair {pair+1}, patch {q}\n'
                for item in sorted([v for v in images if v['dataset']==ds and v['pair']==pair and v['query']==q],key=lambda v:v['image']):
                    text+=f'\n**{item["image_id"]}**\n\n![{ds} {item["image_id"]}, patch {q}, score maps]({item["path"]})\n'
    p=parent/'FINDINGS.md';s=p.read_text();heading='## Score-map comparison'
    if heading in s:s=s[:s.index(heading)].rstrip()+'\n'
    p.write_text(s+'\n'+text)
    atomic_json(root/'COMPLETE.json',dict(passed=True,figures=images,design_sha256=sha256(root/'design.json'),findings_sha256=sha256(p)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--report',type=Path,required=True);ap.add_argument('--stage',choices=['setup','extract','draw'],required=True);a=ap.parse_args();parent=a.report.resolve();torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')));torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.stage=='setup':setup(parent)
    elif a.stage=='extract':
        assert torch.cuda.is_available();extract(parent)
    else:draw(parent)
if __name__=='__main__':main()
