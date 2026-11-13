"""Replace only Wiener losses using full-bank fields and exact cached forwards."""
from __future__ import annotations
import argparse
import csv
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .prepare import atomic_json, sha256
from .imagenet_fields import clean, save
from .donor_full import read, checked, ready, receipt
from .donor_models import Model, observation, patch_losses
from .jit_transfer import order


def covariance(root, d, gallery=False):
    source=Path(d['covariance_report']); expected=d['covariance_design_sha256']
    assert sha256(source/'design.json')==expected
    cd=read(source/'design.json')
    infos=[dict(wnid='face')] if d['dataset']=='celeba128' else (cd['imagenet']['classes'] if gallery else d['classes'])
    fields={}; receipts={}; counts={}
    for info in infos:
        wn=info['wnid']; path=source/('celeba128' if wn=='face' else 'imagenet256/'+wn)
        r=read(path/'COMPLETE.json'); b=read(path/'BANK.json')
        assert r['passed'] and b['passed'] and r['design_sha256']==b['design_sha256']==expected
        assert r['images']==b['images']
        assert sha256(r['path'])==r['sha256']
        with np.load(r['path']) as z: f={k:z[k] for k in z.files}
        assert f['score'].shape==(len(f['times']),len(f['queries']),256) and np.isfinite(f['score']).all()
        assert np.array_equal(f['times'],d['times']) and (f['score']>=0).all()
        fields[wn]=f; counts[wn]=b['images']
        for name in ['COMPLETE.json','BANK.json']:receipts[str(path/name)]=sha256(path/name)
    return fields, receipts, counts


def map_contexts(d, oldplan, fields, evaluation=None):
    ev=d['evaluation'] if evaluation is None else evaluation
    T=len(d['times']); K=len(d['budgets']); oldn=len(oldplan['observations'])
    shape=(len(ev),2,T,8,K); mapping=np.empty(shape,np.int32); masks=[]; observations=[]
    assert np.all(np.diff(oldplan['observations'])>=0)
    ranks={}
    for wn,f in fields.items():
        ranks[wn]={int(q):[order(f['score'][ti,j],int(q)) for ti in range(T)] for j,q in enumerate(f['queries'])}
    for ii,rec in enumerate(ev):
        wn='face' if d['dataset']=='celeba128' else rec['wnid']
        for dr in range(2):
            for ti in range(T):
                oi=(ii*2+dr)*T+ti
                a,b=np.searchsorted(oldplan['observations'],[oi,oi+1])
                dedup={oldplan['masks'][j].tobytes():j for j in range(a,b)}
                for qi,q in enumerate(rec['targets']):
                    rank=ranks[wn][q][ti]
                    assert np.array_equal(np.sort(rank),np.arange(256)) and q in rank[:9]
                    previous=np.zeros(256,bool)
                    for ki,k in enumerate(d['budgets']):
                        keep=np.zeros(256,bool);keep[rank[:k]]=True
                        assert int(keep.sum())==k and np.all(~previous|keep); previous=keep
                        packed=np.packbits(keep); key=packed.tobytes()
                        if key not in dedup:
                            dedup[key]=oldn+len(masks);masks.append(packed);observations.append(oi)
                        mapping[ii,dr,ti,qi,ki]=dedup[key]
    assert (mapping[...,0]<oldn).all() and (mapping[...,-1]<oldn).all()
    return dict(mapping=mapping,masks=np.asarray(masks,dtype=np.uint8).reshape(-1,32),
                observations=np.asarray(observations,dtype=np.int32))


def plan(root,d,name):
    parent=Path(d['forward_cache_report']); ph=d['forward_cache_design_sha256']
    oldplan=checked(parent/name/'PLAN_COMPLETE.json',ph)
    fields,refs,counts=covariance(root,d)
    p=map_contexts(d,oldplan,fields);oldn=len(oldplan['observations'])
    path=Path(d['work_root'])/name/'plan.npz';save(path,**p)
    atomic_json(root/name/'PLAN_COMPLETE.json',receipt(root,d,path,dict(
        original_jobs=oldn,new_forwards=len(p['masks']),covariance_receipts=refs,covariance_images=counts,
        parent_plan_sha256=sha256(parent/name/'PLAN_COMPLETE.json'),
        reused_logical_entries=int(np.sum(p['mapping']<oldn)))))
    print(name,'new Wiener forwards',len(p['masks']),flush=True)


def check_gpu(model,d,baseline,x,truth,tt,labels,queries):
    T=len(d['times']); N=len(d['evaluation']); obs_indices=[0,T//2,2*T-1,(N//2)*2*T+T//2,(N-1)*2*T+T-1]
    ix=torch.tensor(obs_indices); ii=ix//(2*T); xx=x[ix].cuda(); t=tt[ix].cuda(); ll=labels[ix].cuda()
    qs=queries[ii].cuda(); target=truth[ii].cuda(); dense_ref=baseline['independent_dense'].reshape(-1,8)[obs_indices]
    with torch.inference_mode():
        pred=model(xx,t,ll); dense=patch_losses(pred,target,qs).cpu().numpy()
        full=patch_losses(model(xx,t,ll,torch.ones((len(ix),256),dtype=torch.bool,device='cuda')),target,qs).cpu().numpy()
        keep=np.zeros((len(ix),256),bool)
        for j,q in enumerate(qs[:,0].cpu().numpy()):keep[j,order(np.arange(256,dtype=float),int(q))[:24]]=True
        keep=torch.from_numpy(keep).cuda(); masked=model(xx,t,ll,keep)
        p=xx.shape[-1]//16; pixels=keep.reshape(-1,1,16,16).repeat_interleave(p,2).repeat_interleave(p,3)
        altered=model(torch.where(pixels,xx,torch.zeros_like(xx)+2),t,ll,keep)
        leak=float(((masked-altered)*pixels).abs().max())
    err=float(np.max(np.abs(dense-dense_ref))); ferr=float(np.max(np.abs(full-dense)))
    assert err<2e-5 and ferr<2e-5 and leak<2e-6,(err,ferr,leak)
    return dict(passed=True,saved_dense_max_abs=err,full_dense_max_abs=ferr,outside_replacement_max_abs=leak)


def evaluate(root,d,name):
    parent=Path(d['forward_cache_report']); old=read(parent/'design.json'); ph=d['forward_cache_design_sha256']
    base=Path(d['baseline_report']); baseline=checked(base/name/'EVALUATION_COMPLETE.json',d['baseline_design_sha256'])
    original=checked(parent/name/'EVALUATION_COMPLETE.json',ph)
    bench=ready(parent,old,name); batch=bench['inference']['selected']['batch']
    pr=read(root/name/'PLAN_COMPLETE.json'); p=checked(root/name/'PLAN_COMPLETE.json',sha256(root/'design.json'))
    assert sha256(parent/name/'PLAN_COMPLETE.json')==pr['parent_plan_sha256']
    for rp,h in pr['covariance_receipts'].items():assert sha256(rp)==h,rp
    oldplan=checked(parent/name/'PLAN_COMPLETE.json',ph); oldn=pr['original_jobs']
    oldloss=np.empty((oldn,8),np.float32)
    for start in range(0,oldn,batch):
        stop=min(start+batch,oldn)
        got=np.load(Path(old['work_root'])/name/'evaluation'/f'chunk_{start:08d}_{stop:08d}.npy')
        assert got.shape==(stop-start,8) and np.isfinite(got).all();oldloss[start:stop]=got
    assert np.array_equal(oldloss[oldplan['mapping'],np.arange(8)[None,None,None,:,None,None]],original['losses'])
    assert np.array_equal(baseline['losses'][...,:5],original['losses'])
    assert np.array_equal(baseline['independent_dense'],original['independent_dense'])
    obs=p['observations'];masks=p['masks'];N=len(obs);T=len(d['times'])
    work=Path(d['work_root'])/name/'evaluation';work.mkdir(parents=True,exist_ok=True)
    identity=dict(design=sha256(root/'design.json'),plan=pr['sha256'],code=sha256(Path(__file__)),batch=batch)
    if (work/'identity.json').exists():assert read(work/'identity.json')==identity
    else:atomic_json(work/'identity.json',identity)
    model=Model(name,d);model.fused(bench['inference']['selected']['fused'])
    ev=d['evaluation'];cases=[(r,ti,dr) for r in ev for dr in range(2) for ti in range(T)]
    x=torch.stack([observation(r,ti,dr,d) for r,ti,dr in cases]);truth=torch.stack([clean(r) for r in ev])
    tt=torch.tensor([d['times'][ti] for r,ti,dr in cases]);ll=torch.tensor([r.get('class_id',0) for r,ti,dr in cases]);qs=torch.tensor([r['targets'] for r in ev])
    checks=check_gpu(model,d,baseline,x,truth,tt,ll,qs);atomic_json(root/name/'GPU_CHECKS.json',checks)
    torch.cuda.empty_cache();newloss=np.empty((N,8),np.float32);started=time.monotonic()
    with torch.inference_mode():
        for start in range(0,N,batch):
            stop=min(start+batch,N);part=work/f'chunk_{start:08d}_{stop:08d}.npy'
            if part.exists():
                got=np.load(part);assert got.shape==(stop-start,8) and np.isfinite(got).all();newloss[start:stop]=got;continue
            oi=torch.tensor(obs[start:stop].astype(np.int64));ii=oi//(2*T)
            keep=torch.tensor(np.unpackbits(masks[start:stop],axis=1).astype(bool),device='cuda')
            pred=model(x[oi].cuda(),tt[oi].cuda(),ll[oi].cuda(),keep)
            got=patch_losses(pred,truth[ii].cuda(),qs[ii].cuda()).cpu().numpy()
            assert np.isfinite(got).all();newloss[start:stop]=got
            with part.with_suffix('.tmp').open('wb') as f:np.save(f,got)
            part.with_suffix('.tmp').replace(part);del pred,keep
            atomic_json(root/name/'progress.json',dict(completed=stop,total=N,seconds=time.monotonic()-started))
            print(name,stop,'/',N,flush=True)
    alljobs=np.concatenate([oldloss,newloss]); values=alljobs[p['mapping'],np.arange(8)[None,None,None,:,None]]
    dense=baseline['independent_dense'];err=float(np.max(np.abs(values[...,-1]-dense)));assert err<1e-5
    wi=d['methods'].index('lukoianov_covariance'); combined=baseline['losses'].copy();combined[...,wi]=values
    others=[j for j in range(7) if j!=wi]
    assert np.array_equal(combined[...,others],baseline['losses'][...,others])
    assert np.array_equal(combined[...,0,wi],baseline['losses'][...,0,wi])
    assert np.array_equal(combined[...,-1,wi],baseline['losses'][...,-1,wi])
    path=work/'losses.npz';save(path,losses=combined,independent_dense=dense,methods=d['methods'],
        budgets=d['budgets'],times=d['times'],targets=qs.numpy(),image_ids=baseline['image_ids'])
    atomic_json(root/name/'EVALUATION_COMPLETE.json',receipt(root,d,path,dict(new_forwards=N,
        reused_original_jobs=oldn,other_six_methods_identical=True,core_and_full_identical=True,
        dense_mse_max_abs=err,seconds=time.monotonic()-started,model=model.receipt,
        covariance_images=pr['covariance_images'],plan_sha256=pr['sha256'])))


def report(root,d):
    from .report_donor_full import bootstrap, estimates
    # Run the maintained seven-method renderer without changing its CLI logic.
    import subprocess,sys
    subprocess.run([sys.executable,'-m','experiments.attention_context_transfer.report_donor_full','--report',str(root)],check=True)
    indices=bootstrap(d);rows=[]; summaries={}
    for name in d['models']:
        old=checked(Path(d['baseline_report'])/name/'EVALUATION_COMPLETE.json',d['baseline_design_sha256'])
        new=checked(root/name/'EVALUATION_COMPLETE.json',sha256(root/'design.json'))
        wi=d['methods'].index('lukoianov_covariance')
        delta=(new['losses'][...,wi]-old['losses'][...,wi]).mean((1,3))
        mean,ci=estimates(delta,indices)
        summaries[name]=dict(lower_loss_points=int((mean[:,1:-1]<0).sum()),interior_points=int(mean[:,1:-1].size))
        for ti,t in enumerate(d['times']):
            for ki,k in enumerate(d['budgets']):
                rows.append(dict(recipient=name,t=t,budget=k,new_minus_old=float(mean[ti,ki]),low=float(ci[0,ti,ki]),high=float(ci[1,ti,ki])))
    with (root/'wiener_bank_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    atomic_json(root/'REFRESH_COMPLETE.json',dict(passed=True,design_sha256=sha256(root/'design.json'),
        result_receipt_sha256=sha256(root/'COMPLETE.json'),summaries=summaries,
        comparison_sha256=sha256(root/'wiener_bank_comparison.csv')))


def gallery(root):
    from .attention_score_comparison import draw
    replacement={};refs={}
    for ds in ['celeba128','imagenet256']:
        fields,rr,_=covariance(root/ds,read(root/ds/'design.json'),gallery=True);replacement[ds]=fields;refs.update(rr)
    draw(root/'gallery',covariance_override=replacement)
    gd=read(root/'gallery/score_comparison/design.json');old=Path(gd['reused_gallery_parent'])/'score_comparison/figures'
    compared=0
    for path in (root/'gallery/score_comparison/figures').glob('*_raw.npz'):
        with np.load(path) as a,np.load(old/path.name) as b:
            assert np.array_equal(a['scores'][...,[0,2,3,4,5],:],b['scores'][...,[0,2,3,4,5],:])
            for key in ['times','query','distance']:assert np.array_equal(a[key],b[key])
        compared+=1
    assert compared==24,compared
    atomic_json(root/'gallery/WIENER_REFRESH.json',dict(passed=True,covariance_receipts=refs,
        figures_receipt_sha256=sha256(root/'gallery/score_comparison/COMPLETE.json'),
        unchanged_other_score_fields=True,compared_panels=compared))


def findings(root,base):
    from .report_donor_full import MODEL_NAMES
    def link(p):return os.path.relpath(p,base)
    gallery_text=(root/'gallery/FINDINGS.md').read_text().split('## Score-map comparison\n\n',1)[1]
    gallery_text=gallery_text.replace('](score_comparison/figures/',']('+link(root/'gallery/score_comparison/figures')+'/')
    score_notes,gallery_text=gallery_text.split('\n### celeba128\n\n',1)
    face_gallery,imagenet_gallery=gallery_text.split('\n### imagenet256\n\n',1)
    galleries={'celeba128':face_gallery,'imagenet256':imagenet_gallery}
    text='''# Attention experiment: current findings

**Full-training-set Wiener update complete.** The plots and score maps below use the expanded covariance estimates. Only Wiener denoising losses were recomputed; the other six methods, saved dense losses, nine-patch core, images, target patches, noise draws, and context budgets are unchanged. All four evaluation checks passed.

Attention curves retain source-model names and consistent colors: MLP/JiT-B blue, SMLP/JiT-L purple; solid=current image, dashed=calibration average. Distance is black, Sensitivity orange, and Wiener green. Each panel fits its y-axis to the visible mean curves, with one-decimal ticks and a shared power-of-ten multiplier. Shaded bands are omitted; the pointwise 95% bootstrap intervals remain in the saved CSVs and paired comparisons. Larger t means cleaner input.

'''
    text+='[ImageNet256](#imagenet256-jit-b-and-jit-l) · [CelebA128](#celeba128-mlp-and-smlp) · [How to read the score maps](#reading-the-score-maps).\n\n'
    for ds in ['imagenet256','celeba128']:
        d=read(root/ds/'design.json');_,_,counts=covariance(root/ds,d)
        rr=read(root/ds/'REFRESH_COMPLETE.json'); caption='CelebA128: MLP and SMLP' if ds=='celeba128' else 'ImageNet256: JiT-B and JiT-L'
        sample=f"{counts['face']:,} training images" if ds=='celeba128' else f"{min(counts.values()):,}–{max(counts.values()):,} eligible training images per class across the 32 evaluation classes"
        text+=f'## {caption}\n\nWiener uses **{sample}**.\n\n### Score maps\n\n'
        text+=galleries[ds].strip()+'\n\n### Denoising curves\n\n'
        text+=f'![{caption}, refreshed seven-method curves]({link(root/ds/"preview_12_48_primary.png")})\n\n'
        text+=' · '.join(f'[{label}]({link(root/ds/file)})' for label,file in [('All times','preview_12_48_all_times.png'),('Full context range','full_context_primary.png'),('PDF','preview_12_48_primary.pdf'),('Paired comparisons','paired_contrasts.csv'),('Wiener new minus old','wiener_bank_comparison.csv')])+'\n\n'
        text+='At **32 retained patches**, raw excess clean-patch MSE (lower is better):\n\n'
        text+='| Recipient | t | Distance | Sensitivity | Wiener | '+MODEL_NAMES[d['models'][0]]+' attention | '+MODEL_NAMES[d['models'][1]]+' attention |\n|---|---:|---:|---:|---:|---:|---:|\n'
        for name in d['models']:
            with np.load(root/ds/name/'plot_data.npz') as z:means=z['excess_mean']
            for t in [.1,.5,.9]:
                ti=d['times'].index(t);ki=d['budgets'].index(32)
                att=[6 if m==name else 4 for m in d['models']]
                vals=[means[ti,ki,j] for j in [0,1,2]+att]
                text+=f'| {MODEL_NAMES[name]} | {t:g} | '+' | '.join(f'{v:.7f}' for v in vals)+' |\n'
        text+='\nRelative to the smaller-bank Wiener estimate, the new point estimate has lower denoising loss at '
        text+=', '.join(f"{v['lower_loss_points']}/{v['interior_points']} interior noise/budget points for {MODEL_NAMES[k]}" for k,v in rr['summaries'].items())
        text+='. These are descriptive counts, not independent significance tests; the linked paired intervals retain the full grid.\n\n'
    text+='## Reading the score maps\n\n'+score_notes.strip()+'\n\n'
    text+='''## Interpretation and scope

The unchanged attention-versus-Distance results show useful current-image context selection at low noise, with strong geometric performance at high noise. SMLP-supplied attention is better for both CelebA recipients at the previously reported representative medium/low-noise points. JiT donor/self attention comparisons are close at those points; this is not a formal equivalence claim. The new Wiener comparisons are in the tables and paired intervals above, rather than inherited from the smaller-bank analysis.

Both datasets use 256 heldout images, eight balanced target positions/image, two noise draws, and 11 budgets. CelebA has nine noise levels; ImageNet has five and 32 classes. Intervals are paired, pointwise 95% bootstrap intervals over images for CelebA and classes then images for ImageNet. Current-image selectors see the full noisy image; recipient closure is conditional on the selected indices. CelebA models share common-backbone initialization and its heldout set was previously inspected.

Wiener construction remains centered empirical RGB covariance with 1/N normalization, the matching signal/noise coefficients, row-relative absolute coefficients, and maxima across target RGB coordinates, input channels, and input-token pixels. The expansion changes sample coverage, not the readout or a fitted threshold. ImageNet estimates remain class-specific and rank-limited by available observations. More samples do not guarantee smooth score maps or a better Gaussian approximation.

'''
    old=Path(read(root/'celeba128/design.json')['forward_cache_report']).parent
    text+=f'[Earlier smaller-bank findings, preserved]({link(old/"FINDINGS_before_full_wiener.md")}).\n\n'
    return text


def publish(root):
    for ds in ['celeba128','imagenet256']:
        r=read(root/ds/'REFRESH_COMPLETE.json');assert r['passed']
        assert sha256(root/ds/'COMPLETE.json')==r['result_receipt_sha256']
    assert read(root/'gallery/WIENER_REFRESH.json')['passed']
    parent=Path(read(root/'celeba128/design.json')['forward_cache_report']).parent
    original=parent/'FINDINGS.md';backup=parent/'FINDINGS_before_full_wiener.md'
    if not backup.exists():backup.write_bytes(original.read_bytes())
    for target in [root/'FINDINGS.md',original]:
        text=findings(root,target.parent);tmp=target.with_suffix('.tmp');tmp.write_text(text);tmp.replace(target)
    atomic_json(root/'COMPLETE.json',dict(passed=True,findings_sha256=sha256(original),
        previous_findings_sha256=sha256(backup),other_six_methods_preserved=True))
    status=parent.parents[2]/'draft_revision/attention/STATUS_RESULTS.md'
    if status.exists():
        s=status.read_text();begin='<!-- WIENER_REFRESH_STATUS -->';end='<!-- /WIENER_REFRESH_STATUS -->'
        message=f'{begin}\n\nFull-bank Wiener denoising refresh **completed** for all four recipients. Other six methods and dense/core/full endpoint losses were preserved exactly. [Updated findings]({os.path.relpath(original,status.parent)}). New curves, paired intervals, old/new Wiener contrasts, and all 24 score-map panels are published.\n\n{end}'
        if begin in s:s=s[:s.index(begin)]+message+s[s.index(end)+len(end):]
        else:s+='\n\n'+message+'\n'
        status.write_text(s)


def self_test(root):
    """Old Wiener scores must reproduce exactly the old mask references."""
    d=read(root/'imagenet256/design.json');p=Path(d['forward_cache_report']);old=read(p/'design.json');info=old['classes'][0]
    c=checked(p/'covariance'/f"{info['wnid']}.json",d['forward_cache_design_sha256'])
    scores={info['wnid']:dict(c,times=d['times'])};checks={}
    for name in d['models']:
        op=checked(p/name/'PLAN_COMPLETE.json',d['forward_cache_design_sha256'])
        z=map_contexts(d,op,scores,evaluation=d['evaluation'][:2])
        assert len(z['masks'])==0 and np.array_equal(z['mapping'],op['mapping'][:2,...,2])
        checks[name]=dict(exact_original_wiener_mapping=True,logical_entries=z['mapping'].size,new_forwards=0)
        del op;gc.collect()
    atomic_json(root/'CPU_CHECKS.json',dict(passed=True,checks=checks,source_sha256=sha256(Path(__file__))))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--report',required=True,type=Path)
    ap.add_argument('--stage',required=True,choices=['check','plan','evaluate','report','gallery','publish']);ap.add_argument('--model')
    a=ap.parse_args();root=a.report.resolve();code=Path(__file__).resolve().parents[2]
    for rel,h in read(code/'MANIFEST.json').items():assert sha256(code/rel)==h,rel
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.stage=='check':self_test(root);return
    if a.stage=='gallery':gallery(root);return
    if a.stage=='publish':publish(root);return
    d=read(root/'design.json')
    for p,h in d['input_hashes'].items():assert sha256(p)==h,p
    if a.stage=='plan':plan(root,d,a.model)
    elif a.stage=='report':report(root,d)
    else:
        assert torch.cuda.is_available(),'CUDA required; no CPU fallback'
        evaluate(root,d,a.model)


if __name__=='__main__':main()
