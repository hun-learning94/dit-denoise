"""Class-balanced pooled ImageNet Wiener fields, with bounded-memory FP64 algebra."""
from __future__ import annotations
import argparse, csv, gc, json, os, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from scipy import linalg
from .full_covariance_cpu import read, digest, write_json, save_npy, save_npz, load_crop, patch_rows, token_max


def record(root, name, **values):
    import resource
    write_json(root/name, dict(passed=True, design_sha256=digest(root/'design.json'),
        source_sha256=digest(Path(__file__)), job_id=os.environ.get('SLURM_JOB_ID'),
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20, **values))


def checked_record(root, name):
    r=read(root/name); assert r['passed'] and r['design_sha256']==digest(root/'design.json')
    for key in ['path','basis','eigenvalues']:
        if key in r: assert digest(r[key])==r[key+'_sha256']
    return r


def work(root):return Path(read(root/'design.json')['work_root'])


def centered_block(pixels, indices, start, stop, mean=None):
    # uint8 bank is already flattened in CHW order; avoid whole-bank fancy indexing.
    x=np.array(pixels[indices,start:stop], dtype=np.float64, order='C')
    x/=127.5; x-=1.
    if mean is not None:x-=mean[start:stop]
    return x


def block_gram(pixels, indices, mean, block=1024):
    n=len(indices); g=np.zeros((n,n),np.float64,order='F')
    for start in range(0,pixels.shape[1],block):
        x=centered_block(pixels,indices,start,start+block,mean)
        out=linalg.blas.dsyrk(1./n, x, beta=1., c=g, lower=1, overwrite_c=1)
        assert np.shares_memory(out,g)
    return g


def eigensystem(g):
    trace=float(np.trace(g))
    lam,v=linalg.eigh(g,lower=True,driver='evr',overwrite_a=True,check_finite=False)
    assert lam[0]>=-max(1.,lam[-1])*1e-10
    keep=lam>lam[-1]*1e-12
    assert abs(lam[keep].sum()-trace)/trace<1e-9
    return lam[keep],v[:,keep],dict(trace=trace,retained_trace=float(lam[keep].sum()),null_modes=int((~keep).sum()))


def reconstruct(pixels,indices,mean,v,lam,out,block=1024):
    for start in range(0,pixels.shape[1],block):
        x=centered_block(pixels,indices,start,start+block,mean)
        out[start:start+block]=(x.T@v)/np.sqrt(len(indices)*lam)[None,:]


def prepare(root):
    d=read(root/'design.json'); w=work(root); w.mkdir(parents=True,exist_ok=True)
    if (root/'BANK.json').exists():checked_record(root,'BANK.json');return
    forbidden=set(d['pool']['forbidden_pixels']); seen=set(forbidden); manifests=[]
    paths=sorted(p for p in Path(d['pool']['train_root']).iterdir() if p.is_dir() and p.name.startswith('n'))
    assert len(paths)==1000,len(paths)
    shards=w/'classes';shards.mkdir(exist_ok=True)
    started=time.monotonic()
    with ThreadPoolExecutor(16) as executor:
        for ci,folder in enumerate(paths):
            rp=shards/(folder.name+'.json'); pp=shards/(folder.name+'.npy')
            if rp.exists():
                r=read(rp); assert r['design_sha256']==digest(root/'design.json') and digest(pp)==r['path_sha256']
                selected=r['records']; assert len(selected)==50
                assert not (set(x['pixel_sha256'] for x in selected)&seen)
                seen.update(x['pixel_sha256'] for x in selected);manifests.extend(selected);continue
            seed=int.from_bytes(__import__('hashlib').sha256((str(d['pool']['seed'])+folder.name).encode()).digest()[:8],'little')
            files=sorted(p for p in folder.iterdir() if p.is_file()); rng=np.random.default_rng(seed)
            files=[files[i] for i in rng.permutation(len(files))]; selected=[];arrays=[];excluded=[]
            for offset in range(0,len(files),64):
                items=[dict(path=str(p),wnid=folder.name,image_id=p.name,official_split='train') for p in files[offset:offset+64]]
                for rec,arr in executor.map(lambda x:load_crop(x,256),items):
                    reason=rec.get('reason')
                    if reason is None and rec['pixel_sha256'] in seen:reason='duplicate_or_calibration_evaluation_overlap'
                    if reason:excluded.append(dict(rec,reason=reason));continue
                    rec['class_sample_index']=len(selected);selected.append(rec);arrays.append(arr.transpose(2,0,1).ravel());seen.add(rec['pixel_sha256'])
                    if len(selected)==50:break
                if len(selected)==50:break
            assert len(selected)==50,(folder,len(selected))
            save_npy(pp,np.stack(arrays));write_json(rp,dict(design_sha256=digest(root/'design.json'),path_sha256=digest(pp),records=selected,exclusions=excluded))
            manifests.extend(selected)
            if ci%25==0:print('prepared classes',ci+1,'seconds',time.monotonic()-started,flush=True)
    assert len(manifests)==50000 and len({x['pixel_sha256'] for x in manifests})==50000
    path=w/'training_50000.npy';tmp=path.with_suffix('.tmp.npy')
    pixels=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.uint8,shape=(50000,196608))
    for ci,folder in enumerate(paths):pixels[ci*50:(ci+1)*50]=np.load(shards/(folder.name+'.npy'))
    pixels.flush();del pixels;tmp.replace(path)
    write_json(w/'training_manifest.json',manifests)
    indices25=np.concatenate([np.arange(c*50,c*50+25) for c in range(1000)])
    save_npy(w/'indices_25000.npy',indices25);save_npy(w/'indices_50000.npy',np.arange(50000))
    pixels=np.load(path,mmap_mode='r')
    for n in [25000,50000]:
        indices=np.load(w/f'indices_{n}.npy');mean=np.zeros(196608)
        for start in range(0,196608,1024):mean[start:start+1024]=centered_block(pixels,indices,start,start+1024).mean(0)
        save_npy(w/f'mean_{n}.npy',mean)
    record(root,'BANK.json',path=str(path),path_sha256=digest(path),samples=50000,classes=1000,nested_samples=25000,
        manifest_sha256=digest(w/'training_manifest.json'),seconds=time.monotonic()-started)


def cpu_benchmark(root):
    checked_record(root,'BANK.json');w=work(root);pixels=np.load(w/'training_50000.npy',mmap_mode='r')
    mean=np.load(w/'mean_50000.npy');indices=np.linspace(0,49999,8192,dtype=int)
    x=centered_block(pixels,indices,0,2048,mean);tick=time.monotonic()
    g=linalg.blas.dsyrk(1.,x,lower=1);gram_seconds=time.monotonic()-tick
    tick=time.monotonic();lam,v=linalg.eigh(g,lower=True,driver='evr',overwrite_a=True,check_finite=False);eig_seconds=time.monotonic()-tick
    tick=time.monotonic();out=x.T@v;basis_seconds=time.monotonic()-tick
    assert np.isfinite(out).all();estimates={}
    for n in [25000,50000]:
        gram=gram_seconds*(n/8192)**2*(196608/2048)
        basis=basis_seconds*(n/8192)**2*(196608/2048)
        eigen=eig_seconds*(n/8192)**3
        estimates[str(n)]=dict(gram_seconds=gram,basis_seconds=basis,eigen_seconds=eigen,guard_seconds=2*(gram+basis+eigen)+900)
    record(root,'CPU_BENCHMARK.json',gram_seconds=gram_seconds,eigen_seconds=eig_seconds,basis_seconds=basis_seconds,estimates=estimates)
    assert estimates['50000']['guard_seconds']<6*3600,estimates
    record(root,'CPU_PREFLIGHT_PASSED.json',benchmark_sha256=digest(root/'CPU_BENCHMARK.json'))


def basis(root,n):
    checked_record(root,'CPU_PREFLIGHT_PASSED.json');checked_record(root,'BANK.json');w=work(root);n=int(n)
    if (root/f'basis_{n}.json').exists():checked_record(root,f'basis_{n}.json');return
    pixels=np.load(w/'training_50000.npy',mmap_mode='r');indices=np.load(w/f'indices_{n}.npy');mean=np.load(w/f'mean_{n}.npy')
    stage=w/f'basis_{n}';stage.mkdir(exist_ok=True);started=time.monotonic();gp=stage/'gram.npy'
    if not (root/f'gram_{n}.json').exists():
        g=block_gram(pixels,indices,mean);save_npy(gp,g);del g;gc.collect()
        record(root,f'gram_{n}.json',path=str(gp),path_sha256=digest(gp),seconds=time.monotonic()-started)
    checked_record(root,f'gram_{n}.json');vp=stage/'sample_eigenvectors.npy';lp=stage/'eigenvalues.npy'
    if not (root/f'eigen_{n}.json').exists():
        g=np.array(np.load(gp,mmap_mode='r'),order='F');lam,v,checks=eigensystem(g);del g;gc.collect()
        save_npy(vp,v);save_npy(lp,lam);del v;gc.collect()
        record(root,f'eigen_{n}.json',path=str(vp),path_sha256=digest(vp),eigenvalues=str(lp),eigenvalues_sha256=digest(lp),**checks)
    checked_record(root,f'eigen_{n}.json');v=np.load(vp,mmap_mode='r');lam=np.load(lp)
    bp=stage/'pixel_basis.npy';tmp=bp.with_suffix('.tmp.npy')
    u=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float64,shape=(196608,len(lam)))
    reconstruct(pixels,indices,mean,v,lam,u);u.flush();del v;gc.collect()
    jj=np.random.default_rng(97).choice(196608,32,replace=False)
    xx=np.array(pixels[np.ix_(indices,jj)],dtype=np.float64)/127.5-1-mean[jj]
    exact=xx.T@xx/n;err=float(np.max(np.abs((u[jj]*lam)@u[jj].T-exact)));assert err<1e-7,err
    cols=np.linspace(0,len(lam)-1,16,dtype=int);small=np.array(u[:,cols]);ortho=float(np.max(np.abs(small.T@small-np.eye(len(cols)))))
    assert ortho<1e-5,ortho
    del u;tmp.replace(bp)
    record(root,f'basis_{n}.json',basis=str(bp),basis_sha256=digest(bp),eigenvalues=str(lp),eigenvalues_sha256=digest(lp),
        samples=n,rank=len(lam),reconstruction_max_abs=err,orthogonality_max_abs=ortho,seconds=time.monotonic()-started)


def gpu_setup(root,n):
    import torch
    assert torch.cuda.is_available(),'GPU stages must not fall back to CPU'
    torch.backends.cuda.matmul.allow_tf32=False
    r=checked_record(root,f'basis_{n}.json');u=np.load(r['basis'],mmap_mode='r');lam=np.load(r['eigenvalues'])
    ug=torch.as_tensor(u,device='cuda',dtype=torch.float64);lg=torch.as_tensor(lam,device='cuda',dtype=torch.float64)
    return u,lam,ug,lg


def gpu_field(u,lam,t,q):
    import torch
    gain=t*lam/(t*t*lam+(1-t)**2);rows=torch.as_tensor(patch_rows(q,256),device='cuda')
    w=(u[rows]*gain)@u.T
    w.abs_();w.div_(w.amax(1,keepdim=True))
    return w.reshape(-1,3,256,256).amax((0,1))


def gpu_preflight(root):
    import torch
    u,lam,ug,lg=gpu_setup(root,50000);q=136;t=.5;rows=patch_rows(q,256)[[0,31,255,511,767]]
    gain=t*lam/(t*t*lam+(1-t)**2);ref=(u[rows]*gain)@u.T
    got=((ug[torch.tensor(rows,device='cuda')]*(t*lg/(t*t*lg+(1-t)**2)))@ug.T).cpu().numpy()
    err=float(np.max(np.abs(ref-got)));assert err<1e-8,err
    del ref,got;gc.collect();gpu_field(ug,lg,.1,q);torch.cuda.synchronize();times=[]
    torch.cuda.reset_peak_memory_stats()
    for t in [.1,.5,.9]:
        tick=time.monotonic();field=gpu_field(ug,lg,t,q);torch.cuda.synchronize();times.append(time.monotonic()-tick)
        assert torch.isfinite(field).all() and (field>=0).all()
    estimate=max(times)*256*5
    record(root,'GPU_BENCHMARK.json',gpu=torch.cuda.get_device_name(),fp64_reference_max_abs=err,patch_seconds=times,
        fields50k_estimate_seconds=estimate,peak_vram_gib=torch.cuda.max_memory_allocated()/2**30)
    assert estimate*2+600<7200,estimate
    record(root,'GPU_PREFLIGHT_PASSED.json',benchmark_sha256=digest(root/'GPU_BENCHMARK.json'))


def fields(root,n):
    import torch
    checked_record(root,'GPU_PREFLIGHT_PASSED.json');_,_,u,lam=gpu_setup(root,n);d=read(root/'design.json')
    times=d['times'] if n==50000 else d['primary_times'];out=work(root)/f'fields_{n}';out.mkdir(exist_ok=True)
    scores=[];started=time.monotonic()
    for ti,t in enumerate(times):
        rp=f'fields_{n}_{ti}.json';path=out/f't{ti}.npz'
        if (root/rp).exists():
            checked_record(root,rp)
            with np.load(path) as z:scores.append(z['score'])
            continue
        ss=[];native=[]
        for q in range(256):
            f=gpu_field(u,lam,t,q).cpu().numpy();ss.append(token_max(f))
            if q in [85,136,187]:native.append(f)
            if q%32==0:print('fields',n,t,q,time.monotonic()-started,flush=True)
        score=np.stack(ss);assert np.isfinite(score).all();scores.append(score)
        save_npz(path,score=score.astype(np.float32),native=np.stack(native).astype(np.float32),queries=np.arange(256),native_queries=[85,136,187],time=t)
        record(root,rp,path=str(path),path_sha256=digest(path))
    path=out/'scores.npz';save_npz(path,score=np.stack(scores).astype(np.float32),times=times,queries=np.arange(256))
    record(root,f'fields_{n}.json',path=str(path),path_sha256=digest(path),samples=n,seconds=time.monotonic()-started)


def get_fields(root,n):
    r=checked_record(root,f'fields_{n}.json')
    with np.load(r['path']) as z:return {k:z[k] for k in z.files}


def plan(root):
    from .wiener_refresh import map_contexts
    from .donor_full import checked,receipt
    d=read(root/'design.json');p=Path(d['forward_cache_report']);f=get_fields(root,50000)
    for name in d['models']:
        old=checked(p/name/'PLAN_COMPLETE.json',d['forward_cache_design_sha256'])
        mapped=map_contexts(d,old,{c['wnid']:f for c in d['classes']})
        path=work(root)/name/'plan.npz';save_npz(path,**mapped)
        write_json(root/name/'PLAN_COMPLETE.json',receipt(root,d,path,dict(original_jobs=len(old['observations']),new_forwards=len(mapped['masks']),
            parent_plan_sha256=digest(p/name/'PLAN_COMPLETE.json'),fields_sha256=digest(root/'fields_50000.json'))))


def evaluate(root,name):
    import torch
    from .donor_full import checked,receipt,ready
    from .donor_models import Model,observation,patch_losses
    from .imagenet_fields import clean
    from .wiener_refresh import check_gpu
    d=read(root/'design.json');parent=Path(d['forward_cache_report']);old=read(parent/'design.json');base=Path(d['baseline_report'])
    baseline=checked(base/name/'EVALUATION_COMPLETE.json',d['baseline_design_sha256'])
    original=checked(parent/name/'EVALUATION_COMPLETE.json',d['forward_cache_design_sha256'])
    oldplan=checked(parent/name/'PLAN_COMPLETE.json',d['forward_cache_design_sha256'])
    p=checked(root/name/'PLAN_COMPLETE.json',digest(root/'design.json'));pr=read(root/name/'PLAN_COMPLETE.json')
    assert pr['fields_sha256']==digest(root/'fields_50000.json') and pr['parent_plan_sha256']==digest(parent/name/'PLAN_COMPLETE.json')
    bench=ready(parent,old,name);batch=bench['inference']['selected']['batch'];oldn=len(oldplan['observations']);oldloss=np.empty((oldn,8),np.float32)
    for start in range(0,oldn,batch):
        stop=min(start+batch,oldn);oldloss[start:stop]=np.load(Path(old['work_root'])/name/'evaluation'/f'chunk_{start:08d}_{stop:08d}.npy')
    assert np.array_equal(oldloss[oldplan['mapping'],np.arange(8)[None,None,None,:,None,None]],original['losses'])
    assert np.array_equal(baseline['independent_dense'],original['independent_dense'])
    obs=p['observations'];masks=p['masks'];N=len(obs);T=len(d['times']);out=work(root)/name/'evaluation';out.mkdir(parents=True,exist_ok=True)
    identity=dict(design=digest(root/'design.json'),plan=pr['sha256'],code=digest(Path(__file__)),batch=batch)
    if (out/'identity.json').exists():assert read(out/'identity.json')==identity
    else:write_json(out/'identity.json',identity)
    assert torch.cuda.is_available();model=Model(name,d);model.fused(False)
    ev=d['evaluation'];cases=[(r,ti,dr) for r in ev for dr in range(2) for ti in range(T)]
    x=torch.stack([observation(r,ti,dr,d) for r,ti,dr in cases]);truth=torch.stack([clean(r) for r in ev])
    tt=torch.tensor([d['times'][ti] for r,ti,dr in cases]);ll=torch.tensor([r['class_id'] for r,ti,dr in cases]);qs=torch.tensor([r['targets'] for r in ev])
    checks=check_gpu(model,d,baseline,x,truth,tt,ll,qs);write_json(root/name/'GPU_CHECKS.json',checks)
    torch.cuda.empty_cache();newloss=np.empty((N,8),np.float32);started=time.monotonic()
    with torch.inference_mode():
        for start in range(0,N,batch):
            stop=min(start+batch,N);part=out/f'chunk_{start:08d}_{stop:08d}.npy'
            if part.exists():
                got=np.load(part);assert got.shape==(stop-start,8) and np.isfinite(got).all();newloss[start:stop]=got;continue
            oi=torch.tensor(obs[start:stop].astype(np.int64));ii=oi//(2*T)
            keep=torch.tensor(np.unpackbits(masks[start:stop],axis=1).astype(bool),device='cuda')
            pred=model(x[oi].cuda(),tt[oi].cuda(),ll[oi].cuda(),keep)
            got=patch_losses(pred,truth[ii].cuda(),qs[ii].cuda()).cpu().numpy();assert np.isfinite(got).all()
            newloss[start:stop]=got;save_npy(part,got);del pred,keep
            write_json(root/name/'progress.json',dict(completed=stop,total=N,seconds=time.monotonic()-started))
    values=np.concatenate([oldloss,newloss])[p['mapping'],np.arange(8)[None,None,None,:,None]];dense=baseline['independent_dense']
    assert np.array_equal(values[...,0],baseline['losses'][...,0,2]) and np.array_equal(values[...,-1],baseline['losses'][...,-1,2])
    combined=np.concatenate([baseline['losses'],values[...,None]],axis=-1);assert np.array_equal(combined[...,:7],baseline['losses'])
    path=out/'losses.npz';save_npz(path,losses=combined,independent_dense=dense,methods=d['methods'],budgets=d['budgets'],times=d['times'],targets=qs.numpy(),image_ids=baseline['image_ids'])
    write_json(root/name/'EVALUATION_COMPLETE.json',receipt(root,d,path,dict(new_forwards=N,all_seven_original_methods_identical=True,seconds=time.monotonic()-started)))


def calibration(root,name):
    import torch
    from .donor_models import Model,observation,patch_losses
    from .imagenet_fields import clean
    from .jit_transfer import order
    from .donor_full import ready
    d=read(root/'design.json');parent=Path(d['forward_cache_report']);bench=ready(parent,read(parent/'design.json'),name)
    f25=get_fields(root,25000);f50=get_fields(root,50000);fields=[f25,f50];recs=d['pool']['stability_images'];times=d['primary_times'];K=len(d['budgets'])
    assert len(recs)==64 and {r['pixel_sha256'] for r in recs}.isdisjoint(r['pixel_sha256'] for r in d['evaluation'])
    ranks={n:{(t,q):order(f['score'][list(f['times']).index(t),q],q) for t in times for q in range(256)} for n,f in zip([25000,50000],fields)}
    mapping=np.empty((64,2,3,8,K,2),np.int32);masks=[];obs=[]
    for i,r in enumerate(recs):
        for dr in range(2):
            for ti,t in enumerate(times):
                dedup={};oi=(i*2+dr)*3+ti
                for qi,q in enumerate(r['targets']):
                    for ki,k in enumerate(d['budgets']):
                        for mi,n in enumerate([25000,50000]):
                            keep=np.zeros(256,bool);keep[ranks[n][t,q][:k]]=True;packed=np.packbits(keep);key=packed.tobytes()
                            if key not in dedup:dedup[key]=len(masks);masks.append(packed);obs.append(oi)
                            mapping[i,dr,ti,qi,ki,mi]=dedup[key]
    masks=np.stack(masks);obs=np.array(obs);batch=bench['inference']['selected']['batch'];model=Model(name,d);assert torch.cuda.is_available()
    cases=[(r,d['times'].index(t),dr) for r in recs for dr in range(2) for t in times]
    x=torch.stack([observation(r,ti,dr,d,True) for r,ti,dr in cases]);truth=torch.stack([clean(r) for r in recs]);qs=torch.tensor([r['targets'] for r in recs])
    tt=torch.tensor([d['times'][ti] for r,ti,dr in cases]);ll=torch.tensor([r['class_id'] for r,ti,dr in cases]);jobs=np.empty((len(obs),8),np.float32)
    with torch.inference_mode():
        dense=[]
        for start in range(0,len(cases),64):
            oi=torch.arange(start,min(start+64,len(cases)));ii=oi//6
            pred=model(x[oi].cuda(),tt[oi].cuda(),ll[oi].cuda());dense.append(patch_losses(pred,truth[ii].cuda(),qs[ii].cuda()).cpu().numpy())
        dense=np.concatenate(dense).reshape(64,2,3,8)
        for start in range(0,len(obs),batch):
            oi=torch.tensor(obs[start:start+batch]);ii=oi//6;keep=torch.tensor(np.unpackbits(masks[start:start+batch],axis=1).astype(bool),device='cuda')
            pred=model(x[oi].cuda(),tt[oi].cuda(),ll[oi].cuda(),keep)
            jobs[start:start+batch]=patch_losses(pred,truth[ii].cuda(),qs[ii].cuda()).cpu().numpy();del pred,keep
    losses=jobs[mapping,np.arange(8)[None,None,None,:,None,None]]
    assert np.isfinite(losses).all() and np.max(np.abs(losses[...,-1,:]-dense[...,None]))<2e-5
    path=work(root)/name/'stability.npz';save_npz(path,losses=losses,independent_dense=dense,times=times,budgets=d['budgets'],samples=[25000,50000],image_ids=[r['image_id'] for r in recs])
    record(root,name+'/STABILITY_COMPLETE.json',path=str(path),path_sha256=digest(path),images=64,unique_forwards=len(obs))


def self_check(root):
    import tempfile
    from .full_covariance_cpu import sample_basis,field
    rng=np.random.default_rng(52);pixels=rng.integers(0,256,(39,3*16*16),dtype=np.uint8);checks={}
    for n in [19,39]:
        ix=np.arange(n);x=pixels[ix].astype(np.float64)/127.5-1;mean=x.mean(0);x-=mean
        g=block_gram(pixels,ix,mean,block=37);assert np.max(np.abs(np.tril(g)-np.tril(x@x.T/n)))<1e-12
        lam,v,c=eigensystem(g);u=np.empty((pixels.shape[1],len(lam)));reconstruct(pixels,ix,mean,v,lam,u,block=31)
        uu,ll,_=sample_basis(pixels[ix].astype(np.float64)/127.5-1)
        err=max(float(np.max(np.abs(field(u,lam,t,85,16,1)-field(uu,ll,t,85,16,1)))) for t in [.1,.5,.9])
        assert err<1e-6;checks[str(n)]=dict(score_max_abs=err,rank=len(lam))
    record(root,'CPU_CHECKS.json',checks=checks)


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--stage',required=True)
    p.add_argument('--samples',type=int,choices=[25000,50000]);p.add_argument('--model',choices=['jit_b','jit_l']);a=p.parse_args();root=a.report.resolve()
    code=Path(__file__).resolve().parents[2]
    if (code/'MANIFEST.json').exists():
        for rel,h in read(code/'MANIFEST.json').items():assert digest(code/rel)==h,rel
    if a.stage=='check':self_check(root)
    elif a.stage=='prepare':prepare(root)
    elif a.stage=='cpu-benchmark':cpu_benchmark(root)
    elif a.stage=='basis':basis(root,a.samples)
    elif a.stage=='gpu-preflight':gpu_preflight(root)
    elif a.stage=='fields':fields(root,a.samples)
    elif a.stage=='plan':plan(root)
    elif a.stage=='calibration':calibration(root,a.model)
    elif a.stage=='evaluate':evaluate(root,a.model)
    elif a.stage=='report':
        from .report_pooled_wiener import report
        report(root)
    else:raise ValueError(a.stage)

if __name__=='__main__':main()
