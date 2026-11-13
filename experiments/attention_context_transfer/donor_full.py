"""Resumable donor-only attention fields, matched sensitivity and full loss sweeps."""
from __future__ import annotations
import argparse,gc,json,os,time
from pathlib import Path
import numpy as np
import torch
from .prepare import atomic_json,sha256
from .protocol import seed_for
from .imagenet_fields import clean,save
from .jit_transfer import order,geometry
from .donor_models import Model,observation,patch_losses,derivative
from .donor_covariance import tokens


def read(p):return json.loads(Path(p).read_text())
def ready(root,d,name):
    r=read(root/name/'preflight/COMPLETE.json');assert r['passed'] and r['design_sha256']==sha256(root/'design.json')
    p=root/name/'preflight/benchmark.json';assert sha256(p)==r['benchmark_sha256']
    return read(p)
def groups(d):
    if d['dataset']=='celeba128':return [dict(wnid='face',queries=list(range(256)),class_id=0)]
    return d['classes']
def records(d,kind,info):return [r for r in d[kind] if d['dataset']=='celeba128' or r['wnid']==info['wnid']]
def receipt(root,d,path,extra=None):
    return dict(passed=True,path=str(path),sha256=sha256(path),design_sha256=sha256(root/'design.json'),**(extra or {}))
def checked(path,design_hash):
    r=read(path);assert r['passed'] and r['design_sha256']==design_hash
    assert sha256(r['path'])==r['sha256']
    with np.load(r['path']) as z:return {k:z[k] for k in z.files}

def attention(root,d,name):
    bench=ready(root,d,name);model=Model(name,d);model.fused(bench['inference']['selected']['fused'])
    batch=min(2048,max(128,bench['inference']['selected']['batch']//2));work=Path(d['work_root'])/'fields'/name
    started=time.monotonic();T=len(d['times']);D=d['calibration_draws'];dh=sha256(root/'design.json')
    for info in groups(d):
        wn=info['wnid'];rp=root/name/'fields'/f'{wn}.json';path=work/f'{wn}.npz'
        if rp.exists():checked(rp,dh);continue
        cal=records(d,'calibration',info);ev=records(d,'evaluation',info);Q=len(info['queries'])
        sums=np.zeros((2,T,Q,256),np.float64);mass=np.zeros((2,T,Q),np.float64);counts=np.zeros((2,T),np.int64)
        tasks=[(i,ti,draw) for i in range(len(cal)) for ti in range(T) for draw in range(D)];start=0
        with torch.inference_mode():
            while start<len(tasks):
                chunk=tasks[start:start+batch];xx=None;got=None
                try:
                    xx=torch.stack([observation(cal[i],ti,dr,d,True) for i,ti,dr in chunk]).cuda()
                    tt=torch.tensor([d['times'][ti] for i,ti,dr in chunk],device='cuda');ll=torch.full((len(chunk),),info['class_id'],device='cuda',dtype=torch.long)
                    with model.readout(info['queries']) as got:model(xx,tt,ll)
                    assert torch.isfinite(got['score']).all()
                    assert torch.allclose(got['score'].sum(-1)+got['class_mass'],torch.ones_like(got['class_mass']),atol=2e-6)
                    values=got['score'].cpu().numpy();masses=got['class_mass'].cpu().numpy()
                    for j,(i,ti,dr) in enumerate(chunk):
                        h=i//(len(cal)//2);sums[h,ti]+=values[j];mass[h,ti]+=masses[j];counts[h,ti]+=1
                    start+=len(chunk)
                except torch.cuda.OutOfMemoryError:
                    assert batch>32;batch//=2;print('readout batch reduced',batch,flush=True)
                finally:
                    del xx,got;gc.collect();torch.cuda.empty_cache()
        assert (counts==len(cal)//2*D).all();halves=(sums/counts[:,:,None,None]).astype(np.float32);mass=(mass/counts[:,:,None]).astype(np.float32)
        current=np.empty((len(ev),2,T,8,256),np.float32);current_mass=np.empty((len(ev),2,T,8),np.float32)
        tasks=[(i,dr,ti) for i in range(len(ev)) for dr in range(2) for ti in range(T)]
        with torch.inference_mode():
            for start in range(0,len(tasks),batch):
                chunk=tasks[start:start+batch];xx=torch.stack([observation(ev[i],ti,dr,d) for i,dr,ti in chunk]).cuda()
                tt=torch.tensor([d['times'][ti] for i,dr,ti in chunk],device='cuda');ll=torch.full((len(chunk),),info['class_id'],device='cuda',dtype=torch.long)
                qs=np.array([ev[i]['targets'] for i,dr,ti in chunk])
                with model.readout(qs) as got:model(xx,tt,ll)
                values=got['score'].cpu().numpy();masses=got['class_mass'].cpu().numpy()
                for j,(i,dr,ti) in enumerate(chunk):current[i,dr,ti]=values[j];current_mass[i,dr,ti]=masses[j]
                del xx,got
        assert np.isfinite(current).all()
        save(path,current=current,current_class_mass=current_mass,calibration=halves.mean(0),calibration_halves=halves,calibration_class_mass=mass,counts=counts,queries=info['queries'],image_ids=[r['image_id'] for r in ev],targets=np.array([r['targets'] for r in ev]))
        atomic_json(rp,receipt(root,d,path,dict(batch=batch,seconds=time.monotonic()-started,model=model.receipt)))
        print(name,'attention',wn,'seconds',time.monotonic()-started,flush=True)
    atomic_json(root/name/'fields/COMPLETE.json',dict(passed=True,design_sha256=dh,seconds=time.monotonic()-started))


def offsets_for(d,info,q):
    if d['dataset']=='celeba128':
        off=np.concatenate([np.tile(np.random.default_rng(seed_for('smlp_output_offsets',q,h)).permutation(64),8) for h in range(2)])
        return np.stack([q//16*8+off//8,q%16*8+off%8],1)
    offsets=[]
    for half in range(2):
        rng=np.random.default_rng(seed_for('field_offsets',info['wnid'],half))
        oo=np.array([(2*r+int(rng.integers(2)),2*c+int(rng.integers(2))) for r in range(8) for c in range(8)])
        offsets.append(oo[rng.permutation(64)])
    return np.concatenate(offsets)+np.array([q//16*16,q%16*16])


def gradient(root,d,name,index):
    assert name!='smlp','Exact SMLP gradient reused'
    bench=ready(root,d,name);model=Model(name,d);model.fused(False);batch=bench['gradient']['selected']['batch']
    started=time.monotonic();work=Path(d['work_root'])/'gradient'/name;T=len(d['times']);D=d['calibration_draws'];dh=sha256(root/'design.json')
    units=groups(d)
    if d['dataset']=='celeba128':units=[dict(wnid=f'queries_{index:02d}',queries=list(range(index*16,(index+1)*16)),class_id=0)]
    else:units=units[index*2:index*2+2]
    for info in units:
        wn=info['wnid'];rp=root/name/'gradient'/f'{wn}.json';path=work/f'{wn}.npz'
        if rp.exists():checked(rp,dh);continue
        cal=records(d,'calibration',info);Q=len(info['queries']);hsize=len(cal)//2*D;S=128 if d['dataset']=='celeba128' else 256
        offsets=np.stack([offsets_for(d,info,q) for q in info['queries']]);time_scores=[];half_scores=[];stability=[]
        for ti,t in enumerate(d['times']):
            partial=work/f'{wn}_t{ti}.npz'
            if partial.exists():
                with np.load(partial) as z:time_scores.append(z['score']);half_scores.append(z['half_scores']);stability.append(z['half_cosine'])
                continue
            ys=torch.stack([observation(r,ti,dr,d,True) for r in cal for dr in range(D)])
            tasks=[(oi,qi) for oi in range(len(ys)) for qi in range(Q)]
            sums=np.zeros((2,Q,S,S),np.float64);counts=np.zeros((2,Q),np.int64)
            for start in range(0,len(tasks),batch):
                chunk=tasks[start:start+batch];oi=np.array([i for i,q in chunk]);qi=np.array([q for i,q in chunk])
                xx=ys[oi].cuda();tt=torch.full((len(chunk),),t,device='cuda');ll=torch.full((len(chunk),),info['class_id'],device='cuda',dtype=torch.long)
                got=derivative(model,xx,tt,ll,torch.tensor(offsets[qi,oi],device='cuda')).cpu().numpy()
                for j,(i,q) in enumerate(chunk):sums[i//hsize,q]+=got[j];counts[i//hsize,q]+=1
                del xx,got
            assert (counts==hsize).all();halves=(sums/counts[:,:,None,None]).astype(np.float32)
            score=tokens(halves.mean(0));hs=tokens(halves);a,b=halves.reshape(2,Q,-1);cos=(a*b).sum(-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1))
            assert np.isfinite(score).all() and np.isfinite(cos).all()
            save(partial,score=score,half_scores=hs,half_cosine=cos,counts=counts)
            time_scores.append(score);half_scores.append(hs);stability.append(cos)
            print(name,wn,'gradient',ti,'seconds',time.monotonic()-started,flush=True)
        save(path,score=np.stack(time_scores),half_scores=np.stack(half_scores),half_cosine=np.stack(stability),queries=info['queries'],output_offsets=offsets)
        atomic_json(rp,receipt(root,d,path,dict(batch=batch,observations_per_target=len(cal)*D,seconds=time.monotonic()-started,model=model.receipt)))
    atomic_json(root/name/'gradient'/f'SHARD_{index:02d}.json',dict(passed=True,design_sha256=dh,seconds=time.monotonic()-started))


def prepare_plan(root,d,recipient):
    donor=d['donor_for'][recipient];work=Path(d['work_root']);dh=sha256(root/'design.json');T=len(d['times']);K=len(d['budgets'])
    r=read(root/donor/'fields/COMPLETE.json');assert r['passed'] and r['design_sha256']==dh
    fields={};cov={};grad={};field_receipts={}
    def get(rp):
        z=checked(rp,dh);field_receipts[str(rp)]=sha256(rp);return z
    for info in groups(d):fields[info['wnid']]=get(root/donor/'fields'/f"{info['wnid']}.json")
    if d['dataset']=='celeba128':
        assert sha256(d['reused_fields'])==d['reused_fields_sha256']
        with np.load(d['reused_fields']) as z:
            cov['face']=tokens(z['covariance'])
            if donor=='smlp':grad['face']=tokens(z['gradient'])
        if donor=='mlp':
            parts=[get(root/donor/'gradient'/f'queries_{i:02d}.json') for i in range(16)]
            assert np.array_equal(np.concatenate([p['queries'] for p in parts]),np.arange(256))
            grad['face']=np.concatenate([p['score'] for p in parts],1)
    else:
        for info in groups(d):
            wn=info['wnid'];c=get(root/'covariance'/f'{wn}.json');g=get(root/donor/'gradient'/f'{wn}.json')
            assert np.array_equal(c['queries'],info['queries']) and np.array_equal(g['queries'],info['queries'])
            cov[wn]=c['score'];grad[wn]=g['score']
    ev=d['evaluation'];shape=(len(ev),2,T,8,K,5);mapping=np.empty(shape,np.int32);job_masks=[];job_obs=[]
    for ii,rec in enumerate(ev):
        wn='face' if d['dataset']=='celeba128' else rec['wnid'];f=fields[wn];gi=list(f['image_ids']).index(rec['image_id']);qlookup={int(q):j for j,q in enumerate(f['queries'])}
        assert np.array_equal(f['targets'][gi],rec['targets'])
        for draw in range(2):
            for ti in range(T):
                obs=(ii*2+draw)*T+ti;dedup={}
                for qi,q in enumerate(rec['targets']):
                    jj=qlookup[q];orders=[geometry(q),order(grad[wn][ti,jj],q),order(cov[wn][ti,jj],q),order(f['calibration'][ti,jj],q),order(f['current'][gi,draw,ti,qi],q)]
                    for mi,indices in enumerate(orders):
                        assert np.array_equal(np.sort(indices),np.arange(256)) and q in indices[:9]
                        previous=np.zeros(256,bool)
                        for ki,k in enumerate(d['budgets']):
                            keep=np.zeros(256,bool);keep[indices[:k]]=True;assert keep.sum()==k and np.all(~previous|keep);previous=keep
                            packed=np.packbits(keep);key=packed.tobytes()
                            if key not in dedup:dedup[key]=len(job_masks);job_masks.append(packed);job_obs.append(obs)
                            mapping[ii,draw,ti,qi,ki,mi]=dedup[key]
    assert np.all(mapping[:,:,:,:,0,:]==mapping[:,:,:,:,0,:1]) and np.all(mapping[:,:,:,:,-1,:]==mapping[:,:,:,:,-1,:1])
    path=work/recipient/'plan.npz';save(path,mapping=mapping,masks=np.stack(job_masks),observations=np.array(job_obs,np.int32))
    atomic_json(root/recipient/'PLAN_COMPLETE.json',receipt(root,d,path,dict(donor=donor,recipient=recipient,logical_entries=int(np.prod(shape)),unique_forwards=len(job_masks),field_receipts=field_receipts)))
    print(recipient,'unique forwards',len(job_masks),flush=True)


def evaluate(root,d,name):
    bench=ready(root,d,name);r=read(root/name/'PLAN_COMPLETE.json');plan=checked(root/name/'PLAN_COMPLETE.json',sha256(root/'design.json'))
    model=Model(name,d);model.fused(bench['inference']['selected']['fused']);batch=bench['inference']['selected']['batch']
    ev=d['evaluation'];T=len(d['times']);work=Path(d['work_root'])/name/'evaluation';work.mkdir(parents=True,exist_ok=True)
    identity=dict(design=sha256(root/'design.json'),plan=r['sha256'],batch=batch,model=name,code=sha256(Path(__file__)))
    if (work/'identity.json').exists():assert read(work/'identity.json')==identity
    else:atomic_json(work/'identity.json',identity)
    cases=[(rr,ti,dr) for rr in ev for dr in range(2) for ti in range(T)]
    # Cache observations on CPU; reserve GPU memory for the validated inference batch.
    x=torch.stack([observation(rr,ti,dr,d) for rr,ti,dr in cases]);truth=torch.stack([clean(rr) for rr in ev])
    tt=torch.tensor([d['times'][ti] for rr,ti,dr in cases]);labels=torch.tensor([rr.get('class_id',0) for rr,ti,dr in cases]);queries=torch.tensor([rr['targets'] for rr in ev])
    obs=plan['observations'];masks=plan['masks'];mapping=plan['mapping'];losses=np.empty((len(obs),8),np.float32);started=time.monotonic()
    with torch.inference_mode():
        for start in range(0,len(obs),batch):
            stop=min(start+batch,len(obs));part=work/f'chunk_{start:08d}_{stop:08d}.npy'
            if part.exists():
                got=np.load(part);assert got.shape==(stop-start,8) and np.isfinite(got).all();losses[start:stop]=got;continue
            oi=torch.tensor(obs[start:stop].astype(np.int64));ii=oi//(2*T)
            keep=torch.tensor(np.unpackbits(masks[start:stop],axis=1).astype(bool),device='cuda')
            pred=model(x[oi].cuda(),tt[oi].cuda(),labels[oi].cuda(),keep)
            got=patch_losses(pred,truth[ii].cuda(),queries[ii].cuda()).cpu().numpy();assert np.isfinite(got).all();losses[start:stop]=got
            with part.with_suffix('.tmp').open('wb') as f:np.save(f,got)
            part.with_suffix('.tmp').replace(part);del pred,keep
            atomic_json(root/name/'progress.json',dict(completed=stop,total=len(obs),seconds=time.monotonic()-started,batch=batch))
            print(name,stop,'/',len(obs),'seconds',round(time.monotonic()-started,1),flush=True)
        dense=[]
        for start in range(0,len(x),min(batch,1024)):
            stop=min(start+min(batch,1024),len(x));ii=torch.arange(start,stop)//(2*T)
            pred=model(x[start:stop].cuda(),tt[start:stop].cuda(),labels[start:stop].cuda())
            dense.append(patch_losses(pred,truth[ii].cuda(),queries[ii].cuda()).cpu().numpy())
    dense=np.concatenate(dense).reshape(len(ev),2,T,8)
    result=losses[mapping,np.arange(8)[None,None,None,:,None,None]]
    err=float(np.max(np.abs(result[:,:,:,:,-1,0]-dense)));assert err<1e-5,err
    path=work/'losses.npz';save(path,losses=result,independent_dense=dense,methods=d['methods'],budgets=d['budgets'],times=d['times'],targets=queries.numpy(),image_ids=[rr['image_id'] for rr in ev])
    atomic_json(root/name/'EVALUATION_COMPLETE.json',receipt(root,d,path,dict(seconds=time.monotonic()-started,batch=batch,plan_sha256=r['sha256'],dense_mse_max_abs=err,model=model.receipt)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--stage',choices=['attention','gradient','plan','evaluate'],required=True);p.add_argument('--index',type=int,default=None);a=p.parse_args()
    root=a.report.resolve();d=read(root/'design.json');code=Path(__file__).resolve().parents[2]
    for rel,digest in read(code/'MANIFEST.json').items():assert sha256(code/rel)==digest,rel
    for path,digest in d['input_hashes'].items():assert sha256(path)==digest,path
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')));torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.stage=='plan':prepare_plan(root,d,a.model);return
    assert torch.cuda.is_available(),'CUDA required'
    if a.stage=='attention':attention(root,d,a.model)
    elif a.stage=='gradient':gradient(root,d,a.model,a.index if a.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID']))
    else:evaluate(root,d,a.model)
if __name__=='__main__':main()
