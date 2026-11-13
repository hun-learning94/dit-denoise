"""JiT closed-context intervention and bidirectional context-index transfer pilot."""
from __future__ import annotations
import argparse, gc, json, os, time, types
from pathlib import Path
import numpy as np
import torch
from .prepare import atomic_json, sha256
from .protocol import seed_for, distances
from .locality_selectors import core
from .imagenet_fields import clean, observation, load, predict, save


def protected_rows(keep, prefix):
    return torch.cat([torch.ones((len(keep),prefix),device=keep.device,dtype=torch.bool),keep],1)


class ClosedJiT:
    """Keep full grid/positions. Close every protected query over retained keys."""
    def __init__(self, model):
        self.model=model;self.keep=None;self.original=[]
        for block in model.blocks:
            module=block.attn;original=module.forward;self.original.append(original)
            def forward(module,x,rope,original=original):
                if self.keep is None:return original(x,rope)
                b,n,c=x.shape;prefix=n-256;assert prefix in (0,model.in_context_len)
                keep=protected_rows(self.keep,prefix);assert keep.shape==(b,n)
                qkv=module.qkv(x).reshape(b,n,3,module.num_heads,c//module.num_heads).permute(2,0,3,1,4)
                q,k,v=qkv.unbind(0);q,k=rope(module.q_norm(q)),rope(module.k_norm(k))
                # Match upstream FP32 QK and full softmax, changing only forbidden edges.
                weights=(q.float()@k.float().transpose(-2,-1))*((c//module.num_heads)**-.5)
                forbidden=keep[:,:,None]&~keep[:,None,:]
                weights.masked_fill_(forbidden[:,None],-torch.inf)
                weights=weights.softmax(-1)
                y=(weights@v).transpose(1,2).reshape(b,n,c)
                return module.proj_drop(module.proj(y))
            module.forward=types.MethodType(forward,module)
    def __call__(self,x,t,labels,keep=None):
        self.keep=keep
        try:return self.model(x,t,labels)
        finally:self.keep=None


def order(score,q):
    initial=core(q);eligible=np.setdiff1d(np.arange(256),initial)
    assert np.isfinite(score).all() and score.shape==(256,)
    return np.r_[initial,eligible[np.lexsort((eligible,-score[eligible]))]]


def geometry(q):
    rr,cc=np.divmod(np.arange(256),16);r,c=divmod(q,16)
    entry=np.maximum(0,16*np.maximum(abs(rr-r),abs(cc-c))-15)
    initial=core(q);eligible=np.setdiff1d(np.arange(256),initial)
    return np.r_[initial,eligible[np.lexsort((eligible,distances(q)[eligible],entry[eligible]))]]


def tokens(field):
    assert field.shape[-2:]==(256,256)
    return field.reshape(*field.shape[:-2],16,16,16,16).max(axis=(-3,-1)).reshape(*field.shape[:-2],256)


def prepare_plan(root,d):
    work=Path(d['work_root']);recs=[r for r in d['images'] if r['role']=='gallery'];qs=d['queries']
    methods=d['methods'];budgets=d['budgets'];shape=(len(recs),2,len(d['times']),len(qs),len(budgets),len(methods))
    mapping=np.empty(shape,np.int32);job_masks=[];job_obs=[];receipts={};fields={}
    for name in ['jit_b','jit_l','pixeldit_gradient','covariance']:
        receipt=json.loads((root/name/'COMPLETE.json').read_text());assert receipt['passed'] and receipt['design_sha256']==sha256(root/'design.json')
        for info in d['classes']:
            p=work/name/f"{info['wnid']}.npz";receipt=json.loads((root/name/f"{info['wnid']}.json").read_text())
            assert sha256(p)==receipt['sha256'];receipts[str(p)]=receipt['sha256']
            with np.load(p) as z:fields[name,info['wnid']]={k:z[k] for k in z.files}
            assert np.array_equal(fields[name,info['wnid']]['queries'],qs)
    for ii,rec in enumerate(recs):
        wn=rec['wnid'];att={name:fields[name,wn] for name in ['jit_b','jit_l']}
        gi={name:list(att[name]['gallery_ids']).index(rec['image_id']) for name in att}
        grad=tokens(fields['pixeldit_gradient',wn]['halves'].mean(1))
        cov=tokens(fields['covariance',wn]['bank512'])
        for draw in range(2):
            for ti in range(len(d['times'])):
                obs=(ii*2+draw)*len(d['times'])+ti;dedup={}
                for qi,q in enumerate(qs):
                    orders=[geometry(q),order(grad[ti,qi],q),order(cov[ti,qi],q)]
                    for name in ['jit_b','jit_l']:
                        for g in range(6):
                            orders+=[order(att[name]['current'][gi[name],draw,ti,g,qi],q),order(att[name]['calibration_halves'][:,ti,g,qi].mean(0),q)]
                    assert len(orders)==len(methods)
                    for mi,indices in enumerate(orders):
                        assert np.array_equal(np.sort(indices),np.arange(256)) and q in indices[:9]
                        previous=np.zeros(256,bool)
                        for ki,k in enumerate(budgets):
                            keep=np.zeros(256,bool);keep[indices[:k]]=True
                            assert keep.sum()==k and np.all(~previous|keep);previous=keep
                            packed=np.packbits(keep);key=packed.tobytes()
                            if key not in dedup:
                                dedup[key]=len(job_masks);job_masks.append(packed);job_obs.append(obs)
                            mapping[ii,draw,ti,qi,ki,mi]=dedup[key]
    assert np.all(mapping[:,:,:,:,0,:]==mapping[:,:,:,:,0,:1])
    assert np.all(mapping[:,:,:,:,-1,:]==mapping[:,:,:,:,-1,:1])
    path=work/'evaluation_plan.npz';save(path,mapping=mapping,masks=np.stack(job_masks),observations=np.array(job_obs,np.int16),queries=qs,methods=methods,budgets=budgets)
    atomic_json(root/'PLAN_COMPLETE.json',dict(passed=True,path=str(path),sha256=sha256(path),unique_forwards_per_recipient=len(job_masks),logical_entries_per_recipient=int(np.prod(shape)),shape=shape,field_hashes=receipts,design_sha256=sha256(root/'design.json')))
    print('plan',len(job_masks),'unique forwards per recipient',flush=True)


def patch_losses(pred,truth,queries):
    difference=(pred-truth).square().reshape(-1,3,16,16,16,16)
    return difference.mean((1,3,5)).reshape(-1,256)[:,queries]


def preflight(root,d,name):
    out=root/name/'intervention';out.mkdir(parents=True,exist_ok=True)
    model,receipt=load(name,d);wrapped=ClosedJiT(model)
    recs=[r for r in d['images'] if r['role']=='gallery'];qs=d['queries']
    cases=[(recs[i],ti,i%2) for i,ti in zip([0,4,8,12,1,5],[0,1,2,3,4,2])]
    x=torch.stack([observation(r,ti,draw,d) for r,ti,draw in cases]).cuda()
    t=torch.tensor([d['times'][ti] for r,ti,draw in cases],device='cuda');labels=torch.tensor([r['class_id'] for r,ti,draw in cases],device='cuda')
    keeps=[]
    for i in range(len(x)):
        keep=np.zeros(256,bool);keep[geometry(qs[i])[:[9,16,48,96,192,32][i]]]=True;keeps.append(keep)
    keep=torch.tensor(np.stack(keeps),device='cuda')
    full=torch.ones_like(keep);trace=[]
    def hook(module,args,output):
        prefix=output.shape[1]-256;protected=protected_rows(keep,prefix)
        trace.append(output[protected].detach().clone())
    handles=[b.register_forward_hook(hook) for b in model.blocks]
    with torch.inference_mode():
        dense=wrapped(x,t,labels);trace.clear()
        all_context=wrapped(x,t,labels,full);trace.clear()
        masked=wrapped(x,t,labels,keep);original_trace=trace[:];trace.clear()
        pixel_keep=keep.reshape(-1,1,16,16).repeat_interleave(16,2).repeat_interleave(16,3)
        generator=torch.Generator(device='cuda').manual_seed(82729)
        changed=torch.where(pixel_keep,x,torch.randn(x.shape,generator=generator,device='cuda')*3)
        changed_out=wrapped(changed,t,labels,keep);changed_trace=trace[:];trace.clear()
    for h in handles:h.remove()
    full_error=float((dense-all_context).abs().max())
    outside_error=float(((masked-changed_out)*pixel_keep).abs().max())
    hidden=[float((a-b).abs().max()) for a,b in zip(original_trace,changed_trace)]
    with torch.inference_mode():
        individual=torch.cat([wrapped(x[i:i+1],t[i:i+1],labels[i:i+1],keep[i:i+1]) for i in range(len(x))])
    batch_error=float((masked-individual).abs().max())
    assert full_error<1e-5,(name,full_error)
    assert outside_error<1e-6 and max(hidden)<1e-6,(outside_error,hidden)
    assert batch_error<3e-5,batch_error
    assert float((masked-dense).abs().max())>1e-4,'Mask had no effect'
    # Gradient must be exactly zero outside the fixed selected set.
    xx=x[:1].detach().requires_grad_(True)
    with torch.enable_grad():
        yy=wrapped(xx,t[:1],labels[:1],keep[:1]);r,c=divmod(qs[0],16)
        loss=yy[:,:,r*16:(r+1)*16,c*16:(c+1)*16].square().sum()
        grad=torch.autograd.grad(loss,xx)[0]
    grad_out=float((grad*~pixel_keep[:1]).abs().max());assert grad_out==0,grad_out
    checks=dict(passed=True,model=receipt,full_context_max_abs=full_error,fixed_mask_outside_output_max_abs=outside_error,protected_hidden_max_abs_by_block=hidden,batched_vs_single_max_abs=batch_error,outside_gradient_max_abs=grad_out,class_queries_protected=True)
    atomic_json(out/'checks.json',checks)
    del yy,xx,grad,loss,dense,all_context,masked,changed_out,individual,original_trace,changed_trace,trace
    gc.collect();torch.cuda.empty_cache()
    # Benchmark mixed masks, using original model operations, FP32, no compilation.
    total=torch.cuda.get_device_properties(0).total_memory;bench=[]
    for batch in [256,1024,2048,4096,6144,8192]:
        if bench and bench[-1]['passed'] and bench[-1]['peak_bytes']/bench[-1]['batch']*batch>total*.89:break
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        xb=tb=lb=kb=result=None
        try:
            ix=torch.arange(batch,device='cuda')%len(x);xb=x[ix];tb=t[ix];lb=labels[ix];kb=keep[ix]
            with torch.inference_mode():
                result=wrapped(xb,tb,lb,kb);del result;torch.cuda.synchronize();start=time.monotonic()
                result=wrapped(xb,tb,lb,kb);torch.cuda.synchronize();seconds=time.monotonic()-start
                assert torch.isfinite(result).all()
                small=wrapped(xb[:6],tb[:6],lb[:6],kb[:6])
                error=float((small-result[:6]).abs().max());assert error<5e-5,error
            peak=torch.cuda.max_memory_allocated()
            bench.append(dict(batch=batch,passed=True,seconds=seconds,examples_per_second=batch/seconds,peak_bytes=peak,peak_gib=peak/2**30,fraction=peak/total,batch_max_abs=error))
            print(name,'benchmark',bench[-1],flush=True)
        except torch.cuda.OutOfMemoryError:
            bench.append(dict(batch=batch,passed=False,reason='CUDA OOM'));break
        finally:
            del xb,tb,lb,kb,result;gc.collect();torch.cuda.empty_cache()
    good=[v for v in bench if v['passed'] and v['fraction']<.9];assert good
    fastest=max(v['examples_per_second'] for v in good)
    chosen=max([v for v in good if v['examples_per_second']>=fastest*.9],key=lambda v:v['batch'])
    atomic_json(out/'benchmark.json',dict(selected=chosen,batches=bench,total_gib=total/2**30))
    atomic_json(out/'PREFLIGHT_COMPLETE.json',dict(passed=True,design_sha256=sha256(root/'design.json'),checks_sha256=sha256(out/'checks.json'),benchmark_sha256=sha256(out/'benchmark.json')))


def evaluate(root,d,name):
    out=root/name/'intervention';ready=json.loads((out/'PREFLIGHT_COMPLETE.json').read_text());assert ready['passed'] and ready['design_sha256']==sha256(root/'design.json')
    receipt=json.loads((root/'PLAN_COMPLETE.json').read_text());assert receipt['passed'] and receipt['design_sha256']==sha256(root/'design.json') and sha256(receipt['path'])==receipt['sha256']
    with np.load(receipt['path']) as z:mapping=z['mapping'];masks=z['masks'];obs=z['observations']
    model,load_receipt=load(name,d);wrapped=ClosedJiT(model)
    batch=json.loads((out/'benchmark.json').read_text())['selected']['batch']
    recs=[r for r in d['images'] if r['role']=='gallery'];qs=d['queries'];work=Path(d['work_root'])/name/'evaluation';work.mkdir(parents=True,exist_ok=True)
    cases=[(r,ti,draw) for r in recs for draw in range(2) for ti in range(len(d['times']))]
    x=torch.stack([observation(r,ti,draw,d) for r,ti,draw in cases]).cuda()
    clean_bank=torch.stack([clean(r) for r in recs]).cuda()
    tt=torch.tensor([d['times'][ti] for r,ti,draw in cases],device='cuda');labels=torch.tensor([r['class_id'] for r,ti,draw in cases],device='cuda')
    losses=np.empty((len(obs),len(qs)),np.float32);started=time.monotonic()
    for start in range(0,len(obs),batch):
        stop=min(start+batch,len(obs));p=work/f'chunk_{start:07d}_{stop:07d}.npy'
        if p.exists():
            got=np.load(p);assert got.shape==(stop-start,len(qs)) and np.isfinite(got).all();losses[start:stop]=got;continue
        oi=torch.tensor(obs[start:stop].astype(np.int64),device='cuda');keep=torch.tensor(np.unpackbits(masks[start:stop],axis=1).astype(bool),device='cuda')
        with torch.inference_mode():
            pred=wrapped(x[oi],tt[oi],labels[oi],keep)
            got=patch_losses(pred,clean_bank[oi//(2*len(d['times']))],qs).cpu().numpy()
        assert np.isfinite(got).all();losses[start:stop]=got
        with p.with_suffix('.tmp').open('wb') as f:np.save(f,got)
        p.with_suffix('.tmp').replace(p)
        del pred,oi,keep
        atomic_json(out/'progress.json',dict(completed=stop,total=len(obs),seconds=time.monotonic()-started,batch=batch))
        print(name,stop,'/',len(obs),'seconds',round(time.monotonic()-started,1),flush=True)
    result=losses[mapping,np.arange(len(qs))[None,None,None,:,None,None]]
    with torch.inference_mode():
        dense=patch_losses(wrapped(x,tt,labels),clean_bank[torch.arange(len(x),device='cuda')//10],qs).cpu().numpy().reshape(len(recs),2,len(d['times']),len(qs))
    dense_error=float(np.max(np.abs(result[:,:,:,:,-1,0]-dense)));assert dense_error<1e-5,dense_error
    path=work/'losses.npz';save(path,losses=result,independent_dense=dense,methods=d['methods'],budgets=d['budgets'],times=d['times'],queries=qs,image_ids=[r['image_id'] for r in recs])
    atomic_json(out/'EVALUATION_COMPLETE.json',dict(passed=True,path=str(path),sha256=sha256(path),shape=list(result.shape),seconds=time.monotonic()-started,batch=batch,design_sha256=sha256(root/'design.json'),plan_sha256=receipt['sha256'],dense_mse_max_abs=dense_error,load=load_receipt))


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--stage',choices=['plan','preflight','evaluate'],required=True);p.add_argument('--model',choices=['jit_b','jit_l'],default='jit_b');a=p.parse_args()
    root=a.report.resolve();d=json.loads((root/'design.json').read_text());code=Path(__file__).resolve().parents[2]
    for rel,value in json.loads((code/'MANIFEST.json').read_text()).items():assert sha256(code/rel)==value,rel
    for path,value in d['input_hashes'].items():assert sha256(path)==value,path
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')));torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.stage=='plan':prepare_plan(root,d)
    else:
        assert torch.cuda.is_available(),'CUDA required; CPU fallback forbidden'
        if a.stage=='preflight':preflight(root,d,a.model)
        else:evaluate(root,d,a.model)

if __name__=='__main__':main()
