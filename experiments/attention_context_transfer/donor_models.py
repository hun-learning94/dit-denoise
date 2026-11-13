"""Frozen adapters shared by donor-only full fields, gradients and interventions."""
from __future__ import annotations
import gc, json, time, types
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from .prepare import atomic_json, sha256
from .protocol import seed_for
from .imagenet_fields import clean, load as load_jit
from .jit_transfer import ClosedJiT, protected_rows, geometry
from .depth_pilot import model_load
from .smlp_reference import load_reference


def observation(rec,ti,draw,d,calibration=False):
    x=clean(rec);t=d['times'][ti]
    if d['dataset']=='celeba128':
        key=('smlp_locality_noise_v1','template',rec['image_id'],ti,draw) if calibration else ('confirmation_noise_v1','celeba',rec['image_id'],ti,draw)
    else:key=('imagenet_class_fields_v1',rec['image_id'],ti,draw)
    z=torch.randn(x.shape,generator=torch.Generator().manual_seed(seed_for(*key)))
    return t*x+d['noise_scale']*(1-t)*z


def patch_losses(pred,truth,queries):
    p=pred.shape[-1]//16
    loss=(pred-truth).square().reshape(-1,3,16,p,16,p).mean((1,3,5)).reshape(-1,256)
    q=torch.as_tensor(queries,device=pred.device,dtype=torch.long)
    return loss[:,q] if q.ndim==1 else loss.gather(1,q)


class Model:
    def __init__(self,name,d):
        self.name=name;self.keep=None
        if name.startswith('jit'):
            self.model,self.receipt=load_jit(name,d);self.closed=ClosedJiT(self.model)
        else:
            if name=='mlp':self.model,self.receipt=model_load('celeba',d)
            else:self.model,self.receipt,*_=load_reference(d)
            for block in self.model.blocks:
                attn=block.attn
                original=getattr(attn,'_original_forward',attn.forward)
                def forward(attn,x,original=original):
                    if self.keep is None:return original(x)
                    b,n,c=x.shape
                    q,k,v=attn.qkv(x).reshape(b,n,3,attn.num_heads,attn.head_dim).permute(2,0,3,1,4).unbind(0)
                    forbidden=self.keep[:,:,None]&~self.keep[:,None,:]
                    bias=torch.zeros((b,1,n,n),device=x.device).masked_fill_(forbidden[:,None],-torch.inf)
                    y=F.scaled_dot_product_attention(q,k,v,attn_mask=bias,dropout_p=0.)
                    return attn.proj_drop(attn.proj(y.transpose(1,2).contiguous().reshape(b,n,c)))
                attn.forward=types.MethodType(forward,attn)
        self.model.eval().requires_grad_(False)
    def fused(self,value):
        if self.name=='smlp':
            for b in self.model.blocks:b.expansion.fused=value
    def __call__(self,x,t,labels=None,keep=None):
        with sdpa_kernel(SDPBackend.MATH):
            if self.name.startswith('jit'):return self.closed(x,t,labels,keep)
            self.keep=keep
            try:return self.model(x,t)
            finally:self.keep=None
    @contextmanager
    def readout(self,queries):
        result={'score':None,'class_mass':None};handles=[];depth=len(self.model.blocks)
        def hook(attn,args,kwargs):
            x=args[0];b,n,c=x.shape;h=attn.num_heads;prefix=n-256
            q,k,_=attn.qkv(x).reshape(b,n,3,h,c//h).permute(2,0,3,1,4).unbind(0)
            if self.name.startswith('jit'):
                rope=kwargs.get('rope',args[1] if len(args)>1 else None)
                q,k=rope(attn.q_norm(q)),rope(attn.k_norm(k))
            qi=torch.as_tensor(queries,device=x.device,dtype=torch.long)+prefix
            if qi.ndim==1:qi=qi[None].expand(b,-1)
            q=q.gather(2,qi[:,None,:,None].expand(b,h,-1,c//h))
            weights=((q.float()@k.float().transpose(-2,-1))*(c//h)**-.5).softmax(-1).mean(1)
            row=weights[:,:,prefix:];mass=weights[:,:,:prefix].sum(-1)
            if result['score'] is None:
                result['score']=torch.zeros_like(row);result['class_mass']=torch.zeros_like(mass)
            result['score'].add_(row,alpha=1/depth);result['class_mass'].add_(mass,alpha=1/depth)
        for block in self.model.blocks:handles.append(block.attn.register_forward_pre_hook(hook,with_kwargs=True))
        try:yield result
        finally:
            for h in handles:h.remove()


def derivative(model,x,t,labels,offsets):
    yy=x.detach().requires_grad_(True);idx=torch.arange(len(x),device=x.device)
    out=model(yy,t,labels);result=torch.zeros_like(yy[:,0])
    for c in range(3):
        grad=torch.autograd.grad(out[idx,c,offsets[:,0],offsets[:,1]].sum(),yy,retain_graph=c<2)[0]
        result+=grad[:,c].abs()
    assert torch.isfinite(result).all()
    return result.detach()


def preflight(root,d,name):
    out=root/name/'preflight';out.mkdir(parents=True,exist_ok=True)
    model=Model(name,d);recs=d['calibration'][:6];p=8 if d['dataset']=='celeba128' else 16
    qs=[0,15,136,240,255,121];tis=np.linspace(0,len(d['times'])-1,6).astype(int)
    x=torch.stack([observation(r,int(ti),i%2,d,True) for i,(r,ti) in enumerate(zip(recs,tis))]).cuda()
    t=torch.tensor([d['times'][ti] for ti in tis],device='cuda');labels=torch.tensor([r.get('class_id',0) for r in recs],device='cuda')
    keep=torch.zeros((6,256),device='cuda',dtype=torch.bool)
    for i,(q,k) in enumerate(zip(qs,[9,16,48,96,192,32])):keep[i,geometry(q)[:k]]=True
    pixel=keep.reshape(6,1,16,16).repeat_interleave(p,2).repeat_interleave(p,3)
    trace=[]
    def trace_hook(module,args,output):trace.append(output[protected_rows(keep,output.shape[1]-256)].detach().clone())
    handles=[b.register_forward_hook(trace_hook) for b in model.model.blocks]
    with torch.inference_mode():
        dense=model(x,t,labels);trace.clear();full=model(x,t,labels,torch.ones_like(keep));trace.clear()
        masked=model(x,t,labels,keep);first=trace[:];trace.clear()
        changed=model(torch.where(pixel,x,torch.randn_like(x)*3),t,labels,keep);second=trace[:];trace.clear()
        for h in handles:h.remove()
        one=torch.cat([model(x[i:i+1],t[i:i+1],labels[i:i+1],keep[i:i+1]) for i in range(6)])
    for h in handles:h.remove()
    fullerr=float((full-dense).abs().max());out_err=float(((masked-changed)*pixel).abs().max())
    hidden=[float((a-b).abs().max()) for a,b in zip(first,second)];batcherr=float((one-masked).abs().max())
    assert fullerr<2e-5 and out_err<2e-6 and max(hidden)<2e-6,(fullerr,out_err,hidden)
    assert batcherr<2e-4,batcherr
    assert float((dense-masked).abs().max())>1e-4
    with torch.inference_mode():
        with model.readout(qs) as got:readout_pred=model(x,t,labels)
        assert torch.equal(readout_pred,dense)
        assert torch.allclose(got['score'].sum(-1)+got['class_mass'],torch.ones_like(got['class_mass']),atol=2e-6)
        # Independent retained layer/head implementation, not the streaming accumulator.
        if name.startswith('jit'):
            from .imagenet_fields import Readout
            with Readout(model.model,name,qs) as old:model(x,t,labels)
        else:
            from .depth_pilot import LayerMeans
            with LayerMeans(model.model,'celeba',qs) as old:model(x,t,labels)
        reference=torch.stack(old.rows).mean(0)
        readout_err=float((reference-got['score']).abs().max());assert readout_err<2e-7
        per=torch.tensor([qs,qs[::-1]]*3,device='cuda')
        with model.readout(per) as gathered:model(x,t,labels)
        assert torch.allclose(gathered['score'][1],got['score'][1].flip(0),atol=2e-7,rtol=1e-5)
        fused_err=0.
        if name=='smlp':
            model.fused(False);unfused=model(x,t,labels);fused_err=float((dense-unfused).abs().max());assert fused_err<2e-5
    xx=x[:1].detach().requires_grad_(True)
    pred=model(xx,t[:1],labels[:1],keep[:1]);grad=torch.autograd.grad(pred[:,:,:p,:p].square().sum(),xx)[0]
    graderr=float((grad*~pixel[:1]).abs().max());assert graderr==0,graderr
    # Native derivative matches a central finite difference at a nonzero same-channel entry.
    xx=x[:1].detach().requires_grad_(True);pr=model(xx,t[:1],labels[:1]);g=torch.autograd.grad(pr[0,0,p//2,p//2],xx)[0]
    flat=int(g[0,0].abs().argmax());rr,cc=divmod(flat,x.shape[-1]);analytic=float(g[0,0,rr,cc]);eps=.005
    with torch.inference_mode():
        delta=torch.zeros_like(xx);delta[0,0,rr,cc]=eps
        fd=float((model(xx+delta,t[:1],labels[:1])[0,0,p//2,p//2]-model(xx-delta,t[:1],labels[:1])[0,0,p//2,p//2])/(2*eps))
    assert abs(fd-analytic)<max(2e-4,.04*abs(analytic)),(fd,analytic)
    checks=dict(passed=True,model=model.receipt,full_context_max_abs=fullerr,outside_output_max_abs=out_err,hidden_max_abs=hidden,batch_max_abs=batcherr,outside_gradient_max_abs=graderr,readout_max_abs=readout_err,smlp_fused_unfused_max_abs=fused_err,gradient_finite_difference=dict(analytic=analytic,numeric=fd))
    atomic_json(out/'checks.json',checks)
    del full,dense,masked,changed,one,first,second,trace,got,old,reference,gathered,readout_pred,xx,pred,grad,pr,g
    if name=='smlp':del unfused
    gc.collect();torch.cuda.empty_cache()
    total=torch.cuda.get_device_properties(0).total_memory
    def bench(gradient=False,fused=False):
        model.fused(fused);records=[]
        candidates=[32,64,128,256,512,1024] if gradient else [256,1024,2048,4096,6144,8192]
        if name=='smlp' and not fused and not gradient:candidates=[128,256,512,1024,2048]
        for batch in candidates:
            if records and records[-1]['peak_bytes']/records[-1]['batch']*batch>total*.85:break
            gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();answer=None;xb=tb=lb=kb=None
            try:
                ix=torch.arange(batch,device='cuda')%6;xb=x[ix];tb=t[ix];lb=labels[ix];kb=keep[ix]
                offsets=torch.tensor([[p//2,p//2]]*batch,device='cuda')
                if gradient:
                    start=time.monotonic();answer=derivative(model,xb,tb,lb,offsets);torch.cuda.synchronize()
                    elapsed=time.monotonic()-start
                    small=derivative(model,xb[:6],tb[:6],lb[:6],offsets[:6]);error=float((small-answer[:6]).abs().max())
                    assert torch.allclose(small,answer[:6],atol=1e-5,rtol=2e-3),error
                else:
                    with torch.inference_mode():
                        answer=model(xb,tb,lb,kb);del answer;torch.cuda.synchronize();start=time.monotonic()
                        answer=model(xb,tb,lb,kb);torch.cuda.synchronize();elapsed=time.monotonic()-start
                        small=model(xb[:6],tb[:6],lb[:6],kb[:6]);error=float((small-answer[:6]).abs().max());assert error<2e-4,error
                peak=torch.cuda.max_memory_allocated();assert torch.isfinite(answer).all()
                row=dict(batch=batch,peak_bytes=peak,fraction=peak/total,seconds=elapsed,examples_per_second=batch/elapsed,error=error,fused=fused,gradient=gradient)
                records.append(row);print(name,'benchmark',row,flush=True)
            except torch.cuda.OutOfMemoryError:
                print('OOM',name,batch,gradient,fused,flush=True);break
            finally:
                answer=xb=tb=lb=kb=None;gc.collect();torch.cuda.empty_cache()
        good=[r for r in records if r['fraction']<.85];assert good
        fastest=max(r['examples_per_second'] for r in good)
        chosen=max((r for r in good if r['examples_per_second']>=.9*fastest),key=lambda r:r['batch'])
        return dict(selected=chosen,batches=records)
    infer=bench(fused=False)
    if name=='smlp':
        other=bench(fused=True)
        if other['selected']['examples_per_second']>infer['selected']['examples_per_second']:infer=other
        else:infer['alternative_fused']=other
    gradient=bench(gradient=True,fused=False)
    atomic_json(out/'benchmark.json',dict(inference=infer,gradient=gradient,total_gib=total/2**30))
    atomic_json(out/'COMPLETE.json',dict(passed=True,design_sha256=sha256(root/'design.json'),checks_sha256=sha256(out/'checks.json'),benchmark_sha256=sha256(out/'benchmark.json')))

if __name__=='__main__':
    import argparse, os
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True);parser.add_argument('--model',required=True);args=parser.parse_args()
    root=args.report.resolve();d=json.loads((root/'design.json').read_text())
    for path,value in d['input_hashes'].items():assert sha256(path)==value,path
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available(),'CUDA required'
    preflight(root,d,args.model)
