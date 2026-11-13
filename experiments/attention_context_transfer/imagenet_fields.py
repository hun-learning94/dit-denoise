"""Dense ImageNet attention and external sensitivity fields. No masking code."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import types
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from .models import load_model, _import_file
from .prepare import atomic_json, sha256
from .protocol import seed_for


def save(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.npz.tmp').open('wb') as f: np.savez_compressed(f, **arrays)
    path.with_suffix('.npz.tmp').replace(path)


_CLEAN = {}


def clean(rec):
    if rec['crop'] in _CLEAN: return _CLEAN[rec['crop']]
    assert sha256(rec['crop']) == rec['crop_sha256']
    result = torch.from_numpy(np.array(Image.open(rec['crop']).convert('RGB'))).permute(2,0,1).float()/127.5-1
    _CLEAN[rec['crop']] = result
    return result


def observation(rec, ti, draw, d):
    t = d['times'][ti]
    z = torch.randn((3,256,256), generator=torch.Generator().manual_seed(
        seed_for('imagenet_class_fields_v1', rec['image_id'], ti, draw)))
    return t*clean(rec)+(1-t)*z


def load(name, d):
    ck = d['checkpoints'][name]
    assert sha256(ck['path']) == ck['sha256'], name
    if name in ('pixnerd','pixeldit'):
        return load_model(name, ck['path'], d['pix_upstream'], 'cuda')
    source = Path(d['jit_upstream'])
    package = types.ModuleType('util'); package.__path__ = [str(source/'util')]
    sys.modules['util'] = package
    _import_file('util.model_util', source/'util/model_util.py')
    mod = _import_file('gallery_jit_model', source/'model_jit.py')
    ctor = mod.JiT_B_16 if name == 'jit_b' else mod.JiT_L_16
    model = ctor(input_size=256, in_channels=3, num_classes=1000, attn_drop=0., proj_drop=0.)
    # Published release includes argparse.Namespace; local trusted publisher download.
    checkpoint = torch.load(ck['path'], map_location='cpu', weights_only=False, mmap=True)
    state = {k.removeprefix('net.'):v for k,v in checkpoint[d['jit_ema']].items()}
    model.load_state_dict(state, strict=True, assign=True)
    receipt = dict(parameters=sum(p.numel() for p in model.parameters()),
                   state=d['jit_ema'], strict=True, checkpoint_sha256=ck['sha256'],
                   class_tokens=model.in_context_len, insertion_zero_based=model.in_context_start)
    del checkpoint, state
    model.eval().requires_grad_(False)
    return model.cuda(), receipt


def predict(model, name, x, t, labels):
    with sdpa_kernel(SDPBackend.MATH):
        if name.startswith('jit'): return model(x, t, labels)
        return x+(1-t[:,None,None,None])*model(x,t,labels,s=None,mask=None)


class Readout:
    """Unmodified dense attention, accounting for QK normalization, RoPE and class keys."""
    def __init__(self, model, name, queries):
        self.model,self.name,self.queries = model,name,list(queries)
        self.handles=[];self.rows=[];self.class_mass=[]
    def hook(self, module, args, kwargs):
        x=args[0];b,n,c=x.shape;h=module.num_heads
        if self.name=='pixnerd':
            pos=args[1]
            qkv=module.qkv(x).reshape(b,n,3,h,c//h).permute(2,0,1,3,4)
            q,k=module.q_norm(qkv[0]),module.k_norm(qkv[1])
            rotate=sys.modules[module.__class__.__module__].apply_rotary_emb
            q,k=rotate(q,k,freqs_cis=pos);q=q.transpose(1,2);k=k.transpose(1,2)
            prefix=0
        else:
            qkv=module.qkv(x).reshape(b,n,3,h,c//h).permute(2,0,3,1,4)
            q,k=module.q_norm(qkv[0]),module.k_norm(qkv[1])
            rope=kwargs.get('rope',args[1] if len(args)>1 else None)
            assert rope is not None
            q,k=rope(q),rope(k);prefix=n-256
            assert prefix in (0,32)
        qi=torch.as_tensor(self.queries,device=x.device)+prefix
        p=((q[:,:,qi].float()@k.float().transpose(-2,-1))*(c//h)**-.5).softmax(-1).mean(1)
        assert torch.isfinite(p).all() and torch.allclose(p.sum(-1),torch.ones_like(p.sum(-1)),atol=2e-6)
        self.rows.append(p[:,:,prefix:])
        self.class_mass.append(p[:,:,:prefix].sum(-1))
    def __enter__(self):
        blocks=self.model.blocks[:26] if self.name=='pixnerd' else self.model.blocks
        for block in blocks:self.handles.append(block.attn.register_forward_pre_hook(self.hook,with_kwargs=True))
        return self
    def __exit__(self,*args):
        for h in self.handles:h.remove()


def grouped(readout, groups):
    rows=torch.stack(readout.rows,1);mass=torch.stack(readout.class_mass,1)
    return (torch.stack([rows[:,ii].mean(1) for ii in groups.values()],1),
            torch.stack([mass[:,ii].mean(1) for ii in groups.values()],1))


def derivative(model, x, t, labels, offsets):
    yy=x.detach().requires_grad_(True);idx=torch.arange(len(yy),device=yy.device)
    out=predict(model,'pixeldit',yy,t,labels)
    result=torch.zeros_like(yy[:,0])
    for c in range(3):
        grad=torch.autograd.grad(out[idx,c,offsets[:,0],offsets[:,1]].sum(),yy,retain_graph=c<2)[0]
        result+=grad[:,c].abs()
    assert torch.isfinite(result).all()
    return result.detach()


def benchmark(model, name, d, gradient=False):
    # Actual image/time mixture, independent of gallery-field outcomes.
    recs=[r for r in d['images'] if r['role']=='calibration']
    queries=d['classes'][0]['queries'];groups=d['groups'].get(name)
    cases=[(recs[i%len(recs)],i%5,i%4) for i in range(1024)]
    values=[];total=torch.cuda.get_device_properties(0).total_memory
    candidates=[16,32,64,128,256] if gradient else [32,128,256,512,768,1024]
    for b in candidates:
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        try:
            # Benchmark repeat expansion avoids many repeated PIL/hash reads.
            xx=torch.stack([observation(*cases[i],d) for i in range(min(b,16))]).cuda()
            x=xx.repeat((b+len(xx)-1)//len(xx),1,1,1)[:b]
            t=torch.tensor([d['times'][cases[i%len(xx)][1]] for i in range(b)],device='cuda')
            labels=torch.tensor([cases[i%len(xx)][0]['class_id'] for i in range(b)],device='cuda')
            def run():
                if gradient:
                    offset=torch.tensor([[136,136]]*b,device='cuda')
                    return derivative(model,x,t,labels,offset)
                with torch.inference_mode(),Readout(model,name,queries) as rr:
                    out=predict(model,name,x,t,labels);ss,mm=grouped(rr,groups)
                    return out,ss,mm
            result=run();del result;torch.cuda.synchronize();start=time.monotonic()
            result=run();torch.cuda.synchronize();duration=time.monotonic()-start;del result
            peak=torch.cuda.max_memory_allocated()
            values.append(dict(batch=b,seconds=duration,examples_per_second=b/duration,
                               peak_gib=peak/2**30,fraction=peak/total,passed=True))
            del x,xx,t,labels
            if peak>total*.78: break
        except torch.cuda.OutOfMemoryError:
            values.append(dict(batch=b,passed=False,reason='CUDA OOM'));gc.collect();torch.cuda.empty_cache();break
    good=[r for r in values if r['passed'] and r['fraction']<.90]
    assert good,values
    fastest=max(r['examples_per_second'] for r in good)
    best=max((r for r in good if r['examples_per_second']>=.90*fastest),key=lambda r:r['batch'])
    return dict(selected=best,batches=values,device_total_gib=total/2**30)


def attention_checks(model,name,d):
    rec=next(r for r in d['images'] if r['role']=='gallery')
    x=observation(rec,2,0,d)[None].cuda();t=torch.tensor([.5],device='cuda');label=torch.tensor([rec['class_id']],device='cuda')
    q=d['classes'][0]['queries']
    with torch.inference_mode():
        base=predict(model,name,x,t,label)
        with Readout(model,name,q) as rr: hooked=predict(model,name,x,t,label)
        ss,mm=grouped(rr,d['groups'][name])
        err=float((base-hooked).abs().max());assert err==0,err
        with Readout(model,name,q) as rb:
            repeated=predict(model,name,x.repeat(3,1,1,1),t.repeat(3),label.repeat(3))
        bs,bm=grouped(rb,d['groups'][name])
        batch_error=float((bs-ss).abs().max());pred_error=float((repeated-base).abs().max())
        assert batch_error<2e-5 and pred_error<2e-4,(batch_error,pred_error)
        assert torch.allclose(ss.sum(-1)+mm,torch.ones_like(mm),atol=3e-6)
    return dict(passed=True,hook_prediction_max_abs=err,batch_score_max_abs=batch_error,
                batch_prediction_max_abs=pred_error,read_layers=len(rr.rows),class_mass_min=float(mm.min()),
                class_mass_max=float(mm.max()))


def gradient_checks(model,d):
    rec=next(r for r in d['images'] if r['role']=='gallery');x=observation(rec,2,0,d)[None].cuda()
    t=torch.tensor([.5],device='cuda');labels=torch.tensor([rec['class_id']],device='cuda')
    off=torch.tensor([[136,136]],device='cuda')
    one=derivative(model,x,t,labels,off);three=derivative(model,x.repeat(3,1,1,1),t.repeat(3),labels.repeat(3),off.repeat(3,1))
    error=float((one-three).abs().max());assert error<3e-5,error
    xx=x.detach().requires_grad_(True);value=predict(model,'pixeldit',xx,t,labels)[0,0,136,136]
    g=torch.autograd.grad(value,xx)[0]
    direction=torch.randn(x.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(4421))
    expected=float((g*direction).sum());checks=[]
    for eps in [.003,.001]:
        with torch.no_grad():
            plus=predict(model,'pixeldit',x+eps*direction,t,labels)[0,0,136,136]
            minus=predict(model,'pixeldit',x-eps*direction,t,labels)[0,0,136,136]
            actual=float((plus-minus)/(2*eps))
        checks.append(dict(eps=eps,expected=expected,actual=actual,error=abs(actual-expected)))
    assert min(r['error'] for r in checks)<3e-4+.02*abs(expected),checks
    return dict(passed=True,batch_max_abs=error,directional_derivative=checks,
                differentiated='clean prediction y+(1-t)*velocity, including direct y term')


def attention(root,d,name):
    work=Path(d['work_root']);out=root/name;out.mkdir(exist_ok=True)
    model,receipt=load(name,d);atomic_json(out/'load.json',receipt)
    checks=attention_checks(model,name,d);atomic_json(out/'checks.json',checks)
    bench=benchmark(model,name,d);atomic_json(out/'benchmark.json',bench);batch=bench['selected']['batch']
    started=time.monotonic()
    for info in d['classes']:
        wnid=info['wnid'];path=work/name/f'{wnid}.npz'
        if path.exists():continue
        queries=info['queries'];groups=d['groups'][name];nq=len(queries);ng=len(groups);nt=len(d['times'])
        cal=[r for r in d['images'] if r['role']=='calibration' and r['wnid']==wnid]
        gallery=[r for r in d['images'] if r['role']=='gallery' and r['wnid']==wnid]
        sums=np.zeros((2,nt,ng,nq,256),np.float64);mass=np.zeros((2,nt,ng,nq),np.float64)
        counts=np.zeros((2,nt),np.int64)
        current=np.zeros((len(gallery),2,nt,ng,nq,256),np.float32);cmass=np.zeros(current.shape[:-1],np.float32)
        for role,recs,draws in [('calibration',cal,4),('gallery',gallery,2)]:
            tasks=[(i,ti,draw) for i in range(len(recs)) for ti in range(nt) for draw in range(draws)]
            for start in range(0,len(tasks),batch):
                chunk=tasks[start:start+batch]
                x=torch.stack([observation(recs[i],ti,draw,d) for i,ti,draw in chunk]).cuda()
                tt=torch.tensor([d['times'][ti] for i,ti,draw in chunk],device='cuda')
                lab=torch.full((len(chunk),),info['class_id'],device='cuda',dtype=torch.long)
                with torch.inference_mode(),Readout(model,name,queries) as rr:
                    prediction=predict(model,name,x,tt,lab);ss,mm=grouped(rr,groups)
                scores=ss.cpu().numpy();masses=mm.cpu().numpy()
                for j,(i,ti,draw) in enumerate(chunk):
                    if role=='calibration':
                        half=i//16;sums[half,ti]+=scores[j];mass[half,ti]+=masses[j];counts[half,ti]+=1
                    else:current[i,draw,ti]=scores[j];cmass[i,draw,ti]=masses[j]
                del x,tt,lab,prediction,rr,ss,mm
            print(name,wnid,role,'complete',flush=True)
        assert (counts==64).all(),counts
        save(path,calibration_halves=(sums/counts[:,:,None,None,None]).astype(np.float32),
             calibration_class_mass_halves=(mass/counts[:,:,None,None]).astype(np.float32),counts=counts,
             current=current,current_class_mass=cmass,queries=queries,times=d['times'],groups=list(groups),
             gallery_ids=[r['image_id'] for r in gallery])
        atomic_json(out/f'{wnid}.json',dict(path=str(path),sha256=sha256(path),seconds=time.monotonic()-started))
    atomic_json(out/'COMPLETE.json',dict(passed=True,seconds=time.monotonic()-started,design_sha256=sha256(root/'design.json')))


def gradient(root,d):
    work=Path(d['work_root']);out=root/'pixeldit_gradient';out.mkdir(exist_ok=True)
    model,receipt=load('pixeldit',d);atomic_json(out/'load.json',receipt)
    atomic_json(out/'checks.json',gradient_checks(model,d))
    bench=benchmark(model,'pixeldit',d,gradient=True);atomic_json(out/'benchmark.json',bench);batch=bench['selected']['batch']
    started=time.monotonic()
    for info in d['classes']:
        wnid=info['wnid'];queries=info['queries'];recs=[r for r in d['images'] if r['role']=='calibration' and r['wnid']==wnid]
        path=work/'pixeldit_gradient'/f'{wnid}.npz'
        if path.exists():continue
        fields=[]
        for ti,t in enumerate(d['times']):
            part=work/'pixeldit_gradient'/f'{wnid}_t{ti}.npz'
            if part.exists():
                with np.load(part) as z: fields.append(z['halves']);continue
            sums=np.zeros((2,len(queries),256,256),np.float64);counts=np.zeros((2,len(queries)),np.int64)
            offsets=[]
            for half in range(2):
                rng=np.random.default_rng(seed_for('field_offsets',wnid,half))
                # One uniformly jittered native output pixel per 2x2 stratum.
                oo=np.array([(2*r+int(rng.integers(2)),2*c+int(rng.integers(2))) for r in range(8) for c in range(8)])
                offsets.append(oo[rng.permutation(64)])
            tasks=[(i,draw,qi) for i in range(32) for draw in range(4) for qi in range(len(queries))]
            for start in range(0,len(tasks),batch):
                chunk=tasks[start:start+batch]
                xx=torch.stack([observation(recs[i],ti,draw,d) for i,draw,qi in chunk]).cuda()
                native=[]
                for i,draw,qi in chunk:
                    rr,cc=divmod(queries[qi],16);a,b=offsets[i//16][(i%16)*4+draw]
                    native.append((rr*16+int(a),cc*16+int(b)))
                got=derivative(model,xx,torch.full((len(chunk),),t,device='cuda'),
                               torch.full((len(chunk),),info['class_id'],device='cuda',dtype=torch.long),
                               torch.tensor(native,device='cuda')).cpu().numpy()
                for j,(i,draw,qi) in enumerate(chunk):sums[i//16,qi]+=got[j];counts[i//16,qi]+=1
                del xx,got
            assert (counts==64).all()
            half=(sums/counts[:,:,None,None]).astype(np.float32);fields.append(half)
            save(part,halves=half,counts=counts,output_offsets=np.stack(offsets))
            print('gradient',wnid,t,'complete',time.monotonic()-started,flush=True)
        save(path,halves=np.stack(fields),queries=queries,times=d['times'])
        atomic_json(out/f'{wnid}.json',dict(path=str(path),sha256=sha256(path),seconds=time.monotonic()-started))
    atomic_json(out/'COMPLETE.json',dict(passed=True,seconds=time.monotonic()-started,design_sha256=sha256(root/'design.json')))


def basis(z):
    z=z-z.mean(0);n=len(z);gram=z@z.T/n;lam,v=torch.linalg.eigh(gram)
    keep=lam>lam.max()*1e-12;lam=lam[keep];u=z.T@v[:,keep]/(n*lam).sqrt()[None]
    return u,lam


def wiener_field(u,lam,t,q):
    rr,cc=divmod(q,16);spatial=[r*256+c for r in range(rr*16,(rr+1)*16) for c in range(cc*16,(cc+1)*16)]
    rows=torch.tensor([ch*256*256+j for ch in range(3) for j in spatial],device=u.device)
    weight=t*lam/(t*t*lam+(1-t)**2);field=torch.zeros((256,256),device=u.device,dtype=u.dtype)
    for start in range(0,len(rows),32):
        w=(u[rows[start:start+32]]*weight)@u.T
        maxima=w.abs().amax(1,keepdim=True);assert (maxima>0).all()
        relative=w.abs()/maxima
        field=torch.maximum(field,relative.reshape(-1,3,256,256).amax((0,1)))
    return field.float().cpu().numpy()


def covariance(root,d,device='cuda'):
    work=Path(d['work_root']);out=root/'covariance';out.mkdir(exist_ok=True);started=time.monotonic()
    # Independent, small dense-matrix check of sample-space algebra.
    torch.manual_seed(121)
    z=torch.randn(9,17,device=device,dtype=torch.float64);z-=z.mean(0);u,lam=basis(z)
    t=.3;c=z.T@z/len(z)
    direct=t*c@torch.linalg.inv(t*t*c+(1-t)**2*torch.eye(17,device=device,dtype=torch.float64))
    approx=(u*(t*lam/(t*t*lam+(1-t)**2)))@u.T
    error=float((direct-approx).abs().max());assert error<1e-11,error
    atomic_json(out/'checks.json',dict(passed=True,dense_wiener_max_abs=error))
    for info in d['classes']:
        path=work/'covariance'/f"{info['wnid']}.npz"
        if path.exists():continue
        assert sha256(info['bank'])==info['bank_sha256']
        pixels=np.load(info['bank']);results={};ranks={}
        for count in [256,512]:
            z=torch.from_numpy(pixels[:count]).to(device).permute(0,3,1,2).reshape(count,-1).double()/127.5-1
            u,lam=basis(z);ranks[str(count)]=len(lam);assert len(lam)==count-1
            result=np.stack([np.stack([wiener_field(u,lam,t,q) for q in info['queries']]) for t in d['times']])
            results[f'bank{count}']=result;del z,u,lam
            print('covariance',info['wnid'],count,'complete',time.monotonic()-started,flush=True)
        save(path,**results,queries=info['queries'],times=d['times'])
        atomic_json(out/f"{info['wnid']}.json",dict(path=str(path),sha256=sha256(path),ranks=ranks,seconds=time.monotonic()-started))
    atomic_json(out/'COMPLETE.json',dict(passed=True,seconds=time.monotonic()-started,design_sha256=sha256(root/'design.json')))


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True)
    p.add_argument('--stage',choices=['pixnerd','jit_b','jit_l','gradient','covariance'],required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a=p.parse_args();root=a.report.resolve();d=json.loads((root/'design.json').read_text())
    if a.device=='cpu':assert a.stage=='covariance','Only covariance has a deliberate CPU execution mode'
    else:assert torch.cuda.is_available(),'CUDA required; CPU fallback forbidden'
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')));torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for path,expected in d['input_hashes'].items():assert sha256(path)==expected,path
    code=Path(__file__).resolve().parents[2]
    manifest=code/'MANIFEST.json'
    assert manifest.exists(),'Frozen source manifest required'
    for rel,expected in json.loads(manifest.read_text()).items():assert sha256(code/rel)==expected,rel
    if a.stage=='gradient':gradient(root,d)
    elif a.stage=='covariance':covariance(root,d,a.device)
    else:attention(root,d,a.stage)

if __name__=='__main__':main()
