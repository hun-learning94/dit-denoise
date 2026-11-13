"""Frozen, paired depth-group pilots with independent calibration-average masks."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import gc
import gzip
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
from torch.nn.attention import SDPBackend, sdpa_kernel
from .models import load_model, _import_file, ProtectedFeatures
from .protocol import contexts, closed_bias, mask_id, seed_for, distances
from .prepare import atomic_json, sha256


def image_tensor(rec):
    with Image.open(rec['crop']) as im:
        a=np.asarray(im.convert('RGB')).copy()
    return torch.from_numpy(a).permute(2,0,1).float()/127.5-1


def noise_seed(dataset, rec, ti, times):
    if dataset=='imagenet256':
        slots={.2:0,.5:1,.8:2,.1:3,.3:4,.4:5,.6:6,.7:7,.9:8}
        return seed_for('corruption',rec['image_id'],slots[times[ti]],0)
    return seed_for('celeba_depth_corruption_v1',rec['image_id'],str(times[ti]),0)


def observations(dataset, records, spec, times, device='cuda'):
    clean=torch.stack([image_tensor(r) for r in records])
    images=[];meta=[]
    for i,rec in enumerate(records):
        for ti,t in enumerate(times):
            seed=noise_seed(dataset,rec,ti,times)
            z=torch.randn(clean[i].shape,generator=torch.Generator().manual_seed(seed))
            images.append(t*clean[i]+(1-t)*spec['noise_scale']*z)
            meta.append(dict(image_index=i,time_index=ti,t=t,noise_seed=seed))
    return clean.to(device),torch.stack(images).to(device),meta


def model_load(name, design, device='cuda'):
    ck=design['checkpoints'][name]
    print('Verifying checkpoint',name,flush=True)
    assert sha256(ck['path'])==ck['sha256'],'Checkpoint changed'
    if name!='celeba':return load_model(name,ck['path'],design['upstream'],device)
    mod=_import_file('depth_pinned_celeba_models',ck['model_source'])
    saved=torch.load(ck['path'],map_location='cpu',weights_only=False,mmap=True)
    assert saved['step']==63200 and saved['recipe']['noise_scale']==.5
    assert saved['model_cfg']['depth']==12 and saved['model_cfg']['patch_size']==8
    with torch.device('meta'):
        model=mod.DiTBackbone(mod.DiTConfig(**saved['model_cfg']))
        for block in model.blocks:block.attn=mod.FusedMultiHeadSelfAttention(768,12)
    state={k.removeprefix('model.'):v for k,v in saved['ema'].items()}
    model.load_state_dict(state,strict=True,assign=True)
    assert all(torch.isfinite(v).all() for v in state.values())
    model.eval().requires_grad_(False)
    model=model.to(device)
    # Bind a mask-aware variant without modifying trained tensors. Unmasked
    # forwards still use the original implementation for an independent check.
    for block in model.blocks:
        attn=block.attn
        attn._original_forward=attn.forward
        attn._closed_mask=None
        def masked(self,x):
            if self._closed_mask is None:return self._original_forward(x)
            b,n,d=x.shape
            q,k,v=self.qkv(x).view(b,n,3,self.num_heads,self.head_dim).permute(2,0,3,1,4).unbind(0)
            y=F.scaled_dot_product_attention(q,k,v,attn_mask=self._closed_mask,dropout_p=0.)
            return self.proj_drop(self.proj(y.transpose(1,2).contiguous().view(b,n,d)))
        attn.forward=types.MethodType(masked,attn)
    return model,dict(checkpoint=ck['path'],checkpoint_sha256=ck['sha256'],parameters=sum(p.numel() for p in model.parameters()),exact_state_match=True,ema_step=63200,output='direct clean x',conditioning='unconditional',noise_scale=.5)


@torch.inference_mode()
def predict(model,name,y,t,labels,bias=None):
    with sdpa_kernel(SDPBackend.MATH):
        if name=='celeba':
            for block in model.blocks:block.attn._closed_mask=bias
            try:return model(y,t)
            finally:
                for block in model.blocks:block.attn._closed_mask=None
        velocity=model(y,t,labels,s=None,mask=bias)
        return y+(1-t[:,None,None,None])*velocity


class LayerMeans:
    def __init__(self,model,name,queries):
        self.model,self.name,self.queries=model,name,queries
        self.rows=[];self.handles=[]
    def hook(self,module,args,kwargs):
        x=args[0];b,n,d=x.shape;h=module.num_heads
        if self.name=='pixnerd':
            pos=args[1]
            qkv=module.qkv(x).reshape(b,n,3,h,d//h).permute(2,0,1,3,4)
            q,k=module.q_norm(qkv[0]),module.k_norm(qkv[1])
            rotate=sys.modules[module.__class__.__module__].apply_rotary_emb
            q,k=rotate(q,k,freqs_cis=pos);q=q.transpose(1,2);k=k.transpose(1,2)
        else:
            qkv=module.qkv(x).reshape(b,n,3,h,d//h).permute(2,0,3,1,4)
            q,k=qkv[0],qkv[1]
        probs=torch.softmax((q[:,:,self.queries].float()@k.float().transpose(-2,-1))*(module.head_dim**-.5),-1)
        self.rows.append(probs.mean(1))
    def __enter__(self):
        blocks=self.model.blocks[:self.model.num_cond_blocks] if self.name=='pixnerd' else self.model.blocks
        for b in blocks:self.handles.append(b.attn.register_forward_pre_hook(self.hook,with_kwargs=True))
        return self
    def __exit__(self,*args):
        for h in self.handles:h.remove()


@contextmanager
def features(model,name,selected):
    if name!='celeba':
        with ProtectedFeatures(model,name,selected) as trace:yield trace.features
        return
    result={};handles=[]
    for i,block in enumerate(model.blocks):
        def hook(mod,args,output,i=i):result[str(i)]=output[selected].float().cpu()
        handles.append(block.register_forward_hook(hook))
    try:yield result
    finally:
        for h in handles:h.remove()


def patches(images,targets):
    b,c,h,w=images.shape;p=h//16
    grid=images.reshape(b,c,16,p,16,p).permute(0,2,4,1,3,5)
    return grid[torch.arange(b,device=images.device),targets//16,targets%16]


def stats(a,b):
    d=(a-b).float()
    return dict(max_abs=d.abs().max().item(),rms=d.square().mean().sqrt().item())


@torch.inference_mode()
def preflight(model,name,records,spec,times,out):
    x,y,meta=observations('celeba128' if name=='celeba' else 'imagenet256',records[:2],spec,times)
    i=times.index(.5);yy=y[i:i+1];t=torch.tensor([.5],device='cuda');label=torch.tensor([records[0]['class_id']],device='cuda')
    target=records[0]['targets'][0][0]*16+records[0]['targets'][0][1]
    baseline=predict(model,name,yy,t,label)
    full,_=closed_bias([np.arange(256)],'cuda')
    full_error=stats(predict(model,name,yy,t,label,full),baseline)
    assert full_error['max_abs']<2e-5,full_error
    checks=dict(full_mask=full_error,outside=[])
    if name in ['pixnerd','celeba']:
        with LayerMeans(model,name,[target]) as readout:instrumented=predict(model,name,yy,t,label)
        check=stats(instrumented,baseline);assert check['max_abs']==0,check
        layers=torch.stack(readout.rows)
        assert len(readout.rows)==(26 if name=='pixnerd' else 12)
        assert torch.isfinite(layers).all() and (layers>=0).all() and torch.allclose(layers.sum(-1),torch.ones_like(layers.sum(-1)),atol=1e-6)
        checks['instrumentation']=check
    scores=np.random.default_rng(20261107).random(256)
    for k in [9,40,133]:
        idx=contexts(scores,target,[k],records[0]['image_id'],i)[('attention',k)]
        bias,selected=closed_bias([idx],'cuda')
        with features(model,name,selected) as trace:base=predict(model,name,yy,t,label,bias)
        pixels=selected.reshape(1,16,16).repeat_interleave(spec['patch_size'],1).repeat_interleave(spec['patch_size'],2)[:,None]
        gen=torch.Generator(device='cuda').manual_seed(8000+k)
        replacement=torch.randn(yy.shape,device='cuda',generator=gen)
        shuffled=yy.reshape(1,3,16,spec['patch_size'],16,spec['patch_size']).permute(0,2,4,1,3,5).clone()
        outside=np.setdiff1d(np.arange(256),idx);permuted=np.random.default_rng(k).permutation(outside)
        shuffled=shuffled.reshape(1,256,3,spec['patch_size'],spec['patch_size']);shuffled[:,outside]=shuffled[:,permuted]
        shuffled=shuffled.reshape(1,16,16,3,spec['patch_size'],spec['patch_size']).permute(0,3,1,4,2,5).reshape_as(yy)
        for kind,alt in [('independent_noise',replacement),('outside_shuffle',shuffled)]:
            with features(model,name,selected) as trace2:other=predict(model,name,torch.where(pixels,yy,alt),t,label,bias)
            e=stats(patches(other,torch.tensor([target],device='cuda')),patches(base,torch.tensor([target],device='cuda')))
            layer=max(stats(trace[key],trace2[key])['max_abs'] for key in trace)
            assert e['max_abs']<=2e-5 and layer<=2e-5,(e,layer)
            checks['outside'].append(dict(budget=k,replacement=kind,target=e,protected_layer_max_abs=layer))
    checks['passed']=True;atomic_json(out/f'{name}_preflight.json',checks)
    del x,y,baseline,full,base,other,yy


@torch.inference_mode()
def extract(model,name,dataset,spec,design,out):
    pilot=json.loads(Path(design['data'][dataset+'_pilot']).read_text())
    calibration=json.loads(Path(design['data'][dataset+'_calibration']).read_text())
    times=design['times'];groups=spec['groups'];queries=sorted({r*16+c for rec in pilot for r,c in rec['targets']})
    qindex={q:i for i,q in enumerate(queries)};group_names=list(groups)
    sums=np.zeros((len(times),len(groups),len(queries),256),dtype=np.float64);counts=np.zeros(len(times),int)
    # Extract many real image/time observations together; no rollout products.
    batch=512 if name=='pixnerd' else 1536
    checks=[]
    for split,records in [('calibration',calibration),('pilot',pilot)]:
        x,y,meta=observations(dataset,records,spec,times)
        labels=torch.tensor([records[m['image_index']]['class_id'] for m in meta],device='cuda')
        tt=torch.tensor([m['t'] for m in meta],device='cuda')
        saved={} if split=='pilot' else None
        dense={} if split=='pilot' else None
        for start in range(0,len(meta),batch):
            stop=min(start+batch,len(meta))
            with LayerMeans(model,name,queries) as readout:prediction=predict(model,name,y[start:stop],tt[start:stop],labels[start:stop])
            rows=torch.stack(readout.rows,1) # B,L,Q,N
            assert torch.isfinite(rows).all() and (rows>=0).all()
            assert torch.allclose(rows.sum(-1),torch.ones_like(rows.sum(-1)),atol=2e-6)
            grouped=torch.stack([rows[:,indices].mean(1) for indices in groups.values()],1).cpu().numpy()
            if split=='calibration':
                for local,m in enumerate(meta[start:stop]):sums[m['time_index']]+=grouped[local];counts[m['time_index']]+=1
            else:
                cpu_rows=rows.cpu().numpy()
                for local,m in enumerate(meta[start:stop]):
                    rec=records[m['image_index']];targets=[r*16+c for r,c in rec['targets']];qi=[qindex[q] for q in targets]
                    saved[m['image_index'],m['time_index']]=cpu_rows[local][:,qi]
                    target_tensor=torch.tensor(targets,device='cuda')
                    pp=patches(prediction[local:local+1].expand(len(targets),-1,-1,-1),target_tensor)
                    truth=patches(x[m['image_index']:m['image_index']+1].expand(len(targets),-1,-1,-1),target_tensor)
                    dense[m['image_index'],m['time_index']]=(pp-truth).square().flatten(1).mean(1).cpu().numpy()
            print(dataset,split,'readout observations',stop,'/',len(meta),flush=True)
            del prediction,readout,rows,grouped
        if split=='calibration':
            assert (counts==len(records)).all()
            fixed=(sums/counts[:,None,None,None]).astype(np.float32)
            np.savez_compressed(out/'fixed_calibration_scores.npz',scores=fixed,queries=queries,times=times,groups=group_names,counts=counts)
            atomic_json(out/'calibration_receipt.json',dict(images=len(records),forward_observations=len(meta),manifest_sha256=sha256(design['data'][dataset+'_calibration']),scores_sha256=sha256(out/'fixed_calibration_scores.npz'),clean_losses_evaluated=False,frozen_before_pilot_evaluation=True,targets=queries))
        else:
            folder=out/'pilot_scores';folder.mkdir(exist_ok=True)
            for i,rec in enumerate(pilot):
                layers=np.stack([saved[i,ti] for ti in range(len(times))])
                scores=np.stack([layers[:,indices].mean(1) for indices in groups.values()],1)
                np.savez_compressed(folder/f'{i:03d}.npz',layers=layers,scores=scores,targets=[r*16+c for r,c in rec['targets']],times=times,groups=group_names,dense_losses=np.stack([dense[i,ti] for ti in range(len(times))]))
                if dataset=='imagenet256':
                    for oldti,t in enumerate([.2,.5,.8]):
                        oldfile=Path(design['old_pilot_root'])/'mean_comparison_v3/donor_scores'/f"{rec['image_id']}_t{oldti}.npz"
                        previous=np.load(oldfile)['layer_head_means'];current=layers[times.index(t)]
                        delta=float(np.max(np.abs(current-previous)))
                        assert np.allclose(current,previous,atol=2e-6,rtol=1e-4),delta
                        checks.append(dict(image_id=rec['image_id'],t=t,layer_rows_max_abs=delta))
        del x,y,tt,labels
        gc.collect();torch.cuda.empty_cache()
    atomic_json(out/'readout_checks.json',dict(passed=True,calibration_images=len(calibration),pilot_images=len(pilot),groups=groups,legacy_layer_row_comparisons=checks,calibration_and_pilot_disjoint=not ({r['image_id'] for r in calibration}&{r['image_id'] for r in pilot})))


def make_jobs(dataset,spec,design,out):
    records=json.loads(Path(design['data'][dataset+'_pilot']).read_text());times=design['times'];budgets=design['budgets']
    fixed=np.load(out/'fixed_calibration_scores.npz');qindex={int(q):i for i,q in enumerate(fixed['queries'])}
    names=list(spec['groups']);jobs=[];mask_bank=[];mask_ids=[];lookup={};logical=0
    def mask_index(idx):
        ident=mask_id(idx)
        if ident not in lookup:
            lookup[ident]=len(mask_bank);mask_bank.append(np.array(idx,dtype=np.int16));mask_ids.append(ident)
        else:assert np.array_equal(mask_bank[lookup[ident]],idx)
        return lookup[ident]
    for i,rec in enumerate(records):
        cache=np.load(out/'pilot_scores'/f'{i:03d}.npz');targets=[r*16+c for r,c in rec['targets']]
        assert list(cache['targets'])==targets
        for ti,t in enumerate(times):
            for qi,target in enumerate(targets):
                unique={}
                def add(idx,k,group,selector):
                    nonlocal logical
                    mid=mask_index(idx)
                    if mid not in unique:unique[mid]=[]
                    unique[mid].append(dict(budget=k,group=group,selector=selector));logical+=1
                reference=None
                for gi,group in enumerate(names):
                    adaptive=contexts(cache['scores'][ti,gi,qi],target,budgets,rec['image_id'],ti,repeats=3)
                    fixed_masks=contexts(fixed['scores'][ti,gi,qindex[target]],target,budgets,rec['image_id'],ti,repeats=0)
                    for k in budgets[1:-1]:
                        selected=adaptive['attention',k];add(selected,k,group,'adaptive')
                        add(fixed_masks['attention',k],k,group,'fixed')
                        d=distances(target)
                        for rep in range(3):
                            idx=adaptive[f'random_{rep}',k]
                            assert np.array_equal(np.sort(d[idx]),np.sort(d[selected]))
                            add(idx,k,group,f'random_{rep}')
                    reference=adaptive
                for k in budgets[1:-1]:add(reference['nearest',k],k,'common','nearest')
                add(reference['attention',9],9,'common','initial')
                for mid,aliases in unique.items():
                    jobs.append(dict(frame=i*len(times)+ti,target=target,mask_index=mid,aliases=aliases))
    padded=np.full((len(mask_bank),256),-1,dtype=np.int16)
    for i,idx in enumerate(mask_bank):padded[i,:len(idx)]=idx
    np.savez_compressed(out/'mask_bank.npz',indices=padded,mask_ids=mask_ids)
    with gzip.open(out/'jobs.json.gz','wt') as f:json.dump(jobs,f,separators=(',',':'))
    atomic_json(out/'mask_checks.json',dict(passed=True,unique_forward_examples=len(jobs),logical_masked_records=logical,unique_masks=len(mask_bank),exact_distance_profiles=True,initial_included=True,nested=True,mask_bank_sha256=sha256(out/'mask_bank.npz'),jobs_sha256=sha256(out/'jobs.json.gz')))
    return jobs,mask_bank,mask_ids


class EvaluationInputs:
    def __init__(self,dataset,records,spec,times,masks):
        self.x,self.y,self.meta=observations(dataset,records,spec,times)
        self.t=torch.tensor([m['t'] for m in self.meta],device='cuda')
        self.labels=torch.tensor([records[m['image_index']]['class_id'] for m in self.meta],device='cuda')
        self.image_indices=torch.tensor([m['image_index'] for m in self.meta],device='cuda')
        p=spec['patch_size'];self.truth=self.x.reshape(len(records),3,16,p,16,p).permute(0,2,4,1,3,5)
        selected=np.zeros((len(masks),256),dtype=bool)
        for i,idx in enumerate(masks):selected[i,idx]=True
        self.masks=torch.from_numpy(selected).to('cuda')
    def batch(self,jobs):
        ids=torch.tensor([j['frame'] for j in jobs],device='cuda');targets=torch.tensor([j['target'] for j in jobs],device='cuda')
        selected=self.masks[torch.tensor([j['mask_index'] for j in jobs],device='cuda')]
        bias=torch.zeros((len(jobs),256,256),device='cuda').masked_fill_(selected[:,:,None]&~selected[:,None,:],-float('inf'))[:,None]
        truth=self.truth[self.image_indices[ids],targets//16,targets%16]
        return self.y[ids],self.t[ids],self.labels[ids],bias,targets,truth


@torch.inference_mode()
def benchmark(model,name,inputs,jobs,design,out):
    case_ids=np.linspace(0,len(jobs)-1,8,dtype=int)
    cases=[jobs[int(i)] for i in case_ids];references=[]
    for case in cases:
        y,t,labels,bias,targets,truth=inputs.batch([case])
        references.append(predict(model,name,y,t,labels,bias))
    reference=torch.cat(references)
    candidates={'pixnerd':[768],'pixeldit':[1280],'celeba':[256,512,1024,2048,3072,4096,5120,6144,7168,8192]}[name]
    capacity=torch.cuda.get_device_properties(0).total_memory/2**30;limit=.85*capacity
    records=[]
    for b in candidates:
        measured=[r for r in records if 'peak_allocated_gib' in r]
        if len(measured)>=2:
            a,c=measured[-2:];slope=(c['peak_allocated_gib']-a['peak_allocated_gib'])/(c['batch']-a['batch'])
            estimate=(c['peak_allocated_gib']+slope*(b-c['batch']))*1.02
            if estimate>limit:records.append(dict(batch=b,skipped=True,estimated_peak_gib=estimate));break
        y=t=labels=bias=targets=truth=pred=None
        try:
            batch_jobs=[cases[i%8] for i in range(b)]
            y,t,labels,bias,targets,truth=inputs.batch(batch_jobs)
            torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            pred=predict(model,name,y,t,labels,bias);torch.cuda.synchronize();elapsed=time.perf_counter()-start
            err=stats(pred[:8],reference)
            refpatch=patches(reference,targets[:8]);gotpatch=patches(pred[:8],targets[:8])
            refmse=(refpatch-truth[:8]).square().flatten(1).mean(1);mse=(gotpatch-truth[:8]).square().flatten(1).mean(1)
            tol=design['batch_numeric_tolerances']
            err['target_mse_max_abs']=(mse-refmse).abs().max().item()
            err['passed']=err['rms']<=tol['clean_output_rms'] and err['max_abs']<=tol['clean_output_max_abs'] and bool(torch.allclose(mse,refmse,atol=tol['target_mse_atol'],rtol=tol['target_mse_rtol']))
            atomic_json(out/f'{name}_batch{b}_numerics.json',err)
            if not err['passed']:raise RuntimeError(f'{name} batching mismatch: {err}')
            peak=torch.cuda.max_memory_allocated()/2**30
            record=dict(batch=b,seconds=elapsed,examples_per_second=b/elapsed,peak_allocated_gib=peak,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,passed=True,fits=peak<=limit)
            records.append(record);print(name,'batch',b,'peakGiB',round(peak,2),'examples/s',round(b/elapsed,2),flush=True)
            if peak>limit:break
        except torch.cuda.OutOfMemoryError:
            records.append(dict(batch=b,oom=True));break
        finally:
            del y,t,labels,bias,targets,truth,pred;gc.collect();torch.cuda.empty_cache()
    valid=[r for r in records if r.get('passed') and r['fits']]
    assert valid,'No valid large batch'
    selected=max(valid,key=lambda r:r['batch'])
    atomic_json(out/f'{name}_benchmark.json',dict(selected_batch=selected['batch'],device_total_gib=capacity,batches=records,selected=selected,source='Known validated ImageNet batches rechecked; CelebA capacity sweep',complete=True))
    return selected['batch']


@torch.inference_mode()
def evaluate(model,name,dataset,spec,design,out,jobs,masks,mask_ids):
    records=json.loads(Path(design['data'][dataset+'_pilot']).read_text());times=design['times']
    inputs=EvaluationInputs(dataset,records,spec,times,masks)
    batch=benchmark(model,name,inputs,jobs,design,out)
    folder=out/'results'/name;folder.mkdir(parents=True,exist_ok=True)
    # Dense references scored with native conversion, no selected-mask information.
    dense=[]
    for offset in range(0,len(inputs.meta),batch):
        stop=min(offset+batch,len(inputs.meta));prediction=predict(model,name,inputs.y[offset:stop],inputs.t[offset:stop],inputs.labels[offset:stop])
        for local,m in enumerate(inputs.meta[offset:stop]):
            rec=records[m['image_index']]
            for r,c in rec['targets']:
                target=r*16+c;p=spec['patch_size']
                mse=(prediction[local,:,r*p:(r+1)*p,c*p:(c+1)*p]-inputs.x[m['image_index'],:,r*p:(r+1)*p,c*p:(c+1)*p]).square().mean().item()
                dense.append(dict(image_id=rec['image_id'],target=target,time_index=m['time_index'],t=m['t'],noise_seed=m['noise_seed'],budget=256,group='common',selector='full',mask_id='all',mse=mse))
        del prediction
    atomic_json(out/f'{name}_dense.json',dict(rows=dense))
    started=time.perf_counter();logical=0
    for offset in range(0,len(jobs),batch):
        path=folder/f'{offset:07d}.json.gz';subset=jobs[offset:offset+batch]
        if path.exists():
            with gzip.open(path,'rt') as f:old=json.load(f)
            assert old['job_count']==len(subset) and old['offset']==offset
            logical+=len(old['rows']);continue
        y,t,labels,bias,targets,truth=inputs.batch(subset)
        pred=predict(model,name,y,t,labels,bias)
        losses=(patches(pred,targets)-truth).square().flatten(1).mean(1).cpu().tolist()
        assert np.isfinite(losses).all()
        rows=[]
        for job,loss in zip(subset,losses):
            m=inputs.meta[job['frame']];rec=records[m['image_index']]
            for alias in job['aliases']:
                rows.append(dict(image_id=rec['image_id'],target=job['target'],time_index=m['time_index'],t=m['t'],noise_seed=m['noise_seed'],mask_id=mask_ids[job['mask_index']],mse=loss,**alias))
        temp=path.with_suffix('.tmp')
        with gzip.open(temp,'wt') as f:json.dump(dict(offset=offset,job_count=len(subset),rows=rows),f,separators=(',',':'))
        temp.replace(path);logical+=len(rows)
        del y,t,labels,bias,targets,truth,pred
        if offset==0 or offset%(batch*8)==0 or offset+batch>=len(jobs):
            print(dataset,name,'evaluated',min(offset+batch,len(jobs)),'/',len(jobs),'elapsed',round(time.perf_counter()-started,1),flush=True)
            atomic_json(out/f'{name}_progress.json',dict(unique_examples_completed=min(offset+batch,len(jobs)),unique_examples_total=len(jobs),logical_records=logical,seconds=time.perf_counter()-started))
    atomic_json(out/f'{name}_complete.json',dict(passed=True,unique_masked_examples=len(jobs),logical_masked_records=logical,dense_records=len(dense),seconds=time.perf_counter()-started,batch_size=batch))
    del inputs;gc.collect();torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--report',type=Path,required=True);ap.add_argument('--dataset',choices=['imagenet256','celeba128'],required=True)
    args=ap.parse_args();report=args.report.resolve();design=json.loads((report/'design.json').read_text());dataset=args.dataset;spec=design['datasets'][dataset]
    out=report/'results_v1'/dataset;out.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.is_available(),'CUDA required; no CPU fallback'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    frozen=Path(__file__).resolve().parents[2]/'MANIFEST.json'
    if frozen.exists():
        for path,digest in json.loads(frozen.read_text())['files'].items():assert sha256(path)==digest,path
    for p,h in design['input_hashes'].items():assert sha256(p)==h,p
    pilot=json.loads(Path(design['data'][dataset+'_pilot']).read_text());cal=json.loads(Path(design['data'][dataset+'_calibration']).read_text())
    assert len(pilot)==16 and len(cal)==128 and not ({r['image_id'] for r in pilot}&{r['image_id'] for r in cal})
    for r in pilot+cal:assert sha256(r['crop'])==r['crop_sha256'],r['image_id']
    atomic_json(out/'run.json',dict(job_id=os.environ.get('SLURM_JOB_ID'),dataset=dataset,device=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,design_sha256=sha256(report/'design.json'),scope=design['scope']))
    jobs=masks=ids=None
    for name in spec['recipients']:
        model,receipt=model_load(name,design);atomic_json(out/f'{name}_load.json',receipt)
        preflight(model,name,pilot,spec,design['times'],out);print(name,'preflight passed',flush=True)
        if name==spec['donor']:
            extract(model,name,dataset,spec,design,out)
            jobs,masks,ids=make_jobs(dataset,spec,design,out)
            print(dataset,'unique evaluations',len(jobs),'unique masks',len(masks),flush=True)
        evaluate(model,name,dataset,spec,design,out,jobs,masks,ids)
        del model;gc.collect();torch.cuda.empty_cache()
    atomic_json(out/'COMPLETE.json',dict(completed=True,dataset=dataset,recipients=spec['recipients'],pilot_images=16,calibration_images=128,final_test_evaluated=False))


if __name__=='__main__':main()
