"""Audited GPU smoke tests and resumable joint adaptation for pixel DiT / LFM."""
from __future__ import annotations
import argparse
import gc
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from .model import (COUNTS, digest, write, save, seed_all, rng_state, restore_rng,
                    load_source, plain_model, attach_adapters, effective_state,
                    EffectiveEMA, assert_frozen, hybrid_state, update_diagnostics, uvit_module)

STOP = False
def request_stop(*_):
    global STOP
    STOP = True

class Images(Dataset):
    def __init__(self, metadata, split):
        self.metadata=metadata;self.split=split;self.array=None
        record=metadata['splits'][split]
        assert digest(record['path'])==record['sha256'], 'Dataset membership changed'
        if metadata['kind']=='mmap': self.members=np.load(record['path'])
        else: self.members=Path(record['path']).read_text().splitlines()
        assert len(self.members)==record['count']
    def __len__(self): return len(self.members)
    def __getitem__(self,index):
        if self.metadata['kind']=='mmap':
            if self.array is None: self.array=np.load(self.metadata['images'],mmap_mode='r')
            a=np.array(self.array[int(self.members[index])],copy=True)
        else:
            with Image.open(self.members[index]) as im:
                native=self.metadata.get('input_resolution',self.metadata['resolution'])
                assert im.size==(native,)*2 and im.mode=='RGB'
                if native!=self.metadata['resolution']:
                    assert native>self.metadata['resolution'], 'No silent upscaling'
                    im=im.resize((self.metadata['resolution'],)*2,Image.Resampling.LANCZOS)
                a=np.array(im,copy=True)
        return torch.from_numpy(a).permute(2,0,1)

class ResumableBatches(Sampler):
    """Global batches determined solely by seed, target epoch, and update index."""
    def __init__(self,n,seed,start,end,batch=32):
        self.n=n;self.seed=seed;self.start=start;self.end=end;self.batch=batch
        self.steps_per_target_epoch=n//batch
        assert self.steps_per_target_epoch>0
    def __iter__(self):
        previous=None
        for step in range(self.start,self.end):
            epoch,offset=divmod(step,self.steps_per_target_epoch)
            if epoch!=previous:
                generator=torch.Generator().manual_seed(self.seed+1000003*epoch)
                order=torch.randperm(self.n,generator=generator).tolist();previous=epoch
            yield order[offset*self.batch:(offset+1)*self.batch]
    def __len__(self): return self.end-self.start

def loader(dataset,seed,start,end,workers=2):
    return DataLoader(dataset,batch_sampler=ResumableBatches(len(dataset),seed,start,end),
                      num_workers=workers,pin_memory=True,
                      generator=torch.Generator().manual_seed(seed+99991),
                      persistent_workers=workers>0)

def vae_for(spec):
    if spec['family']!='lfm': return None
    from diffusers import AutoencoderKL
    p=Path(spec['vae'])
    assert digest(p/'diffusion_pytorch_model.safetensors')==spec['vae_sha256']
    vae=AutoencoderKL.from_pretrained(str(p),local_files_only=True,use_safetensors=True,torch_dtype=torch.float32)
    assert abs(float(vae.config.scaling_factor)-.18215)<1e-8
    return vae.cuda().eval().requires_grad_(False)

def normalized(images,augment):
    x=images.cuda(non_blocking=True).float().div_(127.5).sub_(1)
    if augment:
        flip=torch.rand(x.shape[0],device=x.device)<.5
        x[flip]=x[flip].flip(-1)
    return x

def loss_values(net,x,spec,vae):
    if spec['family']=='uvit':
        # Same corruption and epsilon target as upstream LSimple/VPSDE.
        sde=uvit_module(spec,'sde').VPSDE()
        t,epsilon,z=sde.sample(x)
        with torch.autocast('cuda',dtype=torch.bfloat16): prediction=net(z,999*t)
        return (prediction.float()-epsilon).square().flatten(1).mean(1)
    if spec['family']=='pixel':
        t=(torch.randn(x.shape[0],1,1,1,device=x.device)*.8-.8).sigmoid()
        noise=torch.randn_like(x)*.5
        z=t*x+(1-t)*noise
        with torch.autocast('cuda',dtype=torch.bfloat16): prediction=net(z,t.flatten())
        error=(prediction.float()-x)/(1-t).clamp_min(.05)
        return error.square().flatten(1).mean(1)
    # Follow the released LFM training path: sample VAE posterior (not mode),
    # scale by .18215, t~U[0,1], and regress velocity along its sigma_min path.
    with torch.no_grad(): z0=vae.encode(x).latent_dist.sample().mul_(.18215)
    t=torch.rand(x.shape[0],1,1,1,device=x.device)
    epsilon=torch.randn_like(z0);sigma=1e-5+(1-1e-5)*t
    z=(1-t)*z0+sigma*epsilon;target=(1-1e-5)*epsilon-z0
    with torch.autocast('cuda',dtype=torch.bfloat16): velocity=net(t.flatten(),z,y=None)
    return (velocity.float()-target).square().flatten(1).mean(1)

def optimizer_for(net,spec):
    betas=(.99,.99) if spec['family']=='uvit' else ((.9,.95) if spec['family']=='pixel' else (.9,.999))
    return torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=spec['lr'],betas=betas,weight_decay=0)

def learning_rate(spec,step,reference_steps):
    if spec['family'] in ('pixel','uvit'):
        # Preserve the source's five-epoch linear warmup and constant schedule,
        # with a lower adaptation LR and reference epochs in this experiment.
        return spec['lr']*min(1.,(step+1)/(5*reference_steps))
    # Released ImageNet DiT-B/2 command uses 1e-4 and --no_lr_decay.
    return spec['lr']

def train_step(net,optimizer,ema,images,spec,vae,micro,lr,check_groups=False):
    net.train();optimizer.zero_grad(set_to_none=True)
    for pg in optimizer.param_groups: pg['lr']=lr
    loss=0.
    for chunk in images.split(micro):
        values=loss_values(net,normalized(chunk,True),spec,vae)
        assert torch.isfinite(values).all(), 'Non-finite target loss'
        part=values.sum()/len(images);part.backward();loss+=float(part.detach())
    params=[p for p in net.parameters() if p.requires_grad]
    norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
    if check_groups:
        qk=vo=mlp=0.
        for name,p in net.named_parameters():
            if not p.requires_grad: continue
            assert p.grad is not None,name
            if '.attn.qkv.weight' in name:
                d=p.shape[1];qk+=float(p.grad[:2*d].square().sum());vo+=float(p.grad[2*d:].square().sum())
            elif '.attn.proj.weight' in name: vo+=float(p.grad.square().sum())
            else: mlp+=float(p.grad.square().sum())
        assert qk>0 and vo>0 and mlp>0,dict(QK=qk,VO=vo,MLP=mlp)
    optimizer.step();ema.update(net)
    return dict(loss=loss,gradient_norm=float(norm),lr=lr)

def online_payload(net,optimizer,ema,step,identity):
    return dict(format='matched-scale-online-v1',identity=identity,step=step,
                net=net.state_dict(),optimizer=optimizer.state_dict(),ema=ema.shadow,
                ema_updates=ema.updates,rng=rng_state())

def restore(payload,net,optimizer,ema,identity):
    assert payload['identity']==identity,'Checkpoint identity mismatch'
    net.load_state_dict(payload['net'],strict=True);optimizer.load_state_dict(payload['optimizer'])
    for name,v in payload['ema'].items(): ema.shadow[name].copy_(v)
    ema.updates=payload['ema_updates'];restore_rng(payload['rng'])
    assert ema.updates==payload['step']
    return payload['step']

def rolling_checkpoint(run,net,optimizer,ema,step,identity):
    save(run/'online_next.pt',online_payload(net,optimizer,ema,step,identity))
    latest=run/'online_latest.pt'
    if latest.exists(): latest.replace(run/'online_previous.pt')
    (run/'online_next.pt').replace(latest)

@torch.no_grad()
def evaluate(state,spec,config,dataset,vae,count=512,seed=20261106):
    keep=rng_state()
    net=plain_model(spec,state,config).cuda().eval().requires_grad_(False)
    seed_all(seed)
    order=torch.randperm(len(dataset),generator=torch.Generator().manual_seed(seed))[:min(count,len(dataset))].tolist()
    values=[]
    # Fixed microbatch and RNG for every hybrid: paired heldout comparisons.
    evaluation_batch=spec.get('evaluation_batch',8)
    for i in range(0,len(order),evaluation_batch):
        images=torch.stack([dataset[j] for j in order[i:i+evaluation_batch]])
        values.extend(loss_values(net,normalized(images,False),spec,vae).cpu().tolist())
    del net;gc.collect();torch.cuda.empty_cache();restore_rng(keep)
    a=np.array(values)
    return dict(mean=float(a.mean()),standard_error=float(a.std(ddof=1)/len(a)**.5),
                count=len(a),per_example=values,indices=order,seed=seed)

def identity_for(manifest_path,run_spec,data_path):
    execution=json.loads(Path(manifest_path).read_text())
    return dict(manifest_sha256=execution.get('checkpoint_manifest_sha256',digest(manifest_path)),
                run_id=run_spec['id'],data_sha256=digest(data_path))

def smoke(manifest,path,family):
    spec=manifest['sources'][family]
    data_path=Path(manifest['smoke_data'][family]);metadata=json.loads(data_path.read_text())
    data=Images(metadata,'train')
    output=Path(manifest['report'])/'smoke'/family;output.mkdir(parents=True,exist_ok=True)
    identity=dict(manifest_sha256=digest(path),family=family,data_sha256=digest(data_path))
    seed_all(0);net,source,config=load_source(spec);net=net.cuda().eval()
    images=torch.stack([data[i] for i in range(32)])
    probe=normalized(images[:2],False)
    if family=='lfm': probe=torch.randn(2,4,32,32,device='cuda')
    t=torch.full((2,),.4,device='cuda')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        before=net(probe,999*t if family=='uvit' else t) if family in ('pixel','uvit') else net(t,probe,y=None)
    counts=attach_adapters(net,0,rank=spec.get('mlp_rank'),expected_counts=spec.get('group_counts'))
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        after=net(probe,999*t if family=='uvit' else t) if family in ('pixel','uvit') else net(t,probe,y=None)
    assert torch.equal(before,after),'Zero LoRA altered source prediction'
    del probe,before,after
    ema=EffectiveEMA(source);optimizer=optimizer_for(net,spec);vae=vae_for(spec)
    micro=spec.get('microbatch',8)
    torch.cuda.reset_peak_memory_stats();timings=[];losses=[]
    for i in range(3):
        torch.cuda.synchronize();started=time.monotonic()
        r=train_step(net,optimizer,ema,images,spec,vae,micro,spec['lr'],True)
        torch.cuda.synchronize();timings.append(time.monotonic()-started);losses.append(r)
    assert_frozen(net,source,ema)
    checkpoint=output/'resume_test.pt'
    save(checkpoint,online_payload(net,optimizer,ema,3,identity))
    reference=train_step(net,optimizer,ema,images,spec,vae,micro,spec['lr'],True)
    expected={n:p.detach().cpu().clone() for n,p in net.named_parameters() if p.requires_grad}
    expected_ema={n:v.cpu().clone() for n,v in ema.shadow.items()}
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    restore(payload,net,optimizer,ema,identity);del payload
    repeated=train_step(net,optimizer,ema,images,spec,vae,micro,spec['lr'],True)
    max_error=max(float((p.detach().cpu()-expected[n]).abs().max()) for n,p in net.named_parameters() if p.requires_grad)
    ema_error=max(float((v.cpu()-expected_ema[n]).abs().max()) for n,v in ema.shadow.items())
    write(output/'resume_comparison.json',dict(reference=reference,repeated=repeated,
        parameter_max_error=max_error,ema_max_error=ema_error,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        largest_parameter_errors=sorted([(n,float((p.detach().cpu()-expected[n]).abs().max()))
            for n,p in net.named_parameters() if p.requires_grad],key=lambda r:r[1],reverse=True)[:10]))
    assert abs(reference['loss']-repeated['loss'])<1e-7,(reference,repeated)
    assert max_error<2e-7 and ema_error<2e-7,(max_error,ema_error)
    assert_frozen(net,source,ema)
    for code,expected_state in [('000',source),('111',ema.shadow)]:
        hybrid=hybrid_state(source,ema.shadow,code)
        assert all(torch.equal(v.cpu(),expected_state[n].cpu()) for n,v in hybrid.items())
    del hybrid,expected,expected_ema
    validation=evaluate(ema.shadow,spec,config,Images(metadata,'valid'),vae,count=16)
    assert math.isfinite(validation['mean'])
    # Verify the native LFM path's clean inversion independently of the model.
    z0=torch.randn(2,4,32,32,device='cuda');e=torch.randn_like(z0);t=torch.rand(2,1,1,1,device='cuda')
    sigma=1e-5+(1-1e-5)*t;z=(1-t)*z0+sigma*e;v=(1-1e-5)*e-z0
    assert torch.allclose((1-1e-5)*z-sigma*v,z0,atol=1e-6)
    if family=='uvit':
        # Check both the original FP32 denoising objective and checkpoint sampler
        # on fixed draws, independently of our BF16 adaptation implementation.
        seed_all(20261107)
        x=normalized(images[:2],False);sde=uvit_module(spec,'sde')
        score=sde.ScoreModel(net,pred='noise_pred',sde=sde.VPSDE())
        keep=rng_state()
        with torch.no_grad(): native=sde.LSimple(score,x)
        restore_rng(keep)
        t,noise,z=score.sde.sample(x)
        with torch.no_grad(): manual=(net(z,t*999)-noise).square().flatten(1).mean(1)
        assert torch.equal(native,manual),'Native U-ViT loss mismatch'
        seed_all(20261107)
        with torch.no_grad():
            initial=torch.randn(4,3,64,64,device='cuda')
            source_net=plain_model(spec,source,config).cuda().eval()
            source_score=sde.ScoreModel(source_net,pred='noise_pred',sde=sde.VPSDE())
            # Native sampler with a short, explicitly qualitative preview grid.
            samples=sde.euler_maruyama(sde.ReverseSDE(source_score),initial,sample_steps=100)
            assert torch.isfinite(samples).all()
            a=((samples.float()+1)*127.5).round().clamp(0,255).byte().permute(0,2,3,1).cpu().numpy()
            np.savez_compressed(output/'source_preview_4.npz',images=a,seed=20261107,sampler='native Euler-Maruyama 100 steps; FP32 qualitative smoke')
            del source_net,source_score,samples,initial
    if family=='lfm' and spec.get('source_preview',False):
        seed_all(20261107)
        source_net=plain_model(spec,source,config).cuda().eval().requires_grad_(False)
        with torch.no_grad():
            initial=torch.randn(4,4,32,32,device='cuda');latent=initial.clone()
            times=torch.linspace(1.,0.,51,device='cuda')
            for start,end in zip(times[:-1],times[1:]):
                dt=end-start
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    v0=source_net(start.expand(4),latent,y=None).float()
                    v1=source_net(end.expand(4),latent+dt*v0,y=None).float()
                latent=latent+dt*(v0+v1)*.5
            assert torch.isfinite(latent).all()
            decoded=torch.cat([vae.decode(chunk/.18215).sample for chunk in latent.split(spec.get('evaluation_batch',2))])
            assert torch.isfinite(decoded).all()
            a=((decoded.float()+1)*127.5).round().clamp(0,255).byte().permute(0,2,3,1).cpu().numpy()
            np.savez_compressed(output/'source_preview_4.npz',images=a,seed=20261107,
                sampler='Heun50 backward1to0; BF16 velocity / FP32 latent and VAE decoder; qualitative only')
        del source_net,initial,latent,decoded,v0,v1
    allocated=torch.cuda.max_memory_allocated()/2**30;reserved=torch.cuda.max_memory_reserved()/2**30
    total=torch.cuda.get_device_properties(0).total_memory/2**30
    assert allocated<.9*total, f'Insufficient memory headroom: {allocated}/{total}'
    result=dict(status='PASS',identity=identity,counts=counts,microbatch=micro,effective_batch=32,
                gpu=torch.cuda.get_device_name(),total_gib=total,peak_allocated_gib=allocated,peak_reserved_gib=reserved,
                update_seconds=timings,updates=losses,resume_parameter_max_error=max_error,resume_ema_max_error=ema_error,
                source_prediction_unchanged_at_initialization=True,frozen_tensors_exact=True,
                hybrid_000_111_exact=True,validation_loss=validation['mean'],job_id=os.getenv('SLURM_JOB_ID'))
    write(output/'PASS.json',result);checkpoint.unlink()
    print(json.dumps(result),flush=True)

def train(manifest,path,index):
    row=manifest['runs'][index];spec=manifest['sources'][row['family']]
    horizon=row.get('reference_epochs',manifest.get('reference_epochs_by_target',{}).get(row['target'],manifest['reference_epochs']))
    manifest=dict(manifest,reference_epochs=horizon)
    smoke_root=Path(manifest.get('smoke_report',manifest['report']))
    approved=json.loads((smoke_root/'smoke'/row['family']/'PASS.json').read_text())
    expected_smoke_manifest=manifest.get('smoke_manifest_sha256',digest(path))
    assert approved['status']=='PASS' and approved['identity']['manifest_sha256']==expected_smoke_manifest
    micro=approved['microbatch'];run=Path(row['output']);run.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(run/'.train.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    metadata=json.loads(Path(row['data']).read_text());identity=identity_for(path,row,row['data'])
    dataset=Images(metadata,'train');valid=Images(metadata,'valid')
    reference_steps=min(spec['source_train_count'],len(dataset))//32
    total_steps=reference_steps*manifest['reference_epochs']
    if (run/'COMPLETE.json').exists():
        assert json.loads((run/'COMPLETE.json').read_text())['identity']==identity
        print('Already complete',flush=True);return
    seed_all(row['seed']);net,source,config=load_source(spec)
    net=net.cuda();counts=attach_adapters(net,row['seed'],rank=spec.get('mlp_rank'),expected_counts=spec.get('group_counts'));ema=EffectiveEMA(source)
    optimizer=optimizer_for(net,spec);vae=vae_for(spec);step=0
    resume=None
    continuation=row.get('continuation')
    candidates=[(run/name,False) for name in ('online_latest.pt','online_previous.pt')]
    if continuation:
        assert run.resolve()!=Path(continuation['parent_output']).resolve(), 'Preserve the parent run'
        candidates.append((Path(continuation['checkpoint']),True))
    for p,is_parent in candidates:
        if p.exists():
            if is_parent:
                assert digest(p)==continuation['checkpoint_sha256'], 'Parent online checkpoint changed'
                parent=json.loads(Path(continuation['completion']).read_text())
                assert digest(continuation['completion'])==continuation['completion_sha256']
                assert parent['identity']==identity and parent['step']==continuation['start_step']
            try: payload=torch.load(p,map_location='cpu',weights_only=False)
            except (OSError,RuntimeError,EOFError) as error:
                print(f'Cannot read {p}: {error}',flush=True);continue
            step=restore(payload,net,optimizer,ema,identity)
            if continuation:
                assert step>=continuation['start_step']
                if is_parent:
                    assert step==continuation['start_step'], 'Continuation must start at the completed endpoint'
                    assert all(g['lr']==spec['lr'] for g in optimizer.param_groups)
                    assert all(tuple(g['betas'])==tuple(optimizer.defaults['betas']) for g in optimizer.param_groups)
                    restored_rng=rng_state()
                    assert torch.equal(restored_rng['cpu'],payload['rng']['cpu'])
                    assert torch.equal(restored_rng['cuda'],payload['rng']['cuda'])
                assert_frozen(net,source,ema)
                audit=dict(parent_checkpoint=continuation['checkpoint'],
                    parent_checkpoint_sha256=continuation['checkpoint_sha256'],
                    resumed_checkpoint=str(p),resumed_step=step,ema_updates=ema.updates,
                    identity=identity,target_epochs=manifest['reference_epochs'],
                    restored=['online dense weights and LoRA factors','AdamW moments and groups',
                        'effective-weight EMA','Python/NumPy/CPU/CUDA RNG','next data batch from step'],
                    changed=['training horizon','output directory','execution provenance'],
                    job_id=os.getenv('SLURM_JOB_ID'))
                write(run/'continuation_resume_latest.json',audit)
                if is_parent and not (run/'continuation_resume_initial.json').exists():
                    write(run/'continuation_resume_initial.json',audit)
                print('CONTINUATION RESUME',json.dumps(audit),flush=True)
            del payload;resume=str(p);break
    if continuation:
        assert resume is not None, 'Refusing to start a requested continuation from initialization'
        assert continuation['start_step']<=step<=total_steps
    write(run/'recipe.json',dict(identity=identity,execution_manifest_sha256=digest(path),source=spec,target=metadata,counts=counts,
          seed=row['seed'],reference_steps=reference_steps,reference_epochs=manifest['reference_epochs'],
          total_steps=total_steps,batch=32,microbatch=micro,ema_decay=ema.decay,
          precision=spec.get('precision','BF16 transformer body; FP32 parameters/optimizer/loss; pixel output head FP32; VAE FP32'),
          frozen=spec.get('frozen','All biases, AdaLN, embeddings, output head, VAE; MLP base weights'),
          resume=resume,resume_step=step,job_id=os.getenv('SLURM_JOB_ID')))
    print(json.dumps(dict(starting=row['id'],step=step,total_steps=total_steps,reference_steps=reference_steps,counts=counts)),flush=True)
    started=time.monotonic();last_log=started;loss_sum=0.;log_count=0
    signal.signal(signal.SIGUSR1,request_stop);signal.signal(signal.SIGTERM,request_stop)
    for images in loader(dataset,row['seed'],step,total_steps):
        r=train_step(net,optimizer,ema,images,spec,vae,micro,learning_rate(spec,step,reference_steps))
        step+=1;loss_sum+=r['loss'];log_count+=1
        if step%100==0 or step<=5:
            now=time.monotonic()
            progress=dict(step=step,reference_epoch=step/reference_steps,total_steps=total_steps,
                loss=loss_sum/log_count,gradient_norm=r['gradient_norm'],lr=r['lr'],
                seconds_per_update=(now-last_log)/log_count,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                job_id=os.getenv('SLURM_JOB_ID'),updated_unix=time.time())
            write(run/'progress.json',progress)
            with (run/'training.jsonl').open('a') as f:f.write(json.dumps(progress)+'\n')
            print(json.dumps(progress),flush=True);last_log=now;loss_sum=0.;log_count=0
        epoch_boundary=step%reference_steps==0
        stopping=STOP or time.monotonic()-started>11.5*3600
        if epoch_boundary or step%500==0 or stopping or step==total_steps:
            rolling_checkpoint(run,net,optimizer,ema,step,identity)
        if epoch_boundary and (step//reference_steps)%25==0:
            epoch=step//reference_steps;assert_frozen(net,source,ema)
            endpoint=dict(format='matched-scale-effective-ema-v1',identity=identity,source=spec,
                          model_cfg=config,step=step,reference_epoch=epoch,ema=ema.shadow)
            save(run/f'ema_epoch{epoch:04d}.pt',endpoint)
            diagnostics=update_diagnostics(source,ema.shadow)
            write(run/f'updates_epoch{epoch:04d}.json',diagnostics)
            if not stopping:
                result=evaluate(ema.shadow,spec,config,valid,vae,count=512)
                write(run/f'validation_epoch{epoch:04d}.json',result)
        if stopping and step<total_steps:
            print(f'Checkpointed update {step}; requesting requeue',flush=True);sys.exit(75)
    assert step==total_steps
    # Common endpoint, paired test inputs/noise, all eight component backgrounds.
    test=Images(metadata,'test');endpoint_results={}
    for code in ('000','001','010','011','100','101','110','111'):
        result_path=run/f'endpoint_loss_{code}.json'
        if not result_path.exists():
            state=hybrid_state(source,ema.shadow,code)
            result=evaluate(state,spec,config,test,vae,count=512);del state
            write(result_path,dict(code=code,identity=identity,**result))
        endpoint_results[code]=json.loads(result_path.read_text())['mean']
        if STOP:
            print('Endpoint evaluation interrupted; checkpoint retained',flush=True);sys.exit(75)
    write(run/'COMPLETE.json',dict(identity=identity,step=step,reference_epochs=manifest['reference_epochs'],
          checkpoint=str(run/f"ema_epoch{manifest['reference_epochs']:04d}.pt"),endpoint_losses=endpoint_results,
          completed_unix=time.time(),job_id=os.getenv('SLURM_JOB_ID')))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','train']);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--family',choices=['pixel','lfm','uvit']);p.add_argument('--run-index',type=int,default=0);a=p.parse_args()
    manifest=json.loads(a.manifest.read_text())
    for relative,expected in manifest['code_sha256'].items():
        assert digest(Path(manifest['code'])/relative)==expected,f'Frozen code changed: {relative}'
    assert torch.cuda.is_available(),'CUDA required; refusing CPU fallback'
    assert torch.cuda.is_bf16_supported(),'BF16 GPU required'
    gpu_name=torch.cuda.get_device_name()
    assert any(name in gpu_name for name in manifest.get('allowed_gpu_names',['A5000'])),gpu_name
    if a.stage=='train' and 'allowed_run_indices' in manifest:
        assert a.run_index in manifest['allowed_run_indices'],'Run not authorized by this execution manifest'
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    if manifest.get('deterministic_algorithms',False) or any(s['family']=='uvit' for s in manifest['sources'].values()):
        torch.use_deterministic_algorithms(True)
    if a.stage=='smoke': smoke(manifest,a.manifest,a.family)
    else: train(manifest,a.manifest,a.run_index)

if __name__=='__main__':main()
