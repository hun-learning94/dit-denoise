"""One model/noise/image shard, with matched input VJPs, attention, and refresh."""
import argparse
import fcntl
import hashlib
import json
import os
import resource
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.experiment0.core import gaussian, rademacher, influence, refresh
from diagnose_dpd import build_folder_dataset, get_images
from diagnose_attention_dit_dpd2 import dit_attention
from eval_model_comparison import load_model
from edm import VPXPred


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def clean_predictor(model, sigma):
    # VPXPred is the actual trained clean-output adapter, with all branches intact.
    t=1/(1+sigma)
    return lambda y: model(t*y, torch.full((len(y),),t,device=y.device))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True); p.add_argument('--task',type=int,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--output-root',default='evals/experiment0')
    a=p.parse_args(); manifest=json.loads(Path(a.manifest).read_text())
    cfg=manifest['config']; task=manifest['tasks'][a.task]
    spec=next(m for m in manifest['models'] if m['label']==task['model'])
    device=torch.device(a.device)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    ckpath=ROOT/spec['checkpoint']; ckhash=digest(ckpath)
    ds=build_folder_dataset(str(ROOT/spec['data_root']),64,3)
    if len(ds)<cfg['num_images']: raise ValueError('Insufficient held-out images')
    selected=np.random.default_rng(cfg['image_seed']).permutation(len(ds))[:cfg['num_images']]
    ids=selected[task['start']:task['stop']]
    # Dataset identities are recorded below, in addition to indices.
    paths=[str(ds.paths[int(i)]) for i in ids]
    source_hashes={name:digest(ROOT/name) for name in [
        'experiments/experiment0/run.py','experiments/experiment0/core.py',
        'models.py','edm.py','data.py','eval_model_comparison.py',
        'diagnose_attention_dit_dpd2.py']}
    identity=dict(config=cfg,task=task,model=spec,checkpoint_sha256=ckhash,
                  source_sha256=source_hashes,indices=ids.tolist(),image_paths=paths,
                  image_sha256=[digest(Path(path)) for path in paths])
    signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    out=ROOT/a.output_root/cfg['stage']/spec['label']/f'sigma{task["sigma_index"]:02d}_images{task["start"]:04d}.npz'
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if out.exists():
            with np.load(out,allow_pickle=False) as saved:
                if str(saved['signature']) != signature:
                    raise RuntimeError(f'Resume signature mismatch: {out}')
                for key in ['influence','attention','refresh','refresh_raw']:
                    if not np.isfinite(saved[key]).all():
                        raise RuntimeError(f'Invalid saved {key}')
            print('Already complete:',out); return
        started=time.perf_counter()
        model,ckpt,kind,_,used_ema,transform=load_model(ckpath,device,True)
        if kind!='dit' or not isinstance(model,VPXPred) or not used_ema:
            raise ValueError('Experiment 0 requires base DiT direct-clean VP with EMA')
        if transform.has_shuffle or transform.has_channel_mix:
            raise ValueError('Only untransformed scratch models are in this panel')
        if int(ckpt.get('step',-1)) != spec['expected_step']:
            raise ValueError('Checkpoint step does not match terminal manifest')
        bb=model.model
        if bb.cfg.patch_size!=cfg['patch'] or bb.cfg.img_resolution!=64:
            raise ValueError('Unexpected patch geometry')
        model.eval().requires_grad_(False)
        nparam=sum(p.numel() for p in model.parameters())
        del ckpt
        sigma=cfg['sigmas'][task['sigma_index']]; predict=clean_predictor(model,sigma)
        energies=[]; attentions=[]; raw_refresh=[]; norm_refresh=[]; previews=[]
        dp=3*cfg['patch']**2
        phase_seconds=dict(influence=0.,attention=0.,refresh=0.)
        def stamp():
            if device.type=='cuda': torch.cuda.synchronize(device)
            return time.perf_counter()
        for i in ids:
            clean=get_images(ds,[int(i)]).to(device)
            eps=gaussian(clean.shape,cfg['noise_seed']+int(i),device)
            probes=torch.stack([torch.stack([rademacher((dp,),cfg['probe_seed']+int(i)*100003+t*101+k,device)
                                            for k in range(cfg['probes'])]) for t in cfg['targets']])
            y=clean+sigma*eps
            phase_start=stamp()
            energies.append(influence(predict,y,cfg['targets'],probes,cfg['patch']))
            phase_seconds['influence']+=stamp()-phase_start
            with torch.no_grad():
                phase_start=stamp()
                states,mats=dit_attention(model,y/(1+sigma),sigma,vp=True)
                attentions.append(np.stack([m[0].mean(0)[cfg['targets']].cpu().numpy() for m in mats]))
                del states,mats
                phase_seconds['attention']+=stamp()-phase_start
                phase_start=stamp()
                xis=[gaussian(clean.shape,cfg['refresh_seed']+int(i)*100003+k,device) for k in range(cfg['refresh_draws'])]
                raw,norm=refresh(predict,clean,eps,sigma,cfg['targets'],cfg['radii'],cfg['cs'],xis,cfg['patch'])
                raw_refresh.append(raw); norm_refresh.append(norm)
                previews.append(clean[0].cpu().numpy())
                phase_seconds['refresh']+=stamp()-phase_start
            print(f'image {int(i)} sigma={sigma:g} elapsed={time.perf_counter()-started:.1f}s',flush=True)
        meta={**identity,'parameter_count':nparam,'model_config':vars(bb.cfg),
              'used_ema':used_ema,'device':str(device),'torch_version':torch.__version__,
              'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
              'wall_seconds':time.perf_counter()-started,
              'phase_seconds':phase_seconds,
              'peak_host_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              'peak_cuda_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
              'precision':'fp32','coordinates':'normalized Y'}
        values=dict(influence=np.asarray(energies),attention=np.asarray(attentions),
                    refresh_raw=np.asarray(raw_refresh),refresh=np.asarray(norm_refresh),
                    images=np.asarray(previews),indices=ids,sigma=sigma,
                    signature=signature,metadata=json.dumps(meta))
        for key in ['influence','attention','refresh_raw','refresh']:
            if not np.isfinite(values[key]).all(): raise RuntimeError(f'Nonfinite {key}')
        tmp=out.with_name(out.name+f'.tmp-{os.getpid()}.npz')
        np.savez_compressed(tmp,**values); os.replace(tmp,out)
        print('Wrote',out)


if __name__=='__main__': main()
