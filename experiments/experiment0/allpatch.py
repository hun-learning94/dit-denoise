"""All-target population maps, sharded by model/noise/images; no finite refresh."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from diagnose_dpd import build_folder_dataset, get_images
from diagnose_attention_dit_dpd2 import dit_attention
from eval_model_comparison import load_model
from edm import VPXPred
from experiments.experiment0.core import gaussian, rademacher, influence, coverage, distances
from experiments.experiment0.run import digest, clean_predictor

MANIFEST=Path('reports/experiment0/allpatch_manifest.json')
OUT=Path('evals/experiment0/allpatch1024')
SOURCES=['experiments/experiment0/allpatch.py','experiments/experiment0/core.py',
         'experiments/experiment0/run.py','models.py','edm.py','data.py',
         'eval_model_comparison.py','diagnose_attention_dit_dpd2.py','diagnose_dpd.py']

def shard_path(task,smoke=False):
    return OUT/('smoke' if smoke else 'shards')/task['model']/f"sigma{task['sigma_index']:02d}_images{task['start']:04d}.npz"

def run(m,task,smoke):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.ones(1,device='cuda').sum().item()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=m['config'];spec=next(s for s in m['models'] if s['label']==task['model'])
    ds=build_folder_dataset(spec['data_root'],64,3)
    if len(ds)<spec['num_images']:raise ValueError('Insufficient test images')
    ids=np.random.default_rng(cfg['image_seed']).permutation(len(ds))[:spec['num_images']][task['start']:task['stop']]
    if smoke:ids=ids[:1]
    identity=dict(config=cfg,model=spec,task=task,smoke=smoke,indices=ids.tolist(),
                  checkpoint_sha256=digest(Path(spec['checkpoint'])),
                  source_sha256={s:digest(Path(s)) for s in SOURCES},
                  image_paths=[str(ds.paths[int(i)]) for i in ids],
                  image_sha256=[digest(Path(ds.paths[int(i)])) for i in ids])
    signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    out=shard_path(task,smoke);out.parent.mkdir(parents=True,exist_ok=True)
    with out.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if out.exists():
            with np.load(out) as z:
                assert str(z['signature'])==signature
                for k in ['sensitivity_sum','attention_sum']:assert np.isfinite(z[k]).all() and (z[k]>=0).all()
            print('Already complete',out,flush=True);return
        started=time.perf_counter()
        model,ckpt,kind,_,ema,transform=load_model(Path(spec['checkpoint']),torch.device('cuda'),True)
        assert kind=='dit' and isinstance(model,VPXPred) and ema
        assert ckpt['step']==spec['expected_step']
        assert not transform.has_shuffle and not transform.has_channel_mix
        assert model.model.cfg.patch_size==4 and model.model.cfg.img_resolution==64
        del ckpt
        model.eval().requires_grad_(False)
        totals={k:np.zeros((256,256),np.float64) for k in ['sensitivity_sum','attention_sum']}
        sigma=cfg['sigmas'][task['sigma_index']]
        for i in ids:
            i=int(i);clean=get_images(ds,[i]).to('cuda')
            y=clean+sigma*gaussian(clean.shape,cfg['noise_seed']+i,'cuda')
            probes=torch.stack([torch.stack([rademacher((48,),cfg['probe_seed']+i*100003+t*101+k,'cuda') for k in range(cfg['probes'])]) for t in cfg['targets']])
            e=influence(clean_predictor(model,sigma),y,cfg['targets'],probes,4).mean(1)
            with torch.no_grad():
                states,mats=dit_attention(model,y/(1+sigma),sigma,vp=True)
                attn=torch.stack([a[0].mean(0) for a in mats]).mean(0).cpu().numpy()
                del states,mats
            for v in [e,attn]:assert v.shape==(256,256) and np.isfinite(v).all() and (v>=0).all()
            np.testing.assert_allclose(attn.sum(-1),1,atol=2e-6)
            totals['sensitivity_sum']+=e;totals['attention_sum']+=attn
            print(f'image={i} sigma={sigma} elapsed={time.perf_counter()-started:.1f}s',flush=True)
        meta={**identity,'seconds':time.perf_counter()-started,'gpu':torch.cuda.get_device_name(),
              'precision':'fp32','torch_version':torch.__version__,'attention_aggregation':'equal head and layer average before population coverage'}
        tmp=out.with_name(out.name+f'.tmp-{os.getpid()}.npz')
        np.savez_compressed(tmp,**totals,indices=ids,metadata=json.dumps(meta),signature=signature)
        os.replace(tmp,out);print('Saved',out,flush=True)

def summarize(m,dest=None):
    cfg=m['config'];d=distances(16,cfg['targets'])
    dest=Path(dest or 'reports/experiment0/figures/allpatch');dest.mkdir(parents=True,exist_ok=True)
    for spec in m['models']:
        results={k:[] for k in ['sensitivity_radius','sensitivity_count','attention_radius','attention_count']}
        ref=None;refids=None
        for si in range(len(cfg['sigmas'])):
            sums={k:np.zeros((256,256),np.float64) for k in ['sensitivity_sum','attention_sum']};ids=[]
            tasks=[t for t in m['tasks'] if t['model']==spec['label'] and t['sigma_index']==si]
            for t in tasks:
                with np.load(shard_path(t)) as z:
                    meta=json.loads(str(z['metadata']))
                    assert meta['config']==cfg and meta['model']==spec and meta['task']==t and not meta['smoke']
                    if ref is None:ref=meta
                    assert all(meta[k]==ref[k] for k in ['source_sha256','checkpoint_sha256'])
                    for k in sums:sums[k]+=z[k]
                    ids.extend(z['indices'].tolist())
            assert len(ids)==len(set(ids))==spec['num_images']
            if refids is None:refids=ids
            assert ids==refids
            for method in ['sensitivity','attention']:
                c=coverage(sums[method+'_sum']/len(ids),d,cfg['qs'],cfg['floor'])
                assert c['valid'].all()
                for metric in ['radius','count']:results[method+'_'+metric].append(c[metric])
        arrays={k:np.asarray(v) for k,v in results.items()}
        summary=dict(model=spec,config=cfg,checkpoint_sha256=ref['checkpoint_sha256'],source_sha256=ref['source_sha256'],
                     aggregation='Population maps first; coverage per target; equal average over all 256 target patches.',
                     **{k:v.mean(1).tolist() for k,v in arrays.items()})
        (dest/(spec['label']+'.json')).write_text(json.dumps(summary,indent=2)+'\n')
        np.savez_compressed(dest/(spec['label']+'.npz'),**arrays,sigmas=cfg['sigmas'],qs=cfg['qs'],metadata=json.dumps(summary))
        print('Summarized',spec['label'],flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',type=int);p.add_argument('--smoke',action='store_true');p.add_argument('--summarize',action='store_true')
    p.add_argument('--manifest',type=Path,default=MANIFEST);a=p.parse_args()
    m=json.loads(a.manifest.read_text())
    if a.summarize:summarize(m)
    else:
        if a.task is None:p.error('--task required')
        run(m,m['tasks'][a.task],a.smoke)
if __name__=='__main__':main()
