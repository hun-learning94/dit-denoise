"""Paired, resumable denoising evaluation over the 18-step EDM schedule."""
from __future__ import annotations
import argparse, json, math, os
from pathlib import Path
import numpy as np
import torch
from eval_celeba32_sampler_rmse import edm_sigmas
from eval_model_comparison import load_model, make_loader, model_call

def atomic_npz(path, **values):
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}.npz")
    np.savez(tmp, **values); os.replace(tmp, path)

def interval(values, seed, draws=2000):
    rng=np.random.default_rng(seed); means=np.empty(draws)
    for start in range(0, draws, 100):
        n=min(100, draws-start); idx=rng.integers(0,len(values),size=(n,len(values)))
        means[start:start+n]=values[idx].mean(1)
    return np.quantile(means,[.025,.975]).tolist()

@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True); p.add_argument('--label',required=True)
    p.add_argument('--data_root',default='datasets/celeba32/test'); p.add_argument('--outdir',default='evals/denoising_full_test')
    p.add_argument('--batch',type=int,default=32); p.add_argument('--num_workers',type=int,default=4)
    p.add_argument('--precision',choices=['fp16','bf16','fp32'],default='fp16'); p.add_argument('--noise_seed',type=int,default=12345)
    p.add_argument('--num_steps',type=int,default=18); p.add_argument('--auc_steps',type=int,default=32); p.add_argument('--sigma_min',type=float,default=.002)
    p.add_argument('--sigma_max',type=float,default=80.); p.add_argument('--rho',type=float,default=7.)
    p.add_argument('--time_min',type=float,default=.05); p.add_argument('--time_max',type=float,default=4.)
    p.add_argument('--direct_sampler_steps',type=int,default=50)
    p.add_argument('--resolution',type=int,default=32); p.add_argument('--channels',type=int,default=3)
    p.add_argument('--folder_tensor_cache',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--eval_folder_tensor_cache_path',default=None); args=p.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); root=Path(args.outdir)/args.label; root.mkdir(parents=True,exist_ok=True)
    model,ckpt,_,_,used_ema,transform=load_model(Path(args.checkpoint),device,True)
    la=argparse.Namespace(**vars(args)); la.eval_data_root=args.data_root; la.eval_batch=args.batch; loader=make_loader(la,0)
    prediction_type=ckpt.get('prediction_type',ckpt.get('args',{}).get('prediction_type','edm_preconditioned'))
    noise_process=ckpt.get('noise_process',ckpt.get('args',{}).get('noise_process','ve'))
    if prediction_type=='x' and noise_process=='vp':
        # Evaluate at VE-equivalent noise ratios sigma=(1-t)/t so the same
        # scientific grid remains comparable across processes.
        sampler_sigmas=np.asarray([.05,.1,.25,.5,1.,2.,4.],dtype=np.float64)
        auc_sigmas=np.geomspace(args.time_min,args.time_max,args.auc_steps)
        eval_sigmas=sampler_sigmas.tolist()
    elif prediction_type=='x':
        sampler_sigmas=np.geomspace(args.time_max,args.time_min,args.direct_sampler_steps)
        auc_sigmas=np.geomspace(args.time_min,args.time_max,args.auc_steps)
        eval_sigmas=[.05,.1,.25,.5,1.,2.,4.]
    else:
        sampler_sigmas=edm_sigmas(args.num_steps,args.sigma_min,args.sigma_max,args.rho)
        auc_sigmas=np.geomspace(args.sigma_min,args.sigma_max,args.auc_steps)
        eval_sigmas=list(sampler_sigmas)
    sampler_indices=[]
    for sigma in sampler_sigmas:
        matches=[i for i,value in enumerate(eval_sigmas) if np.isclose(value,sigma,rtol=1e-12,atol=0)]
        if matches: sampler_indices.append(matches[0])
        else: sampler_indices.append(len(eval_sigmas)); eval_sigmas.append(float(sigma))
    auc_indices=[]
    for sigma in auc_sigmas:
        matches=[i for i,value in enumerate(eval_sigmas) if np.isclose(value,sigma,rtol=1e-12,atol=0)]
        if matches: auc_indices.append(matches[0])
        else: auc_indices.append(len(eval_sigmas)); eval_sigmas.append(float(sigma))
    for i,sigma in enumerate(eval_sigmas):
        path=root/f'errors_step{i:02d}.npz'
        if path.is_file(): print(f'[resume] {path}',flush=True); continue
        gen=torch.Generator(device=device).manual_seed(args.noise_seed+1_000_003*i); raw=[]; clipped=[]
        for bi,batch in enumerate(loader):
            clean=(batch[0] if isinstance(batch,(tuple,list)) else batch).to(device).float(); cm=transform.forward(clean)
            noise=torch.randn(cm.shape,generator=gen,device=device)
            if prediction_type=='x' and noise_process=='vp':
                time_value=1.0/(1.0+float(sigma)); noisy=time_value*cm+(1.0-time_value)*noise
                pred=transform.inverse(model_call(model,noisy,time_value,args.precision))
            else:
                pred=transform.inverse(model_call(model,cm+float(sigma)*noise,float(sigma),args.precision))
            dims=tuple(range(1,pred.ndim))
            raw.append((pred-clean).square().mean(dim=dims).cpu().numpy()); clipped.append((pred.clamp(-1,1)-clean).square().mean(dim=dims).cpu().numpy())
            if (bi+1)%100==0: print(f'[sigma {i:02d}] images={(bi+1)*args.batch}',flush=True)
        atomic_npz(path,sigma=float(sigma),raw_mse=np.concatenate(raw),clipped_mse=np.concatenate(clipped)); print(f'[sigma {i:02d}] saved',flush=True)
    data=[np.load(root/f'errors_step{i:02d}.npz') for i in range(len(eval_sigmas))]
    raw=np.stack([x['raw_mse'] for x in data],1).astype(np.float64); clipped=np.stack([x['clipped_mse'] for x in data],1).astype(np.float64)
    auc_lo=args.time_min if prediction_type=='x' else args.sigma_min; auc_hi=args.time_max if prediction_type=='x' else args.sigma_max
    auc_raw=np.trapezoid(raw[:,auc_indices],np.log(auc_sigmas),axis=1)/(np.log(auc_hi)-np.log(auc_lo))
    auc_clip=np.trapezoid(clipped[:,auc_indices],np.log(auc_sigmas),axis=1)/(np.log(auc_hi)-np.log(auc_lo))
    nxt=np.concatenate([sampler_sigmas[1:],[0.]]); weights=np.square((sampler_sigmas-nxt)/sampler_sigmas); weights/=weights.sum()
    sw_raw=(raw[:,sampler_indices]*weights[None]).sum(1); sw_clip=(clipped[:,sampler_indices]*weights[None]).sum(1)
    result={'label':args.label,'checkpoint':args.checkpoint,'used_ema':used_ema,'num_images':len(raw),'prediction_type':prediction_type,'noise_process':noise_process,
      'sampler_sigmas':sampler_sigmas.tolist(),'auc_sigmas':auc_sigmas.tolist(),'evaluated_sigmas':eval_sigmas,
      'sampler_raw_mse':raw[:,sampler_indices].mean(0).tolist(),'sampler_clipped_mse':clipped[:,sampler_indices].mean(0).tolist(),
      'auc_raw_mse':raw[:,auc_indices].mean(0).tolist(),'auc_clipped_mse':clipped[:,auc_indices].mean(0).tolist(),
      'log_sigma_auc_mse_raw':float(auc_raw.mean()),'log_sigma_auc_mse_raw_ci95':interval(auc_raw,args.noise_seed+91),
      'log_sigma_auc_mse_clipped':float(auc_clip.mean()),'log_sigma_auc_mse_clipped_ci95':interval(auc_clip,args.noise_seed+92),
      'sampler_weighted_rmse_raw':math.sqrt(float(sw_raw.mean())),'sampler_weighted_rmse_raw_ci95':[math.sqrt(x) for x in interval(sw_raw,args.noise_seed+93)],
      'sampler_weighted_rmse_clipped':math.sqrt(float(sw_clip.mean())),'sampler_weighted_rmse_clipped_ci95':[math.sqrt(x) for x in interval(sw_clip,args.noise_seed+94)]}
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
