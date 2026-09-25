"""Compute anatomically selected qualitative maps at two noise levels for the paper panel."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from diagnose_dpd import build_folder_dataset,get_images
from diagnose_attention_dit_dpd2 import dit_attention
from eval_model_comparison import load_model
from experiments.experiment0.core import gaussian,rademacher,influence
from experiments.experiment0.run import clean_predictor,digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--imagenet-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.ones(1,device='cuda').sum().item()
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    base=json.loads(Path('reports/experiment0/main_manifest.json').read_text()); cfg=base['config']
    models=[dict(m,epoch=400) for m in base['models'] if m['label'] in ['celeba64_20m','stl10_64_20m']]
    extended=json.loads(Path('reports/experiment0/stl10_800_manifest.json').read_text())
    models += [dict(m,epoch=800) for m in extended['models']]
    with np.load('evals/experiment0/main/stl10_64_20m/sigma01_images0000.npz') as z:
        stl_index=int(z['indices'][0])
    if args.imagenet_only:
        manifest=json.loads(Path('reports/experiment0/imagenet600_manifest.json').read_text())
        cfg=manifest['config'];models=[dict(m,epoch=600) for m in manifest['models']]
    out=Path('reports/experiment0/figures/consolidated');out.mkdir(parents=True,exist_ok=True)
    for spec in models:
        target=133 if spec["dataset"]=="celeba64" else (106 if spec["dataset"]=="imagenet64" else 170)
        dataset=build_folder_dataset(spec['data_root'],64,3)
        idx=742 if spec['dataset']=='celeba64' else stl_index
        if spec['dataset']=='imagenet64':
            idx=23
            assert Path(dataset.paths[idx]).name=='val_04932.png'
        clean=get_images(dataset,[idx]).to('cuda')
        model,ckpt,kind,_,ema,transform=load_model(Path(spec['checkpoint']),torch.device('cuda'),True)
        assert kind=='dit' and ema and ckpt['step']==spec['expected_step']
        model.eval().requires_grad_(False)
        eps=gaussian(clean.shape,cfg['noise_seed']+idx,'cuda')
        probes=torch.stack([rademacher((48,),cfg['probe_seed']+idx*100003+target*101+k,'cuda') for k in range(cfg['probes'])])[None]
        inputs=[]; sensitivity=[]; attention=[]
        for sigma in [.02,49.]:
            y=clean+sigma*eps; x=y/(1+sigma)
            energy=influence(clean_predictor(model,sigma),y,[target],probes,4)[0].mean(0)
            with torch.no_grad():
                _,mats=dit_attention(model,x,sigma,vp=True)
                attn=torch.stack([a[0].mean(0)[target] for a in mats]).mean(0).cpu().numpy()
            inputs.append(x[0].cpu().numpy());sensitivity.append(energy/energy.sum());attention.append(attn/attn.sum())
        meta=dict(model=spec,checkpoint_sha256=digest(Path(spec['checkpoint'])),image_index=idx,
                  image_path=str(dataset.paths[idx]),image_sha256=digest(Path(dataset.paths[idx])),
                  target=target,target_row=target//16,target_column=target%16,noise_seed=cfg['noise_seed']+idx,
                  probes=cfg['probes'],attention_aggregation='Equal head and layer average',
                  selection='CelebA: first image after the initial sunglasses example with an unobstructed eye; STL: first prespecified held-out image, patch beneath tail base approximating requested anatomical target (exact anus not identifiable at 64x64). Selected by image appearance, not heatmap response.')
        if spec['dataset']=='imagenet64':
            meta['selection']='Cockatoo selected by the user from the held-out ImageNet64 subset; eye patch chosen by appearance before computing heatmaps.'
        np.savez_compressed(out/(spec['label']+'_samples.npz'),noisy_inputs=inputs,
                            sensitivity=sensitivity,attention=attention,sigmas=[.02,49.],metadata=json.dumps(meta))
        (out/(spec['label']+'_samples.json')).write_text(json.dumps(meta,indent=2)+'\n')
        print('Saved',spec['label'],flush=True)
        del model,ckpt;torch.cuda.empty_cache()


if __name__=='__main__': main()
