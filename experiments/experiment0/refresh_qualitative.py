"""Add sigma=0.75 maps for the two previously selected qualitative image/target pairs."""
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
    torch.set_num_threads(1)
    assert torch.cuda.is_available(),'CUDA required'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path('reports/experiment0/figures/consolidated')
    for label in ['celeba64_20m','stl10_64_20m_epoch800']:
        dest=out/(label+'_samples_sigma075.npz')
        if dest.exists():
            print('Exists',dest,flush=True);continue
        with np.load(out/(label+'_samples.npz')) as z:old={k:z[k].copy() for k in z.files}
        meta=json.loads(str(old['metadata']));spec=meta['model'];target=meta['target'];idx=meta['image_index']
        assert digest(Path(spec['checkpoint']))==meta['checkpoint_sha256']
        dataset=build_folder_dataset(spec['data_root'],64,3);clean=get_images(dataset,[idx]).to('cuda')
        model,ckpt,kind,_,ema,_=load_model(Path(spec['checkpoint']),torch.device('cuda'),True)
        assert kind=='dit' and ema and ckpt['step']==spec['expected_step']
        model.eval().requires_grad_(False)
        eps=gaussian(clean.shape,meta['noise_seed'],'cuda');sigma=.75
        probes=torch.stack([rademacher((48,),130363+idx*100003+target*101+k,'cuda') for k in range(meta['probes'])])[None]
        y=clean+sigma*eps;x=y/(1+sigma)
        energy=influence(clean_predictor(model,sigma),y,[target],probes,4)[0].mean(0)
        with torch.no_grad():
            _,mats=dit_attention(model,x,sigma,vp=True)
            attn=torch.stack([a[0].mean(0)[target] for a in mats]).mean(0).cpu().numpy()
        meta.update(map_sigmas=[.75,49.],reference_sigma=0.,probe_seed=130363,
                    low_noise_selection='sigma=0.75 requested for the lower-noise comparison; attention may remain more diffuse than sensitivity.')
        np.savez_compressed(dest,noisy_inputs=np.stack([x[0].cpu().numpy(),old['noisy_inputs'][1]]),
                            reference_input=x[0].cpu().numpy(),clean_input=clean[0].cpu().numpy(),sensitivity=np.stack([energy/energy.sum(),old['sensitivity'][1]]),
                            attention=np.stack([attn/attn.sum(),old['attention'][1]]),sigmas=[.75,49.],metadata=json.dumps(meta))
        print('Saved',dest,flush=True)
        del model,ckpt;torch.cuda.empty_cache()


if __name__=='__main__':main()
