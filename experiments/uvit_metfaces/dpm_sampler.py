"""Official pinned U-ViT DPM-Solver-fast, preserving the native VP/time adapter."""
from pathlib import Path
import hashlib,sys
import torch
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H/'vendor'))
from dpm_solver_pytorch import NoiseScheduleVP,model_wrapper,DPM_Solver
SOLVER_SHA='9e57a29b6469e60be2bdf01f9086b1e5969e8f15e06ea856cfe3e80df8060cff'
assert hashlib.sha256((H/'vendor/dpm_solver_pytorch.py').read_bytes()).hexdigest()==SOLVER_SHA

def identity(steps):
 return dict(algorithm='official U-ViT DPM-Solver-fast',nfe=int(steps),eps=1e-4,
  schedule='linear VP beta_min=0.1 beta_max=20',time_input_type='0',network_time='999*t',
  solver_sha256=SOLVER_SHA,precision='BF16 predictor; FP32 solver and features',
  initial_rng='CPU generator: seed+1000003*shard',quantization='clamp[-1,1], round uint8')

@torch.no_grad()
def sample(m,net,seed,shard,count,steps=None,return_calls=False):
 from experiments.uvit_metfaces import evaluate as old
 from experiments.component_transfer_scale.model import seed_all,uvit_module
 if steps is None:steps=m['sampling_protocol']['nfe']
 g=torch.Generator().manual_seed(seed+1000003*shard)
 x=torch.randn(count,3,64,64,generator=g).cuda()
 seed_all(seed+2000003*shard+31)
 sde=uvit_module(m['source'],'sde')
 score=sde.ScoreModel(old.Predictor(net),pred='noise_pred',sde=sde.VPSDE())
 calls=0
 def noise(x,t):
  nonlocal calls
  calls+=1
  return score.noise_pred(x,t)
 fn=model_wrapper(noise,NoiseScheduleVP(schedule='linear'),time_input_type='0',model_kwargs={})
 out=DPM_Solver(fn,NoiseScheduleVP(schedule='linear')).sample(x,steps=steps,eps=1e-4,
  adaptive_step_size=False,fast_version=True)
 assert calls==steps and torch.isfinite(out).all(),calls
 pixels=((out.clamp(-1,1)+1)*127.5).round().byte()
 return (pixels,calls) if return_calls else pixels
