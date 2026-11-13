import json,os,signal
from pathlib import Path
import numpy as np
import torch
from experiments.component_transfer_scale.model import digest,write,rng_state,restore_rng,seed_all
STOP=False
def stop(*_):
 global STOP
 STOP=True
def setup(manifest,gpu=True):
 m=json.loads(Path(manifest).read_text())
 for rel,h in m['code_sha256'].items():assert digest(Path(m['code'])/rel)==h,rel
 assert digest(m['data'])==m['data_sha256']
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 if gpu:
  assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
  assert torch.cuda.is_available(), "CUDA GPU required"
  torch.ones(1,device='cuda').sum().item()
 torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
 return m
def durable(path,writer):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.tmp-{os.getpid()}')
 with tmp.open('wb') as f:writer(f);f.flush();os.fsync(f.fileno())
 tmp.replace(p)
 fd=os.open(p.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def save(path,value):durable(path,lambda f:torch.save(value,f))
def npy(path,value):durable(path,lambda f:np.save(f,value))
def npz(path,**value):durable(path,lambda f:np.savez(f,**value))
def cpu(value):
 if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
 if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
 if isinstance(value,list):return [cpu(v) for v in value]
 if isinstance(value,tuple):return tuple(cpu(v) for v in value)
 return value
def identity(m,arm):
 return dict(arm=arm,source=m['source']['checkpoint_sha256'],data=m['data_sha256'],recipe=m['recipe'],code_sha256=m['code_sha256'],cache_identity=m['cache_identity'])
def check_pass(m):
 p=Path(m['report'])/'PREFLIGHT_PASS.json';r=json.loads(p.read_text());assert r['manifest_sha256']==digest(m['_manifest']) and r['passed']
