"""Source-faithful backbones, matched component updates, and dense-weight EMA."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import types
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

COUNTS = dict(QK=14155776, VO=14155776, MLP=14100480)
UVIT_COUNTS = dict(QK=6815744, VO=6815744, MLP=6789120)

def uvit_module(spec, name='libs.uvit'):
    """Import the hash-pinned upstream package without modifying its source."""
    package='component_uvit_'+spec['upstream_revision'][:12]
    if package not in sys.modules:
        module=types.ModuleType(package)
        module.__path__=[spec['upstream_code']]
        sys.modules[package]=module
    import importlib
    return importlib.import_module(package+'.'+name)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def write(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.tmp-{os.getpid()}')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def save(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.tmp-{os.getpid()}')
    torch.save(data,tmp);tmp.replace(path)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())

def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['cpu']);torch.cuda.set_rng_state(s['cuda'])

def mutable(name):
    return (name.startswith(('model.blocks.','blocks.','in_blocks.','mid_block.','out_blocks.')) and name.endswith('.weight')
            and any(x in name for x in ('.attn.qkv.','.attn.proj.','.mlp.fc1.','.mlp.fc2.')))

def blocks(net):
    if hasattr(net,'in_blocks'):
        return [*net.in_blocks,net.mid_block,*net.out_blocks]
    return net.model.blocks if hasattr(net,'model') else net.blocks

def plain_model(spec, source=None, config=None):
    if spec['family']=='pixel':
        from models import DiTBackbone, DiTConfig, FusedMultiHeadSelfAttention
        from edm import VPXPred
        cfg=DiTConfig(**config)
        assert cfg.output_head_fp32 and cfg.embed_dim==768 and cfg.depth==12
        assert cfg.img_resolution==128 and cfg.patch_size==8 and cfg.mlp_ratio==4
        raw=DiTBackbone(cfg)
        for b in raw.blocks: b.attn=FusedMultiHeadSelfAttention(768,12)
        net=VPXPred(raw,128,3)
    elif spec['family']=='uvit':
        net=uvit_module(spec).UViT(**spec['model_cfg'])
    else:
        module_spec=importlib.util.spec_from_file_location('lfm_pinned_dit',spec['lfm_model_source'])
        module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
        constructor=spec.get('constructor',dict(img_resolution=32,num_classes=1000,label_dropout=.1,learn_sigma=False))
        net=module.DiT_models[spec.get('model_type','DiT-B/2')](**constructor)
    if source is not None: net.load_state_dict(source,strict=True)
    return net

def load_source(spec):
    assert digest(spec['checkpoint'])==spec['checkpoint_sha256'], 'Source checkpoint hash mismatch'
    saved=torch.load(spec['checkpoint'],map_location='cpu',weights_only=spec['family']=='uvit')
    if spec['family']=='pixel':
        assert saved['step']==63200 and saved['recipe']['noise_scale']==.5
        assert saved['recipe']['arch']=='mlp' and saved['recipe']['epochs']==400
        state=saved['ema'];config=saved['model_cfg']
    elif spec['family']=='uvit':
        state={k.removeprefix('module.'):v for k,v in saved.items()};config=spec['model_cfg']
    else:
        state={k.removeprefix('module.'):v for k,v in saved.items()};config=None
    assert all(torch.isfinite(v).all() for v in state.values())
    net=plain_model(spec,state,config)
    return net,state,config

class LoRALinear(nn.Module):
    def __init__(self, original, rank=153):
        super().__init__()
        self.weight=original.weight;self.bias=original.bias
        self.lora_A=nn.Parameter(torch.empty(rank,original.in_features,device=self.weight.device))
        self.lora_B=nn.Parameter(torch.zeros(original.out_features,rank,device=self.weight.device))
        nn.init.kaiming_uniform_(self.lora_A,a=5**.5)
    def forward(self,x):
        return F.linear(x,self.weight,self.bias)+F.linear(F.linear(x,self.lora_A),self.lora_B)
    def effective(self): return self.weight+self.lora_B@self.lora_A

def attach_adapters(net, seed, rank=None, expected_counts=None):
    is_uvit=hasattr(net,'in_blocks')
    rank=(102 if is_uvit else 153) if rank is None else rank
    expected_counts=(UVIT_COUNTS if is_uvit else COUNTS) if expected_counts is None else expected_counts
    seed_all(seed);net.requires_grad_(False)
    counts=dict(QK=0,VO=0,MLP=0)
    for b in blocks(net):
        d=b.attn.qkv.weight.shape[1]
        b.attn.qkv.weight.requires_grad_(True);b.attn.proj.weight.requires_grad_(True)
        b.mlp.fc1=LoRALinear(b.mlp.fc1,rank);b.mlp.fc2=LoRALinear(b.mlp.fc2,rank)
        counts['QK']+=2*d*d;counts['VO']+=d*d+b.attn.proj.weight.numel()
        counts['MLP']+=sum(p.numel() for l in (b.mlp.fc1,b.mlp.fc2) for p in (l.lora_A,l.lora_B))
    assert counts==expected_counts,counts
    assert sum(p.numel() for p in net.parameters() if p.requires_grad)==sum(counts.values())
    assert all(not p.requires_grad for n,p in net.named_parameters() if n.endswith('.bias'))
    return counts

@torch.no_grad()
def effective_state(net):
    s={n:v.detach() for n,v in net.state_dict().items() if not n.endswith(('.lora_A','.lora_B'))}
    for n,m in net.named_modules():
        if isinstance(m,LoRALinear): s[n+'.weight']=m.effective().detach()
    return s

class EffectiveEMA:
    def __init__(self,source,device='cuda',decay=.9999):
        self.shadow={n:v.to(device).float().clone() for n,v in source.items()}
        self.decay=decay;self.updates=0
    @torch.no_grad()
    def update(self,net):
        for n,v in effective_state(net).items():
            if mutable(n): self.shadow[n].lerp_(v.float(),1-self.decay)
        self.updates+=1

def assert_frozen(net,source,ema):
    state=net.state_dict()
    for n,v in source.items():
        if not mutable(n):
            assert torch.equal(state[n].cpu(),v),f'Frozen online tensor changed: {n}'
            assert torch.equal(ema.shadow[n].cpu(),v),f'Frozen EMA tensor changed: {n}'
        elif '.mlp.' in n: assert torch.equal(state[n].cpu(),v),f'Frozen base MLP changed: {n}'

def hybrid_state(source,adapted,code):
    assert code in ('000','001','010','011','100','101','110','111')
    assert source.keys()==adapted.keys()
    qk,vo,mlp=map(int,code);result={}
    for n,v in source.items():
        endpoint=adapted[n].to(v.device)
        if '.attn.qkv.weight' in n:
            d=v.shape[0]//3; value=v.clone()
            if qk: value[:2*d]=endpoint[:2*d]
            if vo: value[2*d:]=endpoint[2*d:]
        elif '.attn.proj.weight' in n: value=endpoint if vo else v
        elif mutable(n) and '.mlp.' in n: value=endpoint if mlp else v
        else:
            assert torch.equal(v,endpoint),n
            value=v
        result[n]=value
    return result

@torch.no_grad()
def update_diagnostics(source,adapted):
    rows=[]
    for n,v in source.items():
        if not mutable(n): continue
        endpoint=adapted[n].cpu().float();v=v.float()
        chunks=[('QK',slice(0,2*v.shape[1])),('VO',slice(2*v.shape[1],None))] if '.qkv.' in n else [('MLP' if '.mlp.' in n else 'VO',slice(None))]
        for group,selection in chunks:
            original=v[selection];delta=endpoint[selection]-original
            base=float(torch.linalg.vector_norm(original)); change=float(torch.linalg.vector_norm(delta))
            rows.append(dict(tensor=n,group=group,source_frobenius=base,update_frobenius=change,relative_update=change/max(base,1e-30)))
    totals={}
    for group in COUNTS:
        rr=[r for r in rows if r['group']==group]
        b=sum(r['source_frobenius']**2 for r in rr)**.5;d=sum(r['update_frobenius']**2 for r in rr)**.5
        totals[group]=dict(source_frobenius=b,update_frobenius=d,relative_update=d/b)
    return dict(groups=totals,tensors=rows)
