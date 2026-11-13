"""Rank-64 MLP updates, exact frozen tensors, and effective-weight EMA."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from edm import VPXPred
from models import DiTBackbone, DiTConfig


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    torch.save(payload, temporary)
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    torch.cuda.set_rng_state(state['cuda'])


def preflight_cuda():
    assert torch.cuda.is_available(), 'CUDA required; refusing CPU fallback'
    torch.ones(1, device='cuda').sum().item()
    torch.cuda.synchronize()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.cuda.get_device_name()


class LoRALinear(nn.Module):
    """Retain ordinary weight/bias names; factors are separate online state."""
    def __init__(self, original, rank=64):
        super().__init__()
        self.weight = original.weight
        self.bias = original.bias
        self.rank = rank
        self.lora_A = nn.Parameter(torch.empty(rank, original.in_features,
                                             device=self.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, rank,
                                             device=self.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias) + F.linear(F.linear(x, self.lora_A), self.lora_B)

    def effective(self):
        return self.weight + self.lora_B @ self.lora_A


def mutable(name):
    return (name.startswith('model.blocks.') and name.endswith('.weight')
            and any(part in name for part in ('.attn.qkv.', '.attn.proj.', '.mlp.fc1.', '.mlp.fc2.')))


def plain_model(metadata, state, device):
    cfg = dict(metadata['backbone']['cfg'])
    cfg['print_params'] = False
    net = VPXPred(DiTBackbone(DiTConfig(**cfg)), 64, 3).to(device)
    net.load_state_dict(state, strict=True)
    return net


def load_source(record, device='cpu'):
    assert digest(record['path']) == record['sha256'], 'Source checkpoint changed'
    checkpoint = torch.load(record['path'], map_location='cpu', weights_only=False)
    assert checkpoint['prediction_type'] == 'x' and checkpoint['noise_process'] == 'vp'
    assert not checkpoint.get('shuffle') and not checkpoint.get('channel_mix')
    cfg = checkpoint['backbone']['cfg']
    assert (cfg['embed_dim'], cfg['depth'], cfg['num_heads'], cfg['patch_size']) == (320, 10, 10, 4)
    assert cfg['mlp_ratio'] == 4 and cfg['img_resolution'] == 64
    metadata = {k: checkpoint[k] for k in ('backbone', 'prediction_type', 'noise_process', 'vp_denominator_floor')}
    metadata.update(model_type='dit', model_cfg=cfg, vp_logit_mean=-.8,
                    vp_logit_std=.8, loss_type='velocity_mse')
    source = {k: v.to(device).clone() for k, v in checkpoint['ema'].items()}
    return plain_model(metadata, source, device), metadata, source


def attach_adapters(net, seed):
    seed_all(seed)
    net.requires_grad_(False)
    for block in net.model.blocks:
        block.attn.qkv.weight.requires_grad_(True)
        block.attn.proj.weight.requires_grad_(True)
        block.mlp.fc1 = LoRALinear(block.mlp.fc1)
        block.mlp.fc2 = LoRALinear(block.mlp.fc2)
    counts = dict(QK=0, VO=0, MLP=0)
    for block in net.model.blocks:
        dim = block.attn.qkv.weight.shape[1]
        counts['QK'] += 2 * dim * dim
        counts['VO'] += dim * dim + block.attn.proj.weight.numel()
        counts['MLP'] += sum(p.numel() for layer in (block.mlp.fc1, block.mlp.fc2)
                             for p in (layer.lora_A, layer.lora_B))
    assert counts == dict(QK=2048000, VO=2048000, MLP=2048000), counts
    assert sum(p.numel() for p in net.parameters() if p.requires_grad) == 6144000
    assert all(not p.requires_grad for name, p in net.named_parameters() if name.endswith('.bias'))
    return counts


@torch.no_grad()
def effective_state(net):
    state = {name: value.detach() for name, value in net.state_dict().items()
             if not name.endswith(('.lora_A', '.lora_B'))}
    for name, layer in net.named_modules():
        if isinstance(layer, LoRALinear):
            state[name + '.weight'] = layer.effective().detach()
    return state


class EffectiveEMA:
    def __init__(self, source):
        self.shadow = {k: v.detach().float().clone() for k, v in source.items()}
        self.updates = 0

    @torch.no_grad()
    def update(self, net):
        for name, value in effective_state(net).items():
            if mutable(name):
                self.shadow[name].mul_(.999).add_(value.float(), alpha=.001)
        self.updates += 1


def assert_frozen(net, source, ema):
    online = net.state_dict()
    for name, value in source.items():
        if not mutable(name):
            assert torch.equal(online[name], value), f'Frozen online tensor changed: {name}'
            assert torch.equal(ema.shadow[name], value), f'Frozen EMA tensor changed: {name}'
        elif '.mlp.' in name:
            assert torch.equal(online[name], value), f'Frozen base MLP changed: {name}'


def hybrid_state(source, adapted, code):
    assert source.keys() == adapted.keys()
    qk, vo, mlp = map(int, code)
    result = {}
    for name, original in source.items():
        endpoint = adapted[name]
        if '.attn.qkv.weight' in name:
            dim = original.shape[0] // 3
            value = original.clone()
            if qk:
                value[:2 * dim] = endpoint[:2 * dim]
            if vo:
                value[2 * dim:] = endpoint[2 * dim:]
        elif '.attn.proj.weight' in name:
            value = endpoint if vo else original
        elif mutable(name) and '.mlp.' in name:
            value = endpoint if mlp else original
        else:
            assert torch.equal(original, endpoint), f'Frozen endpoint tensor changed: {name}'
            value = original
        result[name] = value
    if code in ('000', '111'):
        reference = source if code == '000' else adapted
        assert all(torch.equal(result[k], v) for k, v in reference.items())
    return result
