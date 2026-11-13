"""Independent Q/K/V adapters with immutable pretrained weights and biases."""
from __future__ import annotations

import hashlib
import math
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from edm import VPXPred
from models import DiTBackbone, DiTConfig


GROUPS = {'qk': ('q', 'k'), 'qk_vo': ('q', 'k', 'v', 'o'),
          'vo_mlp': ('v', 'o', 'fc1', 'fc2'), 'mlp': ('fc1', 'fc2')}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    torch.cuda.set_rng_state_all(state['cuda'])


class LowRank(nn.Module):
    def __init__(self, in_features, out_features, rank, seed, device):
        super().__init__()
        assert 0 < rank <= min(in_features, out_features)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device))
        # Per-matrix initialization is independent of the other selected groups.
        generator = torch.Generator(device=device).manual_seed(seed)
        self.lora_A.data.uniform_(-1 / math.sqrt(in_features),
                                 1 / math.sqrt(in_features), generator=generator)

    def forward(self, x):
        # alpha = rank, so the conventional alpha/r scaling is exactly one.
        return F.linear(F.linear(x, self.lora_A), self.lora_B)

    def delta(self):
        return self.lora_B @ self.lora_A


class LoRALinear(nn.Module):
    def __init__(self, original, rank, seed):
        super().__init__()
        self.weight, self.bias = original.weight, original.bias
        self.adapter = LowRank(original.in_features, original.out_features,
                               rank, seed, self.weight.device)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias) + self.adapter(x)


class SplitQKVLoRA(nn.Module):
    def __init__(self, original, selected, rank, seeds):
        super().__init__()
        self.weight, self.bias = original.weight, original.bias
        self.dim = original.in_features
        assert original.out_features == 3 * self.dim
        self.adapters = nn.ModuleDict({part: LowRank(self.dim, self.dim,
            rank[part] if isinstance(rank, dict) else rank,
            seeds[part], self.weight.device) for part in ('q', 'k', 'v') if part in selected})

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias).split(self.dim, dim=-1)
        return torch.cat([value + self.adapters[part](x) if part in self.adapters else value
                          for part, value in zip(('q', 'k', 'v'), base)], dim=-1)


def attach(net, group, rank, seed, update_modes=None):
    if update_modes is not None:
        from .dense_model import attach_mixed
        return attach_mixed(net, group, rank, seed, update_modes)
    selected = GROUPS[group]
    net.requires_grad_(False)
    records = []

    def matrix_seed(name):
        return int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:8], 'little') % (2**63 - 1)

    def matrix_rank(name):
        return rank[name] if isinstance(rank, dict) else rank

    for i, block in enumerate(net.model.blocks):
        prefix = f'model.blocks.{i}'
        if any(part in selected for part in ('q', 'k', 'v')):
            name = prefix + '.attn.qkv'
            seeds = {part: matrix_seed(name + '.' + part) for part in ('q', 'k', 'v')}
            qkv_ranks = {part: matrix_rank(name + '.' + part) for part in ('q', 'k', 'v') if part in selected}
            block.attn.qkv = SplitQKVLoRA(block.attn.qkv, selected, qkv_ranks, seeds)
            for index, part in enumerate(('q', 'k', 'v')):
                if part in selected:
                    records.append(dict(weight=name + '.weight', start=index * block.attn.qkv.dim,
                        stop=(index + 1) * block.attn.qkv.dim,
                        adapter=name + '.adapters.' + part, component=part))
        for part, owner, attribute in [('o', block.attn, 'proj'),
                                       ('fc1', block.mlp, 'fc1'), ('fc2', block.mlp, 'fc2')]:
            if part in selected:
                name = prefix + ('.attn.proj' if part == 'o' else '.mlp.' + part)
                original = getattr(owner, attribute)
                setattr(owner, attribute, LoRALinear(original, matrix_rank(name), matrix_seed(name)))
                records.append(dict(weight=name + '.weight', start=0,
                    stop=original.out_features, adapter=name + '.adapter', component=part))
    trainable = {name: p.numel() for name, p in net.named_parameters() if p.requires_grad}
    assert trainable and all(name.endswith(('.lora_A', '.lora_B')) for name in trainable)
    return dict(group=group, rank=rank, alpha=rank, scaling=1.0, dropout=0,
                bias='frozen', parameters=sum(trainable.values()), tensors=trainable, matrices=records)


def adapter_state(net):
    return {name: p.detach().cpu().clone() for name, p in net.named_parameters() if p.requires_grad}


def load_adapters(net, state):
    parameters = {name: p for name, p in net.named_parameters() if p.requires_grad}
    assert parameters.keys() == state.keys(), 'Adapter tensor selection changed'
    with torch.no_grad():
        for name, p in parameters.items():
            p.copy_(state[name])


def assert_frozen(net, source, counts=None):
    if counts and counts.get('update_family') == 'dense-mixed-v1':
        from .dense_model import assert_mixed_frozen
        return assert_mixed_frozen(net, source, counts)
    current = net.state_dict()
    assert all(torch.equal(current[name], value) for name, value in source.items()), 'Pretrained tensor changed'


@torch.no_grad()
def merged_state(net, source, counts):
    if counts.get('update_family') == 'dense-mixed-v1':
        from .dense_model import merged_mixed
        return merged_mixed(net, source, counts)
    merged = {name: value.detach().clone() for name, value in source.items()}
    modules = dict(net.named_modules())
    for item in counts['matrices']:
        merged[item['weight']][item['start']:item['stop']] += modules[item['adapter']].delta()
    return merged


def load_source(record, device):
    checkpoint = torch.load(record['path'], map_location='cpu', weights_only=False)
    assert checkpoint['step'] == record.get('step', 2034400)
    assert checkpoint['prediction_type'] == 'x' and checkpoint['noise_process'] == 'vp'
    cfg = dict(checkpoint['backbone']['cfg'], print_params=False)
    assert cfg['img_resolution'] == 64 and cfg['img_channels'] == 3 and cfg['patch_size'] == 4
    assert all(cfg[key] == value for key, value in record['model_cfg'].items() if key != 'print_params')
    assert checkpoint['vp_logit_mean'] == -.8 and checkpoint['vp_logit_std'] == .8
    assert checkpoint['vp_denominator_floor'] == .05
    net = VPXPred(DiTBackbone(DiTConfig(**cfg)), 64, 3).to(device)
    net.load_state_dict(checkpoint['ema'], strict=True)
    source = {name: value.detach().clone() for name, value in net.state_dict().items()}
    assert sum(p.numel() for p in net.parameters()) == record['parameters']
    return net, source
