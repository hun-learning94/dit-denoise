"""Exact selected dense weights and optional MLP LoRA, with frozen complements."""
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from .model import GROUPS, LoRALinear

CONFIGURATIONS = (
    ('full_qk', 'qk', {'q': 'dense', 'k': 'dense'}, 2048000),
    ('full_qk_vo', 'qk_vo', {'q': 'dense', 'k': 'dense', 'v': 'dense', 'o': 'dense'}, 4096000),
    ('full_mlp', 'mlp', {'fc1': 'dense', 'fc2': 'dense'}, 8192000),
    ('full_vo_mlp', 'vo_mlp', {'v': 'dense', 'o': 'dense', 'fc1': 'dense', 'fc2': 'dense'}, 10240000),
    ('mlp_rank64', 'mlp', {'fc1': 'lora', 'fc2': 'lora'}, 2048000),
    ('dense_vo_mlp_rank64', 'vo_mlp', {'v': 'dense', 'o': 'dense', 'fc1': 'lora', 'fc2': 'lora'}, 4096000),
)


class SplitQKVDense(nn.Module):
    """Keep source storage immutable; expose only selected complete matrices.

    Reassemble a single QKV weight for the original fused linear operation.
    Unselected Q/K/V slices have neither gradients nor optimizer state.
    """
    def __init__(self, original, selected):
        super().__init__()
        self.weight, self.bias = original.weight, original.bias
        self.dim = original.in_features
        assert original.out_features == 3 * self.dim
        self.dense = nn.ParameterDict({part: nn.Parameter(self.weight[i*self.dim:(i+1)*self.dim].detach().clone())
            for i, part in enumerate(('q', 'k', 'v')) if part in selected})

    def forward(self, x):
        weight = torch.cat([self.dense[part] if part in self.dense else self.weight[i*self.dim:(i+1)*self.dim]
            for i, part in enumerate(('q', 'k', 'v'))], dim=0)
        return F.linear(x, weight, self.bias)


def attach_mixed(net, group, rank, seed, modes):
    selected = GROUPS[group]
    assert set(modes) == set(selected)
    assert all(value in ('dense', 'lora') for value in modes.values())
    net.requires_grad_(False)
    records = []
    for i, block in enumerate(net.model.blocks):
        prefix = f'model.blocks.{i}'
        qkv_parts = [part for part in ('q', 'k', 'v') if part in selected]
        if qkv_parts:
            assert all(modes[part] == 'dense' for part in qkv_parts)
            name = prefix + '.attn.qkv'
            block.attn.qkv = SplitQKVDense(block.attn.qkv, qkv_parts)
            dim = block.attn.qkv.dim
            for index, part in enumerate(('q', 'k', 'v')):
                if part in qkv_parts:
                    records.append(dict(weight=name + '.weight', start=index*dim, stop=(index+1)*dim,
                        parameter=name + '.dense.' + part, component=part, mode='dense'))
        for part, owner, attribute in [('o', block.attn, 'proj'),
                                       ('fc1', block.mlp, 'fc1'), ('fc2', block.mlp, 'fc2')]:
            if part not in selected:
                continue
            name = prefix + ('.attn.proj' if part == 'o' else '.mlp.' + part)
            original = getattr(owner, attribute)
            record = dict(weight=name + '.weight', start=0, stop=original.out_features,
                component=part, mode=modes[part])
            if modes[part] == 'dense':
                original.weight.requires_grad_(True)
                record['parameter'] = name + '.weight'
            else:
                matrix_seed = int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:8], 'little') % (2**63 - 1)
                matrix_rank = rank[name] if isinstance(rank, dict) else rank
                setattr(owner, attribute, LoRALinear(original, matrix_rank, matrix_seed))
                record['adapter'] = name + '.adapter'
            records.append(record)
    trainable = {name: p.numel() for name, p in net.named_parameters() if p.requires_grad}
    assert trainable and all(not name.endswith('.bias') for name in trainable)
    return dict(update_family='dense-mixed-v1', group=group, modes=modes, rank=rank,
        alpha=rank, scaling=1.0, dropout=0, bias='frozen',
        parameters=sum(trainable.values()), tensors=trainable, matrices=records)


def assert_mixed_frozen(net, source, counts):
    parameters = dict(net.named_parameters())
    actual = {name: p.numel() for name, p in parameters.items() if p.requires_grad}
    assert actual == counts['tensors'], 'Trainable tensor selection changed'
    # Ordinary dense linear weights are the only original source keys allowed
    # to change. Dense QKV selections live in separate selected parameters.
    mutable = {item['weight'] for item in counts['matrices']
        if item['mode'] == 'dense' and item['parameter'] == item['weight']}
    state = net.state_dict()
    for name, value in source.items():
        if name not in mutable:
            assert torch.equal(state[name], value), f'Frozen source tensor changed: {name}'
        else:
            assert name.endswith('.weight') and parameters[name].requires_grad


@torch.no_grad()
def merged_mixed(net, source, counts):
    merged = {name: value.detach().clone() for name, value in source.items()}
    modules, parameters = dict(net.named_modules()), dict(net.named_parameters())
    for item in counts['matrices']:
        target = merged[item['weight']][item['start']:item['stop']]
        if item['mode'] == 'dense':
            target.copy_(parameters[item['parameter']])
        else:
            target.add_(modules[item['adapter']].delta())
    return merged
