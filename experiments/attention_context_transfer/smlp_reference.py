"""Frozen CelebA128 SMLP adapter; input gradients only, no parameter fitting."""
from __future__ import annotations

import sys
from pathlib import Path
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from .models import _import_file
from .prepare import sha256


def load_reference(design, device='cuda'):
    ck = design['reference_checkpoint']
    assert sha256(ck['path']) == ck['sha256'], 'SMLP checkpoint changed'
    source = Path(ck['source'])
    for name, digest in ck['source_sha256'].items():
        assert sha256(source / name) == digest, name
    # Bind imports to the training snapshot without replacing modules used by
    # the MLP adapter. No initialization is used: every tensor is loaded strictly.
    previous = {name: sys.modules.get(name) for name in ['models', 'fused_grouped_gelu']}
    try:
        _import_file('models', source / 'models.py')
        _import_file('fused_grouped_gelu', source / 'fused_grouped_gelu.py')
        mod = _import_file('smlp_reference_snapshot', source / 'model_grouped_dpd6.py')
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    wide = _import_file('smlp_reference_wide', source / 'fused_grouped_gelu_wide.py')
    saved = torch.load(ck['path'], map_location='cpu', weights_only=False, mmap=True)
    assert saved['step'] == 63200 and saved['recipe']['noise_scale'] == .5
    cfg = mod.GroupedDPD6Config(**saved['model_cfg'])
    assert cfg.depth == 12 and cfg.patch_size == 8 and cfg.img_resolution == 128
    with torch.device('meta'):
        model = mod.GroupedDPD6Backbone(cfg)
    state = {k.removeprefix('model.'): v for k, v in saved['ema'].items()}
    model.load_state_dict(state, strict=True, assign=True)
    assert all(torch.isfinite(v).all() for v in state.values())
    model = model.eval().requires_grad_(False).to(device)
    # Frozen K/V projections do not depend on the noisy image. Caching them is
    # exact for input derivatives and avoids repeated shared-bank projections.
    model.prepare_projected_cache()

    class QueryOnly(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, keys, biases, values):
            assert not any(x.requires_grad for x in (keys, biases, values))
            shape = q.shape
            q, keys, biases, values = (x.contiguous() for x in (q, keys, biases, values))
            q = q.reshape(-1, shape[-2], shape[-1])
            m, h, d = q.shape
            r = keys.shape[0]
            config = wide._BF16_CONFIG if q.dtype == torch.bfloat16 else wide.BASELINE_CONFIG
            bm, br, warps, stages = config[0]
            out = torch.empty_like(q)
            wide._forward[(wide.triton.cdiv(m, bm), h)](
                q, keys, biases, values, out, m, h, r, d,
                wide.triton.next_power_of_2(d), BM=bm, BR=br,
                num_warps=warps, num_stages=stages)
            ctx.save_for_backward(q, keys, biases, values)
            ctx.shape, ctx.config = shape, config
            return out.reshape(shape)

        @staticmethod
        def backward(ctx, gradient):
            q, keys, biases, values = ctx.saved_tensors
            m, h, d = q.shape
            bm, br, warps, stages = ctx.config[1]
            grad = torch.empty_like(q)
            wide._backward_query[(wide.triton.cdiv(m, bm), h)](
                q, keys, biases, values, gradient.contiguous().reshape_as(q), grad,
                m, h, keys.shape[0], d, wide.triton.next_power_of_2(d),
                BM=bm, BR=br, num_warps=warps, num_stages=stages)
            return grad.reshape(ctx.shape), None, None, None

    mod.fused_grouped_gelu = QueryOnly.apply
    for block in model.blocks:
        block.expansion.fused = True
    receipt = dict(checkpoint=ck['path'], checkpoint_sha256=ck['sha256'],
                   parameters=sum(p.numel() for p in model.parameters()),
                   exact_state_match=True, ema_step=63200, precision='FP32; TF32 disabled',
                   source=ck['source'], source_sha256=ck['source_sha256'],
                   input_gradient='frozen snapshot forward/query-backward kernels; bank gradients skipped',
                   output='direct clean x', noise_scale=.5, projected_bank_cache=True)
    return model, receipt, mod, wide


@torch.no_grad()
def forward(model, y, t, batch=128):
    with sdpa_kernel(SDPBackend.MATH):
        return torch.cat([model(y[i:i+batch], t[i:i+batch]) for i in range(0, len(y), batch)])
