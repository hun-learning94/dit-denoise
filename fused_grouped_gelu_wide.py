"""Masked wider-head grouped GELU kernels; the original D=32 path is unchanged.

Key/value partial gradients and reduced outputs use packed allocations with
explicit disjoint offsets.
This prevents compiled allocation merging from aliasing separate write-only
Triton scratch arguments. The bank backward is split over tokens to avoid serializing an entire training
microbatch in one CTA. Partial bank gradients are reduced in FP32.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# (token tile, bank tile, warps, pipeline stages), separately for each pass.
# Keep configuration outside the graph; configure before torch.compile.
BASELINE_CONFIG = ((32, 32, 4, 3), (32, 32, 4, 3), (32, 32, 4, 3), 16, False)
# H200 H8/D96/R2304: checked against the unfused implementation, including
# raw shared-bank gradients, and timed in complete compiled training updates.
# Records: reports/2026-10-19/celeba128_kernel_pipeline/results.json.
TUNED_BF16_CONFIG = ((64, 32, 4, 1), (32, 128, 4, 2), (64, 64, 4, 1), 16, True)
_BF16_CONFIG = TUNED_BF16_CONFIG


def configure_bf16_kernel(config):
    """Configure BF16 launch geometry before compiling a model (benchmark use)."""
    global _BF16_CONFIG
    forward, query, bank, splits, reduction = config
    tiles = [tuple(tile) if len(tile) == 4 else (*tile, 3) for tile in (forward, query, bank)]
    for bm, br, warps, stages in tiles:
        if bm not in (32, 64, 128) or br not in (32, 64, 128) or warps not in (4, 8):
            raise ValueError("unsupported wide-kernel launch geometry")
        if stages not in (1, 2, 3):
            raise ValueError("unsupported pipeline depth")
    if splits not in (1, 2, 4, 8, 16, 32, 64):
        raise ValueError("unsupported bank-gradient split count")
    _BF16_CONFIG = (*tiles, splits, bool(reduction))


@triton.jit
def _gelu(x):
    return 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))


@triton.jit
def _grad(x):
    return 0.5 * (1.0 + tl.erf(x * 0.7071067811865476)) + x * 0.3989422804014327 * tl.exp(-0.5 * x * x)


@triton.jit
def _forward(Q, K, B, V, Y, M: tl.constexpr, H: tl.constexpr, R: tl.constexpr,
             D: tl.constexpr, DP: tl.constexpr, BM: tl.constexpr, BR: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    h = tl.program_id(1)
    d = tl.arange(0, DP)
    q = tl.load(Q + m[:, None] * H * D + h * D + d[None, :],
                (m[:, None] < M) & (d[None, :] < D), 0)
    acc = tl.zeros((BM, DP), tl.float32)
    for r0 in range(0, R, BR):
        r = r0 + tl.arange(0, BR)
        mask = (r[:, None] < R) & (d[None, :] < D)
        k = tl.load(K + r[:, None] * H * D + h * D + d[None, :], mask, 0)
        v = tl.load(V + r[:, None] * H * D + h * D + d[None, :], mask, 0)
        b = tl.load(B + r * H + h, r < R, 0)
        logits = tl.dot(q, tl.trans(k), input_precision="ieee") * (D ** -0.5) + b[None, :]
        coeff = tl.where(r[None, :] < R, _gelu(logits), 0).to(q.dtype)
        acc += tl.dot(coeff, v, input_precision="ieee")
    tl.store(Y + m[:, None] * H * D + h * D + d[None, :], acc,
             (m[:, None] < M) & (d[None, :] < D))


@triton.jit
def _backward_query(Q, K, B, V, G, GQ, M: tl.constexpr, H: tl.constexpr,
                    R: tl.constexpr, D: tl.constexpr, DP: tl.constexpr,
                    BM: tl.constexpr, BR: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    h = tl.program_id(1)
    d = tl.arange(0, DP)
    mask_m = (m[:, None] < M) & (d[None, :] < D)
    q = tl.load(Q + m[:, None] * H * D + h * D + d[None, :], mask_m, 0)
    g = tl.load(G + m[:, None] * H * D + h * D + d[None, :], mask_m, 0)
    acc = tl.zeros((BM, DP), tl.float32)
    for r0 in range(0, R, BR):
        r = r0 + tl.arange(0, BR)
        mask = (r[:, None] < R) & (d[None, :] < D)
        k = tl.load(K + r[:, None] * H * D + h * D + d[None, :], mask, 0)
        v = tl.load(V + r[:, None] * H * D + h * D + d[None, :], mask, 0)
        b = tl.load(B + r * H + h, r < R, 0)
        logits = tl.dot(q, tl.trans(k), input_precision="ieee") * (D ** -0.5) + b[None, :]
        gl = tl.dot(g, tl.trans(v), input_precision="ieee") * _grad(logits)
        gl = tl.where(r[None, :] < R, gl, 0)
        acc += tl.dot(gl.to(q.dtype), k, input_precision="ieee") * (D ** -0.5)
    tl.store(GQ + m[:, None] * H * D + h * D + d[None, :], acc, mask_m)


@triton.jit
def _backward_bank(Q, K, B, V, G, PKV, PB, M: tl.constexpr, H: tl.constexpr,
                   R: tl.constexpr, D: tl.constexpr, DP: tl.constexpr,
                   CHUNK: tl.constexpr, V_OFFSET: tl.constexpr, BM: tl.constexpr, BR: tl.constexpr):
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    h, split = tl.program_id(1), tl.program_id(2)
    d = tl.arange(0, DP)
    mask_r = (r[:, None] < R) & (d[None, :] < D)
    k = tl.load(K + r[:, None] * H * D + h * D + d[None, :], mask_r, 0)
    v = tl.load(V + r[:, None] * H * D + h * D + d[None, :], mask_r, 0)
    b = tl.load(B + r * H + h, r < R, 0)
    gk, gv = tl.zeros((BR, DP), tl.float32), tl.zeros((BR, DP), tl.float32)
    gb = tl.zeros((BR,), tl.float32)
    for offset in range(0, CHUNK, BM):
        m = split * CHUNK + offset + tl.arange(0, BM)
        mask = (m[:, None] < M) & (d[None, :] < D)
        q = tl.load(Q + m[:, None] * H * D + h * D + d[None, :], mask, 0)
        g = tl.load(G + m[:, None] * H * D + h * D + d[None, :], mask, 0)
        logits = tl.dot(q, tl.trans(k), input_precision="ieee") * (D ** -0.5) + b[None, :]
        coeff = tl.where(m[:, None] < M, _gelu(logits), 0)
        gl = tl.dot(g, tl.trans(v), input_precision="ieee") * _grad(logits)
        gk += tl.dot(tl.trans(gl.to(q.dtype)), q, input_precision="ieee") * (D ** -0.5)
        gv += tl.dot(tl.trans(coeff.to(g.dtype)), g, input_precision="ieee")
        gb += tl.sum(gl, axis=0)
    base = split * R * H * D + r[:, None] * H * D + h * D + d[None, :]
    tl.store(PKV + base, gk, mask_r)
    tl.store(PKV + V_OFFSET + base, gv, mask_r)
    tl.store(PB + split * R * H + r * H + h, gb, r < R)


@triton.jit
def _reduce_bank(PKV, PB, GKV, GB, SIZE: tl.constexpr, BIAS_SIZE: tl.constexpr,
                 SPLITS: tl.constexpr, SP: tl.constexpr, BLOCK: tl.constexpr):
    """Sum FP32 partials and cast once, without materializing FP32 totals."""
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    split = tl.arange(0, SP)
    offsets = split[:, None] * SIZE + index[None, :]
    mask = (split[:, None] < SPLITS) & (index[None, :] < SIZE)
    gk = tl.sum(tl.load(PKV + offsets, mask, 0), axis=0)
    gv = tl.sum(tl.load(PKV + SPLITS * SIZE + offsets, mask, 0), axis=0)
    tl.store(GKV + index, gk, index < SIZE)
    tl.store(GKV + SIZE + index, gv, index < SIZE)
    if tl.program_id(0) * BLOCK < BIAS_SIZE:
        bias_offsets = split[:, None] * BIAS_SIZE + index[None, :]
        bias_mask = (split[:, None] < SPLITS) & (index[None, :] < BIAS_SIZE)
        gb = tl.sum(tl.load(PB + bias_offsets, bias_mask, 0), axis=0)
        tl.store(GB + index, gb, index < BIAS_SIZE)


class _WideGroupedGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, keys, biases, values):
        shape = q.shape
        q, keys, biases, values = (x.contiguous() for x in (q, keys, biases, values))
        q = q.reshape(-1, shape[-2], shape[-1])
        m, h, d = q.shape
        r = keys.shape[0]
        if not q.is_cuda or d not in (64, 96, 128):
            raise ValueError("wide grouped GELU requires CUDA and head width 64, 96, or 128")
        if keys.shape != (r, h, d) or values.shape != keys.shape or biases.shape != (r, h):
            raise ValueError("incompatible grouped GELU input shapes")
        if any(t.dtype != q.dtype or t.device != q.device for t in (keys, biases, values)):
            raise ValueError("grouped GELU inputs must have the same dtype/device")
        config = _BF16_CONFIG if q.dtype == torch.bfloat16 else BASELINE_CONFIG
        bm, br, warps, stages = config[0]
        out = torch.empty_like(q)
        _forward[(triton.cdiv(m, bm), h)](q, keys, biases, values, out, m, h, r, d,
            triton.next_power_of_2(d), BM=bm, BR=br, num_warps=warps, num_stages=stages)
        ctx.save_for_backward(q, keys, biases, values)
        ctx.shape = shape
        ctx.config = config
        return out.reshape(shape)

    @staticmethod
    def backward(ctx, gradient):
        q, keys, biases, values = ctx.saved_tensors
        gradient = gradient.contiguous().reshape_as(q)
        m, h, d = q.shape
        r = keys.shape[0]
        gq = torch.empty_like(q)
        bm, br, warps, stages = ctx.config[2]
        splits = min(ctx.config[3], max(1, triton.cdiv(m, 512)))
        chunk = triton.cdiv(triton.cdiv(m, splits), bm) * bm
        # Align the number of split chunks with the rounded boundary.
        splits = triton.cdiv(m, chunk)
        partials = torch.empty((2, splits, r, h, d), device=q.device, dtype=torch.float32)
        pb = torch.empty((splits, r, h), device=q.device, dtype=torch.float32)
        qm, qr, qw, qs = ctx.config[1]
        _backward_query[(triton.cdiv(m, qm), h)](q, keys, biases, values, gradient, gq,
            m, h, r, d, triton.next_power_of_2(d), BM=qm, BR=qr, num_warps=qw, num_stages=qs)
        _backward_bank[(triton.cdiv(r, br), h, splits)](q, keys, biases, values,
            gradient, partials, pb, m, h, r, d, triton.next_power_of_2(d),
            chunk, splits*r*h*d, BM=bm, BR=br, num_warps=warps, num_stages=stages)
        if ctx.config[4]:
            # Separate empty_like outputs can also be merged by Inductor,
            # even after the partial workspaces have been packed. Keep both
            # writable output planes in one allocation with explicit offsets.
            gkv = torch.empty((2, r, h, d), device=keys.device, dtype=keys.dtype)
            gb = torch.empty_like(biases)
            _reduce_bank[(triton.cdiv(r*h*d, 256),)](partials, pb, gkv, gb,
                r*h*d, r*h, splits, triton.next_power_of_2(splits), BLOCK=256, num_warps=4)
            gk, gv = gkv[0], gkv[1]
        else:
            gk, gb, gv = partials[0].sum(0).to(keys.dtype), pb.sum(0).to(biases.dtype), partials[1].sum(0).to(values.dtype)
        return gq.reshape(ctx.shape), gk, gb, gv


def fused_grouped_gelu_wide(q, keys, biases, values):
    return _WideGroupedGELU.apply(q, keys, biases, values)
