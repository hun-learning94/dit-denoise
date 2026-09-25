"""Triton fused dense GELU expansion for grouped DPD6."""
from __future__ import annotations
import torch
import triton
import triton.language as tl

@triton.jit
def _gelu(x):
    return 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))

@triton.jit
def _gelu_grad(x):
    return 0.5 * (1.0 + tl.erf(x * 0.7071067811865476)) + x * 0.3989422804014327 * tl.exp(-0.5 * x * x)

@triton.jit
def _fwd(q, k, bias, value, out, M: tl.constexpr, H: tl.constexpr,
         R: tl.constexpr, D: tl.constexpr, scale: tl.constexpr,
         BM: tl.constexpr, BR: tl.constexpr):
    pid, h = tl.program_id(0), tl.program_id(1)
    ms, ds = pid * BM + tl.arange(0, BM), tl.arange(0, D)
    mask = ms[:, None] < M
    qt = tl.load(q + ms[:, None] * H * D + h * D + ds[None, :], mask=mask, other=0.)
    acc = tl.zeros((BM, D), tl.float32)
    for r0 in range(0, R, BR):
        rs = r0 + tl.arange(0, BR)
        kt = tl.load(k + rs[:, None] * H * D + h * D + ds[None, :])
        logits = tl.dot(qt, tl.trans(kt)) * scale + tl.load(bias + rs * H + h)[None, :]
        coeff = _gelu(logits).to(qt.dtype)
        vt = tl.load(value + rs[:, None] * H * D + h * D + ds[None, :])
        acc += tl.dot(coeff, vt)
    tl.store(out + ms[:, None] * H * D + h * D + ds[None, :], acc, mask=mask)

@triton.jit
def _bwd_q(q, k, bias, value, go, gq, M: tl.constexpr, H: tl.constexpr,
           R: tl.constexpr, D: tl.constexpr, scale: tl.constexpr,
           BM: tl.constexpr, BR: tl.constexpr):
    pid, h = tl.program_id(0), tl.program_id(1)
    ms, ds = pid * BM + tl.arange(0, BM), tl.arange(0, D)
    mask = ms[:, None] < M
    qt = tl.load(q + ms[:, None] * H * D + h * D + ds[None, :], mask=mask, other=0.)
    got = tl.load(go + ms[:, None] * H * D + h * D + ds[None, :], mask=mask, other=0.)
    acc = tl.zeros((BM, D), tl.float32)
    for r0 in range(0, R, BR):
        rs = r0 + tl.arange(0, BR)
        kt = tl.load(k + rs[:, None] * H * D + h * D + ds[None, :])
        vt = tl.load(value + rs[:, None] * H * D + h * D + ds[None, :])
        logits = tl.dot(qt, tl.trans(kt)) * scale + tl.load(bias + rs * H + h)[None, :]
        gl = tl.dot(got, tl.trans(vt)) * _gelu_grad(logits)
        acc += tl.dot(gl.to(qt.dtype), kt) * scale
    tl.store(gq + ms[:, None] * H * D + h * D + ds[None, :], acc, mask=mask)

@triton.jit
def _bwd_lib(q, k, bias, value, go, gk, gb, gv, M: tl.constexpr,
             H: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
             scale: tl.constexpr, BM: tl.constexpr, BR: tl.constexpr):
    rb, h = tl.program_id(0), tl.program_id(1)
    rs, ds = rb * BR + tl.arange(0, BR), tl.arange(0, D)
    kt = tl.load(k + rs[:, None] * H * D + h * D + ds[None, :])
    vt = tl.load(value + rs[:, None] * H * D + h * D + ds[None, :])
    b = tl.load(bias + rs * H + h)
    dkt, dvt = tl.zeros((BR, D), tl.float32), tl.zeros((BR, D), tl.float32)
    dbt = tl.zeros((BR,), tl.float32)
    for m0 in range(0, M, BM):
        ms = m0 + tl.arange(0, BM)
        mask = ms[:, None] < M
        qt = tl.load(q + ms[:, None] * H * D + h * D + ds[None, :], mask=mask, other=0.)
        got = tl.load(go + ms[:, None] * H * D + h * D + ds[None, :], mask=mask, other=0.)
        logits = tl.dot(qt, tl.trans(kt)) * scale + b[None, :]
        coeff = _gelu(logits)
        gl = tl.dot(got, tl.trans(vt)) * _gelu_grad(logits)
        dkt += tl.dot(tl.trans(gl.to(qt.dtype)), qt) * scale
        dvt += tl.dot(tl.trans(coeff.to(got.dtype)), got)
        dbt += tl.sum(gl, axis=0)
    tl.store(gk + rs[:, None] * H * D + h * D + ds[None, :], dkt)
    tl.store(gv + rs[:, None] * H * D + h * D + ds[None, :], dvt)
    tl.store(gb + rs * H + h, dbt)

class _Fused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, bias, value):
        q, k, bias, value = (x.contiguous() for x in (q, k, bias, value))
        shape = q.shape
        q = q.reshape(-1, shape[-2], shape[-1])
        M, H, D = q.shape; R = k.shape[0]
        if not q.is_cuda or D != 32 or R % 32:
            raise ValueError("fused grouped GELU requires CUDA, D=32, and R divisible by 32")
        out = torch.empty_like(q)
        _fwd[(triton.cdiv(M, 16), H)](q, k, bias, value, out, M, H, R, D,
            D ** -0.5, BM=16, BR=32, num_warps=4)
        ctx.save_for_backward(q, k, bias, value); ctx.shape = shape
        return out.reshape(shape)
    @staticmethod
    def backward(ctx, go):
        q, k, bias, value = ctx.saved_tensors
        go = go.contiguous().reshape_as(q)
        M, H, D = q.shape; R = k.shape[0]
        gq, gk, gb, gv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(bias), torch.empty_like(value)
        _bwd_q[(triton.cdiv(M, 16), H)](q, k, bias, value, go, gq, M, H, R, D,
            D ** -0.5, BM=16, BR=32, num_warps=4)
        _bwd_lib[(triton.cdiv(R, 32), H)](q, k, bias, value, go, gk, gb, gv,
            M, H, R, D, D ** -0.5, BM=16, BR=32, num_warps=4)
        return gq.reshape(ctx.shape), gk, gb, gv

def fused_grouped_gelu(q, keys, biases, values):
    return _Fused.apply(q, keys, biases, values)
