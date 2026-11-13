"""CUDA correctness checks required before timing the wider fused kernel."""
from __future__ import annotations

import gc
import torch
import torch.nn.functional as F

from fused_grouped_gelu import fused_grouped_gelu
from model_grouped_dpd6 import GroupedDPD6Backbone, GroupedDPD6Config


def error(actual, expected):
    delta = actual.detach().float() - expected.detach().float()
    scale = expected.detach().float().norm().clamp_min(1e-12)
    return {"relative_l2": (delta.norm()/scale).item(), "max_abs": delta.abs().max().item(),
            "reference_norm": scale.item()}


def reference(q, k, bias, value):
    logits = torch.einsum("mhd,rhd->mhr", q, k) * (q.shape[-1] ** -0.5)
    return torch.einsum("mhr,rhd->mhd", F.gelu(logits + bias.T[None]), value)


def primitive_check(dtype, width, tokens, rows):
    torch.manual_seed(1234)
    shapes = ((tokens, 8, width), (rows, 8, width), (rows, 8), (rows, 8, width))
    left = [torch.randn(shape, dtype=dtype, device="cuda").requires_grad_() for shape in shapes]
    with torch.no_grad():
        left[-1].div_(width)
    right = [x.detach().clone().requires_grad_() for x in left]
    probe = torch.randn_like(left[0])
    a, b = reference(*left), fused_grouped_gelu(*right)
    a.backward(probe)
    b.backward(probe)
    results = {"output": error(b, a)}
    results.update({name: error(y.grad, x.grad) for name, x, y in
                    zip(("query", "keys", "bias", "values"), left, right)})
    limit = 0.04 if dtype == torch.bfloat16 else 0.0005
    assert all(x["relative_l2"] < limit for x in results.values()), results
    return {"dtype": str(dtype), "width": width, "tokens": tokens, "rows": rows,
            "relative_l2_limit": limit, "errors": results}


def end_to_end(dtype):
    torch.manual_seed(4321)
    cfg = GroupedDPD6Config(img_resolution=128, img_channels=3, patch_size=8,
        embed_dim=768, depth=12, num_heads=12, prototype_count=2304,
        coefficient_groups=8, group_dim=96, prototype_key_width=1840,
        prototype_value_width=1840, fused_expansion=False, print_params=False)
    reference_model = GroupedDPD6Backbone(cfg).cuda()
    with torch.no_grad():
        for block in reference_model.blocks:
            block.adaLN.bias[2*768:3*768].fill_(0.1)
            block.adaLN.bias[5*768:6*768].fill_(0.1)
    candidate = GroupedDPD6Backbone(cfg).cuda()
    candidate.load_state_dict(reference_model.state_dict())
    for block in candidate.blocks:
        block.expansion.fused = True
    x = torch.randn(2, 3, 128, 128, device="cuda")
    times = torch.tensor([0.3, 0.7], device="cuda")
    probe = torch.randn_like(x)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        actual = candidate(x, times)
        expected = reference_model(x, times)
        loss_a = (actual.float() * probe).mean()
        loss_b = (expected.float() * probe).mean()
    loss_a.backward()
    loss_b.backward()
    results = {"output": error(actual, expected)}
    expected_params = dict(reference_model.named_parameters())
    for name, parameter in candidate.named_parameters():
        if (name in {"key_prototypes", "value_prototypes"}
                or ".expansion." in name):
            other = expected_params[name]
            assert parameter.grad is not None and other.grad is not None, name
            metrics = error(parameter.grad, other.grad)
            assert other.grad.float().norm().item() > 1e-12, (name, "zero reference gradient")
            results[name] = metrics
    limit = 0.05 if dtype == torch.bfloat16 else 0.001
    failures = {name: value for name, value in results.items() if value["relative_l2"] >= limit}
    assert not failures, failures
    del candidate, reference_model, expected_params, actual, expected, loss_a, loss_b
    gc.collect()
    torch.cuda.empty_cache()
    return {"dtype": str(dtype), "depth": 12, "batch": 2, "gate_bias": 0.1,
            "relative_l2_limit": limit, "errors": results}


def run_checks():
    results = {"primitive": [], "end_to_end": []}
    # Non-power-of-two token and bank dimensions exercise every mask and split.
    for dtype in (torch.float32, torch.bfloat16):
        results["primitive"].append(primitive_check(dtype, 96, 1037, 2307))
        print(f"CHECK primitive width96 {dtype}: PASS", flush=True)
        results["end_to_end"].append(end_to_end(dtype))
        print(f"CHECK full 12-layer shared banks {dtype}: PASS", flush=True)
    results["primitive"].append(primitive_check(torch.bfloat16, 32, 67, 256))
    results["passed"] = True
    return results


def compiled_model_check(raw, compiled, images):
    """Check compiled gradients after warmup, when predictor gates are nonzero."""
    # Match the production update's zero_grad(set_to_none=True) precondition;
    # compare saved gradient values independently of the previous update buffers.
    raw.zero_grad(set_to_none=True)
    times = torch.tensor([0.3, 0.7], device=images.device)
    raw.eval()
    parameters = [(name, p) for name, p in raw.named_parameters()
                  if (name in {"key_prototypes", "value_prototypes"}
                      or ".expansion." in name or ".mlp." in name)]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = raw(images[:2], times)
        expected_loss = (expected.float() * images[:2]).mean()
    expected_gradients = tuple(g.detach().clone() for g in
        torch.autograd.grad(expected_loss, [p for _, p in parameters]))
    torch.cuda.synchronize()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = compiled(images[:2], times)
        actual_loss = (actual.float() * images[:2]).mean()
    actual_gradients = torch.autograd.grad(actual_loss, [p for _, p in parameters])
    metrics = {"output": error(actual, expected), "gradients": {}}
    assert metrics["output"]["relative_l2"] < 0.02, metrics["output"]
    for (name, _), actual_grad, expected_grad in zip(parameters, actual_gradients, expected_gradients):
        assert expected_grad.float().norm().item() > 1e-12, (name, "zero reference gradient")
        measurement = error(actual_grad, expected_grad)
        assert measurement["relative_l2"] < 0.05, (name, measurement)
        metrics["gradients"][name] = measurement
    return metrics
