"""Minimal adapters around pinned upstream code; model weights are never changed."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import torch


def _import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_model(name, upstream):
    upstream = Path(upstream)
    if name == "pixnerd":
        module = _import_file("attention_upstream_pixnerd", upstream / "pixnerd/src/models/transformer/pixnerd_c2i.py")
        with torch.device("meta"):
            model = module.PixNerDiT(in_channels=3, patch_size=16, num_groups=16, hidden_size=1152,
                                    hidden_size_x=64, num_blocks=30, num_cond_blocks=26,
                                    nerf_mlpratio=2, num_classes=1000)
    elif name == "pixeldit":
        # The upstream model imports this package; keep the exact pinned modules.
        pkg = types.ModuleType("pixdit_core")
        pkg.__path__ = [str(upstream / "pixeldit/pixdit_core")]
        sys.modules["pixdit_core"] = pkg
        _import_file("pixdit_core.modules", upstream / "pixeldit/pixdit_core/modules.py")
        module = _import_file("pixdit_core.pixeldit_c2i", upstream / "pixeldit/pixdit_core/pixeldit_c2i.py")
        with torch.device("meta"):
            model = module.PixDiT(in_channels=3, patch_size=16, num_groups=16, hidden_size=1152,
                                 patch_depth=26, pixel_depth=4, num_classes=1000, pixel_hidden_size=16)
    else:
        raise ValueError(name)
    return model


def load_model(name, path, upstream, device="cpu"):
    model = build_model(name, upstream)
    expected = model.state_dict()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    roots = [("root", checkpoint)]
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "ema", "model", "net"):
            if isinstance(checkpoint.get(key), dict):
                roots.append((key, checkpoint[key]))
    # Prefer explicit inference EMA keys. Never silently choose a partial match.
    candidates = []
    for root_name, root in roots:
        if not isinstance(root, dict):
            continue
        for prefix in ("ema_denoiser.", "denoiser_ema.", "ema_net.", "model_ema.", "ema.",
                       "denoiser.", "net.", "model.", "module.", ""):
            state = {k[len(prefix):]: v for k, v in root.items()
                     if isinstance(k, str) and k.startswith(prefix) and torch.is_tensor(v)}
            if set(state) == set(expected) and all(state[k].shape == expected[k].shape for k in expected):
                score = (0 if "ema" in (root_name + prefix) else 1, len(prefix) == 0)
                candidates.append((score, root_name, prefix, state))
    if not candidates:
        summary = {r: list(d)[:12] for r, d in roots if isinstance(d, dict)}
        raise RuntimeError(f"No exact denoiser state match in {name}: {summary}")
    candidates.sort(key=lambda x: x[0])
    _, root_name, prefix, state = candidates[0]
    model.load_state_dict(state, strict=True, assign=True)
    record = {"model": name, "checkpoint": str(path), "state_root": root_name, "state_prefix": prefix,
              "exact_state_match": True, "tensor_count": len(state),
              "parameters": sum(p.numel() for p in model.parameters()),
              "checkpoint_top_keys": list(checkpoint)[:15],
              "candidate_states": [(r, p) for _, r, p, _ in candidates]}
    del checkpoint, roots, candidates, state, expected
    model.eval().requires_grad_(False)
    return model.to(device), record


class PixNerdRollout:
    """Read original dense Q/K without replacing the upstream SDPA computation."""

    def __init__(self, model, collect_mean=False, targets=None):
        self.model = model
        self.handles = []
        self.matrix = None
        self.layer_count = 0
        self.max_attention_row_error = 0.0
        self.collect_mean = collect_mean
        self.mean_sum = None
        self.targets = targets
        self.layer_rows = []

    @property
    def mean_matrix(self):
        if self.mean_sum is None or not self.layer_count:
            raise RuntimeError("Layer-mean collection was not enabled")
        return self.mean_sum / self.layer_count

    def _hook(self, module, args, kwargs):
        x, pos = args[:2]
        mask = kwargs.get("mask", args[2] if len(args) > 2 else None)
        if mask is not None:
            raise RuntimeError("Donor rollout must be collected on an unmasked forward")
        b, n, c = x.shape
        h = module.num_heads
        qkv = module.qkv(x).reshape(b, n, 3, h, c // h).permute(2, 0, 1, 3, 4)
        q, k = module.q_norm(qkv[0]), module.k_norm(qkv[1])
        rotate = sys.modules[module.__class__.__module__].apply_rotary_emb
        q, k = rotate(q, k, freqs_cis=pos)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        with torch.autocast(device_type=x.device.type, enabled=False):
            attn = torch.softmax((q.float() @ k.float().transpose(-2, -1)) * (module.head_dim ** -0.5), dim=-1)
            self.max_attention_row_error = max(self.max_attention_row_error,
                                               (attn.sum(-1) - 1).abs().max().item())
            eye = torch.eye(n, device=x.device).expand(b, n, n)
            if self.matrix is None:
                self.matrix = eye.clone()
            head_mean = attn.mean(1)
            self.matrix = (0.5 * (head_mean + eye)) @ self.matrix
            if self.collect_mean:
                self.mean_sum = head_mean.clone() if self.mean_sum is None else self.mean_sum + head_mean
                if self.targets is not None:
                    self.layer_rows.append(head_mean[:, self.targets].cpu())
        self.layer_count += 1

    def __enter__(self):
        for block in self.model.blocks[:self.model.num_cond_blocks]:
            self.handles.append(block.attn.register_forward_pre_hook(self._hook, with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


class ProtectedFeatures:
    """Capture only protected feature rows to locate any end-to-end leakage."""
    def __init__(self, model, name, selected):
        self.model, self.name, self.selected = model, name, selected
        self.handles, self.features = [], {}

    def _hook(self, label, pixel=False):
        def hook(module, args, output):
            if pixel:
                b = self.selected.shape[0]
                output = output.reshape(b, 256, *output.shape[1:])
            self.features[label] = output[self.selected].detach().float().cpu()
        return hook

    def __enter__(self):
        if self.name == "pixnerd":
            for i, block in enumerate(self.model.blocks[:self.model.num_cond_blocks]):
                self.handles.append(block.register_forward_hook(self._hook(f"conditioning_{i}")))
        else:
            for i, block in enumerate(self.model.patch_blocks):
                self.handles.append(block.register_forward_hook(self._hook(f"patch_{i}")))
            for i, block in enumerate(self.model.pixel_blocks):
                self.handles.append(block.register_forward_hook(self._hook(f"pixel_{i}", pixel=True)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
