"""Deterministic masks, shared corruption, and target-patch scoring."""
from __future__ import annotations
import hashlib
import json

import numpy as np
import torch

N = 256
GRID = 16
PATCH = 16


def seed_for(*parts):
    text = json.dumps(parts, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little") % (2**63 - 1)


def distances(target):
    rr, cc = np.divmod(np.arange(N), GRID)
    r, c = divmod(int(target), GRID)
    return (rr - r)**2 + (cc - c)**2


def contexts(scores, target, budgets, image_id, time_index, noise_id=0, repeats=3):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (N,) or not np.isfinite(scores).all():
        raise ValueError("Expected finite score for every input patch")
    d2 = distances(target)
    tie = np.random.default_rng(seed_for("geometry", target)).permutation(N)
    nearest = np.lexsort((tie, d2))
    start = nearest[:9]
    if target not in start:
        raise AssertionError("Starting context must include target")
    eligible = np.setdiff1d(np.arange(N), start)
    attention_rank = eligible[np.lexsort((eligible, -scores[eligible]))]
    geometric_rank = nearest[~np.isin(nearest, start)]
    permutations = []
    for repeat in range(repeats):
        rng = np.random.default_rng(seed_for("distance_random", image_id, target, time_index, noise_id, repeat))
        permutations.append({int(d): rng.permutation(eligible[d2[eligible] == d]) for d in np.unique(d2[eligible])})
    result = {}
    previous = {}
    for k in sorted(set(budgets)):
        if not 9 <= k <= N:
            raise ValueError(k)
        attention = np.concatenate([start, attention_rank[:k-9]])
        sels = {"attention": attention, "nearest": np.concatenate([start, geometric_rank[:k-9]])}
        counts = {int(d): int(np.count_nonzero(d2[attention[9:]] == d)) for d in np.unique(d2[eligible])}
        for repeat, perm in enumerate(permutations):
            pieces = [start] + [perm[d][:count] for d, count in counts.items() if count]
            sels[f"random_{repeat}"] = np.concatenate(pieces)
        for label, idx in sels.items():
            idx = np.sort(idx).astype(np.int64)
            idxset = set(idx.tolist())
            if len(idxset) != k or not set(start).issubset(idxset):
                raise AssertionError("Invalid context count/base")
            if not previous.get(label, set()).issubset(idxset):
                raise AssertionError("Contexts not nested")
            previous[label] = idxset
            if label.startswith("random") and not np.array_equal(np.sort(d2[idx]), np.sort(d2[attention])):
                raise AssertionError("Distance profile mismatch")
            result[(label, k)] = idx
    return result


def closed_bias(indices, device):
    selected = torch.zeros(len(indices), N, dtype=torch.bool, device=device)
    for b, idx in enumerate(indices):
        selected[b, torch.as_tensor(idx, dtype=torch.long, device=device)] = True
    forbidden = selected[:, :, None] & ~selected[:, None, :]
    bias = torch.zeros((len(indices), N, N), device=device, dtype=torch.float32)
    bias.masked_fill_(forbidden, -float("inf"))
    return bias[:, None], selected


def patch(x, target):
    r, c = divmod(int(target), GRID)
    return x[..., r*PATCH:(r+1)*PATCH, c*PATCH:(c+1)*PATCH]


def clean_prediction(noisy, velocity, t):
    return noisy.float() + (1 - t.float().view(-1, 1, 1, 1)) * velocity.float()


def mask_id(indices):
    return hashlib.sha256(np.asarray(indices, dtype=np.int16).tobytes()).hexdigest()[:20]
