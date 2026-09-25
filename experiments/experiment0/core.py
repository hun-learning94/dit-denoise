"""Input-Jacobian and finite-refresh measurements; no model-loading side effects."""
from __future__ import annotations
import numpy as np
import torch


def patches(x, patch=4):
    """NCHW -> N, row-major patch, (channel,row,column)."""
    n, c, h, w = x.shape
    if h % patch or w % patch:
        raise ValueError('Image dimensions must be multiples of patch size')
    return x.reshape(n, c, h//patch, patch, w//patch, patch).permute(
        0, 2, 4, 1, 3, 5).reshape(n, (h//patch)*(w//patch), c*patch*patch)


def distances(grid, targets, metric='chebyshev'):
    coords = np.array([(r, c) for r in range(grid) for c in range(grid)])
    delta = np.abs(coords[None] - coords[np.asarray(targets), None])
    return delta.max(-1) if metric == 'chebyshev' else delta.sum(-1)


def coverage(energy, dist, qs=(.75, .9, .95), floor=1e-12):
    """Last dimension is source; dist broadcasts over leading dimensions."""
    energy = np.asarray(energy, dtype=np.float64)
    if not np.isfinite(energy).all() or (energy < 0).any():
        raise ValueError('Nonfinite or negative energy')
    total = energy.sum(-1)
    valid = total > floor
    w = energy / np.where(valid, total, 1)[..., None]
    dist = np.broadcast_to(dist, energy.shape)
    radii = np.arange(int(dist.max())+1)
    mass = np.stack([(w*(dist <= r)).sum(-1) for r in radii], -1)
    sorted_mass = np.sort(w, axis=-1)[..., ::-1].cumsum(-1)
    # Clip accumulation roundoff, without changing the definition of coverage.
    rs = np.stack([np.argmax(mass >= q-1e-12, axis=-1) for q in qs], -1).astype(float)
    ks = np.stack([1+np.argmax(sorted_mass >= q-1e-12, axis=-1) for q in qs], -1).astype(float)
    rs[~valid] = np.nan
    ks[~valid] = np.nan
    tails = np.stack([(energy*(dist > r)).sum(-1) for r in radii], -1)
    relative = tails / np.where(valid, total, 1)[..., None]
    relative[~valid] = np.nan
    return dict(radius=rs, count=ks, total=total, tail=tails,
                relative_tail=relative, valid=valid)


def gaussian(shape, seed, device='cpu'):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed)).to(device)


def rademacher(shape, seed, device='cpu'):
    return (2*torch.randint(2, shape, generator=torch.Generator().manual_seed(seed))-1).float().to(device)


def influence(predict, y, targets, probes, patch=4):
    """Return target,probe,source energies. One image; input in normalized Y units."""
    if y.shape[0] != 1:
        raise ValueError('Use one image per call for explicit image/probe accounting')
    y = y.detach().requires_grad_(True)
    out = patches(predict(y), patch)[0]
    result = []
    for ti, target in enumerate(targets):
        row = []
        for pi, u in enumerate(probes[ti]):
            retain = not (ti == len(targets)-1 and pi == len(probes[ti])-1)
            grad, = torch.autograd.grad((out[target]*u).sum(), y, retain_graph=retain)
            row.append(patches(grad, patch).square().mean(-1)[0].detach().cpu().numpy())
        result.append(row)
    return np.asarray(result)


@torch.no_grad()
def refresh(predict, clean, eps, sigma, targets, radii, cs, xis, patch=4):
    """Return raw squared patch response / d_p and its variance-normalized value."""
    if sigma <= 0 or any(not 0 <= c < 1 for c in cs):
        raise ValueError('Refresh needs sigma>0 and 0<=c<1')
    grid = clean.shape[-1]//patch
    dist = distances(grid, targets)
    y = clean + sigma*eps
    base = patches(predict(y), patch)[0]
    raw = np.zeros((len(targets), len(radii), len(cs), len(xis)), dtype=np.float64)
    for ti, target in enumerate(targets):
        for ri, radius in enumerate(radii):
            mask = torch.as_tensor(dist[ti] > radius, device=clean.device).reshape(grid, grid)
            mask = mask.repeat_interleave(patch, 0).repeat_interleave(patch, 1)[None, None]
            if not mask.any():
                continue
            for ci, c in enumerate(cs):
                for xi, noise in enumerate(xis):
                    yp = torch.where(mask, clean+sigma*(c*eps+(1-c*c)**.5*noise), y)
                    diff = patches(predict(yp), patch)[0, target]-base[target]
                    raw[ti, ri, ci, xi] = diff.square().mean().item()
    denom = 2*sigma*sigma*(1-np.asarray(cs))[None, None, :, None]
    return raw, raw/denom
