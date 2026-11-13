"""Shared geometry and reduction for the functional context-size experiment."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

H = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n');tmp.replace(path)


def save(path, **values):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **values)
    tmp.replace(path)


def ranks():
    """Nearest tokens: Chebyshev distance, squared Euclidean distance, index."""
    ids = np.arange(256);yy, xx = np.divmod(ids, 16);result = []
    for q in ids:
        dy, dx = yy-q//16, xx-q%16
        order = np.lexsort((ids, dy*dy+dx*dx, np.maximum(abs(dy), abs(dx))))
        assert order[0] == q and np.array_equal(np.sort(order), ids)
        rank = np.empty(256, np.int64);rank[order] = ids;result.append(rank)
    return np.stack(result)


def required_counts(mean_error, tolerances):
    """All larger counts must pass too; no assumed monotonicity in error/noise."""
    values = np.asarray(mean_error)
    assert values.shape[-1] == 256 and np.isfinite(values).all()
    assert (values >= 0).all() and np.all(values[..., -1] == 0)
    envelope = np.maximum.accumulate(values[..., ::-1], axis=-1)[..., ::-1]
    return np.stack([1 + (envelope > epsilon).sum(-1) for epsilon in tolerances], -1)
