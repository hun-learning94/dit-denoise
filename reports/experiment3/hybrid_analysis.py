"""Selected reporting scope and paired, per-latent cube statistics.

Keep the broader historical exports intact: the summary is edited separately.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAIRS = ([('celeba64', t) for t in ('aahq64', 'ffhq64', 'afhq_v2_64', 'stl10_64')]
         + [(s, t) for s in ('stl10_64', 'imagenet64')
            for t in ('aahq64', 'ffhq64', 'afhq_v2_64', 'celeba64')])
CODES = ['000', '100', '010', '001', '110', '101', '011', '111']
LINEAR = {
    'joint_nonadditivity': {'111': 1, '100': -1, '010': -1, '001': -1, '000': 2},
    'QK_restoration': {'111': 1, '011': -1},
    'VO_restoration': {'111': 1, '101': -1},
    'MLP_restoration': {'111': 1, '110': -1},
    'three_way_interaction': {'111': 1, '110': -1, '101': -1, '011': -1,
                              '100': 1, '010': 1, '001': 1, '000': -1},
    'QK_only_minus_source': {'100': 1, '000': -1},
    'QK_given_VO': {'110': 1, '010': -1},
    'QK_given_MLP': {'101': 1, '001': -1},
    'VO_given_MLP': {'011': 1, '001': -1},
    'MLP_given_VO': {'011': 1, '010': -1},
    'VO_MLP_minus_QK': {'011': 1, '100': -1},
    'MLP_minus_QK': {'001': 1, '100': -1},
    'MLP_only_minus_source': {'001': 1, '000': -1},
}
# All six cube-face interactions; the opposite-face difference is the same I3.
for name, a, b, other in [('QK_VO', 0, 1, 2), ('QK_MLP', 0, 2, 1), ('VO_MLP', 1, 2, 0)]:
    for context in (0, 1):
        weights = {}
        for av, bv, sign in [(1, 1, 1), (1, 0, -1), (0, 1, -1), (0, 0, 1)]:
            bits = [0, 0, 0]; bits[a] = av; bits[b] = bv; bits[other] = context
            weights[''.join(map(str, bits))] = sign
        LINEAR[f'{name}_interaction_context{context}'] = weights
CONTRAST_NAMES = [*LINEAR, 'minimum_reversion']


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def resolve(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        path = Path(str(path).replace('/experiment3/primary_metrics_1024/',
                                     '/experiment3/archive/legacy_reports/primary_metrics_1024/'))
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def selected_rows():
    rows = list(csv.DictReader((HERE/'metrics/metrics_by_seed.csv').open()))
    rows = [r for r in rows if (r['source'], r['target']) in PAIRS]
    rows.sort(key=lambda r: (PAIRS.index((r['source'], r['target'])), int(r['adaptation_seed'])))
    for pair in PAIRS:
        for seed in ('0', '1'):
            subset = [r for r in rows if (r['source'], r['target'], r['adaptation_seed']) == (*pair, seed)]
            assert len(subset) == 32 and len({r['id'] for r in subset}) == 1, (pair, seed)
            assert {(r['metric'], r['state']) for r in subset} == {(m, c) for m in ('P', 'D_rel', 'E', 'D_raw') for c in CODES}
    assert all(r['scale'] == '20m' for r in rows)
    return rows


def locate(row):
    """Return the exact measurement/cache, including corrected k=50 density."""
    ident = row['id']; density = None
    if row['source'] == 'celeba64':
        base = ROOT/'reports/experiment3/archive/legacy_reports/primary_metrics_1024'
        manifest = read(base/'evaluation/manifest.json')
        task = next(i for i, r in enumerate(manifest['rows']) if r['id'] == ident)
        cache = base/f'evaluation/tasks/{task}'
        density = base/f'density_checks/task{task}_n1000_k50.npz'
    elif row['source'] == 'stl10_64':
        base = ROOT/'reports/2026-09-19/experiment3_completion'
        spec = next(s for s in read(base/'selected_manifest.json')['rows'] if s['row']['id'] == ident)
        cache = (Path(spec['reuse_campaign'])/'cache' if spec['reuse_campaign']
                 else base/'hybrids'/ident/'cache')
    elif '_source800_' in ident:
        cache = ROOT/'reports/2026-09-19/imagenet800_replacement/hybrids'/ident/'cache'
    else:
        name = 'imagenet600_celeba_hybrid_metrics' if row['target'] == 'celeba64' else 'imagenet600_hybrid_metrics'
        cache = ROOT/'reports/2026-09-18'/name/'cache'
    return cache/'measurements'/f'{ident}.npz', density, cache/'cubes'/f'{ident}.json'


def measurements(row):
    path, density_path, cube_path = locate(row)
    with np.load(path) as a:
        assert list(a['codes']) == CODES
        distances = a['distances'].astype(float)
        densities = a['densities'].astype(float)
        latent_hash = str(a['latent_hash'])
    if density_path is not None:
        with np.load(density_path) as a:
            assert list(a['codes']) == CODES and str(a['latent_hash']) == latent_hash
            densities = a['density'].astype(float)
    assert distances.shape == densities.shape == (8, 1024)
    assert np.isfinite(distances).all() and np.isfinite(densities).all()
    cube = read(cube_path)
    assert cube['row']['id'] == row['id'] and cube['latent_hash'] == latent_hash
    provenance = dict(measurement=str(path.relative_to(ROOT)), measurement_sha256=digest(path),
                      density_override=str(density_path.relative_to(ROOT)) if density_path else None,
                      density_override_sha256=digest(density_path) if density_path else None,
                      cube=str(cube_path.relative_to(ROOT)), cube_sha256=digest(cube_path),
                      latent_hash=latent_hash, density_k=50)
    return distances, densities, provenance


def contrast_values(values):
    """values: [..., 8]; apply nonlinear minimum within each bootstrap draw."""
    result = {name: sum(weight * values[..., CODES.index(c)] for c, weight in weights.items())
              for name, weights in LINEAR.items()}
    result['minimum_reversion'] = np.minimum.reduce([result[f'{c}_restoration'] for c in ('QK', 'VO', 'MLP')])
    return result


def summary(value, draws):
    good = np.isfinite(draws)
    fraction = float(good.mean())
    interval = np.quantile(draws[good], [.025, .975]) if fraction >= .95 and np.isfinite(value) else [None, None]
    return dict(value=float(value) if np.isfinite(value) else None,
                ci95_low=interval[0], ci95_high=interval[1], bootstrap_valid_fraction=fraction)


def bootstrap(distances, densities, geometry=None):
    """Shared latent resamples; normalization and nonlinear contrasts recomputed."""
    n = distances.shape[1]
    rng = np.random.default_rng(20260911)
    indices = rng.integers(0, n, (2000, n))
    def means(array):
        return np.concatenate([array.mean(1)[None], np.stack([x[indices].mean(1) for x in array], axis=1)])
    e, d = means(distances), means(densities)
    def normalized(array, denominator, guard, recovery=False):
        result = np.full_like(array, np.nan)
        good = denominator >= guard
        result[good] = array[good] / denominator[good, None]
        return 1 - result if recovery else result
    scores = {'E': e, 'D_raw': d, 'P': normalized(e, e[:, 0], .01, True),
              'D_rel': normalized(d, d[:, 7], .01)}
    if geometry is not None:
        g = means(geometry)
        scores.update(E_structure=g, G=normalized(g, g[:, 0], 1e-6, True))
    states, contrasts = [], []
    for metric, values in scores.items():
        for i, state in enumerate(CODES):
            states.append(dict(metric=metric, state=state, **summary(values[0, i], values[1:, i])))
        if metric not in ('P', 'D_rel', 'G'):
            continue
        for name, values in contrast_values(values).items():
            contrasts.append(dict(metric=metric, contrast=name, **summary(values[0], values[1:])))
    return states, contrasts
