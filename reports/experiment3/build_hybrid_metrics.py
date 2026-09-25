"""Build the selected image/metric reporting view without editing the summary."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from notation import display_text, style_figure

from hybrid_analysis import (HERE, ROOT, PAIRS, CODES, LINEAR, CONTRAST_NAMES,
                            bootstrap, measurements, read, selected_rows, digest)

from association_analysis import CubeAssociations, DISPLAY_CODES

OUT = HERE/'metrics'
NAMES = {'celeba64': 'CelebA', 'aahq64': 'AAHQ', 'ffhq64': 'FFHQ',
         'afhq_v2_64': 'AFHQ-v2', 'stl10_64': 'STL-10', 'imagenet64': 'ImageNet64'}
COLORS = ['#1768ac', '#c16c16']
METRICS = [('P', 'Endpoint agreement P'),
           ('G', 'Relative spatial endpoint agreement G')]


def title(pair):
    return ' → '.join(NAMES[p] for p in pair)


def save(fig, stem):
    style_figure(fig)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT/f'{stem}.{ext}', dpi=165, bbox_inches='tight')
    plt.close(fig)


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def group_rows(ax, labels=True, source_epochs=True):
    def label(pair):
        epoch = (800 if (HERE/'inputs/imagenet800_endpoints.json').exists() else 600) if pair[0] == 'imagenet64' else 800 if pair[0] == 'stl10_64' else None
        return title(pair) + (f' (source{epoch})' if source_epochs and epoch else '')
    ax.set_yticks(range(12), [label(p) for p in PAIRS] if labels else [])
    ax.set_ylim(11.6, -.6)
    for split in (3.5, 7.5):
        ax.axhline(split, color='#c7c7c7', lw=.9)
    ax.tick_params(axis='y', labelsize=9, length=0)
    ax.spines[['top', 'right']].set_visible(False)


def state_heatmap(states):
    def mean(pair, metric, state):
        vals = [r['value'] for r in states if (r['source'], r['target'], r['metric'], r['state']) == (*pair, metric, state)]
        assert len(vals) == 2
        return np.nan if any(v is None for v in vals) else float(np.mean(vals))
    for settings, stem in [(METRICS, 'cube_metrics_20m')]:
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 7.2), layout='constrained')
        for j, (ax, (metric, label)) in enumerate(zip(axes, settings)):
            values = np.array([[mean(p, metric, c) for c in DISPLAY_CODES] for p in PAIRS])
            vmax = max(1, np.nanmax(values))
            im = ax.imshow(values, aspect='auto', cmap='cividis', vmin=min(0, np.nanmin(values)), vmax=vmax)
            ax.set_xticks(range(8), DISPLAY_CODES, fontsize=11)
            group_rows(ax, labels=j == 0, source_epochs=False)
            ax.tick_params(axis='y', labelsize=11.5)
            ax.set_title('Endpoint agreement (EA)' if metric == 'P' else 'Spatial agreement (SA)', fontsize=13)
            ax.set_xlabel('State (QK / VO / MLP)', fontsize=11)
            for (y, x), v in np.ndenumerate(values):
                label = f'{0.0 if abs(v) < .005 else v:.2f}' if np.isfinite(v) else 'N/A'
                ax.text(x, y, label, ha='center', va='center', fontsize=11.5,
                        color='black' if not np.isfinite(v) or im.norm(v) > .6 else 'white')
            colorbar = fig.colorbar(im, ax=ax, shrink=.76, fraction=.03, pad=.015)
            colorbar.ax.tick_params(labelsize=11)
        style_figure(fig)
        fig.savefig(ROOT/'draft/figs/experiment3_cube_agreement.pdf', bbox_inches='tight')
        save(fig, stem)
    return mean


def dataset_figures(result, main_only=False):
    domains = list(result['domain_summaries'])
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout='constrained')
    for ax, variant, label in zip(axes, ['spatial_normalized', 'raw_tokens'], ['Spatially normalized patches', 'Raw patches (sensitivity check)']):
        values = np.zeros((len(domains), len(domains)))
        for r in result['rows']:
            if r['variant'] != variant: continue
            i, j = domains.index(r['source']), domains.index(r['target'])
            values[i, j] = values[j, i] = r['mmd2']
        im = ax.imshow(values, cmap='magma_r', vmin=min(0, values.min()), vmax=values.max())
        ax.set_xticks(range(len(domains)), [NAMES[d] for d in domains], rotation=45, ha='right')
        ax.set_yticks(range(len(domains)), [NAMES[d] for d in domains]); ax.set_title(label)
        for (i, j), v in np.ndenumerate(values):
            ax.text(j, i, f'{v:.4f}', ha='center', va='center', fontsize=8, color='white' if im.norm(v) > .55 else 'black')
        fig.colorbar(im, ax=ax, shrink=.7, label='Spatial MMD² ↓')
    fig.suptitle('Real datasets · spatial self-similarity distributions\nDiagonal: independent within-dataset halves')
    save(fig, 'dataset_geometry_20m')
    if main_only:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), layout='constrained')
    for ax, variant, label in zip(axes, ['spatial_normalized', 'raw_tokens'], ['Spatially normalized patches', 'Raw patches (sensitivity check)']):
        for y, pair in enumerate(PAIRS):
            row = next(r for r in result['rows'] if r['variant'] == variant and {r['source'], r['target']} == set(pair))
            ax.plot([row['ci95_low'], row['ci95_high']], [y, y], color=COLORS[0])
            ax.plot(row['mmd2'], y, 'o', color=COLORS[0], markersize=4)
        group_rows(ax, labels=ax is axes[0], source_epochs=False); ax.set_title(label)
        ax.axvline(0, color='gray', ls='--', lw=.8); ax.set_xlabel('Spatial MMD² ↓ (descriptive 95% image-bootstrap interval)')
    save(fig, 'dataset_geometry_pairs')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout='constrained')
    for domain, values in result['domain_summaries'].items():
        axes[0].plot(range(1, 31), values['correlogram'], label=NAMES[domain])
    axes[0].set(xlabel='Manhattan patch distance', ylabel='Mean raw patch cosine similarity', title='Spatial correlograms'); axes[0].legend(fontsize=8)
    names = ['brightness', 'saturation', 'shift4', 'shift8']
    for i, domain in enumerate(domains):
        vals = result['validation'][domain]['medians']
        axes[1].plot(range(4), [vals[k] for k in names], 'o-', label=NAMES[domain], markersize=3)
    axes[1].set_xticks(range(4), ['Brightness\n×0.8', 'Saturation\n×0.5', 'Shift\n4 px', 'Shift\n8 px'])
    axes[1].set(ylabel='Median structural descriptor MSE', title='Appearance and spatial perturbation controls')
    save(fig, 'dataset_geometry_validation')


def paper_dataset_geometry_figure(result):
    """Export the normalized-distance heatmap at paper-readable scale."""
    domains = list(result['domain_summaries'])
    values = np.zeros((len(domains), len(domains)))
    for row in result['rows']:
        if row['variant'] != 'spatial_normalized':
            continue
        i, j = domains.index(row['source']), domains.index(row['target'])
        values[i, j] = values[j, i] = row['mmd2'] * 100
    fig, ax = plt.subplots(figsize=(9.5, 4.4), layout='constrained')
    im = ax.imshow(values, aspect='auto', cmap='magma_r', vmin=min(0, values.min()), vmax=values.max())
    ax.set_xticks(range(len(domains)), [NAMES[d] for d in domains], fontsize=12)
    ax.set_yticks(range(len(domains)), [NAMES[d] for d in domains], fontsize=12.5)
    for (i, j), value in np.ndenumerate(values):
        ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=12.5,
                color='white' if im.norm(value) > .55 else 'black')
    colorbar = fig.colorbar(im, ax=ax, shrink=.85)
    colorbar.set_label(r'Spatial MMD$^2$ ($10^{-2}$ units)', fontsize=12)
    colorbar.ax.tick_params(labelsize=12)
    style_figure(fig)
    fig.savefig(ROOT/'draft/figs/experiment3_dataset_geometry.pdf', bbox_inches='tight')
    fig.savefig(ROOT/'draft/figs/experiment3_dataset_geometry.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def compute_statistics():
    rows = selected_rows(); states, contrasts, provenance = [], [], {}
    for first in rows[::32]:
        e, d, source = measurements(first)
        geometry_path = OUT/'geometry_results'/f"{first['id']}.npz"
        spatial_meta = read(geometry_path.with_suffix('.json'))
        assert spatial_meta['id'] == first['id'] and spatial_meta['cube_sha256'] == source['cube_sha256']
        assert spatial_meta['measurement_sha256'] == digest(geometry_path)
        assert spatial_meta['dataset_manifest_sha256'] == read(OUT/'dataset_geometry.json')['manifest_sha256']
        with np.load(geometry_path) as a:
            assert list(a['codes']) == CODES and str(a['latent_hash']) == source['latent_hash']
            spatial, edge_data = a['distances'], a['qk_edge_distances']
        assert spatial.shape == (8, 1024) and edge_data.shape == (4, 1024)
        assert np.isfinite(spatial).all() and np.isfinite(edge_data).all()
        assert np.all(spatial >= 0) and np.all(edge_data >= 0) and np.all(spatial[7] == 0)
        np.testing.assert_array_equal(spatial[6], edge_data[3])
        current_states, current_contrasts = bootstrap(e, d, geometry=spatial)
        current_states = [r for r in current_states if r['metric'] in ('P', 'G')]
        current_contrasts = [r for r in current_contrasts if r['metric'] in ('P', 'G')]
        meta = {k: first[k] for k in ('id', 'scale', 'source', 'target', 'adaptation_seed', 'selected_epoch')}
        old = {(r['metric'], r['state']): r for r in rows if r['id'] == first['id']}
        for r in current_states:
            if r['metric'] == 'G': continue
            expected = old[r['metric'], r['state']]['value']
            assert (r['value'] is None and not expected) or np.isclose(r['value'], float(expected), atol=1e-10, rtol=1e-10), (meta, r)
        states.extend(dict(**meta, **r) for r in current_states)
        contrasts.extend(dict(**meta, **r) for r in current_contrasts)
        for unused in ('density_k', 'density_override', 'density_override_sha256'):
            source.pop(unused, None)
        source['geometry'] = dict(measurement=str(geometry_path.relative_to(HERE)),
                                  measurement_sha256=digest(geometry_path), provenance=spatial_meta)
        provenance[first['id']] = source
        print('Computed paired statistics:', first['id'], flush=True)
    write_csv(OUT/'selected_metrics_by_seed.csv', states)
    write_csv(OUT/'selected_contrasts_by_seed.csv', contrasts)
    protocol = dict(directions=PAIRS, states=CODES, reported_scores=['P', 'G'], images_per_state=1024,
                    bootstrap_repeats=2000, bootstrap_seed=20260911,
                    normalization_guards=dict(P=.01, G=1e-6), minimum_valid_fraction=.95,
                    linear_contrasts=LINEAR, nonlinear_contrasts=dict(minimum_reversion='min(QK_restoration, VO_restoration, MLP_restoration), recomputed within each bootstrap resample'),
                    implementation_sha256=digest(HERE/'hybrid_analysis.py'), cubes=provenance,
                    geometry_stage='Real-dataset controls passed before applying the same descriptor to matched-latent generated cubes')
    (OUT/'selected_protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    return rows, states, contrasts, provenance


def build(reuse_statistics=False):
    if reuse_statistics:
        # A presentation-only refresh preserves saved measurements and intervals.
        rows = selected_rows()
        def saved(name):
            with (OUT/name).open() as f:
                entries = list(csv.DictReader(f))
            for r in entries:
                for key in ('value', 'ci95_low', 'ci95_high', 'bootstrap_valid_fraction'):
                    r[key] = float(r[key]) if r[key] else None
            return entries
        states = saved('selected_metrics_by_seed.csv')
        contrasts = saved('selected_contrasts_by_seed.csv')
        provenance = read(OUT/'selected_protocol.json')['cubes']
    else:
        rows, states, contrasts, provenance = compute_statistics()
    mean = state_heatmap(states)
    associations = CubeAssociations()
    associations.mmd_associations()
    associations.pairwise_comparison()
    geometry_path = OUT/'dataset_geometry.json'
    geometry = read(geometry_path) if geometry_path.exists() else None
    if geometry: dataset_figures(geometry)
    text = (HERE/'inputs/hybrid_metrics_template.md').read_text()
    bandwidth_table = '| Descriptor variant | Frozen bandwidth $\\sigma^2$ |\n|---|---:|\n'
    for variant, label in [('spatial_normalized', 'Spatially normalized patches'), ('raw_tokens', 'Raw patches (sensitivity check)')]:
        bandwidth_table += f'| {label} | {geometry["bandwidth_squared"][variant]:.8f} |\n'
    text = text.replace('CALIBRATION_BANDWIDTHS', bandwidth_table)
    table = '| Transfer | Endpoint reference epoch | Recovery $P_{011}$ | Spatial recovery $G_{011}$ |\n|---|---:|---:|---:|\n'
    for pair in PAIRS:
        epoch = next(r['selected_epoch'] for r in rows if (r['source'], r['target']) == pair)
        table += f'| {title(pair)} | {epoch} | {mean(pair, "P", "011"):.3f} | {mean(pair, "G", "011"):.3f} |\n'
    text = text.replace('ENDPOINT_TABLE', table)
    if geometry:
        validation = '| Dataset | LDS | CDS | RMSC | Spatial controls |\n|---|---:|---:|---:|---|\n'
        for domain, domain_summary in geometry['domain_summaries'].items():
            s = domain_summary['statistics']; status = 'Pass' if geometry['validation'][domain]['passed'] else 'Needs review'
            validation += f'| {NAMES[domain]} | {s["LDS"]:.4f} | {s["CDS"]:.4f} | {s["RMSC"]:.4f} | {status} |\n'
        geometry_text = ('![Real-dataset spatial distances](metrics/dataset_geometry_20m.png)\n\n'
                         '![Dataset pairs in transfer-report order](metrics/dataset_geometry_pairs.png)\n\n'
                         '![Spatial correlograms and perturbation controls](metrics/dataset_geometry_validation.png)\n\n'+validation+
                         '\n[Complete dataset statistics and validation](metrics/dataset_geometry.json)\n')
        def distance(a, b, variant='spatial_normalized'):
            return next(r['mmd2'] for r in geometry['rows'] if r['variant'] == variant and {r['source'], r['target']} == {a, b})
        geometry_text += (f'\nFor CelebA, spatially normalized MMD² is **{distance("celeba64", "aahq64"):.4f} to AAHQ**, '
                          f'**{distance("celeba64", "ffhq64"):.4f} to FFHQ**, '
                          f'**{distance("celeba64", "afhq_v2_64"):.4f} to AFHQ-v2**, and '
                          f'**{distance("celeba64", "stl10_64"):.4f} to STL-10**. '
                          'The smaller distances to the two human-face datasets are also present with raw patch similarities; '
                          'the fine ordering of AAHQ and FFHQ changes under normalization. '
                          'This supports a broad dataset-structure distinction under the fixed descriptor, '
                          'without establishing that their geometry is identical or that QK causes the distinction.\n')
        if not geometry['validated']:
            geometry_text += '\n**Validation is incomplete:** interpret these as exploratory descriptor differences; at least one domain fails the predefined appearance/spatial sensitivity checks.\n'
    else:
        geometry_text = 'The real-dataset spatial measurement is being computed from the frozen image banks. No generated-image geometry conclusion is drawn before these checks finish.\n'
    text = text.replace('DATASET_RESULTS', geometry_text)
    # The common formatter retains source600 until the replacement is validated.
    import build_reports
    text = build_reports.imagenet_report_text(text)
    (HERE/'experiment3_metrics.md').write_text(display_text(text))
    validation = dict(directions=12, cubes=24, state_rows=len(states), contrast_rows=len(contrasts),
                      contrasts_per_score=len(CONTRAST_NAMES), point_estimates_match_existing_exports=True,
                      row_order=PAIRS, all_intervals_valid=all(r['bootstrap_valid_fraction'] >= .95 for r in states+contrasts),
                      dataset_geometry_complete=geometry is not None,
                      dataset_geometry_validated=bool(geometry and geometry['validated']),
                      spatial_cubes=len(provenance), reported_scores=['P', 'G'])
    (OUT/'selected_validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    print('Selected reports:', validation, flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-statistics', action='store_true', help='Render from existing selected P/G tables without recomputing intervals')
    parser.add_argument('--paper-dataset-figure', action='store_true', help='Export only the paper spatial-MMD heatmap')
    args = parser.parse_args()
    if args.paper_dataset_figure:
        paper_dataset_geometry_figure(read(OUT/'dataset_geometry.json'))
    else:
        build(reuse_statistics=args.reuse_statistics)
