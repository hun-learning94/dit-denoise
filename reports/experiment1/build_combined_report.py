"""Build the unified Experiment 1 report from unique runs and finalized metrics."""
from pathlib import Path
import csv
import json
import math
import re
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reports.experiment1.build_report_assets64 import mse_at
from reports.experiment1.build_v2_report import GRID, read_metrics, label
HERE = Path(__file__).resolve().parent
FIG = HERE / 'figures_combined'
METRICS = [('vp_velocity_loss', 'vp_velocity_loss', 'Held-out VP velocity loss'),
           ('denoising', 'log_sigma_auc_mse_raw', 'Denoising AUC'),
           ('fid50k', 'fid', 'FID-50k')]
from reports.experiment1.plot_style import COLORS, HEAD_MARKERS, head_count, width_linestyle, efficiency_legend, trajectory_layout, trajectory_legend, EFFICIENCY_METRICS, efficiency_layout, add_efficiency_legend


def display_name(name):
    """Paper notation for display; run IDs and baseline data labels stay intact."""
    name = name.replace('Original DPD6', 'SMLP (ungrouped auxiliary)')
    name = name.replace('Grouped DPD6', 'DPD6').replace('gDPD6', 'DPD6')
    name = name.replace('DPD5', 'SCA').replace('DPD6', 'SMLP').replace('Learned PLD', 'LPLD')
    name = re.sub(r'F(\d+)', lambda m: '$R='+m[1]+'$', name)
    name = re.sub(r'H(\d+)', lambda m: r'$H='+m[1]+'$', name)
    name = re.sub(r'\s+\bd\d+\b', '', name)  # Derived from H in the active grid.
    name = re.sub(r'W(\d+)', lambda m: "$d_h'="+m[1]+'$', name)
    return name


def inventory():
    # The fixed-width manifest is the sole source of active baselines.
    # Do not merge the historical H14/H18 grid or its DiT F1280 control.
    models = {}
    for r in GRID:
        run = r['run_dir']
        family = {'dit': 'DiT', 'dpd5': 'DPD5', 'grouped_dpd6': 'DPD6'}[r['model']]
        if family == 'DiT':
            assert r['width'] in (640, 1024, 1792)
        else:
            assert r['H'] in (1, 8) and r['H'] * r['d'] == 256
        name = label(r) if family == 'DiT' else f'{family} H{r["H"]} d{r["d"]} W{r["width"]}'
        assert run not in models
        models[run] = dict(run_dir=run, model=name, family=family, parameters=r['parameters'],
                           grids=['fixed-width'], data=read_metrics(r))
    assert len(models) == 15
    order = list(COLORS)
    result = sorted(models.values(), key=lambda r: (order.index(r['family']), r['parameters'], r['model']))
    for r in result:
        h = head_count(r['model'])
        r['marker'] = HEAD_MARKERS[h]
        r['efficiency_marker'] = HEAD_MARKERS[h]
        r['efficiency_label'] = r['family']
        r['linestyle'] = width_linestyle(r['model'], r['family'])
        assert (ROOT / r['run_dir'] / 'checkpoints/ckpt_step2034400.pt').is_file()
        for e in range(25, 401, 25):
            for kind, key, _ in METRICS:
                obj = r['data'][e][kind]
                assert math.isfinite(float(obj[key]))
                if kind == 'fid50k': assert obj['generated_images'] == 50000
    return result


def save(fig, stem, models):
    trajectory = stem == 'metric_trajectories'
    if trajectory:
        trajectory_legend(fig, [(display_name(r['model']), r['family']) for r in models])
    else:
        add_efficiency_legend(fig, [r['family'] for r in models])
    for ext in ('png', 'pdf'): fig.savefig(FIG / f'{stem}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    models = inventory(); trajectory = []; endpoints = []
    for r in models:
        for e in range(25, 401, 25):
            for kind, key, _ in METRICS:
                trajectory.append(dict(model=r['model'], run_dir=r['run_dir'], epoch=e,
                                       metric=kind, value=r['data'][e][kind][key]))
        d, v, f = (r['data'][400][k] for k in ('denoising', 'vp_velocity_loss', 'fid50k'))
        endpoints.append(dict(model=r['model'], family=r['family'], parameters=r['parameters'],
                              grids=' + '.join(r['grids']), run_dir=r['run_dir'], epoch=400,
                              mse_low=mse_at(d,.1), mse_mid=mse_at(d,1), mse_high=mse_at(d,4),
                              denoising_auc=d['log_sigma_auc_mse_raw'], vp_velocity_loss=v['vp_velocity_loss'], fid=f['fid']))
    for filename, rows in [('metric_trajectories_combined.csv', trajectory), ('epoch400_metrics_combined.csv', endpoints)]:
        with (HERE / filename).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    render(models, endpoints)
    manifest = [{k:v for k,v in r.items() if k != 'data'} for r in models]
    (HERE/'combined_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # The maintained report includes the evolving corrected PLD panel.
    # This legacy builder only refreshes baseline assets and never rewrites it.
    return


def render(models, endpoints):
    """Render baseline figures using already selected metric values."""
    FIG.mkdir(exist_ok=True)
    efficiency_pairs = list(zip(models, endpoints))
    assert len(efficiency_pairs) == 15
    plot_models = [r for r, _ in efficiency_pairs]
    fig, axes = trajectory_layout()
    for ax, (kind, key, title) in zip(axes, METRICS):
        for r in plot_models:
            ax.plot(range(25,401,25), [r['data'][e][kind][key] for e in range(25,401,25)],
                    color=COLORS[r['family']], marker=r['marker'], ls=r['linestyle'], lw=1.6, ms=4, alpha=.9)
        ax.set_ylabel(title, fontsize=15)
    save(fig, 'metric_trajectories', plot_models)
    fig, axes = efficiency_layout()
    for ax, (key, _) in zip(axes, EFFICIENCY_METRICS):
        for r, point in efficiency_pairs:
            ax.scatter(point['parameters']/1e6, point[key], color=COLORS[r['family']],
                       marker=r['efficiency_marker'], s=65)
    save(fig, 'parameter_efficiency_epoch400', [r for r, _ in efficiency_pairs])


if __name__ == '__main__': main()
