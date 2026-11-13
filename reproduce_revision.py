"""Re-render revised-paper numerical figures and tables on CPU."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'revision/data'
COLORS = {'QK': '#0072B2', 'MLP': '#E69F00', 'VO+MLP': '#D55E00'}
MARKERS = {'uvit64': 'o', 'dit128': 's', 'lfm256': '^'}


def rows(name):
    with (DATA / name).open() as stream:
        return list(csv.DictReader(stream))


def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
        'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    return plt


def save(fig, out, name):
    for suffix in ['pdf', 'png']:
        fig.savefig(out / f'{name}.{suffix}', dpi=220, bbox_inches='tight')


def locality(out):
    plt = setup()
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.5))
    for model, color in [('jit_b', '#0072B2'), ('jit_l', '#D55E00')]:
        with np.load(DATA / f'{model}_locality.npz') as z:
            sigma = z['sigmas']
            excess = z['mean_clean_mse'] - z['dense_clean_mse'][:, None]
            envelope = np.maximum.accumulate(excess[:, ::-1], axis=1)[:, ::-1]
            needed = np.argmax(envelope <= .003, axis=1) + 1
            selected = sigma <= 32
            axes[0].plot(z['budgets'], excess[np.flatnonzero(sigma == 1)[0]], color=color)
            axes[1].plot(1 / (1 + sigma[selected]), needed[selected], color=color, label=model.replace('_', '-').replace('jit', 'JiT').upper().replace('JIT', 'JiT'))
    axes[0].set(xscale='log', xlabel='Retained patches k', ylabel='Excess denoising MSE', xlim=(1, 256), title='t = 0.5')
    axes[0].set_xticks([1, 16, 256], ['1', '16', '256'])
    axes[1].set(xlim=(1, 1 / 33), ylim=(0, 256), xlabel='Time t', ylabel='Required context k', title='Excess MSE tolerance 0.003')
    axes[1].set_xticks([1, .75, .5, .25, 1 / 33], ['1', '.75', '.5', '.25', '.03'])
    axes[1].set_yticks([0, 128, 256])
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save(fig, out, 'locality_context')
    plt.close(fig)


def attention(out):
    plt = setup()
    data = rows('attention_curves.csv')
    fig, axes = plt.subplots(2, 2, figsize=(7, 4.8))
    methods = {'donor_current': 'Current attention', 'donor_average': 'Average attention',
        'kamb_geometry': 'Distance', 'donor_sensitivity': 'Gradient', 'wiener': 'Covariance'}
    datasets = sorted({r['dataset'] for r in data})
    for i, dataset in enumerate(datasets):
        recipients = sorted({r['recipient'] for r in data if r['dataset'] == dataset})
        recipient = recipients[0]
        for j, t in enumerate([.1, .9]):
            ax = axes[i, j]
            selected = [r for r in data if r['dataset'] == dataset and r['recipient'] == recipient and np.isclose(float(r['t']), t)]
            for method in dict.fromkeys(r['method'] for r in selected):
                points = sorted([r for r in selected if r['method'] == method], key=lambda r: float(r['budget']))
                ax.plot([float(r['budget']) for r in points], [float(r['excess']) for r in points], label=methods.get(method, method.replace('_', ' ')))
            ax.set(title=f'{dataset}: {recipient}, t={t}', xlabel='Retained patches', ylabel='Excess denoising MSE')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, .12, 1, 1))
    save(fig, out, 'attention_context_curves')
    plt.close(fig)


def predictors(out):
    plt = setup()
    data = rows('small_predictors.csv')
    colors = {'MLP': '#0072B2', 'SMLP': '#D55E00', 'SCA': '#009E73', 'LPLD': '#CC79A7'}
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.9))
    for ax, metric, label in zip(axes, ['vp_velocity_loss', 'denoising_auc', 'fid'], ['Velocity loss', 'Denoising error', 'FID-50k']):
        for family, color in colors.items():
            for heads in ([1] if family == 'MLP' else [1, 8]):
                points = sorted([r for r in data if r['family'] == family and int(r['heads']) == heads], key=lambda r: int(r['parameters']))
                x = [int(r['parameters']) / 1e6 for r in points]
                y = [float(r[metric]) for r in points]
                ax.plot(x, y, color=color, linestyle='-' if heads == 8 or family == 'MLP' else ':',
                        marker='^' if heads == 8 else 'o', label=family + (f' H={heads}' if family != 'MLP' else ''))
                if metric == 'fid':
                    ax.errorbar(x, y, yerr=[[v - float(r['fid_min']) for v, r in zip(y, points)],
                        [float(r['fid_max']) - v for v, r in zip(y, points)]], fmt='none', color=color, capsize=2)
                    for xx, r in zip(x, points):
                        if r['recovery'] == 'True':
                            ax.annotate('†', (xx, float(r['fid_max'])), xytext=(3, 3), textcoords='offset points', color=color)
        ax.set(xlabel='Parameters (millions)', ylabel=label)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=4, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, .18, 1, 1))
    save(fig, out, 'small_predictors_1x3')
    plt.close(fig)


def distances(out, spatial=False, norm_matched=False):
    plt = setup()
    data = rows('norm_matched_dino_distances.csv' if norm_matched else
                ('spatial_dino_distances.csv' if spatial else 'dino_distances.csv'))
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5), sharex=True, sharey=True)
    for ax, target in zip(axes, ['aahq', 'afhq', 'ffhq']):
        for r in data:
            if r['target'] != target:
                continue
            name = {'100': 'QK', '001': 'MLP', '011': 'VO+MLP'}[r['bits']]
            ax.plot(float(r['source_mean']), float(r['target_mean' if spatial else 'joint_mean']), marker=MARKERS[r['family']],
                    color=COLORS[name], markerfacecolor='none', linestyle='none', markersize=6)
        ax.plot([0, 1.2], [0, 1.2], '--', color='.7', linewidth=.8)
        ax.scatter([0, 1], [1, 0], c=['#008B5A', '#B23A8A'], marker='X', s=35)
        ax.text(.03, 1.03, 'Source', fontsize=7, color='#008B5A')
        ax.text(.70, .03, 'Target', fontsize=7, color='#B23A8A')
        ax.set(xlim=(-.04, 1.2), ylim=(-.04, 1.2), xlabel='Distance to Source',
               title=f'{target.upper()} · epoch {50 if target == "ffhq" else 100}')
        ax.set_aspect('equal')
    axes[0].set_ylabel('Distance to Target')
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, label=name) for name, c in COLORS.items()]
    handles += [Line2D([], [], color='.4', marker=m, markerfacecolor='none', linestyle='none', label=name)
                for name, m in MARKERS.items()]
    fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, .12, 1, 1))
    save(fig, out, 'norm_matched_dino_distance' if norm_matched else
         ('spatial_dino_distance' if spatial else 'dino_distance'))
    plt.close(fig)


def metfaces(out):
    plt = setup()
    data = rows('metfaces_summary.csv')
    groups = ['qk', 'vo', 'mlp', 'vo_mlp', 'qk_vo_mlp', 'full']
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
    for ax, family in zip(axes, ['uvit', 'dit', 'lfm']):
        selected = [next(r for r in data if r['family'] == family and r['arm'] == arm) for arm in groups]
        ax.bar(np.arange(6), [1000 * float(r['kid_mean']) for r in selected],
               yerr=[1000 * float(r['kid_sd']) for r in selected], color=['#0072B2', '#009E73', '#E69F00', '#D55E00', '#CC79A7', '.6'], capsize=2)
        ax.set_xticks(np.arange(6), ['QK', 'VO', 'MLP', 'VO+MLP', 'All', 'Full'], rotation=50, ha='right')
        ax.set(title=f'{family.upper()}: ' + ('3 generation seeds' if family == 'lfm' else '3 training seeds'), ylabel='KID × 1,000')
    fig.tight_layout()
    save(fig, out, 'metfaces_kid')
    plt.close(fig)


def tables(out):
    chunks = []
    large = rows('large_predictors.csv')
    chunks.append('## Larger MLP/SMLP models\n\nDataset | MLP FID-50k | SMLP FID-50k\n---|---:|---:\n')
    for dataset in ['celeba128', 'imagenet128', 'imagenet256']:
        columns = []
        means = {}
        for arch in ['mlp', 'smlp']:
            values = [float(r['fid']) for r in large if r['dataset'] == dataset and r['arch'] == arch]
            assert len(values) == 3
            means[arch] = statistics.mean(values)
            columns.append(f'{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}')
        winner = 0 if means['mlp'] < means['smlp'] else 1
        columns[winner] = '**' + columns[winner] + '**'
        chunks.append(dataset + ' | ' + ' | '.join(columns) + '\n')
    chunks.append('\nOne training seed per model; SD across three generation seeds. ImageNet128 MLP has a documented FP32/lower-LR recovery.\n')
    chunks.append('\n## Spatial dataset MMD²\n\nNative view | Target | MMD² | Pointwise 95% interval\n---|---|---:|---:\n')
    for r in rows('spatial_dataset_mmd.csv'):
        if r['source'] == 'source' and r['target'] != 'source':
            chunks.append(f"{r['view']} | {r['target']} | {float(r['mmd2']):.5f} | [{float(r['ci95_low']):.5f}, {float(r['ci95_high']):.5f}]\n")
    chunks.append('\n## MetFaces at epoch 1,000\n\nFamily | Allocation | KID × 1,000 | FID | SD axis\n---|---|---:|---:|---\n')
    for r in rows('metfaces_summary.csv'):
        chunks.append(f"{r['family']} | {r['arm']} | {1000*float(r['kid_mean']):.2f} ± {1000*float(r['kid_sd']):.2f} | {float(r['fid_mean']):.2f} ± {float(r['fid_sd']):.2f} | {r['sd_axis']}\n")
    chunks.append('\n## Component transplantation performance\n\nBackbone | Target | Epoch | State | Source FID-10k | Target FID-10k | Source loss | Target loss\n---|---|---:|---|---:|---:|---:|---:\n')
    for r in rows('transplant_performance.csv'):
        chunks.append(' | '.join([r['backbone'], r['target'], r['epoch'], r['state'],
            *[f'{float(r[k]):.4f}' for k in ['source_fid10k', 'target_fid10k', 'source_loss', 'target_loss']]]) + '\n')
    (out / 'tables.md').write_text(''.join(chunks))


def check():
    assert len(rows('large_predictors.csv')) == 18
    assert len(rows('small_predictors.csv')) == 21
    for name in ['dino_distances.csv', 'spatial_dino_distances.csv']:
        data = rows(name)
        assert len(data) == 27
        assert len({(r['family'], r['target'], r['bits']) for r in data}) == 27
        assert all(int(r['epoch']) == (50 if r['target'] == 'ffhq' else 100) for r in data)
    summary = rows('metfaces_summary.csv')
    observations = rows('metfaces_per_seed.csv')
    for r in summary:
        values = [x for x in observations if x['family'] == r['family'] and x['arm'] == r['arm']]
        assert len(values) == 3
        for metric in ['kid', 'fid']:
            v = [float(x[metric]) for x in values]
            np.testing.assert_allclose([statistics.mean(v), statistics.stdev(v)], [float(r[metric + '_mean']), float(r[metric + '_sd'])], rtol=1e-10)
    for model in ['jit_b', 'jit_l']:
        with np.load(DATA / f'{model}_locality.npz') as z:
            assert len(z['sigmas']) == 14 and np.all(np.isfinite(z['mean_clean_mse']))
            np.testing.assert_array_equal(z['mean_clean_mse'][:, -1], z['dense_clean_mse'])
    print('Verified revised numerical inputs, paired settings, and seed aggregation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('asset', choices=['all', 'locality', 'attention', 'predictors', 'transplants', 'metfaces', 'tables', 'schematics', 'check'])
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/revision')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    check()
    selected = ['locality', 'attention', 'predictors', 'transplants', 'metfaces', 'tables', 'schematics'] if args.asset == 'all' else [args.asset]
    for name in selected:
        if name == 'transplants':
            distances(args.output)
            distances(args.output, spatial=True)
            distances(args.output, norm_matched=True)
        elif name == 'schematics':
            for script in ['overview.py', 'architecture.py']:
                subprocess.run([sys.executable, '-m', 'revision.figures.' + Path(script).stem], check=True, cwd=ROOT)
        elif name != 'check':
            globals()[name](args.output)
    (args.output / 'reproduction.json').write_text(json.dumps(dict(
        assets=selected, inference_performed=False, inputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.iterdir() if p.is_file()}), indent=2) + '\n')
    print(f'Wrote revised-paper artifacts to {args.output}')


if __name__ == '__main__':
    main()
