"""Descriptive MMD associations from saved, seed-specific P/G cube scores.

No inference of independent transfers: shared datasets, pretraining, and score
normalization make these correlations descriptive. Keep array order immutable.
"""
import csv
from itertools import combinations
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import numpy as np
from scipy.stats import spearmanr
from notation import display_text, style_figure

from hybrid_analysis import HERE, PAIRS, CODES, digest

DISPLAY_CODES = ['000', '100', '010', '110', '001', '101', '011', '111']
NAMES = {'celeba64': 'CelebA', 'aahq64': 'AAHQ', 'ffhq64': 'FFHQ',
         'afhq_v2_64': 'AFHQ-v2', 'stl10_64': 'STL-10', 'imagenet64': 'ImageNet64'}
SOURCES = ('celeba64', 'stl10_64', 'imagenet64')
COLORS = dict(zip(SOURCES, ('#197a9b', '#b65b29', '#76549c')))
QK = [('QK_only_minus_source', 'QK effect · source VO, MLP\n100 − 000'),
      ('QK_given_VO', 'QK effect · adapted VO\n110 − 010'),
      ('QK_given_MLP', 'QK effect · adapted MLP\n101 − 001'),
      ('QK_restoration', 'QK reversion penalty · adapted VO, MLP\n111 − 011')]
COORDINATION = [('joint_nonadditivity', 'Joint non-additivity S'),
                ('minimum_reversion', 'Minimum reversion penalty Rmin'),
                ('three_way_interaction', 'Three-way interaction I3')]
PAIR_INTERACTIONS = [(f'{pair}_interaction_context{c}',
                      f'{pair.replace("_", "–")} interaction · {other} = {c}')
                     for pair, other in [('QK_VO', 'MLP'), ('QK_MLP', 'VO'), ('VO_MLP', 'QK')]
                     for c in (0, 1)]
LABELS = dict(QK + COORDINATION + PAIR_INTERACTIONS + [
    ('VO_restoration', 'VO reversion penalty R(VO)'),
    ('MLP_restoration', 'MLP reversion penalty R(MLP)'),
    ('VO_given_MLP', 'VO benefit with adapted MLP · 011 − 001')])

# Keep one representative of each normalized endpoint-complement pair.
# For every mapping below, the omitted difference is 1 minus its representative.
STATE_DIFFERENCES = [((b, a), f'State difference · {b} − {a}') for a, b in combinations(CODES, 2)]
SUMMARY_EQUIVALENTS = {
    ('111', '100'): ('100', '000'),
    ('111', '010'): ('010', '000'),
    ('111', '001'): ('001', '000'),
    ('110', '000'): ('111', '110'),
    ('101', '000'): ('111', '101'),
    ('011', '000'): ('111', '011'),
}


def nested_difference(key):
    """One state's adapted-component set is a subset of the other's."""
    a, b = key
    return all(x <= y for x, y in zip(a, b)) or all(y <= x for x, y in zip(a, b))


SUMMARY_STATE_DIFFERENCES = [(key, desc) for key, desc in STATE_DIFFERENCES
                             if nested_difference(key) and key not in SUMMARY_EQUIVALENTS
                             and key != ('111', '000')]
SUMMARY_USES = {key: ('mmd_comparison', '4.1', 'Additional MMD correlation table', 'P,G')
                for key, _ in SUMMARY_STATE_DIFFERENCES}
SUMMARY_USES['111', '011'] = ('mmd_evidence', '4.1', 'Primary metric: QK reversion penalty', 'P,G')
SUMMARY_STYLES = {
    'mmd_evidence': ('#246b3d', 'Summary: primary metric'),
    'mmd_comparison': ('#365e8a', 'Summary: comparison table'),
}
HIGH_CORRELATION_THRESHOLD = .70


def consistently_high(p, g):
    return bool(p is not None and g is not None and p*g > 0
                and min(abs(p), abs(g)) >= HIGH_CORRELATION_THRESHOLD)


def summary_use(key, metric):
    role, section, description, scores = SUMMARY_USES.get(key, ('comparison_only', '', '', ''))
    if metric not in scores.split(','):
        role, section, description = 'comparison_only', '', ''
    return dict(summary_role=role, summary_section=section, summary_use=description,
                summary_included=role in SUMMARY_STYLES,
                summary_evidence=role == 'mmd_evidence',
                summary_mmd_evidence=role == 'mmd_evidence')


def read_csv(name):
    with (HERE / 'metrics' / name).open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def correlation(x, y):
    if np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12:
        return None
    return float(spearmanr(x, y).statistic)


def save(fig, directory, name):
    style_figure(fig)
    for ext in ('png', 'pdf'):
        fig.savefig(directory / f'{name}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


class CubeAssociations:
    def __init__(self):
        self.states = read_csv('selected_metrics_by_seed.csv')
        self.contrasts = read_csv('selected_contrasts_by_seed.csv')
        self.geometry = json.loads((HERE / 'metrics/dataset_geometry.json').read_text())
        self.state_index = {((r['source'], r['target']), r['metric'], r['state'], int(r['adaptation_seed'])):
                            float(r['value']) for r in self.states}
        self.contrast_index = {((r['source'], r['target']), r['metric'], r['contrast'], int(r['adaptation_seed'])):
                               float(r['value']) for r in self.contrasts}
        assert len(self.state_index) == 12 * 2 * 8 * 2
        assert self.geometry['validated']

    def value(self, pair, metric, key, seed=None):
        if seed is None:
            return float(np.mean([self.value(pair, metric, key, s) for s in (0, 1)]))
        if isinstance(key, tuple):
            a, b = key
            return self.value(pair, metric, a, seed) - self.value(pair, metric, b, seed)
        index = self.state_index if key in CODES else self.contrast_index
        return index[pair, metric, key, seed]

    def vector(self, metric, key, seed=None):
        return np.array([self.value(pair, metric, key, seed) for pair in PAIRS])

    def mmd(self, pair, variant='spatial_normalized'):
        return next(r['mmd2'] for r in self.geometry['rows']
                    if r['variant'] == variant and {r['source'], r['target']} == set(pair))

    def rho(self, metric, key, seed=None, dataset=False):
        x = [self.mmd(pair) for pair in PAIRS] if dataset else self.vector(metric, 'joint_nonadditivity', seed)
        return correlation(x, self.vector(metric, key, seed))

    def scatter(self, ax, metric, key, dataset=False, ylabel=None):
        x = np.array([self.mmd(pair) for pair in PAIRS]) if dataset else self.vector(metric, 'joint_nonadditivity')
        y = self.vector(metric, key)
        for i, pair in enumerate(PAIRS):
            color = COLORS[pair[0]]
            sx = [x[i]]*2 if dataset else [self.value(pair, metric, 'joint_nonadditivity', s) for s in (0, 1)]
            sy = [self.value(pair, metric, key, s) for s in (0, 1)]
            ax.plot(sx, sy, color=color, alpha=.35, lw=1, marker='.', ms=5, zorder=2)
            ax.scatter(x[i], y[i], s=43, color=color, edgecolors='white', linewidths=.5, zorder=3)
            ax.annotate(str(i+1), (x[i], y[i]), xytext=(5, 7 if i % 2 == 0 else -12),
                        textcoords='offset points', fontsize=9, color=color, weight='bold')
        ax.set_xlabel('Real-dataset spatial MMD²' if dataset else 'Joint non-additivity S')
        ax.set_ylabel(ylabel or LABELS[key])
        rho = self.rho(metric, key, dataset=dataset)
        ax.set_title(f'{metric} · Spearman ρ = {rho:.2f}' if rho is not None else f'{metric} · constant score; ρ undefined')
        if y.min() < 0 < y.max():
            ax.axhline(0, color='#888888', ls=':', lw=.7)
        ax.grid(alpha=.18); ax.margins(x=.16, y=.2)

    @staticmethod
    def place_labels(fig, axes):
        """Keep point IDs readable as endpoint values move the points."""
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in np.asarray(axes).flat:
            occupied = []
            for collection in ax.collections:
                for xy in ax.transData.transform(collection.get_offsets()):
                    occupied.append(Bbox.from_bounds(xy[0]-6, xy[1]-6, 12, 12))
            for annotation in ax.texts:
                candidates = [annotation.get_position(), (6, 8), (6, -13), (-18, 8), (-18, -13),
                              (0, 17), (0, -23), (17, 0), (-27, 0), (18, 16), (-28, 16)]
                scored = []
                for offset in candidates:
                    annotation.set_position(offset)
                    box = annotation.get_window_extent(renderer).expanded(1.15, 1.15)
                    intersections = [Bbox.intersection(box, other) for other in occupied]
                    area = sum(b.width*b.height for b in intersections if b is not None)
                    inside = Bbox.intersection(box, ax.bbox)
                    area += box.width*box.height - (inside.width*inside.height if inside is not None else 0)
                    scored.append((area + .01*np.hypot(*offset), offset))
                annotation.set_position(min(scored, key=lambda item: item[0])[1])
                occupied.append(annotation.get_window_extent(renderer).expanded(1.15, 1.15))

    def scatter_grid(self, settings, title, dataset=True, mark_summary=False):
        count = len(settings)
        height = 3.4*count + (1.7 if mark_summary else 1.2)
        fig, axes = plt.subplots(count, 2, figsize=(11.8, height), squeeze=False)
        for row, (key, ylabel) in enumerate(settings):
            for col, metric in enumerate(('P', 'G')):
                ax = axes[row, col]
                self.scatter(ax, metric, key, dataset, ylabel)
                use = summary_use(key, metric)
                if mark_summary and use['summary_role'] in SUMMARY_STYLES:
                    color, badge = SUMMARY_STYLES[use['summary_role']]
                    ax.set_title(ax.get_title()+f"\n{badge} · §{use['summary_section']}", color=color, fontsize=10)
                    for spine in ax.spines.values():
                        spine.set_color(color); spine.set_linewidth(1.7)
        fig.subplots_adjust(top=1-(1.65 if mark_summary else .95)/height, bottom=.65/height,
                            left=.09, right=.98, wspace=.28, hspace=.48)
        heading = '20M · '+title+'\nP: endpoint agreement · G: relative spatial endpoint agreement'
        if mark_summary:
            heading += '\nGreen: summary primary metric · Blue: summary correlation-table comparison'
        fig.suptitle(heading, fontsize=11.5 if mark_summary else 13, y=.995)
        legend = [Line2D([], [], marker='o', ls='', color=COLORS[s], label=f'{NAMES[s]} source') for s in SOURCES]
        fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, 1-(.78 if mark_summary else .44)/height), ncol=3, frameon=False)
        self.place_labels(fig, axes)
        return fig

    def correlation_rows(self, settings, mark_summary=False):
        rows = []
        for key, description in settings:
            for metric in ('P', 'G'):
                rows.append(dict(metric=metric, effect='-'.join(key) if isinstance(key, tuple) else key,
                                 description=description.replace('\n', ' '),
                                 spearman_mean=self.rho(metric, key, dataset=True),
                                 spearman_seed0=self.rho(metric, key, 0, dataset=True),
                                 spearman_seed1=self.rho(metric, key, 1, dataset=True), directions=12))
                if mark_summary:
                    rows[-1].update(summary_use(key, metric))
                    rows[-1]['consistently_high_P_G'] = consistently_high(
                        self.rho('P', key, dataset=True), self.rho('G', key, dataset=True))
        return rows

    def mmd_associations(self):
        out = HERE / 'metrics'
        for settings, name, title in [(QK, 'mmd_qk_effects_20m', 'Spatial distance and all four QK effects'),
                                      (COORDINATION, 'mmd_coordination_20m', 'Spatial distance and component coordination')]:
            save(self.scatter_grid(settings, title), out, name)
        write_csv(out/'mmd_correlations.csv', self.correlation_rows(QK + COORDINATION))

    def pairwise_comparison(self):
        out = HERE / 'metrics/mmd_pairwise_comparison'
        out.mkdir(exist_ok=True)
        # A fixed, stated orientation includes every unordered pair exactly once.
        differences = STATE_DIFFERENCES
        pages = []
        with PdfPages(out/'all_pairwise_metrics_vs_mmd.pdf') as pdf:
            for settings, stem, title in [(differences, 'state_differences', 'All 28 state differences versus MMD²'),
                                          (PAIR_INTERACTIONS, 'component_interactions', 'All six component-pair interactions versus MMD²')]:
                for start in range(0, len(settings), 4):
                    batch = settings[start:start+4]
                    fig = self.scatter_grid(batch, f'{title} · {start+1}–{start+len(batch)}', mark_summary=True)
                    style_figure(fig)
                    pdf.savefig(fig, bbox_inches='tight')
                    name = f'{stem}_{start//4+1:02d}.png'
                    fig.savefig(out/name, dpi=165, bbox_inches='tight')
                    plt.close(fig)
                    pages.append((title, name))
        settings = differences + PAIR_INTERACTIONS
        write_csv(out/'correlations.csv', self.correlation_rows(settings, mark_summary=True))
        rows = []
        for key, desc in settings:
            for i, pair in enumerate(PAIRS):
                for metric in ('P', 'G'):
                    for seed in (0, 1):
                        rows.append(dict(number=i+1, source=pair[0], target=pair[1], metric=metric,
                                         effect='-'.join(key) if isinstance(key, tuple) else key,
                                         adaptation_seed=seed, mmd2=self.mmd(pair), value=self.value(pair, metric, key, seed)))
        write_csv(out/'values_by_seed.csv', rows)
        self.comparison_provenance()
        lines = ['# All pairwise cube metrics versus spatial MMD', '',
                 '[Complete nine-page PDF](all_pairwise_metrics_vs_mmd.pdf) · [Correlations](correlations.csv) · [Values by seed](values_by_seed.csv)', '',
                 'All 28 unordered pairs of cube states are included for both P and G. The difference is later minus earlier in the fixed analysis order `'+', '.join(CODES)+'`; reversing it changes only the sign of the correlation. All six component-pair interactions are included separately. These are endpoint-score differences, not direct distances between generated images.', '',
                 'Each numbered point averages the two separately normalized adaptation-seed scores; small connected points show the seeds. Spearman correlations use twelve direction means, with separate seed correlations in the CSV. MMD uses the frozen spatial-normalized descriptor. The 111 − 000 score is exactly 1, so its correlation is undefined.', '',
                 'These exhaustive comparisons are descriptive controls for specificity, not independent tests. Directions share datasets and source models, many metrics share cube terms, and the same symmetric MMD is reused for reciprocal directions. With P and G normalized to source 0 / joint 1, subtracting either endpoint gives algebraically related correlations. Comparing correlation magnitudes does not establish a statistically significant difference; no p-values or multiple-testing claims are made.', '',
                 '**Summary evidence markers.** Green titles and borders mark the sole primary metric in [summary §4.1](../../experiment3_summary.md#41-spatial-similarity-source-qk-compatibility-and-component-coordination): the QK reversion penalty, 111 − 011. Blue marks the 11 other nested state differences retained in the summary correlation table. After removing algebraic duplicates and the constant difference, all 12 retained differences switch one component while holding its partners fixed. Unmarked panels—including the nine non-nested comparisons and all six component-pair interactions—remain exhaustive controls outside that table. The CSV flags `consistently_high_P_G` when both correlations have the same sign and absolute magnitude at least 0.70. This is a descriptive flag, not a significance test.', '',
                 '| Comparison | Scores | Summary use |', '|---|---|---|']
        for key, (role, section, description, scores) in SUMMARY_USES.items():
            effect = ' − '.join(key) if isinstance(key, tuple) else LABELS[key]
            marker = {'mmd_evidence': '**Green: primary metric**', 'mmd_comparison': 'Blue: comparison table'}[role]
            lines.append(f'| {effect} | {scores.replace(",", ", ")} | {marker} · §{section}: {description} |')
        lines += ['', '**Coordination comparisons in the summary table:** joint non-additivity S, minimum reversion penalty Rmin, and three-way interaction I3 are also tabulated in the summary. They are not two-state differences or component-pair interactions, so they are outside the 34 comparisons above. Their [full coordination plots](../mmd_coordination_20m.png) and the [four QK-context plots](../mmd_qk_effects_20m.png) remain in the detailed metrics report. The summary may reverse a subtraction when both P and G correlations are negative; its CSV records the original difference and orientation. This exhaustive atlas retains the original subtraction order, so a blue panel can appear with the opposite sign in the summary table. The summary displays only the [QK reversion scatter](../../figures_summary/mmd_qk_reversion_20m.png) for §4.1.', '',
                  '**Nested comparisons and duplicate removal in the summary:** one adapted-component set must contain the other. Comparisons that exchange one component for another, such as QK-only versus VO+MLP, are excluded. Since F000 = 0 and F111 = 1, endpoint-anchored pairs Fx − F000 and F111 − Fx sum to 1. Only one is retained per intermediate state; 011 − 000 is omitted in favor of the primary QK penalty. The constant 111 − 000 is omitted. This exhaustive comparison still retains every original contrast. Summary rows are sorted separately within state differences and coordination by decreasing min(|ρP|, |ρG|), then mean absolute correlation.', '',
                  '| Point | Transfer |', '|---:|---|']
        lines += [f'| {i+1} | {NAMES[a]} → {NAMES[b]} |' for i, (a,b) in enumerate(PAIRS)]
        for title, name in pages:
            lines += ['', f'![{title}]({name})']
        (out/'README.md').write_text(display_text('\n'.join(lines)+'\n'))

    @staticmethod
    def comparison_provenance():
        out = HERE/'metrics/mmd_pairwise_comparison'
        (out/'provenance.json').write_text(json.dumps(dict(
            inputs={name: digest(HERE/name) for name in (
                'metrics/selected_metrics_by_seed.csv', 'metrics/selected_contrasts_by_seed.csv',
                'metrics/dataset_geometry.json', 'experiment3_summary.md', 'association_analysis.py')},
            directions=PAIRS, analysis_state_order=CODES, display_state_order=DISPLAY_CODES,
            differences=28, component_interactions=6, scores=['P', 'G'],
            descriptor='spatial_normalized', correlation='Spearman on 12 direction means; seed sensitivity separate',
            summary_markers={'green': 'Primary metric in summary 4.1', 'blue': 'Additional summary correlation-table comparison'},
            summary_omitted_equivalents={'-'.join(k): '-'.join(v) for k, v in SUMMARY_EQUIVALENTS.items()},
            summary_nested_only=True, summary_state_differences=len(SUMMARY_STATE_DIFFERENCES),
            consistently_high_rule=dict(minimum_absolute_rho=HIGH_CORRELATION_THRESHOLD, same_sign=True),
            constant_difference='111-000; undefined correlation', pdf_pages=9), indent=2)+'\n')

    def directional_reuse(self, directory):
        from paper_summary_assets import directional_figure
        directional_figure(self, directory)


if __name__ == '__main__':
    model = CubeAssociations()
    model.mmd_associations()
    model.pairwise_comparison()
