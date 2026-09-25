"""Build the five-part summary using only the selected image and P/G reports.

Recompute numerical findings when the selected source/endpoints change, without
altering measurements or the other reports. Historical summary assets stay intact.
"""
import csv
import hashlib
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import paper_summary_assets as paper
from notation import display_text, style_figure

from hybrid_analysis import HERE, PAIRS, CODES
from association_analysis import (CubeAssociations, DISPLAY_CODES, QK, COORDINATION,
                                  NAMES, SOURCES, COLORS, save as save_association,
                                  SUMMARY_STATE_DIFFERENCES, SUMMARY_EQUIVALENTS,
                                  HIGH_CORRELATION_THRESHOLD)

OUT = HERE / 'figures_summary'
IMAGE_PICKS = {'celeba64': [(0, 3), (0, 9), (0, 32)],
               'stl10_64': [(0, 3), (1, 9), (0, 14)],
               'imagenet64': [(1, 1), (0, 12), (0, 40)]}

TITLES = {'000': 'Source', '100': 'QK', '010': 'VO', '110': 'QK+VO',
          '001': 'MLP', '101': 'QK+MLP', '011': 'VO+MLP', '111': 'Joint'}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label(pair):
    return ' → '.join(NAMES[d] for d in pair)


def difference_name(key):
    """Name the displayed, signed difference; a cost is without minus with."""
    a, b = key
    components = ('QK', 'VO', 'MLP')
    changed = [i for i in range(3) if a[i] != b[i]]
    common = [components[i] for i in range(3) if a[i] == b[i] == '1']
    if len(changed) == 1:
        i = changed[0]
        if a == '111':
            return f'{components[i]} reversion penalty'
        kind = 'gain' if a[i] == '1' else 'cost'
        if not common:
            return f'{components[i]}-only adaptation {kind}'
        return f'{components[i]} adaptation {kind} ({"+".join(common)} adapted)'
    def varying(code):
        active = [components[i] for i in changed if code[i] == '1']
        return '+'.join(active) + (' only' if len(active) == 1 and not common else '')
    return f'{varying(a)} vs {varying(b)}' + (f' ({"+".join(common)} adapted)' if common else '')


def save(fig, name):
    style_figure(fig)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


class Summary(CubeAssociations):
    def __init__(self):
        super().__init__()
        self.endpoints = {(r['source'], r['target']): r for r in
                          json.loads((HERE / 'tables/endpoint_selections.json').read_text())}
        image_report = HERE / 'experiment3_images.md'
        self.images_text = (image_report.read_text() if image_report.exists()
                            else '\n'.join(f'## {label(pair)}' for pair in PAIRS))
        assert re.findall(r'^## (.+)$', self.images_text, re.M) == list(map(label, PAIRS))
        for rows in (self.states, self.contrasts):
            assert list(dict.fromkeys((r['source'], r['target']) for r in rows)) == PAIRS
            assert {r['metric'] for r in rows} == {'P', 'G'}
            assert all(r['scale'] == '20m' for r in rows)
            assert all(float(r['selected_epoch']) == self.endpoints[r['source'], r['target']]['reference_epoch'] for r in rows)
        assert self.geometry['validated']
        self.source_epochs = {'celeba64': 400}
        for src in SOURCES[1:]:
            epochs = {int(re.search(r'_source(\d+)_', r['id'])[1]) for r in self.states if r['source'] == src}
            assert len(epochs) == 1
            self.source_epochs[src] = epochs.pop()

    def positive_intervals(self, metric, key):
        return sum(float(r['ci95_low']) > 0 for r in self.contrasts
                   if r['metric'] == metric and r['contrast'] == key)

    def image_overview(self):
        """User-selected examples, target rows and the original eight state columns."""
        root = HERE.parents[1]
        gallery = Path(os.environ.get('EXPERIMENT3_SELECTION_GALLERY', root / 'reports/experiment3_selection64'))
        provenance = []
        for source in SOURCES:
            catalog = json.loads((gallery / source / 'provenance.json').read_text())
            targets = [t for src, t in PAIRS if src == source]
            banks = {}
            for seed, index in IMAGE_PICKS[source]:
                for target in targets:
                    if (target, seed) in banks:
                        continue
                    bank, = [b for b in catalog['banks'] if b['target'] == target and b['seed'] == seed]
                    assert bank['source_training_epoch'] == self.source_epochs[source], 'Gallery source does not match active report'
                    assert bank['reference_epoch'] == self.endpoints[source, target]['reference_epoch'], 'Gallery endpoint does not match active report'
                    if 'adaptation_step' in bank:
                        assert bank['adaptation_step'] == self.endpoints[source, target]['step']
                    path = root / bank['raw_bank'] / 'images.npz'
                    assert sha(path) == bank['images_sha256']
                    with np.load(path) as a:
                        banks[target, seed] = dict(images=a['images'].copy(), codes=list(a['codes']),
                                                   latents=a['latents'].copy(), path=path, bank=bank)
            with plt.rc_context(paper.STYLE):
                fig = plt.figure(figsize=(5.5, 6.8))
                grid = fig.add_gridspec(14, 8, left=.245, right=.995, top=.94, bottom=.01,
                                       wspace=.025, hspace=.025,
                                       height_ratios=[1]*4+[.25]+[1]*4+[.25]+[1]*4)
                for group, (seed, index) in enumerate(IMAGE_PICKS[source]):
                    ref = banks[targets[0], seed]
                    for row, target in enumerate(targets):
                        pair = (source, target)
                        bank = banks[target, seed]
                        images, codes = bank['images'], bank['codes']
                        assert set(codes) == set(CODES)
                        assert np.array_equal(bank['latents'][index-1], ref['latents'][index-1])
                        assert np.array_equal(images[codes.index('000'), index-1],
                                              ref['images'][ref['codes'].index('000'), index-1])
                        for col, code in enumerate(DISPLAY_CODES):
                            ax = fig.add_subplot(grid[group*5+row, col])
                            ax.imshow(images[codes.index(code), index-1].transpose(1, 2, 0), interpolation='nearest')
                            ax.set_xticks([]); ax.set_yticks([])
                            for spine in ax.spines.values():
                                spine.set_visible(False)
                            if group == 0 and row == 0:
                                ax.set_title(f'{code}\n{TITLES[code]}', fontsize=6.5, pad=3)
                            if col == 0:
                                ax.set_ylabel(f'{label(pair)}\nMMD² = {self.mmd(pair):.5f}',
                                              rotation=0, ha='right', va='center', fontsize=6.5,
                                              labelpad=4, color=COLORS[source])
                        path = bank['path']
                        provenance.append(dict(source=source, target=target, adaptation_seed=seed,
                            source_training_epoch=self.source_epochs[source],
                            reference_epoch=bank['bank']['reference_epoch'],
                            quantitative_reference_epoch=self.endpoints[pair]['reference_epoch'],
                            matches_quantitative_endpoint=(bank['bank']['reference_epoch'] == self.endpoints[pair]['reference_epoch']),
                            figure=f'selected_images_{source}_20m', group_one_based=group+1,
                            source_npz=os.path.relpath(path, HERE), source_sha256=bank['bank']['images_sha256'],
                            source_json_sha256=sha(path.with_name('provenance.json')),
                            selected_rows_one_based=[index], latent_seed=bank['bank']['latent_rng_seed'],
                            selected_latent_sha256=hashlib.sha256(bank['latents'][index-1].tobytes()).hexdigest(),
                            spatial_mmd2=self.mmd(pair)))
                fig.suptitle(f'{NAMES[source]} source', fontsize=9, y=.993)
                paper.finish(fig, OUT, f'selected_images_{source}_20m')
        (OUT / 'selection_manifest.json').write_text(json.dumps(dict(
            selection='User-selected illustrative groups; three (adaptation seed, one-based latent index) choices per source. Target rows share source pixels and raw latents exactly. No row or group numbers displayed.',
            picks={source: [dict(seed=seed, index=index) for seed,index in picks] for source,picks in IMAGE_PICKS.items()},
            states=DISPLAY_CODES, entries=provenance), indent=2) + '\n')

    def reversion_overlay(self, key, title, dataset=False):
        """Overlay score-specific direction means and separate descriptive OLS fits."""
        fig, ax = plt.subplots(figsize=(7.2, 5.3))
        fig.subplots_adjust(left=.13, right=.97, bottom=.13, top=.77)
        metric_handles = []
        for metric, marker, style in (('P', 'o', '-'), ('G', '^', '--')):
            x = np.array([self.mmd(pair) for pair in PAIRS]) if dataset else self.vector(metric, 'joint_nonadditivity')
            y = self.vector(metric, key)
            for source in SOURCES:
                mask = np.array([pair[0] == source for pair in PAIRS])
                ax.scatter(x[mask], y[mask], marker=marker, s=65, color=COLORS[source],
                           edgecolor='white', linewidth=.7, zorder=3)
            slope, intercept = np.polyfit(x, y, 1)
            domain = np.array([x.min(), x.max()])
            ax.plot(domain, slope * domain + intercept, linestyle=style, color='#343434',
                    linewidth=1.6, zorder=2)
            metric_handles.append(Line2D([], [], color='#343434', marker=marker,
                linestyle=style, markersize=7, label=rf'{metric}: Spearman $\rho$ = {self.rho(metric, key, dataset=dataset):.2f}'))
        fig.suptitle(title, fontsize=13, y=.98)
        fig.legend(handles=metric_handles, loc='upper center', bbox_to_anchor=(.5, .93),
                   ncol=2, frameon=False, fontsize=10, handlelength=2.7)
        fig.legend(handles=[Line2D([], [], marker='s', linestyle='', color=COLORS[s],
                                  label=NAMES[s]) for s in SOURCES],
                   loc='upper center', bbox_to_anchor=(.5, .855), ncol=3, frameon=False,
                   fontsize=10, title='Source dataset', title_fontsize=9)
        ax.set_xlabel('Dataset spatial MMD²' if dataset else 'Joint non-additivity S')
        ax.set_ylabel('QK reversion penalty (111 − 011)' if dataset else 'VO reversion penalty (111 − 101)')
        ax.grid(alpha=.2)
        ax.set_axisbelow(True)
        ax.margins(x=.06, y=.1)
        return fig

    def associations(self):
        fig = self.reversion_overlay('QK_restoration', 'Spatial distance and source-QK compatibility', dataset=True)
        save_association(fig, OUT, 'mmd_qk_reversion_20m')
        fig = self.reversion_overlay('VO_restoration', 'VO importance and joint non-additivity')
        save_association(fig, OUT, 'vo_and_coadaptation_20m')
        self.directional_reuse(OUT)
        paper.figures(self, OUT)

    def fragments(self):
        f = {}
        epoch = self.source_epochs['imagenet64']
        f['SOURCE_NOTE'] = ('The current ImageNet64 results use the 20M epoch-600 source; we plan to replace them with epoch-800 results after the full transfer and evaluation pipeline completes, retaining epoch 600 as the fallback if time is limited.'
                            if epoch == 600 else f'The ImageNet64 results use the validated epoch-{epoch} source replacement.')
        f['SOURCE_EPOCHS'] = '; '.join(f'{NAMES[s]} epoch {self.source_epochs[s]}' for s in SOURCES)
        f['BANDWIDTH'] = f"{self.geometry['bandwidth_squared']['spatial_normalized']:.8f}"
        f['DATASET_FINDINGS'] = (f"For CelebA, AAHQ and FFHQ are comparatively close in spatial-descriptor distribution "
            f"(MMD² {self.mmd(PAIRS[0]):.5f} and {self.mmd(PAIRS[1]):.5f}); AFHQ-v2 is farther "
            f"({self.mmd(PAIRS[2]):.5f}) and STL-10 farther still ({self.mmd(PAIRS[3]):.5f}). "
            "This is a relative ordering, with no calibrated boundary between ‘same’ and ‘different’ geometry. "
            "The raw-token sensitivity check preserves this broad ordering but swaps the two human-face targets; "
            "the small AAHQ/FFHQ difference should not carry the interpretation.")
        table = ['| Transfer / complete images | Reference epoch | Spatial MMD² | $P_{011}$ | $G_{011}$ |',
                 '|---|---:|---:|---:|---:|']
        for i, pair in enumerate(PAIRS):
            anchor = label(pair).lower().replace(' → ', '--')
            table.append(f'| [{label(pair)}](experiment3_images.md#{anchor}) | '
                         f"{self.endpoints[pair]['reference_epoch']:g} | {self.mmd(pair):.5f} | "
                         f"{self.value(pair, 'P', '011'):.3f} | {self.value(pair, 'G', '011'):.3f} |")
        f['ENDPOINT_TABLE'] = '\n'.join(table)
        ca, cf, cs, sc = PAIRS[0], PAIRS[1], PAIRS[3], PAIRS[7]
        f['GENERATED_FINDINGS'] = (
            f"- **CelebA → AAHQ:** QK alone reaches $P_{{100}}={self.value(ca, 'P', '100'):.3f}$; "
            f"MLP alone reaches $P_{{001}}={self.value(ca, 'P', '001'):.3f}$; adapted VO+MLP with source QK reaches "
            f"$P_{{011}}={self.value(ca, 'P', '011'):.3f}$ and $G_{{011}}={self.value(ca, 'G', '011'):.3f}$. "
            "The source selector remains partly compatible with the adapted computation.\n"
            f"- **CelebA → FFHQ versus CelebA → STL-10:** Source-QK spatial agreement is "
            f"{self.value(cf, 'G', '011'):.3f} versus {self.value(cs, 'G', '011'):.3f}. "
            "The larger measured dataset difference accompanies a much larger loss on reverting QK.\n"
            "- **Broad-domain sources:** For STL-10 and ImageNet64, source-QK agreement varies by target; "
            f"AAHQ gives $P_{{011}}={self.value(PAIRS[4], 'P', '011'):.3f}$ / "
            f"${self.value(PAIRS[8], 'P', '011'):.3f}$, whereas CelebA gives "
            f"{self.value(sc, 'P', '011'):.3f} / {self.value(PAIRS[11], 'P', '011'):.3f}. "
            "A broad source alone does not ensure uniformly high reuse.")
        f['QK_FINDINGS'] = (
            rf"**Figure 1(a)** shows positive correlations between spatial MMD² and "
            rf"$F_{{111}}-F_{{011}}$: $\rho_P={self.rho('P', 'QK_restoration', dataset=True):.3f}$ "
            rf"and $\rho_G={self.rho('G', 'QK_restoration', dataset=True):.3f}$. "
            "Smaller dataset distance accompanies greater source-QK compatibility with adapted VO+MLP. "
            rf"For CelebA → AAHQ, the mean penalties are {self.value(ca, 'P', 'QK_restoration'):.3f} "
            rf"($P$) and {self.value(ca, 'G', 'QK_restoration'):.3f} ($G$), compared with "
            rf"{self.value(cs, 'P', 'QK_restoration'):.3f} and {self.value(cs, 'G', 'QK_restoration'):.3f} "
            "for CelebA → STL-10. The face-domain result supports partial compatibility, "
            "rather than a negligible QK contribution.")
        f['COADAPTATION_FINDINGS'] = (
            rf"**Figure 2** extends the association beyond the QK difference. Spatial MMD² correlates "
            rf"with $S$ at $\rho_P={self.rho('P', 'joint_nonadditivity', dataset=True):.3f}$ and "
            rf"$\rho_G={self.rho('G', 'joint_nonadditivity', dataset=True):.3f}$; with the minimum "
            rf"reversion penalty at {self.rho('P', 'minimum_reversion', dataset=True):.3f} and "
            rf"{self.rho('G', 'minimum_reversion', dataset=True):.3f}; and with $I_3$ at "
            rf"{self.rho('P', 'three_way_interaction', dataset=True):.3f} and "
            rf"{self.rho('G', 'three_way_interaction', dataset=True):.3f}. "
            "Thus larger spatial shifts accompany stronger functional coordination under all three "
            "summaries in these transfers. "
            rf"$S$ has a positive lower 95% latent-bootstrap bound in "
            rf"{self.positive_intervals('P', 'joint_nonadditivity')}/24 seed-specific cubes for $P$ "
            rf"and {self.positive_intervals('G', 'joint_nonadditivity')}/24 for $G$; the minimum "
            rf"penalty does so in {self.positive_intervals('P', 'minimum_reversion')}/24 and "
            rf"{self.positive_intervals('G', 'minimum_reversion')}/24. "
            "These are separate, unadjusted within-cube intervals. The three statistics share terms "
            "and are not independent confirmations; the minimum explicitly contains the QK difference. "
            "The result concerns dependence among jointly trained components, not proof that larger "
            "spatial shifts require more optimization or that joint training beats restricted training.")
        ic = PAIRS[11]
        common_faces = [pair for pair in PAIRS if pair[1] in paper.FACE_TARGETS[:3]]
        face_ranges = {metric: [self.value(pair, metric, '011') for pair in common_faces]
                       for metric in ('P', 'G')}
        f['DIRECTION_FINDINGS'] = (
            "**Figure 1(c)** now retains all eleven face-target transfers and the CelebA → STL-10 "
            "reference. AAHQ, FFHQ, and CelebA are human-face targets; AFHQ-v2 is an animal-face target. "
            "Each source is compared on the same AAHQ, FFHQ, and AFHQ-v2 targets, avoiding an average "
            "over different target sets. Across those nine transfers, source-QK agreement spans "
            rf"$P_{{011}}={min(face_ranges['P']):.3f}$–${max(face_ranges['P']):.3f}$ and "
            rf"$G_{{011}}={min(face_ranges['G']):.3f}$–${max(face_ranges['G']):.3f}$. "
            "Both natural-domain sources therefore retain substantial partial endpoint agreement "
            "on these face targets, alongside the CelebA-source baseline. "
            "The pattern is target-dependent, not a universal advantage for natural-domain pretraining. "
            "For example, on AAHQ, CelebA gives "
            rf"$(P_{{011}},G_{{011}})=({self.value(ca, 'P', '011'):.3f},{self.value(ca, 'G', '011'):.3f})$, "
            "compared with "
            rf"$({self.value(PAIRS[4], 'P', '011'):.3f},{self.value(PAIRS[4], 'G', '011'):.3f})$ from STL-10 "
            "and "
            rf"$({self.value(PAIRS[8], 'P', '011'):.3f},{self.value(PAIRS[8], 'G', '011'):.3f})$ from ImageNet64. "
            "On CelebA, the natural-source scores are lower: "
            rf"$({self.value(sc, 'P', '011'):.3f},{self.value(sc, 'G', '011'):.3f})$ from STL-10 and "
            rf"$({self.value(ic, 'P', '011'):.3f},{self.value(ic, 'G', '011'):.3f})$ from ImageNet64. "
            "Both still exceed the reverse-reference CelebA → STL-10 scores "
            rf"$({self.value(cs, 'P', '011'):.3f},{self.value(cs, 'G', '011'):.3f})$. "
            "Only CelebA/STL-10 is reciprocal, with the same MMD² "
            rf"({self.mmd(cs):.5f}) in both directions. No unmeasured reverse or self-transfer is "
            "filled in; source training histories, endpoints, and normalization gaps differ.")
        f.update(paper.tables(self, OUT, difference_name))
        paper.preview_document(OUT)
        f['HIGH_CORRELATION_THRESHOLD'] = f'{HIGH_CORRELATION_THRESHOLD:.2f}'
        f['VO_MAIN_FINDINGS'] = (
            rf"**Figure 1(b)** relates joint non-additivity $S$ to the loss caused by reverting VO, "
            rf"$F_{{111}}-F_{{101}}$. Its correlations are "
            rf"$\rho_P={self.rho('P', 'VO_restoration'):.3f}$ and "
            rf"$\rho_G={self.rho('G', 'VO_restoration'):.3f}$. "
            "Transfers with stronger non-additivity exhibit larger dependence on adapted VO in the "
            "joint model. The component comparison below shows that this association is stronger "
            "for VO than for QK or MLP in both scores.")
        dominance = {m: int(np.sum((self.vector(m, 'VO_restoration') > self.vector(m, 'QK_restoration')) &
                                   (self.vector(m, 'VO_restoration') > self.vector(m, 'MLP_restoration')))) for m in ('P', 'G')}
        f['VO_FINDINGS'] = (
            rf"The $S$ versus $F_{{111}}-F_{{101}}$ association remains positive separately for seeds "
            rf"0 and 1: $\rho_P={self.rho('P', 'VO_restoration', 0):.2f}$ / "
            rf"${self.rho('P', 'VO_restoration', 1):.2f}$ and "
            rf"$\rho_G={self.rho('G', 'VO_restoration', 0):.2f}$ / "
            rf"${self.rho('G', 'VO_restoration', 1):.2f}$. "
            rf"VO has the largest mean reversion penalty in {dominance['P']}/12 directions for $P$ "
            rf"and {dominance['G']}/12 for $G$. Increasing association with coordination therefore "
            "does not imply that VO is generally the most important component.")
        f['VO_CONTEXT_FINDINGS'] = (
            rf"The partner context also matters. For CelebA → STL-10, adding VO to adapted MLP "
            rf"with source QK gives $F_{{011}}-F_{{001}}={self.value(cs, 'P', 'VO_given_MLP'):.3f}$ "
            rf"($P$) and {self.value(cs, 'G', 'VO_given_MLP'):.3f} ($G$). With both QK and MLP adapted, "
            rf"$F_{{111}}-F_{{101}}$ is {self.value(cs, 'P', 'VO_restoration'):.3f} and "
            rf"{self.value(cs, 'G', 'VO_restoration'):.3f}. "
            "The much larger joint-context benefit matches the qualitative contrast between states.")
        return f

    def build(self, render=True):
        OUT.mkdir(exist_ok=True)
        if render:
            self.image_overview()
            self.associations()
        text = (HERE / 'inputs/summary_template.md').read_text()
        for key, value in self.fragments().items():
            text = text.replace(f'@@{key}@@', value)
        assert not re.search(r'@@[A-Z_]+@@', text)
        (HERE / 'experiment3_summary.md').write_text(display_text(text))
        association_rows = []
        for i, pair in enumerate(PAIRS):
            for metric in ('P', 'G'):
                for seed in (0, 1):
                    row = dict(number=i+1, source=pair[0], target=pair[1], metric=metric, adaptation_seed=seed,
                               source_training_epoch=self.source_epochs[pair[0]], reference_epoch=self.endpoints[pair]['reference_epoch'],
                               mmd2=self.mmd(pair))
                    row.update({key: self.value(pair, metric, key, seed) for key in
                                tuple(key for key, _ in QK + COORDINATION) + ('VO_restoration', 'MLP_restoration', 'VO_given_MLP')})
                    association_rows.append(row)
        with (OUT / 'associations_by_seed.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(association_rows[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(association_rows)
        inputs = ['experiment3_images.md', 'experiment3_metrics.md', 'metrics/selected_metrics_by_seed.csv',
                  'metrics/selected_contrasts_by_seed.csv', 'metrics/dataset_geometry.json',
                  'metrics/selected_protocol.json', 'tables/endpoint_selections.json', 'inputs/summary_template.md',
                  'build_summary_assets.py', 'association_analysis.py', 'paper_summary_assets.py', 'notation.py']
        (OUT / 'summary_provenance.json').write_text(json.dumps(dict(
            directions=PAIRS, states=DISPLAY_CODES, measurement_states=CODES, scores=['P', 'G'], source_training_epochs=self.source_epochs,
            inputs={name: sha(HERE / name) for name in inputs},
            association_method='Spearman ranks of 12 direction means; separate adaptation-seed sensitivity; no p-values or independence assumption.',
            reversion_plot_method='P circles and solid OLS trend; G triangles and dashed OLS trend. Separate fits to 12 two-seed direction means per score, on original axes. VO uses score-specific S. Trends are descriptive, without confidence bands; annotations report Spearman correlations. Source dataset determines point color; no point numbers.',
            primary_mmd_metric='QK_restoration',
            display_symbols=dict(P='EA', G='SA', F='Agr', S='JNA'),
            paper_layout=dict(width_inches=5.5, font='DejaVu Sans', body_font_pt=8,
                              main_figure='main_figure1_20m', coordination_figure='mmd_coordination_paper_20m',
                              panel_order=['MMD vs QK reversion', 'S vs VO reversion', 'source-QK agreement by target'],
                              state_difference_table_rows=12, vo_comparison_table_rows=3),
            paper_assets={name: sha(OUT / name) for name in (
                'main_figure1_20m.pdf', 'main_figure1_20m.png',
                'mmd_coordination_paper_20m.pdf', 'mmd_coordination_paper_20m.png',
                'nested_mmd_correlations.tex', 'mmd_comparison_correlations.csv',
                'vo_nonadditivity_correlations.tex', 'vo_nonadditivity_correlations.csv',
                'directional_reuse_20m.pdf', 'directional_reuse_20m.png',
                'directional_reuse_by_target.csv', 'selection_manifest.json',
                'selected_images_celeba64_20m.png', 'selected_images_celeba64_20m.pdf',
                'selected_images_stl10_64_20m.png', 'selected_images_stl10_64_20m.pdf',
                'selected_images_imagenet64_20m.png', 'selected_images_imagenet64_20m.pdf')},
            directional_plot_method='Main panel: target datasets on x-axis, source-QK agreement on y-axis; P solid and G hatched grouped bars, with source colors. Expanded bar plot retains seed dots and ranges. All 12 directions shown separately; 11 face-target transfers and CelebA-to-STL10 reference. Face targets shaded; no pooled target averages or imputed transfers. Limits follow data.',
            comparison_rule=dict(minimum_absolute_rho=HIGH_CORRELATION_THRESHOLD, same_sign=True,
                                 compared_scores=['P', 'G'], inferential=False,
                                 sort='Within family: descending minimum absolute rho, then mean absolute rho',
                                 orientation='Reverse a state difference only when both P and G correlations are negative; retain mixed signs',
                                 omitted_equivalents={'-'.join(k): '-'.join(v) for k, v in SUMMARY_EQUIVALENTS.items()},
                                 omitted_constant='111-000', pair_interactions_in_table=False,
                                 nested_only=True, state_difference_rows=len(SUMMARY_STATE_DIFFERENCES)),
            spearman={metric: {'mmd': {key: self.rho(metric, key, dataset=True) for key, _ in QK + COORDINATION},
                               'nonadditivity_vs_vo': self.rho(metric, 'VO_restoration')}
                      for metric in ('P', 'G')}), indent=2) + '\n')
        validate_summary()
        print('Rebuilt five-part summary: 12 selected transfers, P/G only, correlation figures, directional reuse, and numerical provenance.')


def validate_summary():
    text = (HERE / 'experiment3_summary.md').read_text()
    assert re.findall(r'^## (.+)$', text, re.M) == [
        '1. Hypotheses', '2. Image datasets analysis: spatial similarity before transfer',
        '3. Generated image analysis: quantitative endpoint agreement',
        '4. Metrics analysis: spatial compatibility, coordination, and directional reuse', '5. Conclusion']
    assert not any(word in text.lower() for word in ('icartoon', 'density', '@@'))
    assert not any(path in text for path in ('metrics/paired_contrasts_20m.png',
        'metrics/qk_contexts_20m.png', 'metrics/coadaptation_20m.png',
        'metrics/pair_interactions_20m.png', 'metrics/partner_contrasts_20m.png',
        'mmd_pairwise_comparison/'))
    assert 'figures_summary/main_figure1_20m.png' in text
    assert 'figures_summary/mmd_coordination_paper_20m.png' in text
    assert r'R_{' not in text and r'R_{{' not in text
    with (OUT / 'mmd_comparison_correlations.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12 and all(r['family'] == 'State difference' for r in rows)
    with (OUT / 'vo_nonadditivity_correlations.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert [r['component'] for r in rows] == ['VO', 'QK', 'MLP']
    assert '![Dataset spatial MMD against all four QK effects' not in text
    assert '![Dataset spatial MMD against joint non-additivity' not in text
    assert re.findall(r'^### 4\..+$', text, re.M) == [
        '### 4.1. Similar spatial structure supports QK compatibility; larger shifts accompany coordination',
        '### 4.2. VO matters more when functional coadaptation is stronger',
        '### 4.3. Source-QK reuse across face targets and transfer directions']
    for direction in ('AAHQ → CelebA', 'FFHQ → CelebA', 'AFHQ-v2 → CelebA'):
        assert direction not in text
    assert not any(ord(c) < 32 and c not in '\n\r\t' for c in text)
    assert re.findall(r'^\| \[([^\]]+)\]\(experiment3_images', text, re.M) == list(map(label, PAIRS))
    entries = json.loads((OUT / 'selection_manifest.json').read_text())['entries']
    assert list(dict.fromkeys((r['source'], r['target']) for r in entries)) == PAIRS
    assert len(entries) == 36
    for source, picks in IMAGE_PICKS.items():
        actual = list(dict.fromkeys((r['adaptation_seed'], r['selected_rows_one_based'][0]) for r in entries if r['source'] == source))
        assert actual == picks
        assert f'figures_summary/selected_images_{source}_20m.png' in text
    for link in re.findall(r'\]\(([^)]+)\)', text):
        if not link.startswith(('http:', 'https:', '#')):
            assert (HERE / link.split('#')[0]).exists(), link


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--text-only', action='store_true', help='Refresh prose, tables, and provenance using existing figures')
    Summary().build(render=not parser.parse_args().text_only)
