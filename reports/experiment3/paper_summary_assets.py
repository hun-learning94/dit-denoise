"""Publication-sized Experiment 3 figures and booktabs tables from frozen scores.

The 5.5-inch width matches draft/iclr2027_conference.sty. No measurements,
endpoint selections, image selections, or manuscript files are changed.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
from notation import display_text, style_figure

from association_analysis import (
    PAIRS, SOURCES, COLORS, NAMES, SUMMARY_STATE_DIFFERENCES,
    consistently_high, write_csv,
)

STYLE = {
    'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans',
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7.5,
    'axes.linewidth': .6, 'xtick.major.width': .6, 'ytick.major.width': .6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.major.pad': 2, 'ytick.major.pad': 2,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
}


def finish(fig, directory, stem, trim_vertical=False):
    style_figure(fig)
    save_options = {}
    if trim_vertical:
        from matplotlib.transforms import Bbox
        fig.canvas.draw()
        bounds = fig.get_tightbbox(fig.canvas.get_renderer())
        # Crop unused vertical canvas while keeping the 5.5-inch width and type size.
        save_options['bbox_inches'] = Bbox.from_extents(
            0, max(0, bounds.y0 - .025), fig.get_figwidth(),
            min(fig.get_figheight(), bounds.y1 + .025))
        save_options['pad_inches'] = 0
    # Preserve exact physical width and font sizes: no tight-bbox rescaling.
    for ext in ('pdf', 'png'):
        fig.savefig(directory / f'{stem}.{ext}', dpi=320, facecolor='white', **save_options)
    plt.close(fig)


def common_legend(fig, right=False):
    handles = [Line2D([], [], marker='s', linestyle='', color=COLORS[s],
                      markersize=4.5, label=NAMES[s]) for s in SOURCES]
    handles += [Line2D([], [], marker=m, linestyle=ls, color='#303030',
                       markersize=4, linewidth=1, label=metric)
                for metric, m, ls in [('P', 'o', '-'), ('G', '^', '--')]]
    if right:
        fig.legend(handles=handles[:3], title='Source', ncol=1,
                   loc='upper left', bbox_to_anchor=(.852, .89), frameon=False,
                   fontsize=7, title_fontsize=7.5, borderaxespad=0,
                   handlelength=1.2, handletextpad=.4, labelspacing=.65)
        fig.legend(handles=handles[3:], title=r'$F$', ncol=1,
                   loc='upper left', bbox_to_anchor=(.852, .47), frameon=False,
                   fontsize=7, title_fontsize=7.5, borderaxespad=0,
                   handlelength=2.8, handletextpad=.4, labelspacing=.65)
        return
    fig.legend(handles=handles, ncol=5, loc='upper center',
               bbox_to_anchor=(.52, 1.0), frameon=False,
               borderaxespad=.15, columnspacing=1.2, handletextpad=.4,
               handlelength=1.6)


def scatter(ax, model, key, dataset=True):
    for metric, marker, style in [('P', 'o', '-'), ('G', '^', '--')]:
        x = (np.array([model.mmd(pair) for pair in PAIRS]) if dataset
             else model.vector(metric, 'joint_nonadditivity'))
        y = model.vector(metric, key)
        for source in SOURCES:
            mask = np.array([pair[0] == source for pair in PAIRS])
            ax.scatter(x[mask], y[mask], s=17, marker=marker, color=COLORS[source],
                       edgecolor='white', linewidth=.35, zorder=3)
        fit = np.polyfit(x, y, 1)
        domain = np.array([x.min(), x.max()])
        ax.plot(domain, np.polyval(fit, domain), style, color='#303030',
                linewidth=1, zorder=2)
    ax.text(.035, .97,
            rf'$\rho_P={model.rho("P", key, dataset=dataset):.2f}$' '\n'
            rf'$\rho_G={model.rho("G", key, dataset=dataset):.2f}$',
            transform=ax.transAxes, va='top', fontsize=7,
            bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=.7))
    ax.grid(alpha=.2, linewidth=.5)
    ax.set_axisbelow(True)
    ax.margins(x=.09, y=.18)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    if dataset:
        ax.set_xticks([.02, .05, .08])
    else:
        ax.xaxis.set_major_locator(MaxNLocator(3))


FACE_TARGETS = ('aahq64', 'ffhq64', 'afhq_v2_64', 'celeba64')
REUSE_TARGETS = FACE_TARGETS + ('stl10_64',)


def reuse_rows(model):
    return [dict(source=source, target=target, metric=metric,
                 mean=model.value((source, target), metric, '011'),
                 seed0=model.value((source, target), metric, '011', 0),
                 seed1=model.value((source, target), metric, '011', 1))
            for target in REUSE_TARGETS for source in SOURCES
            if (source, target) in PAIRS for metric in ('P', 'G')]


def directional_bars(ax, model, compact=False):
    """Target-grouped bars; seed detail is retained in the expanded view."""
    ax.axvspan(-.5, 3.5, color='#f1f1f1', zorder=0)
    ax.axvline(3.5, color='#999999', linewidth=.6, linestyle='--')
    values = []
    for i, target in enumerate(REUSE_TARGETS):
        sources = [source for source in SOURCES if (source, target) in PAIRS]
        offsets = (np.arange(len(sources))-(len(sources)-1)/2)*.25
        for source, offset in zip(sources, offsets):
            for metric, metric_offset, hatch in [('P', -.055, ''), ('G', .055, '///')]:
                y = [model.value((source, target), metric, '011', seed) for seed in (0, 1)]
                values.extend(y)
                x = i + offset + metric_offset
                ax.bar(x, np.mean(y), width=.105, color=COLORS[source],
                       edgecolor='white', linewidth=.4, hatch=hatch, zorder=2)
                if not compact:
                    ax.plot([x, x], y, color='#303030', linewidth=.6, zorder=3)
                    ax.scatter([x-.012, x+.012], y, s=5, color='#303030',
                               edgecolor='white', linewidth=.2, zorder=4)
    low, high = min(0, min(values)), max(0, max(values))
    span = max(high-low, .05)
    ax.set_ylim(low-.06*span if low < 0 else 0, high+.18*span)
    ax.set_xlim(-.5, 4.5)
    ax.set_xticks(range(5), ['AAHQ', 'FFHQ', 'AFHQ', 'CelebA', 'STL-10'])
    if compact:
        ax.text(1.5, .97, r'$\rightarrow$ faces', transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=8)
        ax.tick_params(axis='x', labelrotation=55)
        for label in ax.get_xticklabels():
            label.set_ha('right')
        ax.set_xlabel('')
    else:
        ax.text(1.5, .97, 'Face targets (human and animal)', transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=8)
        ax.text(4, .97, 'Natural target', transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=8)
        ax.set_xlabel('')
    ax.set_ylabel(r'Source-QK agreement $F_{011}$', labelpad=2)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.grid(axis='y', alpha=.2, linewidth=.5)
    ax.set_axisbelow(True)
    ax.axhline(0, color='#555555', linewidth=.6)


def directional_figure(model, directory):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 2.35))
        fig.subplots_adjust(left=.09, right=.99, bottom=.18, top=.81)
        directional_bars(ax, model)
        handles = [Patch(facecolor=COLORS[source], label=NAMES[source]) for source in SOURCES]
        handles += [Patch(facecolor='#777777', edgecolor='white', label='P'),
                    Patch(facecolor='#777777', edgecolor='white', hatch='///', label='G')]
        fig.legend(handles=handles, ncol=5, loc='upper center',
                   bbox_to_anchor=(.53, 1), frameon=False,
                   columnspacing=1.2, handlelength=1.6)
        finish(fig, directory, 'directional_reuse_20m')
    write_csv(directory/'directional_reuse_by_target.csv', reuse_rows(model))


def figures(model, directory):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.05),
                                 gridspec_kw={'width_ratios': [1, 1, 1.05]})
        fig.subplots_adjust(left=.075, right=.835, bottom=.28, top=.80, wspace=.40)
        common_legend(fig, right=True)
        scatter(axes[0], model, 'QK_restoration')
        scatter(axes[1], model, 'VO_restoration', dataset=False)
        directional_bars(axes[2], model, compact=True)
        for ax, title, ylabel in zip(axes,
                ['(a) QK compatibility', '(b) VO coordination', '(c) Directional reuse'],
                [r'$F_{111}-F_{011}$', r'$F_{111}-F_{101}$', r'$F_{011}$']):
            ax.set_title(title, loc='left', pad=4, fontsize=7)
            ax.set_ylabel(ylabel, labelpad=2)
        axes[0].set_xlabel(r'Spatial MMD$^2$', labelpad=2)
        axes[1].set_xlabel('Joint non-additivity', labelpad=2)
        finish(fig, directory, 'main_figure1_20m', trim_vertical=True)

        fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.35), sharex=True)
        fig.subplots_adjust(left=.075, right=.835, bottom=.28, top=.80, wspace=.40)
        common_legend(fig, right=True)
        for ax, key, title, ylabel in zip(axes,
                ['joint_nonadditivity', 'minimum_reversion', 'three_way_interaction'],
                ['(a) Non-additivity', '(b) Min. reversion', '(c) Three-way interaction'],
                [r'$S$', r'$R_{\min}$', r'$\mathrm{TWI}$']):
            scatter(ax, model, key)
            ax.set_title(title, loc='left', pad=4, fontsize=7)
            ax.set_ylabel(ylabel, labelpad=2)
            ax.set_xlabel(r'Spatial MMD$^2$', labelpad=2)
        finish(fig, directory, 'mmd_coordination_paper_20m', trim_vertical=True)


def latex_table(path, caption, label, header, rows, widths='llrr'):
    lines = [
        r'\begin{table}[t]', r'\centering', r'\small',
        r'\setlength{\tabcolsep}{3pt}', r'\renewcommand{\arraystretch}{1.13}',
        r'\caption{' + caption + '}', r'\label{' + label + '}',
        r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}' + widths + '@{}}',
        r'\toprule', ' & '.join(header) + r' \\', r'\midrule',
    ]
    lines += [' & '.join(row) + r' \\' for row in rows]
    lines += [r'\bottomrule', r'\end{tabular*}', r'\end{table}', '']
    path.write_text(display_text('\n'.join(lines)))


def tables(model, directory, difference_name):
    rows = []
    for key, _ in SUMMARY_STATE_DIFFERENCES:
        cp, cg = (model.rho(metric, key, dataset=True) for metric in ('P', 'G'))
        orientation = -1 if cp < 0 and cg < 0 else 1
        displayed = key[::-1] if orientation == -1 else key
        p, g = orientation*cp, orientation*cg
        rows.append(dict(
            family='State difference', name=difference_name(displayed),
            effect='-'.join(displayed), canonical_effect='-'.join(key),
            orientation=orientation, spearman_P=p, spearman_G=g,
            primary=key == ('111', '011'), canonical_spearman_P=cp,
            canonical_spearman_G=cg, minimum_absolute_rho=min(abs(p), abs(g)),
            consistently_high=consistently_high(p, g),
            direction='positive' if p >= 0 and g >= 0 else 'mixed',
        ))
    rows.sort(key=lambda r: (-r['minimum_absolute_rho'],
                            -np.mean([abs(r['spearman_P']), abs(r['spearman_G'])]),
                            r['canonical_effect']))
    write_csv(directory/'mmd_comparison_correlations.csv', rows)
    markdown = ['| Interpretation | State difference | $\\rho_P$ | $\\rho_G$ |',
                '|---|---|---:|---:|']
    tex = []
    for row in rows:
        a, b = row['effect'].split('-')
        formula = rf'$F_{{{a}}}-F_{{{b}}}$'
        p, g = (f'{row[f"spearman_{m}"]:.3f}' for m in ('P', 'G'))
        name = row['name']
        if row['primary']:
            name += ' (primary)'
        name_tex = name
        if row['consistently_high']:
            markdown.append(f'| **{name}** † | {formula} | **{p}** | **{g}** |')
            name_tex = r'\textbf{' + name_tex + r'}$^\dagger$'
            p, g = (r'\textbf{' + val + '}' for val in (p, g))
        else:
            markdown.append(f'| {name} | {formula} | {p} | {g} |')
        tex.append([name_tex, formula, p, g])
    latex_table(directory/'nested_mmd_correlations.tex',
        r'Spatial distance and nested state differences. Spearman correlations with dataset spatial '
        r'MMD$^2$ use 12 two-seed direction means for each score. Each row changes one component '
        r'with its partners fixed. Endpoint-complement duplicates and the constant endpoint gap '
        r'are omitted. Rows are sorted by decreasing $\min(|\rho_P|,|\rho_G|)$. '
        r'Both-negative rows are reversed for display; mixed signs are retained. '
        r'Bold/$^\dagger$: same sign and both $|\rho|\geq0.70$, a descriptive criterion, '
        r'not a significance test.',
        'tab:exp3-nested-mmd', ['Interpretation', 'State difference', r'$\rho_P$', r'$\rho_G$'], tex)

    vo_rows = []
    for component, key, code in [('VO', 'VO_restoration', '101'),
                                 ('QK', 'QK_restoration', '011'),
                                 ('MLP', 'MLP_restoration', '110')]:
        vo_rows.append(dict(component=component, effect=f'111-{code}',
            spearman_P=model.rho('P', key), spearman_G=model.rho('G', key)))
    write_csv(directory/'vo_nonadditivity_correlations.csv', vo_rows)
    vo_md = ['| Reverted component | State difference | $\\rho_P$ with $S_P$ | $\\rho_G$ with $S_G$ |',
             '|---|---|---:|---:|']
    tex = []
    for row in vo_rows:
        formula = rf'$F_{{111}}-F_{{{row["effect"].split("-")[1]}}}$'
        p, g = (f'{row[f"spearman_{m}"]:.3f}' for m in ('P', 'G'))
        component = row['component']
        pmax = row['spearman_P'] == max(r['spearman_P'] for r in vo_rows)
        gmax = row['spearman_G'] == max(r['spearman_G'] for r in vo_rows)
        vo_md.append(f'| {component} | {formula} | ' +
                     (f'**{p}**' if pmax else p) + ' | ' +
                     (f'**{g}**' if gmax else g) + ' |')
        tex.append([component, formula, r'\textbf{'+p+'}' if pmax else p,
                    r'\textbf{'+g+'}' if gmax else g])
    latex_table(directory/'vo_nonadditivity_correlations.tex',
        r'Component dependence and joint non-additivity. Spearman correlations between '
        r'$S_F$ and each displayed difference use 12 two-seed direction means, separately '
        r'for $F=P,G$. Bold marks the largest correlation in each column. All quantities '
        r'share cube terms; differences between correlations are descriptive.',
        'tab:exp3-vo-nonadditivity',
        ['Reverted component', 'State difference', r'$\rho_P$', r'$\rho_G$'], tex)

    mixed = [r for r in rows if r['direction'] == 'mixed']
    exception = ('The mixed-sign row is retained: ' + '; '.join(
        f'{r["name"]} has $\\rho_P={r["spearman_P"]:.3f}$ and $\\rho_G={r["spearman_G"]:.3f}$'
        for r in mixed) + '.' if mixed else 'Every displayed row has nonnegative correlations.')
    primary = next(r for r in rows if r['primary'])
    rank = next(i for i, r in enumerate(rows, 1) if r['primary'])
    other_high = sum(r['consistently_high'] and not r['primary'] for r in rows)
    findings = (
        f'The QK reversion difference ranks **{rank}/12** by the weaker of its two absolute '
        f'correlations ({primary["minimum_absolute_rho"]:.3f}); '
        + ('No other retained difference meets the cross-score highlighting rule. '
           if other_high == 0 else
           f'{other_high} other retained differences meet the cross-score highlighting rule. ')
        +
        'This makes the association comparatively consistent across both scores. Some other '
        'differences have higher $P$-only correlations. The table cannot rule out coincidence '
        'or establish a statistically significant advantage: the comparisons were examined '
        'together, share cube terms, and use the same twelve dependent directions.'
    )
    return dict(MMD_CORRELATIONS='\n'.join(markdown), VO_CORRELATIONS='\n'.join(vo_md),
                ORIENTATION_EXCEPTION=exception, COMPARISON_FINDINGS=findings)


def preview_document(directory):
    """A standalone review sheet; figure and table snippets remain reusable."""
    (directory/'paper_assets_preview.tex').write_text(display_text(r"""
\documentclass[10pt]{article}
\usepackage[paperwidth=7in,paperheight=9in,margin=0.75in]{geometry}
\usepackage{times,amsmath,amssymb,graphicx,booktabs}
\pagestyle{empty}
\begin{document}
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{main_figure1_20m.pdf}
\caption{Spatial distance, component coordination, and source-QK reuse.
(a) Dataset spatial MMD$^2$ versus $F_{111}-F_{011}$.
(b) Joint non-additivity $S$ versus $F_{111}-F_{101}$.
(c) Source-QK agreement $F_{011}$ for all 12 transfers, grouped by target.
Circles/solid lines: $P$; triangles/dashed lines: $G$; colors: source datasets.
Scatter points are two-seed direction means; lines are separate descriptive linear fits
and annotations are Spearman correlations. In (c), the horizontal axis lists target
datasets and bar height is agreement. Solid/hatched bars show $P/G$ two-seed means;
shading groups human/animal face targets, with STL-10 as the natural-target reference.
Seed detail is shown in Figure 3. Missing combinations are unmeasured, not zero.
Only CelebA/STL-10 is reciprocal.}
\end{figure}
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{mmd_coordination_paper_20m.pdf}
\caption{Spatial distance and functional coordination.
All panels use dataset spatial MMD$^2$ and twelve direction means per score.
The minimum in (b) is over $h\in\{011,101,110\}$ and is computed within each seed before averaging.
Marker, color, trend, and correlation conventions follow Figure 1.
The statistics share cube terms and are not independent evidence.}
\end{figure}
\clearpage
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{directional_reuse_20m.pdf}
\caption{Expanded source-QK reuse across targets.
All 12 measured source/target pairs are retained separately. Source colors follow Figure 1;
solid/hatched bars show $P_{011}/G_{011}$ means.
Dots and segments show the two seed values and their range, not confidence intervals.
Shading groups human and animal face targets; the separate STL-10 target supplies
the CelebA-to-natural reference. No target averages or missing transfers are imputed.
Only CelebA/STL-10 is reciprocal.}
\end{figure}
\clearpage
\renewcommand{\thetable}{A\arabic{table}}
\input{nested_mmd_correlations.tex}
\renewcommand{\thetable}{\arabic{table}}
\setcounter{table}{0}
\input{vo_nonadditivity_correlations.tex}
\end{document}
""".lstrip()))
