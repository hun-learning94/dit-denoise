"""Combine selected transfer latents by source for the draft appendix."""
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
import numpy as np

from build_summary_assets import Summary, HERE, OUT, DISPLAY_CODES, TITLES, sha
import paper_summary_assets as paper


ROOT = HERE.parents[1]
NAMES = {'aahq64': 'AAHQ', 'ffhq64': 'FFHQ', 'afhq_v2_64': 'AFHQ',
         'stl10_64': 'STL-10', 'celeba64': 'CelebA'}


def build(source, blocks):
    selection = json.loads((OUT / 'selection_manifest.json').read_text())
    model = Summary()
    targets = ['aahq64', 'ffhq64', 'afhq_v2_64',
               'stl10_64' if source == 'celeba64' else 'celeba64']
    assert blocks == {
        'celeba64': [(3, 32), (1, 14), (1, 3)],
        'stl10_64': [(2, 9), (3, 14), (1, 3)],
        'imagenet64': [(1, 1), (2, 12), (3, 40)],
    }[source]
    entries = []
    images_by_entry = []
    for figure_block, (group, index) in enumerate(blocks, 1):
        rows = [entry for entry in selection['entries']
                if entry['source'] == source and entry['group_one_based'] == group]
        assert [entry['target'] for entry in rows] == targets
        pick = selection['picks'][source][group - 1]
        source_image = None
        shared_latent_hash = None
        for entry in rows:
            assert entry['adaptation_seed'] == pick['seed']
            assert entry['matches_quantitative_endpoint']
            path = HERE / entry['source_npz']
            assert sha(path) == entry['source_sha256']
            with np.load(path) as data:
                codes = list(data['codes'])
                images = data['images'][:, index - 1].copy()
                latent_hash = hashlib.sha256(data['latents'][index - 1].tobytes()).hexdigest()
            if index == pick['index']:
                assert entry['selected_rows_one_based'] == [index]
                assert entry['selected_latent_sha256'] == latent_hash
            if shared_latent_hash is None:
                shared_latent_hash = latent_hash
            else:
                assert latent_hash == shared_latent_hash
            assert set(codes) == set(DISPLAY_CODES)
            image_map = {code: images[codes.index(code)] for code in DISPLAY_CODES}
            if source_image is None:
                source_image = image_map['000']
            else:
                assert np.array_equal(source_image, image_map['000'])
            displayed_entry = dict(entry)
            displayed_entry['source_selection_group_one_based'] = group
            displayed_entry['group_one_based'] = figure_block
            displayed_entry['selected_rows_one_based'] = [index]
            displayed_entry['selected_latent_sha256'] = latent_hash
            entries.append(displayed_entry)
            images_by_entry.append(image_map)

    # Match Figure 1's image size and column labels. Extra space separates latents.
    width, gap, header, group_gap = 5.5, .016, .25, .075
    tile = (width - .95 - 7 * gap) / 8
    nrows = len(entries)
    height = header + nrows * tile + (nrows - 1) * gap + (len(blocks) - 1) * group_gap
    image_left = .34
    image_right = image_left + 8 * tile + 7 * gap
    stem = f'appendix_{source}_combined_20m'
    with plt.rc_context(paper.STYLE):
        fig = plt.figure(figsize=(width, height))
        axes = []
        for row, (entry, images) in enumerate(zip(entries, images_by_entry)):
            y = height - header - tile - row * (tile + gap) - (row // 4) * group_gap
            row_axes = []
            for col, code in enumerate(DISPLAY_CODES):
                x = image_left + col * (tile + gap)
                ax = fig.add_axes((x / width, y / height, tile / width, tile / height))
                ax.imshow(images[code].transpose(1, 2, 0), interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if row == 0:
                    ax.set_title(f'{TITLES[code]} ({code})', fontsize=5.1, pad=3)
                row_axes.append(ax)
            axes.append(row_axes)

        top_image = axes[0][0].get_position().y1
        for first, last, label, color in (
                (1, 3, 'Target attention weights only', '#547697'),
                (4, 6, 'Target MLP weights included', '#986F8B')):
            x0 = axes[0][first].get_position().x0 - .002
            x1 = axes[0][last].get_position().x1 + .002
            y0 = axes[-1][first].get_position().y0 + .006
            y1 = top_image + .242 / height
            fig.add_artist(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                     transform=fig.transFigure, fill=False,
                                     edgecolor=color, linewidth=.9,
                                     clip_on=False, zorder=20))
            fig.text((x0 + x1) / 2, top_image + .178 / height, label,
                     ha='center', va='center', fontsize=6.4, color=color)
        x_label = (image_right + .15) / width
        for row, entry in enumerate(entries):
            pair = source, entry['target']
            y = (axes[row][0].get_position().y0 + axes[row][0].get_position().y1) / 2
            fig.text(x_label, y,
                     f'→ {NAMES[entry["target"]]}\n({100 * model.mmd(pair):.2f})',
                     rotation=270, ha='center', va='center', fontsize=5.8)
        fig.text(x_label, top_image + .16 / height,
                 '→ Target\n' + r'(MMD$^2$)', ha='center', va='center', fontsize=5.7)

        paper.style_figure(fig)
        bounds = Bbox.from_extents(0, 0, width, height)
        for extension in ('pdf', 'png'):
            fig.savefig(OUT / f'{stem}.{extension}', dpi=320, facecolor='white',
                        bbox_inches=bounds, pad_inches=0)
        plt.close(fig)

    draft = ROOT / 'draft/figs' / f'experiment3_{source}_combined.pdf'
    shutil.copy2(OUT / f'{stem}.pdf', draft)
    metadata = {
        'source': source,
        'selected_latents': [dict(seed=selection['picks'][source][group - 1]['seed'],
                                  index=index) for group, index in blocks],
        'states': DISPLAY_CODES, 'targets': targets,
        'mmd_format': 'MMD squared in units of 10^-2, two decimals',
        'layout': 'Four target rows per latent; shared state columns and full-height component boxes',
        'entries': entries, 'builder_sha256': sha(Path(__file__)),
        'pdf_sha256': sha(draft), 'draft_figure': str(draft.relative_to(ROOT)),
        'width_inches': width, 'height_inches': height, 'image_side_inches': tile,
    }
    (OUT / f'{stem}.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'{source}: {len(blocks)} latents, {nrows} rows -> {draft}')


if __name__ == '__main__':
    build('celeba64', [(3, 32), (1, 14), (1, 3)])
    build('stl10_64', [(2, 9), (3, 14), (1, 3)])
    build('imagenet64', [(1, 1), (2, 12), (3, 40)])
