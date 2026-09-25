"""Export a source's second selected example group in the draft flagship style."""
import argparse
from pathlib import Path
import hashlib
import json
import shutil

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.transforms import Bbox
import numpy as np

from build_summary_assets import Summary, HERE, OUT, DISPLAY_CODES, TITLES, sha
import paper_summary_assets as paper

ROOT=HERE.parents[1]

def scientific(value):
    """Exactly three significant digits, with an explicit power of ten."""
    mantissa, exponent=f'{value:.2e}'.split('e')
    return rf'{mantissa}\times 10^{{{int(exponent)}}}'


def build(source='celeba64', group=2, appendix=False):
    model=Summary()
    selection=json.loads((OUT/'selection_manifest.json').read_text())
    pick=selection['picks'][source][group-1]
    seed,index=pick['seed'],pick['index']
    entries=[e for e in selection['entries'] if e['source']==source and e['group_one_based']==group]
    assert all(e['adaptation_seed']==seed and e['selected_rows_one_based']==[index] for e in entries)
    targets=['aahq64','ffhq64','afhq_v2_64','stl10_64' if source=='celeba64' else 'celeba64']
    assert [e['target'] for e in entries]==targets
    names={'aahq64':'AAHQ','ffhq64':'FFHQ','afhq_v2_64':'AFHQ','stl10_64':'STL-10','celeba64':'CelebA'}
    stem='flagship_celeba_20m' if source=='celeba64' else 'flagship_imagenet64_20m'
    if appendix:
        stem=f'appendix_{source}_group{group}_20m'
    with plt.rc_context({**paper.STYLE, 'font.family': 'DejaVu Sans',
                         'mathtext.fontset': 'dejavusans'}):
        # The main figure places target labels and a geometry guide at right.
        # Appendix cubes retain their existing layout.
        main_flagship = source == 'celeba64' and not appendix
        width,gap,header=5.5,.016,.25 if main_flagship else .26
        sidebar=.95 if main_flagship else .48
        tile=(width-sidebar-7*gap)/8
        height=4*tile+3*gap+header
        image_left=.23 if main_flagship else sidebar
        image_right=image_left+8*tile+7*gap if main_flagship else width
        fig,axes=plt.subplots(4,8,figsize=(width,height))
        fig.subplots_adjust(left=image_left/width,
                            right=image_right/width,
                            top=1-header/height,bottom=0,
                            wspace=gap/tile,hspace=gap/tile)
        for row,entry in enumerate(entries):
            path=HERE/entry['source_npz']
            assert sha(path)==entry['source_sha256']
            assert entry['matches_quantitative_endpoint']
            with np.load(path) as a:
                images=a['images'][:,index-1].copy();codes=list(a['codes'])
            pair=source,entry['target']
            for col,code in enumerate(DISPLAY_CODES):
                ax=axes[row,col]
                ax.imshow(images[codes.index(code)].transpose(1,2,0),interpolation='nearest')
                ax.set_xticks([]);ax.set_yticks([])
                for spine in ax.spines.values():spine.set_visible(False)
                if row==0:
                    title=(('Fine-tuned\n(111)' if code=='111' else f'{TITLES[code]} ({code})') if main_flagship
                           else f'{code}\n{TITLES[code]}')
                    ax.set_title(title,fontsize=5.1 if main_flagship else 6.8,pad=3)
                if col==0 and not main_flagship:
                    ax.set_ylabel('→ '+names[entry['target']]+'\n'+
                                  rf'${scientific(model.mmd(pair))}$',
                                  rotation=0,ha='right',va='center',fontsize=5.5,labelpad=2,
                                  color='black')
        fig.canvas.draw()
        if main_flagship:
            top_image=axes[0,0].get_position().y1
            for first,last,label,color in (
                    (1,3,'Source MLP retained','#547697'),
                    (4,6,'Source MLP replaced','#986F8B')):
                x0=axes[0,first].get_position().x0-.002
                x1=axes[0,last].get_position().x1+.002
                y0=axes[-1,first].get_position().y0+.006
                y1=top_image+.242/height
                fig.add_artist(Rectangle((x0,y0),x1-x0,y1-y0,
                                         transform=fig.transFigure,fill=False,
                                         edgecolor=color,linewidth=.9,
                                         clip_on=False,zorder=20))
                fig.text((x0+x1)/2,top_image+.178/height,label,
                         ha='center',va='center',fontsize=6.4,color=color)
            x_label=(image_right+.15)/width
            for row,entry in enumerate(entries):
                pair=source,entry['target']
                y=(axes[row,0].get_position().y0+axes[row,0].get_position().y1)/2
                mmd_label=rf'$({100*model.mmd(pair):.2f}\times 10^{{-2}})$'
                fig.text(x_label,y,f'→ {names[entry["target"]]}\n{mmd_label}',
                         rotation=270,ha='center',va='center',fontsize=5.8,color='black')
            fig.text(x_label,top_image+.16/height,
                     r'(MMD$^2$)',ha='center',va='center',fontsize=5.7)
            arrow_x=(image_right+.305)/width
            fig.add_artist(FancyArrowPatch((arrow_x,axes[-1,0].get_position().y0+.012),
                                          (arrow_x,axes[0,0].get_position().y1-.012),
                                          transform=fig.transFigure,arrowstyle='-|>',
                                          mutation_scale=7,linewidth=.8,color='#4A4A4A',
                                          clip_on=False))
            fig.text((image_right+.435)/width,.5,'Similar geometry',rotation=90,
                     ha='center',va='center',fontsize=6,fontweight='bold',color='#4A4A4A')
        else:
            renderer=fig.canvas.get_renderer()
            for code in ('100','001','011'):
                col=DISPLAY_CODES.index(code)
                top=axes[0,col].get_position()
                bottom=axes[-1,col].get_position()
                title=axes[0,col].title.get_window_extent(renderer).transformed(fig.transFigure.inverted())
                x0,x1=top.x0-.001,top.x1+.001
                # Keep the bottom outline stroke inside the image boundary.
                y0,y1=bottom.y0+.9/144/height,title.y1+.004
                fig.add_artist(Rectangle((x0,y0),x1-x0,y1-y0,transform=fig.transFigure,
                                         fill=False,edgecolor='#ECA0A0',linewidth=.9,
                                         clip_on=False,zorder=20))
        paper.style_figure(fig)
        fig.canvas.draw()
        bounds=fig.get_tightbbox(fig.canvas.get_renderer())
        if main_flagship:
            # Keep the paper's 5.5-inch width when the right guide moves inward.
            bounds=Bbox.from_extents(0,bounds.y0,width,bounds.y1)
        else:
            # Patch bounds omit half the outline stroke; retain it at the top edge.
            bounds=Bbox.from_extents(bounds.x0,bounds.y0,bounds.x1,bounds.y1+.9/144)
        for ext in ('pdf','png'):
            fig.savefig(OUT/f'{stem}.{ext}',dpi=320,facecolor='white',
                        bbox_inches=bounds,pad_inches=0)
        plt.close(fig)
    draft=ROOT/'draft/figs'/('experiment3_flagship.pdf' if source=='celeba64' else 'experiment3_imagenet64.pdf')
    if appendix:
        draft=ROOT/'draft/figs'/f'experiment3_{source}_group{group}.pdf'
    shutil.copy2(OUT/(stem+'.pdf'),draft)
    epochs={e['source_training_epoch'] for e in entries}
    assert len(epochs)==1
    metadata=dict(source=source,adaptation_seed=seed,latent_index_one_based=index,group_one_based=group,
        states=DISPLAY_CODES,targets=targets,source_training_epoch=epochs.pop(),
        mmd_format=('MMD squared as mantissa times 10^-2, two decimals' if main_flagship else
                    'MMD squared; scientific notation with three significant digits; quantity named in caption'),
        label_style=('Clockwise vertical target labels at right with upward similar-geometry arrow' if main_flagship else
                     'Black target-only arrows; DejaVu Sans and matching math, as in other draft figures'),
        outlined_states=[] if main_flagship else ['100','001','011'],
        outline_groups={'Source MLP retained':['100','010','110'],
                        'Source MLP replaced':['001','101','011']} if main_flagship else {},
        outline_color=None if main_flagship else '#ECA0A0',
        spatial_mmd2={t:model.mmd((source,t)) for t in targets},
        entries=entries,builder_sha256=sha(Path(__file__)),
        pdf_sha256=sha(draft),draft_figure=str(draft.relative_to(ROOT)),
        width_inches=bounds.width,height_inches=bounds.height,
        row_label_fontsize=5.8 if main_flagship else 5.5,
        image_side_inches=tile,outer_padding_inches=0)
    (OUT/(stem+'.json')).write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Flagship: {draft}; {source} seed {seed}, index {index:03d}, all four targets.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',choices=['celeba64','imagenet64'],default='celeba64')
    parser.add_argument('--appendix',action='store_true',help='Export all selected groups absent from the main figures')
    args=parser.parse_args()
    if args.appendix:
        selection=json.loads((OUT/'selection_manifest.json').read_text())
        included={}
        for source,stem in [('celeba64','flagship_celeba_20m'),('imagenet64','flagship_imagenet64_20m')]:
            main=json.loads((OUT/f'{stem}.json').read_text())
            assert sha(ROOT/main['draft_figure'])==main['pdf_sha256']
            included[source]=(main['adaptation_seed'],main['latent_index_one_based'])
        for source,picks in selection['picks'].items():
            for group,pick in enumerate(picks,1):
                if (pick['seed'],pick['index']) != included.get(source):
                    build(source,group,appendix=True)
    else:
        build(args.source)
