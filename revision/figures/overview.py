"""PLD/DiT operations on schematic cartoon faces, at manuscript text width.

All images, attention weights, features, and outputs are illustrations. No
dataset examples or measured model quantities are used. Retains the published
figure filenames so the manuscript and existing links continue to work.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.path import Path as MplPath
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'outputs/revision'
WIDTH, HEIGHT = 384, 230
GRID, SIZE, TARGET = 6, 240, (2, 2)
C = dict(ink='#263344', muted='#6A7582', selection='#0072B2',
         pale_selection='#EAF4FA', prediction='#D55E00', pale_prediction='#FFF2E8',
         edge='#D4DAE1', key='#4D4D4D', target_patch='#D74353')

for name in ['Cantarell-Regular.otf', 'Cantarell-Bold.otf']:
    font_manager.fontManager.addfont(str(ROOT / 'revision/fonts' / name))
font_manager.fontManager.ttflist = [f for f in font_manager.fontManager.ttflist
                                  if Path(f.fname).name != 'Cantarell-VF.otf']
font_manager.fontManager._findfont_cached.cache_clear()
plt.rcParams.update({'font.family': 'Cantarell', 'mathtext.fontset': 'stix',
                     'font.size': 8.2, 'pdf.fonttype': 42,
                     'ps.fonttype': 42, 'svg.fonttype': 'none'})
fig = plt.figure(figsize=(WIDTH/72, HEIGHT/72), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0))
ax.axis('off')
texts = []


def text(x, y, label, size=8.2, color=None, weight='normal', **kwargs):
    ha = kwargs.pop('ha', 'center')
    a = ax.text(x, y, label, fontsize=size, color=color or C['ink'],
                va='center', ha=ha, weight=weight, zorder=30, **kwargs)
    texts.append(a)
    return a


def box(x, y, w, h, fill, edge='none', lw=.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle='round,pad=0,rounding_size=2.5',
                 facecolor=fill, edgecolor=edge, linewidth=lw, zorder=1))


def arrow(x1, y1, x2, y2, color=None, lw=.65, scale=4.7):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                 color=color or C['muted'], linewidth=lw,
                 mutation_scale=scale, shrinkA=.6, shrinkB=.6, zorder=12))


def face(index):
    """Original cartoons with varied location, hairstyle, complexion, and gaze."""
    styles = [
        (101, 109, 57, 73, '#F3BF8D', '#4A3530', '#5E87A3', False, 0, '#DAE8ED'),
        (158, 134, 66, 83, '#DFA174', '#423344', '#B0778D', True, 0, '#E1E8D8'),
        (84, 124, 62, 78, '#F0C8A0', '#664B32', '#818D78', True, -.30, '#E8E1DB'),
        (150, 102, 61, 74, '#B17A58', '#332F32', '#7B9BA6', False, .30, '#DCE4EA'),
        (135, 137, 67, 81, '#E8AA86', '#8C4E38', '#8A86A6', True, .10, '#E9E4D8'),
    ]
    cx, cy, rx, ry, skin, hair, shirt, woman, gaze, bg = styles[index]
    im = Image.new('RGB', (SIZE, SIZE), bg)
    d = ImageDraw.Draw(im)
    d.ellipse((cx-rx-25, cy+ry-9, cx+rx+25, cy+ry+85), fill=shirt)
    if woman:
        d.rounded_rectangle((cx-rx-12, cy-ry-7, cx+rx+12, cy+ry+27), radius=43, fill=hair)
    d.ellipse((cx-rx-3, cy-ry-7, cx+rx+3, cy+ry), fill=hair)
    d.ellipse((cx-rx, cy-ry+6, cx+rx, cy+ry), fill=skin)
    d.pieslice((cx-rx-3, cy-ry-7, cx+rx+3, cy+20), 180, 337, fill=hair)
    ey = cy-7
    for ex in [cx-rx*.37, cx+rx*.37]:
        ex += gaze*rx*.45
        d.ellipse((ex-5, ey-6, ex+5, ey+6), fill='#30323A')
        if woman:
            d.line((ex-5, ey-4, ex-9, ey-8), fill='#30323A', width=2)
    nose = cx + gaze*rx
    d.line((nose+1, cy+3, nose-4, cy+18, nose+5, cy+18), fill='#A66B52', width=3)
    mouth = cx + gaze*rx*.6
    d.arc((mouth-rx*.35, cy+9, mouth+rx*.35, cy+43), 12, 166,
          fill='#A44357' if woman else '#803E42', width=5)
    return np.asarray(im).astype(float)/255


clean = [face(0), face(1)]
rng = np.random.default_rng(20261113)
noisy = [np.clip(a + rng.normal(0, .045, a.shape), 0, 1) for a in clean]
fixed = np.zeros((GRID, GRID), bool)
fixed[1:4, 1:4] = True
rr, cc = np.meshgrid((np.arange(GRID)+.5)/GRID,
                     (np.arange(GRID)+.5)/GRID, indexing='ij')
fields = []
for cx, cy, sx, sy in [(.42, .45, .235, .30), (.68, .57, .30, .365)]:
    v = np.exp(-.5*((cc-cx)/sx)**2-.5*((rr-cy)/sy)**2)
    fields.append(.025 + .975*v/v.max())
# Match PLD's retained-patch budget, always including the target patch.
# These masks illustrate selection only; the architectural attention is soft.
attention_masks = []
for field in fields:
    mask = np.zeros((GRID, GRID), dtype=bool)
    mask[TARGET] = True
    for index in np.argsort(-field.ravel(), kind='stable'):
        if mask.sum() == fixed.sum():
            break
        mask.flat[index] = True
    attention_masks.append(mask)


def tile(rgb, x, y, w, weights=None, mark=True, grid=True, z=4, memory=False):
    pixels = np.asarray(rgb).copy()
    if weights is not None:
        assert weights.shape == (GRID, GRID)
        a = np.repeat(np.repeat(weights, SIZE//GRID, 0), SIZE//GRID, 1)
        # Excluded patches are black in both schematic binary selections.
        pixels = pixels*a[..., None]
    ax.imshow(pixels, extent=(x, x+w, y+w, y), interpolation='nearest', zorder=z)
    if grid:
        for t in np.linspace(0, w, GRID+1):
            ax.plot([x+t, x+t], [y, y+w], color='white', alpha=.85, lw=.28, zorder=z+1)
            ax.plot([x, x+w], [y+t, y+t], color='white', alpha=.85, lw=.28, zorder=z+1)
    if mark:
        r, c = TARGET
        ax.add_patch(Rectangle((x+c*w/GRID, y+r*w/GRID), w/GRID, w/GRID,
                     facecolor=C['target_patch'], alpha=.93, edgecolor='none', zorder=z+2))
    ax.add_patch(Rectangle((x, y), w, w, fill=False,
                 edgecolor=C['edge'], linewidth=.45, zorder=z+3))


def feature(x, y, face_width=10, depth=34):
    """One spatial patch, with a long axis for feature channels."""
    dx, dy = depth, -8
    ax.add_patch(Polygon([(x, y), (x+dx, y+dy),
                 (x+face_width+dx, y+dy), (x+face_width, y)],
                 facecolor='#B3D5E8', edgecolor=C['selection'], lw=.55, zorder=5))
    ax.add_patch(Polygon([(x+face_width, y), (x+face_width+dx, y+dy),
                 (x+face_width+dx, y+face_width+dy), (x+face_width, y+face_width)],
                 facecolor='#66AAD0', edgecolor=C['selection'], lw=.55, zorder=5))
    for t in [.22, .45, .68, .88]:
        ax.plot([x+t*dx, x+face_width+t*dx], [y+t*dy]*2,
                color='white', lw=.4, alpha=.75, zorder=6)
        ax.plot([x+face_width+t*dx]*2, [y+t*dy, y+face_width+t*dy],
                color='white', lw=.4, alpha=.75, zorder=6)
    ax.add_patch(Rectangle((x, y), face_width, face_width,
                 facecolor='#338EC1', edgecolor=C['selection'], lw=.65, zorder=7))


AGG_X, AGG_W = 85, 147
PRED_X, PRED_W = 249, 133
BODY_H = 102
IMAGE_X, CONTEXT_X, IMAGE_W = 36, 98, 34
FEATURE_X = 163

# The population maps are column headers, not an extra illustrated row.
text(53, 8, r'$Y_t$', 11)
arrow(71, 9, 88, 9)
text(AGG_X+AGG_W/2, 8, r'Aggregator  $\mathcal{S}^*_{m,t}$', 9.4, C['selection'], 'bold')
arrow(AGG_X+AGG_W+1, 9, PRED_X, 9, C['selection'])
text(PRED_X+PRED_W/2, 8, r'Predictor  $g^*_{m,t}$', 9.4, C['prediction'], 'bold')


def aggregate(y, learned):
    box(AGG_X, y, AGG_W, BODY_H, C['pale_selection'])
    text(AGG_X+AGG_W/2, y+8, 'Input-adaptive selection' if learned else 'Fixed selection for all inputs',
         8.3, C['selection'], 'bold')
    if learned:
        text(CONTEXT_X, y+21, r'$[\operatorname{Attn}(Y_t;Q,K)]_{m,l}\,OV$',
             8.7, C['selection'], ha='left')
    else:
        text(CONTEXT_X, y+21, r'$S_{m,t}$', 10, C['selection'], ha='left')
    text(FEATURE_X+21, y+60, 'aggregated patch', 7.5, C['selection'])
    for i, offset in enumerate([29, 64]):
        row_y = y+offset
        tile(noisy[i], IMAGE_X, row_y, IMAGE_W)
        arrow(IMAGE_X+IMAGE_W+2, row_y+IMAGE_W/2, CONTEXT_X-3,
              row_y+IMAGE_W/2)
        tile(noisy[i], CONTEXT_X, row_y, IMAGE_W,
             weights=attention_masks[i].astype(float) if learned else fixed.astype(float))
        points = ([(1, 2), (2, 2), (3, 2)] if i == 0 else
                  [(2, 3), (3, 4), (4, 4)]) if learned else [(1, 1), (2, 2), (3, 3)]
        for j, (r, c) in enumerate(points):
            assert (attention_masks[i] if learned else fixed)[r,c]
            arrow(CONTEXT_X+(c+.5)*IMAGE_W/GRID, row_y+(r+.5)*IMAGE_W/GRID,
                  FEATURE_X-1, row_y+12+j*3, C['selection'], lw=.5, scale=3.4)
        feature(FEATURE_X, row_y+12, depth=26)
        text(FEATURE_X+43, row_y+17, r'$s$', 10, C['selection'])


def predict(y, learned):
    box(PRED_X, y, PRED_W, BODY_H, C['pale_prediction'])
    text(PRED_X+PRED_W/2, y+8, 'MLP (GeLU retrieval)' if learned else
         'RBF (Softmax retrieval)', 8.1, C['prediction'], 'bold')
    bank_rng = np.random.default_rng(37)
    for i, x in enumerate([PRED_X+8, PRED_X+51, PRED_X+94]):
        if learned:
            for dx, dy, edge in [(0, 0, C['key']), (4, 4, C['prediction'])]:
                rgb = bank_rng.uniform(.12, .88, (6, 6, 3))
                ax.imshow(rgb, extent=(x+dx, x+dx+21, y+41+dy, y+20+dy),
                          interpolation='nearest', zorder=4+dy)
                ax.add_patch(Rectangle((x+dx, y+20+dy), 21, 21,
                             fill=False, edgecolor=edge, lw=.7, zorder=5+dy))
        else:
            tile(face(2+i), x, y+20, 25, weights=fixed.astype(float), memory=True)
        arrow(x+12, y+47, PRED_X+19+29*i, y+55, C['prediction'], lw=.6, scale=4)
    op_x, op_y, op_w, op_h = PRED_X+6, y+57, 83, 24
    box(op_x, op_y, op_w, op_h, 'white', C['prediction'], lw=.7)
    formula = (r'$\sum\;\operatorname{gelu}(\cdot)\,\tilde{v}^{(r)}$' if learned else
               r'$\sum\;\operatorname{sm}_i(\cdot)\,x_m^{(i)}$')
    text(op_x+op_w/2, op_y+op_h/2, formula, 10.5, C['prediction'])
    text(op_x+op_w/2, y+94,
         r'$g_t^{\mathrm{mlp}}(s)$' if learned else r'$g_{m,t}^{\mathrm{rbf}}(s)$', 10)
    # Only the upper example is connected. The lower aggregate is not pooled.
    line = MplPath([(FEATURE_X+49, y+46), (PRED_X-9, y+46), (PRED_X-9, op_y+op_h/2),
                    (op_x-1, op_y+op_h/2)],
                   [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO])
    ax.add_patch(FancyArrowPatch(path=line, arrowstyle='-|>',
                 color=C['selection'], linewidth=.75, mutation_scale=4.8, zorder=12))
    r, c = TARGET
    patch = clean[0][r*SIZE//GRID:(r+1)*SIZE//GRID,
                     c*SIZE//GRID:(c+1)*SIZE//GRID]
    tile(patch, PRED_X+113, y+61.5, 15, mark=False, grid=False)
    arrow(op_x+op_w+2, y+69, PRED_X+110, y+69, C['prediction'], scale=4.4)


for y, learned in [(18, False), (126, True)]:
    text(15, y+52, 'DiT' if learned else 'PLD', 9.5, weight='bold')
    if learned:
        text(15, y+65, '1-layer', 7.5, C['muted'])
    aggregate(y, learned)
    predict(y, learned)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
outside, overlaps = [], []
for i, a in enumerate(texts):
    b = a.get_window_extent(renderer)
    if b.x0 < 0 or b.y0 < 0 or b.x1 > fig.bbox.x1 or b.y1 > fig.bbox.y1:
        outside.append(a.get_text())
    for other in texts[i+1:]:
        if b.overlaps(other.get_window_extent(renderer)):
            overlaps.append([a.get_text(), other.get_text()])
OUT.mkdir(parents=True, exist_ok=True)
for suffix in ['pdf', 'svg', 'png']:
    fig.savefig(OUT/f'aggregator_predictor_overview.{suffix}', dpi=320,
                metadata={'Creator': 'pixel-dit reproducible schematic'} if suffix == 'pdf' else None)
checks = dict(palette=C, text_outside_canvas=outside, text_overlaps=overlaps,
              patch_grid=[GRID, GRID], target_zero_based=list(TARGET),
              pld_selected_patches=int(fixed.sum()),
              dit_weights='Binary schematic selection from soft scores, target retained, same nine-patch budget as PLD',
              dit_selected_patches=[int(mask.sum()) for mask in attention_masks],
              dit_masks=[np.argwhere(mask).tolist() for mask in attention_masks],
              inputs_per_estimator=2, predictors_per_estimator=1,
              predictor_connection='Upper example only; no pooling between examples',
              feature_label='aggregated patch (once per aggregator box); s on each feature', citations_in_art=False,
              context_encoding='Binary masks for PLD and attention illustrations; excluded cells black; white grid',
              matrix_labels='S_m,t above masked input; [Attn(Y_t;Q,K)]_m,l O V for attention',
              context_to_feature_gap_pt=FEATURE_X-(CONTEXT_X+IMAGE_W),
              input_faces='Man and woman; training cartoons vary position and gaze',
              population_display='Column headings only',
              image_origin='Original procedural cartoon illustrations',
              scientific_scope='Schematic; no measured model outputs or attention',
              point_size=[WIDTH, HEIGHT], smallest_text_pt=min(a.get_fontsize() for a in texts),
              output_sha256={s: hashlib.sha256((OUT/f'aggregator_predictor_overview.{s}').read_bytes()).hexdigest()
                             for s in ['pdf', 'svg', 'png']})
(HERE/'layout_checks.json').write_text(json.dumps(checks, indent=2)+'\n')
assert not outside, outside
assert not overlaps, overlaps
plt.close(fig)
print(json.dumps(checks, indent=2))
