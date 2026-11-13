"""Half-text-width vertical MLP/SMLP schematic; native vector exports."""
from revision.runtime import path as _release_path
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT/'outputs/revision'
OUT.mkdir(parents=True, exist_ok=True)
REFERENCE = Path(_release_path('assets/external/draw_dit_equation_mapping_compact.py'))
NOTATION = ROOT/'reports/experiment1/predictor_notation.md'
W, H = 198, 235
INK, GRAY = '#203040', '#758394'
BLUE, TEAL, PURPLE = '#2675B8', '#087F7B', '#8B52A1'
EDGE = '#A9B5C1'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':7.5,
    'mathtext.fontset':'dejavusans','pdf.fonttype':42,'svg.fonttype':'none',
    'text.color':INK})
fig=plt.figure(figsize=(W/72,H/72),facecolor='white')
ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(H,0));ax.axis('off')
contained=[]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).is_file() else None

def label(x,y,s,size=7.5,color=INK,**kw):
    return ax.text(x,y,s,ha=kw.pop('ha','center'),va='center',fontsize=size,
        color=color,zorder=10,linespacing=1.12,**kw)

def box(x,y,w,h,edge=EDGE,face='white',dashed=False,lw=.65,z=2):
    patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=2',
        edgecolor=edge,facecolor=face,linewidth=lw,
        linestyle=(0,(2.6,2)) if dashed else 'solid',zorder=z)
    ax.add_patch(patch)
    return (x,y,w,h)

def arrow(x0,y0,x1,y1,color=GRAY,lw=.65):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle='-|>',
        mutation_scale=4.5,color=color,linewidth=lw,shrinkA=0,shrinkB=0,zorder=4))

def stage(cx,y,w,h,text,edge=EDGE,face='white',size=7.5,color=INK):
    rect=box(cx-w/2,y,w,h,edge,face)
    artist=label(cx,y+h/2,text,size,color)
    contained.append((artist,rect))

for cx,learned in [(39,False),(126,True)]:
    label(cx if not learned else 143,8,
          '(a) DiT + MLP' if not learned else '(b) DiT + SMLP',8.0,weight='medium')
    label(cx,24,r'$Y_t$',9)
    arrow(cx,30,cx,35)
    stage(cx,35,57,20,'Patch embed\n+ position',BLUE,'#EAF3FC',7.0)
    # The first and last layers stand for the complete depth-L residual stack.
    box(cx-31,60,62,122,GRAY,'#F5F7FA',dashed=True,z=0)
    arrow(cx,55,cx,66)
    stage(cx,66,47,16,r'Attention$_1$',size=7.1)
    arrow(cx,82,cx,90)
    edge,face=(TEAL,'#EAF6F3') if learned else (PURPLE,'#F4EDF8')
    for y,index in [(90,'1'),(153,'L')]:
        rect=box(cx-23.5,y,47,24,edge,face,lw=.8)
        name='SMLP' if learned else 'MLP'
        contained.append((label(cx,y+7,rf'{name}$_{index}$',8.0,edge),rect))
        contained.append((label(cx,y+18,'layer maps' if learned else 'own K/V',6.8,GRAY),rect))
    arrow(cx,114,cx,117)
    label(cx,121,r'$\vdots$',10,GRAY)
    arrow(cx,125,cx,129)
    stage(cx,129,47,16,r'Attention$_L$',size=7.1)
    arrow(cx,145,cx,153)
    arrow(cx,177,cx,187)
    stage(cx,187,57,22,'Final AdaLN\nLinear head',size=7.0)
    arrow(cx,209,cx,217)
    label(cx,224,r'$\widehat{X}_0$',9)

# A single base K/V pair supplies every layer's own SMLP projection maps.
label(179,95,'Shared\nbank',7.0,TEAL,weight='medium')
k=box(165,105,13,19,TEAL,'#EAF6F3',lw=.8)
v=box(181,105,13,19,TEAL,'#EAF6F3',lw=.8)
contained += [(label(171.5,114.5,r'$K$',8.0,TEAL),k),
              (label(187.5,114.5,r'$V$',8.0,TEAL),v)]
ax.plot([178,179.5,181],[114.5,114.5,114.5],color=TEAL,lw=.65,zorder=3)
# Both banks feed the same bus; arrows enter the layer-specific maps.
ax.plot([179.5,179.5,160,160],[114.5,137,137,102],color=TEAL,lw=.65,zorder=3)
ax.plot([160,160],[137,165],color=TEAL,lw=.65,zorder=3)
arrow(160,102,149.5,102,TEAL)
arrow(160,165,149.5,165,TEAL)

fig.canvas.draw();renderer=fig.canvas.get_renderer()
outside=[];overlaps=[];overflow=[]
for i,a in enumerate(ax.texts):
    b=a.get_window_extent(renderer)
    if b.x0<0 or b.y0<0 or b.x1>fig.bbox.x1 or b.y1>fig.bbox.y1:outside.append(a.get_text())
    for other in ax.texts[i+1:]:
        if b.overlaps(other.get_window_extent(renderer)):overlaps.append([a.get_text(),other.get_text()])
for a,(x,y,w,h) in contained:
    b=a.get_window_extent(renderer).transformed(ax.transData.inverted())
    left,right=sorted([b.x0,b.x1]);top,bottom=sorted([b.y0,b.y1])
    if left<x+1 or right>x+w-1 or top<y+.5 or bottom>y+h-.5:overflow.append(a.get_text())
assert not outside,outside
assert not overlaps,overlaps
assert not overflow,overflow
for suffix in ['pdf','svg','png']:
    fig.savefig(OUT/('mlp_vs_smlp.'+suffix),dpi=400,facecolor='white')
checks=dict(passed=True,canvas_inches=[W/72,H/72],text_width_fraction=.5,
    font='DejaVu Sans',minimum_font_pt=min(t.get_fontsize() for t in ax.texts),
    text_outside_canvas=outside,text_overlaps=overlaps,labels_outside_boxes=overflow,
    reference_style_source='editorial style reference',reference_style_sha256=sha(REFERENCE),
    notation_source='predictor notation',notation_sha256=sha(NOTATION),
    implementation_sources={p:sha(ROOT/p) for p in ['models.py','model_grouped_dpd6.py']},
    architecture='First and last layers shown, with an ellipsis for intermediate depth. Independent MLP banks versus common SMLP base K/V banks and layer-specific maps.',
    omitted='Residual additions, AdaLN within branches, time embedding/gates, biases and headwise details are suppressed in the schematic; they are retained in the models.',
    no_tied_smlp_projections=True,no_training_images_in_shared_bank=True,
    outputs_sha256={s:sha(OUT/('mlp_vs_smlp.'+s)) for s in ['pdf','svg','png']})
(OUT/'architecture_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
plt.close(fig)
print(json.dumps(checks,indent=2))
