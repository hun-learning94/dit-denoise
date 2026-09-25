"""CPU-only sparse GMRF study on a reproducible random weighted patch graph."""
import os
# Set BLAS limits before importing NumPy/SciPy; no GPU or PyTorch is needed.
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]=os.environ.get('EXPERIMENT0_CPU_THREADS','1')
import argparse
import json
from pathlib import Path
import resource
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.sparse.csgraph import shortest_path, connected_components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


def precision(grid,alpha,beta):
    diag=np.r_[1,np.full(grid-2,2),1]
    L=sparse.diags([-np.ones(grid-1),diag,-np.ones(grid-1)],[-1,0,1],format='csc')
    return alpha*sparse.eye(grid*grid,format='csc')+beta*sparse.kronsum(L,L,format='csc')


def target_indices(grid):
    """Two fixed, spatially separated display targets, independent of graph RNG."""
    positions=[(max(0,grid//4-1),max(0,grid//4-1)),
               (3*grid//4,3*grid//4)]
    return np.array([r*grid+c for r,c in positions])


def random_precision(grid,alpha,beta,seed=20260916,edge_count=None):
    """Uniform G(n,E): choose E distinct unordered pairs, with no spatial bias."""
    n=grid*grid
    edge_count=round(1.5*n) if edge_count is None else edge_count
    if grid<3 or alpha<=0 or beta<=0 or not 0<=edge_count<=n*(n-1)//2:
        raise ValueError('Invalid graph parameters')
    rng=np.random.default_rng(seed)
    candidates=np.column_stack(np.triu_indices(n,k=1))
    pairs=candidates[rng.choice(len(candidates),size=edge_count,replace=False)]
    weights=beta*rng.uniform(.5,1.5,len(pairs))
    rows=np.r_[pairs[:,0],pairs[:,1]]; cols=np.r_[pairs[:,1],pairs[:,0]]
    adjacency=sparse.csc_matrix((np.r_[weights,weights],(rows,cols)),shape=(n,n))
    degrees=np.asarray(adjacency.sum(1)).ravel()
    Q=sparse.diags(alpha+degrees,format='csc')-adjacency
    return Q,adjacency,pairs,weights


def certificate(lo,hi,sigma,r):
    chi=(1+sigma*sigma*hi)/(1+sigma*sigma*lo)
    rho=(np.sqrt(chi)-1)/(np.sqrt(chi)+1)
    return (1-chi**-.5)**2/(lo*(1+sigma*sigma*lo))*rho**(2*np.asarray(r))


def certified_radius(lo,hi,sigma,epsilon):
    c0=certificate(lo,hi,sigma,0)
    if c0<=epsilon: return 0
    rho=(np.sqrt((1+sigma*sigma*hi)/(1+sigma*sigma*lo))-1)/(np.sqrt((1+sigma*sigma*hi)/(1+sigma*sigma*lo))+1)
    return max(0,int(np.ceil(np.log(c0/epsilon)/(2*np.log(1/rho)))))


def local_error(Q,sigma,target,inside,w):
    """Stable resolvent identity, avoiding subtraction of posterior variances."""
    e=np.zeros(Q.shape[0]); e[target]=1
    A=Q+sparse.diags(inside.astype(float)/(sigma*sigma),format='csc')
    factor=splu(A,permc_spec='MMD_AT_PLUS_A')
    x=factor.solve(e)
    residual=float(np.linalg.norm(A@x-e))
    outside=~inside
    return float(w[outside]@x[outside]),residual,factor.L.nnz+factor.U.nnz


def calculate(grid=16,alpha=.2,beta=1.,noise_count=41,progress=True,seed=20260916,edge_count=None):
    if grid<3 or alpha<=0 or beta<=0: raise ValueError('Need grid>=3, alpha>0, beta>0')
    start=time.perf_counter(); n=grid*grid
    Q,adjacency,edges,weights=random_precision(grid,alpha,beta,seed,edge_count)
    lo=alpha; hi=alpha+2*float(np.asarray(adjacency.sum(1)).max())
    endpoint=np.sqrt((lo**(1/3)+hi**(1/3))/(lo**(2/3)*hi**(2/3)))
    sigmas=np.unique(np.r_[0.,np.geomspace(.01,100,noise_count),.1,1.,10.,endpoint])
    targets=target_indices(grid)
    rr,cc=np.indices((grid,grid)); coords=np.c_[rr.ravel(),cc.ravel()]
    dist=shortest_path(adjacency,directed=False,unweighted=True,indices=targets)
    radii=np.arange(int(dist[np.isfinite(dist)].max())+1); qs=np.array([.75,.9,.95]); tolerances=np.array([1e-2,1e-3,1e-4])
    shape=(len(sigmas),len(targets),len(radii))
    errors=np.full(shape,np.nan); tails=np.zeros(shape); bounds=np.zeros((len(sigmas),len(radii)))
    exact=np.zeros((len(sigmas),len(targets),len(tolerances)),int)
    cert=np.zeros((len(sigmas),len(tolerances)),int)
    sens=np.zeros((len(sigmas),len(targets),len(qs)),int)
    total=np.zeros((len(sigmas),len(targets))); max_residual=0.; max_fill=0; solves=0
    for si,sigma in enumerate(sigmas):
        bounds[si]=certificate(lo,hi,sigma,radii)
        cert[si]=[certified_radius(lo,hi,sigma,e) for e in tolerances]
        if sigma==0:
            errors[si]=0; total[si]=1; continue
        e=np.zeros((n,len(targets))); e[targets,np.arange(len(targets))]=1
        H=sparse.eye(n,format='csc')+sigma*sigma*Q
        factor=splu(H,permc_spec='MMD_AT_PLUS_A'); W=factor.solve(e).T
        max_residual=max(max_residual,float(np.linalg.norm(H@W.T-e)))
        max_fill=max(max_fill,factor.L.nnz+factor.U.nnz)
        del factor
        for ti,target in enumerate(targets):
            energy=W[ti]**2; total[si,ti]=energy.sum()
            reachable=np.isfinite(dist[ti])
            if np.any(energy[~reachable]>1e-25): raise ArithmeticError('Disconnected nodes have influence')
            shells=np.bincount(dist[ti,reachable].astype(int),weights=energy[reachable],minlength=len(radii))
            tails[si,ti]=np.r_[np.cumsum(shells[:0:-1])[::-1],0.]
            cdf=np.cumsum(shells)/energy.sum()
            sens[si,ti]=[int(np.flatnonzero(cdf>=q-1e-12)[0]) for q in qs]
            for r in range(int(dist[ti,reachable].max())+1):
                inside=dist[ti]<=r
                if inside.all(): value=0.; residual=0.; fill=0
                else:
                    value,residual,fill=local_error(Q,sigma,int(target),inside,W[ti]); solves+=1
                max_residual=max(max_residual,residual); max_fill=max(max_fill,fill)
                errors[si,ti,r]=value
                lower=(1/hi+sigma*sigma)*tails[si,ti,r]
                upper=(1/lo+sigma*sigma)*tails[si,ti,r]
                tol=1e-8*max(value,upper,bounds[si,r])+1e-25
                if not lower-tol<=value<=upper+tol: raise ArithmeticError('Sensitivity spectral bounds violated')
                if value>bounds[si,r]+tol: raise ArithmeticError('Chebyshev certificate violated')
                # Exact error is nonincreasing as contexts nest. Once below the
                # smallest tolerance, all reported minimum radii are identified.
                if r>=8 and value<=tolerances.min(): break
            available=errors[si,ti,np.isfinite(errors[si,ti])]
            if np.any(np.diff(available)>1e-10): raise ArithmeticError('Locality error increased with radius')
            exact[si,ti]=[int(np.flatnonzero(errors[si,ti]<=eps)[0]) for eps in tolerances]
        if progress and (si%5==0 or si==len(sigmas)-1):
            print(f'grid={grid} noise={si+1}/{len(sigmas)} sigma={sigma:g} elapsed={time.perf_counter()-start:.1f}s',flush=True)
    if max_residual>1e-7: raise ArithmeticError(f'Large linear solve residual: {max_residual}')
    if np.any(np.diff(cert[sigmas<=endpoint],axis=0)<0): raise ArithmeticError('Certificate monotonicity check failed')
    meta=dict(grid=grid,nodes=n,edges=len(edges),alpha=alpha,beta=beta,q_min=lo,q_max=hi,
              seed=seed,weight_range=[.5*beta,1.5*beta],
              graph_construction='Uniform G(n,E), E distinct unordered pairs sampled without replacement; no connectivity conditioning',
              components=int(connected_components(adjacency,directed=False)[0]),
              displayed_ball_radius=3,ball_sizes=(dist<=3).sum(1).tolist(),
              reachable_nodes=np.isfinite(dist).sum(1).tolist(),
              spectral_bounds='q_min=alpha; q_max=alpha+2*maximum weighted degree (valid upper bound)',
              sigma_abs=endpoint,seconds=time.perf_counter()-start,local_factorizations=solves,
              max_linear_residual=max_residual,max_factor_nnz=max_fill,
              process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              threads=int(os.environ.get('EXPERIMENT0_CPU_THREADS','1')),
              target_names=['Patch 1','Patch 2'],target_coordinates=[divmod(int(t),grid) for t in targets],distance='unweighted shortest-path hops on the retained random graph',
              checks='Spectral bounds, Chebyshev bound, nested-context error, bounded-range certificate monotonicity passed',
              omitted_errors='NaN after r>=8 and error<=1e-4; minimum radii identified exactly up to floating-point error')
    return dict(edge_pairs=edges,edge_weights=weights,distances=dist,sigmas=sigmas,radii=radii,targets=targets,qs=qs,tolerances=tolerances,error=errors,
                tail=tails,total=total,certificate=bounds,certified_radius=cert,exact_radius=exact,
                sensitivity_radius=sens,sigma_abs=endpoint,q_min=lo,q_max=hi,metadata=json.dumps(meta))


def save(fig,path):
    for suffix in ['png','pdf']: fig.savefig(path.with_suffix('.'+suffix),dpi=240,bbox_inches='tight',pad_inches=.025)
    plt.close(fig)


TARGET_COLORS=['#0072B2','#D55E00']
NEIGHBOR_MARKERS=['s','^']


def graph_panel(ax,z):
    meta=json.loads(str(z['metadata'])); g=meta['grid']
    coords=np.array([(c,r) for r in range(g) for c in range(g)])
    pairs=z['edge_pairs']; segments=coords[pairs]
    background='#e5e7e9'
    ax.add_collection(LineCollection(segments,colors=background,linewidths=.5,zorder=0))
    ax.scatter(coords[:,0],coords[:,1],s=10,c=background,zorder=1)
    for ti,(target,name,color,marker) in enumerate(zip(z['targets'],meta['target_names'],TARGET_COLORS,NEIGHBOR_MARKERS)):
        adjacent=(z['distances'][ti]<=3)&(z['distances'][ti]>0)
        incident=np.any(pairs==target,axis=1)
        ax.add_collection(LineCollection(segments[incident],colors=color,linewidths=1.2,alpha=.5,zorder=2))
        ax.scatter(coords[adjacent,0],coords[adjacent,1],s=50+ti*38,marker=marker,
                   facecolors='none',edgecolors=color,linewidths=1.5,zorder=3,
                   label=name.capitalize()+f': $|B_3|$={int((z["distances"][ti]<=3).sum())}')
        ax.scatter(*coords[target],s=85,c=color,marker='o',edgecolors='white',linewidths=.8,zorder=4)
        ax.annotate(name.capitalize(),coords[target],xytext=(5,-13 if ti else 7),
                    textcoords='offset points',fontsize=12,color=color,weight='bold',zorder=5)
    ax.set(aspect='equal',xlim=(-.7,g-.3),ylim=(g-.3,-.7),xlabel='Patch column',ylabel='Patch row',
           title=f'(a) $G_0$: {g}×{g} nodes, {meta["edges"]} edges')
    ax.set_xticks([0,4,8,12,15]); ax.set_yticks([0,4,8,12,15])
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.115),ncol=2,fontsize=11,frameon=False,
              handletextpad=.3,columnspacing=.7)


def plot(z,out):
    with plt.rc_context({'font.size':13,'axes.titlesize':14,'axes.labelsize':13,
                         'xtick.labelsize':11,'ytick.labelsize':11,'axes.labelpad':3,
                         'axes.titlepad':7,'legend.fontsize':11,'pdf.fonttype':42,
                         'ps.fonttype':42}):
        _plot(z,out)


def _plot(z,out):
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='compressed')
    fig.get_layout_engine().set(w_pad=.015,h_pad=.015,wspace=.025,hspace=.01)
    for ax in axes:
        ax.set_box_aspect(1)
    graph_panel(axes[0],z)
    s=z['sigmas']; keep=s>0; meta=json.loads(str(z['metadata']))
    for ti,(name,color) in enumerate(zip(meta['target_names'],TARGET_COLORS)):
        axes[1].step(s[keep],z['exact_radius'][keep,ti,1],where='post',color=color,
                     label=name.capitalize()+' exact',linewidth=1.8,
                     marker=NEIGHBOR_MARKERS[ti],markersize=3,markevery=(ti*2,9))
        for q,style in [(.90,'-'),(.95,'--')]:
            qi=int(np.flatnonzero(np.isclose(z['qs'],q))[0])
            axes[2].step(s[keep],z['sensitivity_radius'][keep,ti,qi],where='post',
                         color=color,linestyle=style,linewidth=1.7,label=f'{name.capitalize()} {q:.0%}',
                         marker=NEIGHBOR_MARKERS[ti],markersize=3,markevery=(ti*2,9))
    axes[1].step(s[keep],z['certified_radius'][keep,1],where='post',color='#222222',
                 linestyle='--',linewidth=1.8,label='Certified (both)')
    for ax in axes[1:]:
        ax.axvline(z['sigma_abs'],color='#777777',linestyle=(0,(1,2)),linewidth=1.6,dash_capstyle='round')
        ax.set(xscale='log',xlabel='Effective noise σ',ylabel='Graph radius (hops)')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis='y',alpha=.15)
    axes[1].set_title('(b) Exact / certified radius (ε = 10⁻³)')
    axes[2].set_title('(c) Sensitivity radius: 90% / 95%')
    handles,labels=axes[1].get_legend_handles_labels()
    handles.append(Line2D([],[],color='#777777',linestyle=(0,(1,2)),linewidth=1.6,dash_capstyle='round',label=r'$\sigma_{\rm abs}$'))
    axes[1].legend(handles=handles,loc='upper left',fontsize=11,handlelength=3.2,handletextpad=.5,borderpad=.35,labelspacing=.3)
    axes[2].legend(loc='upper left',ncol=2,fontsize=11,columnspacing=.7,handlelength=2.8,handletextpad=.4,borderpad=.35,labelspacing=.3)
    save(fig,out/'synthetic_locality')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--grids',nargs='+',type=int,default=[16])
    p.add_argument('--alpha',type=float,default=.2); p.add_argument('--beta',type=float,default=1.)
    p.add_argument('--noise-count',type=int,default=41); p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=20260916)
    p.add_argument('--edges',type=int,default=384)
    a=p.parse_args()
    if a.grids!=[16]: p.error('This report uses only a 16x16 random graph')
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    for g in a.grids:
        z=calculate(g,a.alpha,a.beta,a.noise_count,seed=a.seed,edge_count=a.edges)
        tmp=out/f'grid{g}.tmp.npz'; np.savez_compressed(tmp,**z); tmp.replace(out/f'grid{g}.npz')
        (out/f'grid{g}_summary.json').write_text(json.dumps(json.loads(z['metadata']),indent=2)+'\n')
    plot(z,out)
    print('Completed CPU-only study:',out,flush=True)


if __name__=='__main__': main()
