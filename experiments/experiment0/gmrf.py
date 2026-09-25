"""Exact scalar-patch GMRF reference for the draft's locality corollary."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from experiments.experiment0.core import coverage, distances


def grid_precision(grid=8, alpha=.2, beta=1.):
    lap=np.zeros((grid*grid,grid*grid))
    for r in range(grid):
        for c in range(grid):
            i=r*grid+c
            for rr,cc in [(r+1,c),(r,c+1)]:
                if rr<grid and cc<grid:
                    j=rr*grid+cc
                    lap[i,i]+=1; lap[j,j]+=1; lap[i,j]-=1; lap[j,i]-=1
    return alpha*np.eye(grid*grid)+beta*lap


def exact_locality(Q, sigma, target, context):
    n=len(Q); eye=np.eye(n)
    W=np.linalg.solve(eye+sigma*sigma*Q,eye)
    Sigma=np.linalg.solve(Q,eye)+sigma*sigma*eye
    B=np.asarray(context,dtype=int); C=np.setdiff1d(np.arange(n),B)
    if not len(C): return 0.,0.
    conditional=Sigma[np.ix_(C,C)]-Sigma[np.ix_(C,B)]@np.linalg.solve(
        Sigma[np.ix_(B,B)],Sigma[np.ix_(B,C)])
    row=W[target,C]
    return float(row@conditional@row),float(row@row)


def certificate(qlo,qhi,sigma,radius,dp=1):
    chi=(1+sigma*sigma*qhi)/(1+sigma*sigma*qlo)
    rho=(np.sqrt(chi)-1)/(np.sqrt(chi)+1)
    prefactor=dp*(1-1/np.sqrt(chi))**2/(qlo*(1+sigma*sigma*qlo))
    return prefactor*rho**(2*np.asarray(radius))


def certified_radius(qlo,qhi,sigma,tolerance,dp=1):
    c0=certificate(qlo,qhi,sigma,0,dp)
    if c0<=tolerance: return 0
    chi=(1+sigma*sigma*qhi)/(1+sigma*sigma*qlo)
    rho=(np.sqrt(chi)-1)/(np.sqrt(chi)+1)
    return max(0,int(np.ceil(np.log(c0/tolerance)/(2*np.log(1/rho)))))


def calculate(grid=8,alpha=.2,beta=1.,tolerance=1e-3):
    Q=grid_precision(grid,alpha,beta); eig=np.linalg.eigvalsh(Q); lo,hi=eig[0],eig[-1]
    sigma_abs=np.sqrt((lo**(1/3)+hi**(1/3))/(lo**(2/3)*hi**(2/3)))
    sigmas=np.unique(np.r_[0.,np.geomspace(.01,100,41),sigma_abs])
    target=(grid//2)*grid+grid//2; d=distances(grid,[target],'manhattan')[0]
    radii=np.arange(int(d.max())+1); error=[]; tail=[]; bounds=[]; rc=[]; re=[]; rs=[]
    for sigma in sigmas:
        pairs=[exact_locality(Q,sigma,target,np.flatnonzero(d<=r)) for r in radii]
        e,t=np.array(pairs).T
        bound=certificate(lo,hi,sigma,radii)
        assert np.all(e<=(1/lo+sigma*sigma)*t+1e-10)
        assert np.all(e>=(1/hi+sigma*sigma)*t-1e-10)
        assert np.all(e<=bound+1e-10)
        W=np.linalg.solve(np.eye(len(Q))+sigma*sigma*Q,np.eye(len(Q)))
        rs.append(coverage(W[target]**2,d,qs=[.9])['radius'][0])
        error.append(e); tail.append(t); bounds.append(bound)
        rc.append(certified_radius(lo,hi,sigma,tolerance))
        re.append(int(np.flatnonzero(e<=tolerance)[0]))
    rc=np.asarray(rc)
    assert np.all(np.diff(rc[sigmas<=sigma_abs])>=0)
    return dict(sigmas=sigmas,radii=radii,error=np.asarray(error),tail=np.asarray(tail),
                certificate=np.asarray(bounds),certified_radius=rc,exact_radius=np.asarray(re),
                sensitivity_radius=np.asarray(rs),sigma_abs=sigma_abs,Q=Q,
                q_min=lo,q_max=hi,target=target,tolerance=tolerance)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); a=p.parse_args()
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    result=calculate(); np.savez_compressed(out/'gmrf.npz',**result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    s=result['sigmas']; nonzero=s>0
    for key,label in [('certified_radius','Certified absolute-error radius'),('exact_radius','Exact absolute-error radius'),('sensitivity_radius','90% sensitivity radius')]:
        ax[0].plot(s[nonzero],result[key][nonzero],label=label)
    ax[0].axvline(result['sigma_abs'],color='k',ls=':',label='Certified interval endpoint')
    ax[0].set(xscale='log',xlabel='Effective noise',ylabel='Graph hops'); ax[0].legend(fontsize=7)
    for wanted in [.1,1,10]:
        i=int(np.argmin(abs(s-wanted)))
        error=result['error'][i]
        line,=ax[1].plot(result['radii'],np.where(error>0,error,np.nan),label=f'Exact, sigma={s[i]:g}')
        ax[1].plot(result['radii'],result['certificate'][i],ls='--',color=line.get_color())
    ax[1].set(yscale='log',xlabel='Graph radius',ylabel='Absolute locality error / certificate'); ax[1].legend(fontsize=7)
    r=2
    ax[2].plot(s[nonzero],result['error'][nonzero,r],label='Exact locality error')
    ax[2].plot(s[nonzero],(1/result['q_min']+s[nonzero]**2)*result['tail'][nonzero,r],ls='--',label='Spectral upper bound')
    ax[2].plot(s[nonzero],(1/result['q_max']+s[nonzero]**2)*result['tail'][nonzero,r],ls=':',label='Spectral lower bound')
    ax[2].set(xscale='log',yscale='log',xlabel='Effective noise',ylabel='Radius-2 absolute error'); ax[2].legend(fontsize=7)
    for ext in ['png','pdf']: fig.savefig(out/f'gmrf.{ext}',dpi=180)
    plt.close(fig)
    (out/'summary.json').write_text(json.dumps({k:float(result[k]) for k in ['q_min','q_max','sigma_abs','tolerance']},indent=2)+'\n')
    print('Exact GMRF inequalities and bounded-range monotonicity verified:',out)


if __name__=='__main__': main()
