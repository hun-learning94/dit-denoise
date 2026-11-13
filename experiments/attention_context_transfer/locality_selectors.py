"""Patch-context policies and covariance/gradient algebra; no model fitting side effects."""
from __future__ import annotations
import numpy as np
import torch
from .protocol import distances, seed_for

def core(target):
    tie=np.random.default_rng(seed_for("geometry",int(target))).permutation(256)
    return np.lexsort((tie,distances(int(target))))[:9]

def mass_context(scores,target,mass=.90):
    scores=np.asarray(scores,dtype=np.float64)
    if scores.shape!=(256,) or not np.isfinite(scores).all() or (scores<0).any() or scores.sum()<=0:
        raise ValueError("Expected nonnegative finite nonzero spatial scores")
    selected=np.zeros(256,bool);selected[core(target)]=True
    order=np.lexsort((np.arange(256),-scores))
    total=scores.sum();have=scores[selected].sum()
    for j in order:
        if have>=mass*total:break
        if not selected[j]:selected[j]=True;have+=scores[j]
    return selected

def relative_context(scores,target,threshold=.02):
    scores=np.asarray(scores)
    if scores.shape!=(256,) or not np.isfinite(scores).all() or (scores<0).any() or scores.max()<=0:
        raise ValueError("Invalid covariance scores")
    selected=scores>=threshold*scores.max()
    selected[core(target)]=True
    return selected

def square_context(target,radius):
    rr,cc=np.divmod(np.arange(256),16);r,c=divmod(int(target),16)
    selected=(np.abs(rr-r)<=radius)&(np.abs(cc-c)<=radius)
    selected[core(target)]=True
    return selected

def square_schedule(mean_gradients,mass=.90):
    fields=np.asarray(mean_gradients,dtype=np.float64)
    assert fields.ndim==3 and fields.shape[1:]==(256,256)
    assert np.isfinite(fields).all() and (fields.sum(-1)>0).all()
    normalized=fields/fields.sum(-1,keepdims=True)
    radii=[];captured=[]
    for rows in normalized:
        for radius in range(16):
            fraction=np.mean([rows[m,square_context(m,radius)].sum() for m in range(256)])
            if fraction>=mass:break
        radii.append(radius);captured.append(fraction)
    return np.array(radii),np.array(captured)

def patch_basis(basis,patch=8):
    """Flattened CHW RGB basis -> [patch, channel*pixel-in-patch, rank]."""
    dim,rank=basis.shape;side=int((dim//3)**.5)
    assert dim==3*side*side and side//patch==16
    return basis.reshape(3,16,patch,16,patch,rank).permute(1,3,0,2,4,5).reshape(256,3*patch*patch,rank)

def covariance_scores(basis,eigenvalues,times,patch=8):
    """Exact block Frobenius scores for the supplied finite-rank covariance."""
    u=patch_basis(basis,patch)
    gram=u.transpose(1,2)@u
    flat=gram.flatten(1);result=[]
    for t in times:
        a=float(t);b=.5*(1-a)
        weight=a*eigenvalues/(a*a*eigenvalues+b*b)
        weighted=(gram*weight[None,:,None]*weight[None,None,:]).flatten(1)
        squared=weighted@flat.T
        result.append((squared.clamp_min(0)/(3*patch*patch)).sqrt())
    return torch.stack(result)

def patch_energy(gradient,patch=8):
    b,c,h,w=gradient.shape
    return gradient.square().reshape(b,c,16,patch,16,patch).sum((1,3,5)).reshape(b,256)/(c*patch*patch)

def patch_output(images,targets):
    b,c,h,w=images.shape;p=h//16
    grid=images.reshape(b,c,16,p,16,p).permute(0,2,4,1,3,5)
    return grid[torch.arange(b,device=images.device),targets//16,targets%16].flatten(1)

def comparison(a,b):
    """Compare nonnegative score fields at their own relative-threshold supports."""
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    an=a/a.sum(-1,keepdims=True);bn=b/b.sum(-1,keepdims=True)
    aa=np.empty(a.shape,bool);bb=np.empty(b.shape,bool)
    for index in np.ndindex(a.shape[:-1]):
        m=index[-1];aa[index]=relative_context(a[index],m);bb[index]=relative_context(b[index],m)
    intersection=(aa&bb).sum(-1);union=(aa|bb).sum(-1)
    count_a=aa.sum(-1);count_b=bb.sum(-1)
    return dict(normalized_l1_mean=float(np.abs(an-bn).sum(-1).mean()),
                support_jaccard_mean=float((intersection/union).mean()),
                support_jaccard_p10=float(np.quantile(intersection/union,.1)),
                count_a_mean=float(count_a.mean()),count_b_mean=float(count_b.mean()),
                count_abs_change_mean=float(np.abs(count_a-count_b).mean()))
