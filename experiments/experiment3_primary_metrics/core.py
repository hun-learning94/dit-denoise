"""CPU numerical definitions. No checkpoint selection or model execution here."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

CODES=('000','100','010','001','110','101','011','111')
CONTRASTS={'MLP_minus_QK':{'001':1,'100':-1},'VO_MLP_minus_QK':{'011':1,'100':-1},
 'QK_restoration':{'111':1,'011':-1},'VO_given_MLP':{'011':1,'001':-1},
 'MLP_given_VO':{'011':1,'010':-1},'VO_MLP_interaction':{'011':1,'001':-1,'010':-1,'000':1}}

def unit(x):
 x=np.asarray(x,dtype=np.float64)
 if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('Expected finite feature matrix')
 n=np.linalg.norm(x,axis=1,keepdims=True)
 if np.any(n<1e-12):raise ValueError('Zero feature vector')
 return x/n

def fit_probe(x,y,regularization):
 x=unit(x);y=np.asarray(y,float)
 if set(y)!={0.,1.} or len(x)!=len(y):raise ValueError('Need both real-data classes')
 def loss(v):
  z=x@v[:-1]+v[-1];err=expit(z)-y
  return np.mean(np.logaddexp(0,z)-y*z)+regularization*np.dot(v[:-1],v[:-1])/2, np.r_[x.T@err/len(y)+regularization*v[:-1],err.mean()]
 r=minimize(loss,np.zeros(x.shape[1]+1),jac=True,method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-12})
 if not r.success:raise RuntimeError(r.message)
 return r.x

def probe_scores(features,weights):return expit(unit(features)@weights[:-1]+weights[-1])
def discrimination(y,p):
 y=np.asarray(y);p=np.asarray(p);n1=int(y.sum());n0=len(y)-n1
 if not n0 or not n1:raise ValueError('Need both classes')
 auc=(rankdata(p)[y==1].sum()-n1*(n1+1)/2)/(n0*n1)
 return dict(auroc=float(auc),balanced_accuracy=float(((p[y==1]>=.5).mean()+(p[y==0]<.5).mean())/2),log_loss=float(-np.mean(y*np.log(p.clip(1e-12,1))+(1-y)*np.log((1-p).clip(1e-12,1)))))

def real_radii(real,k=5,chunk=256):
 real=unit(real)
 if not 0<k<len(real):raise ValueError('Require 0 < k < reference size')
 out=[]
 for start in range(0,len(real),chunk):
  d=cdist(real[start:start+chunk],real,metric='sqeuclidean')
  d[np.arange(len(d)),np.arange(start,start+len(d))]=np.inf
  out.extend(np.partition(d,k-1,axis=1)[:,k-1])
 return np.asarray(out)

def density_per_image(generated,real,k=5,chunk=256):
 real=unit(real);generated=unit(generated);r=real_radii(real,k,chunk)
 return np.concatenate([(cdist(generated[i:i+chunk],real,'sqeuclidean')<=r[None,:]).sum(1)/k for i in range(0,len(generated),chunk)])

def summarize(scores,densities,distances,indices=None,min_score_gap=.05,min_endpoint_distance=.01):
 arrays=[scores,densities,distances]
 if any(set(x)!=set(CODES) for x in arrays):raise ValueError('Need all eight states')
 n=len(scores['000']);idx=np.arange(n) if indices is None else indices
 if any(len(a[c])!=n for a in arrays for c in CODES):raise ValueError('Unmatched latents')
 s={c:float(np.mean(scores[c][idx])) for c in CODES};d={c:float(np.mean(densities[c][idx])) for c in CODES};e={c:float(np.mean(distances[c][idx])) for c in CODES}
 gap=s['111']-s['000'];base=e['000']
 # Require positive, material directed change. Negative gap is reported, not normalized.
 t={c:(s[c]-s['000'])/gap if gap>=min_score_gap else None for c in CODES}
 p={c:1-e[c]/base if base>=min_endpoint_distance else None for c in CODES}
 groups={'s':s,'D':d,'E':e,'T':t,'P':p}
 contrasts={name:{label:sum(w*vals[c] for c,w in weights.items()) if all(vals[c] is not None for c in weights) else None for label,weights in CONTRASTS.items()} for name,vals in groups.items() if name!='E'}
 return dict(metrics=groups,contrasts=contrasts,score_gap=gap,endpoint_denominator=base)

def bootstrap(scores,densities,distances,repeats=2000,seed=20260911,**guards):
 point=summarize(scores,densities,distances,**guards);rng=np.random.default_rng(seed);n=len(scores['000'])
 reps=[summarize(scores,densities,distances,rng.integers(0,n,n),**guards) for _ in range(repeats)]
 intervals={}
 for group in ['metrics','contrasts']:
  intervals[group]={}
  for metric,values in point[group].items():
   intervals[group][metric]={}
   for name,value in values.items():
    vals=[r[group][metric][name] for r in reps];valid=[v for v in vals if v is not None]
    intervals[group][metric][name]=dict(ci95=list(map(float,np.quantile(valid,[.025,.975]))) if value is not None and len(valid)>=.95*repeats else None,valid_fraction=len(valid)/repeats)
 point['intervals']=intervals
 return point

def routing(source,target,k=16):
 a=np.asarray(source,float);b=np.asarray(target,float)
 if a.shape!=b.shape or not 0<k<=a.shape[-1] or np.any(a<0) or np.any(b<0):raise ValueError('Invalid routing arrays')
 if not np.allclose(a.sum(-1),1) or not np.allclose(b.sum(-1),1):raise ValueError('Rows must sum to one')
 m=(a+b)/2
 def kl(x):return np.sum(x*np.log(np.maximum(x,1e-300)/np.maximum(m,1e-300)),axis=-1)
 ia=np.argsort(-a,axis=-1,kind='stable')[...,:k];ib=np.argsort(-b,axis=-1,kind='stable')[...,:k]
 overlap=np.any(ia[..., :,None]==ib[...,None,:],axis=-1).sum(-1)/k
 return dict(js=(kl(a)+kl(b))/2,overlap=overlap)
