"""CPU sample-space covariance for the new class-conditional full campaign."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from .prepare_donor_full import bank
from .imagenet_fields import basis,wiener_field,save
from .prepare import atomic_json,sha256

def tokens(field):
    p=field.shape[-1]//16
    return field.reshape(*field.shape[:-2],16,p,16,p).max(axis=(-3,-1)).reshape(*field.shape[:-2],256)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--report',type=Path,required=True);ap.add_argument('--index',type=int,required=True);a=ap.parse_args()
    root=a.report.resolve();d=json.loads((root/'design.json').read_text());info=d['classes'][a.index];wn=info['wnid'];started=time.monotonic()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','16')))
    bank(a.index)
    receipt=json.loads((root/'banks'/f'{wn}.json').read_text());assert sha256(info['bank'])==receipt['sha256']
    torch.manual_seed(121);z=torch.randn(9,17,dtype=torch.float64);z-=z.mean(0);u,lam=basis(z);t=.3;c=z.T@z/len(z)
    error=float((t*c@torch.linalg.inv(t*t*c+(1-t)**2*torch.eye(17,dtype=torch.float64))-(u*(t*lam/(t*t*lam+(1-t)**2)))@u.T).abs().max());assert error<1e-11
    pix=np.load(info['bank']);z=torch.from_numpy(pix).permute(0,3,1,2).reshape(512,-1).double()/127.5-1
    u,lam=basis(z);del z,pix;assert len(lam)==511
    path=Path(d['work_root'])/'covariance'/f'{wn}.npz';scores=[]
    for ti,t in enumerate(d['times']):
        part=path.parent/f'{wn}_t{ti}.npz'
        if part.exists():
            with np.load(part) as zz:score=zz['score']
        else:
            score=np.stack([tokens(wiener_field(u,lam,t,q)) for q in info['queries']]);save(part,score=score)
        assert score.shape==(len(info['queries']),256) and np.isfinite(score).all();scores.append(score)
        print(wn,'time',t,'seconds',time.monotonic()-started,flush=True)
    save(path,score=np.stack(scores),queries=info['queries'])
    atomic_json(root/'covariance'/f'{wn}.json',dict(passed=True,path=str(path),sha256=sha256(path),design_sha256=sha256(root/'design.json'),rank=len(lam),bank_sha256=receipt['sha256'],dense_algebra_max_abs=error,seconds=time.monotonic()-started))
if __name__=='__main__':main()
