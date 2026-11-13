"""Paired, resumable KID evaluation; features match the project's prior metrics."""
import argparse,json,os,signal,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.models import inception_v3
from experiments.component_transfer_scale.model import load_source,plain_model,uvit_module,digest,write,seed_all
from experiments.component_transfer_scale.runner import Images

STOP=False
def stop(*_):
    global STOP
    STOP=True

class Features(torch.nn.Module):
    def __init__(self,m):
        super().__init__();assert digest(m['feature_weights'])==m['feature_weights_sha256']
        self.net=inception_v3(weights=None,aux_logits=True,transform_input=False,init_weights=False)
        self.net.load_state_dict(torch.load(m['feature_weights'],map_location='cpu',weights_only=True))
        self.net.fc=torch.nn.Identity();self.net.eval().requires_grad_(False);self.cuda()
    @torch.no_grad()
    def forward(self,x):
        x=torch.nn.functional.interpolate(x.float(),(299,299),mode='bilinear',align_corners=False)
        return self.net((x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None])

class Predictor(torch.nn.Module):
    def __init__(self,net):super().__init__();self.net=net
    def forward(self,x,t):
        with torch.autocast('cuda',dtype=torch.bfloat16):out=self.net(x,t)
        return out.float()

@torch.no_grad()
def sample(m,net,seed,shard,count):
    # Every shard has its own initial and path RNG, independent of restart/order.
    generator=torch.Generator().manual_seed(seed+1000003*shard)
    x=torch.randn(count,3,64,64,generator=generator).cuda()
    seed_all(seed+2000003*shard+31)
    sde=uvit_module(m['source'],'sde')
    score=sde.ScoreModel(Predictor(net),pred='noise_pred',sde=sde.VPSDE())
    out=sde.euler_maruyama(sde.ReverseSDE(score),x,sample_steps=1000)
    assert torch.isfinite(out).all()
    return ((out.clamp(-1,1)+1)*127.5).round().byte()

def array_save(path,array):
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.save(f,array)
    tmp.replace(path)

def metrics(real,gen):
    # Exact unbiased polynomial MMD, no subset Monte Carlo or zero-clipping.
    real=np.asarray(real,dtype=np.float64);gen=np.asarray(gen,dtype=np.float64);d=real.shape[1]
    def kernel_mean(a,b,diagonal=False):
        total=0.
        for i in range(0,len(a),128):
            k=(a[i:i+128]@b.T/d+1)**3
            if diagonal:
                j=np.arange(len(k));k[j,i+j]=0
            total+=k.sum()
        return total/(len(a)*(len(b)-1 if diagonal else len(b)))
    kid=kernel_mean(real,real,True)+kernel_mean(gen,gen,True)-2*kernel_mean(real,gen)
    ar=(real-real.mean(0))/np.sqrt(len(real)-1);ag=(gen-gen.mean(0))/np.sqrt(len(gen)-1)
    # Exact empirical-covariance Bures cross term using the small real sample axis.
    cross=np.linalg.svd(ar@ag.T,compute_uv=False).sum()
    fid=((real.mean(0)-gen.mean(0))**2).sum()+(ar**2).sum()+(ag**2).sum()-2*cross
    return dict(kid=float(kid),fid=float(max(fid,0)),real_images=len(real),generated_images=len(gen))

@torch.no_grad()
def real_features(m):
    output=Path(m['eval_output']);output.mkdir(parents=True,exist_ok=True);feature=Features(m)
    data=json.loads(Path(m['data']).read_text())
    for split in ('train','test'):
        ds=Images(data,split);parts=[]
        for start in range(0,len(ds),32):
            x=torch.stack([ds[i] for i in range(start,min(start+32,len(ds)))]).cuda().float()/255
            parts.append(feature(x).cpu().numpy())
        array_save(output/f'real_{split}.npy',np.concatenate(parts))
    write(output/'REAL_READY.json',dict(data_sha256=digest(m['data']),weights_sha256=m['feature_weights_sha256'],
          files={s:digest(output/f'real_{s}.npy') for s in ('train','test')}))

@torch.no_grad()
def evaluate(m,task):
    start=time.monotonic();row=m['evaluation_tasks'][task];base=Path(m['eval_output']);output=base/row['label'];output.mkdir(parents=True,exist_ok=True)
    real_provenance=json.loads((base/'REAL_READY.json').read_text());assert real_provenance['data_sha256']==digest(m['data'])
    for split in ('train','test'):assert digest(base/f'real_{split}.npy')==real_provenance['files'][split]
    if row['arm']=='source':
        net,_,_=load_source(m['source']);checkpoint=m['source']['checkpoint'];epoch=0
    else:
        run=Path(m['runs'][row['arm']]);assert (run/'COMPLETE.json').exists()
        checkpoint=str(run/(row['endpoint']+'.pt'));payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert payload['identity']['arm']==row['arm'] and payload['identity']['data']==digest(m['data'])
        assert payload['identity']['source']==m['source']['checkpoint_sha256'] and payload['identity']['code_sha256']==m['code_sha256']
        net=plain_model(m['source'],payload['ema']);epoch=payload['epoch']
    identity=dict(task=row,checkpoint_sha256=digest(checkpoint),data_sha256=digest(m['data']),
                  code_sha256=m['code_sha256'],feature_weights_sha256=m['feature_weights_sha256'])
    if (output/'identity.json').exists():assert json.loads((output/'identity.json').read_text())==identity
    else:write(output/'identity.json',identity)
    if (output/'result.json').exists():return
    net.cuda().eval().requires_grad_(False);feature=Features(m)
    n=m['evaluation']['generated_images'];batch=m['evaluation']['batch'];parts=[]
    for shard,first in enumerate(range(0,n,batch)):
        path=output/f'features_{shard:04d}.npy';count=min(batch,n-first)
        if not path.exists():
            images=sample(m,net,row['seed'],shard,count)
            features=feature(images.float()/255).cpu().numpy();assert np.isfinite(features).all()
            array_save(path,features)
            if shard==0:
                for i,im in enumerate(images[:32].permute(0,2,3,1).cpu().numpy()):Image.fromarray(im).save(output/f'example_{i:02d}.png')
            write(output/'progress.json',dict(images=first+count,total=n,seconds=time.monotonic()-start,epoch=epoch,job=os.getenv('SLURM_JOB_ID')))
        part=np.load(path);assert part.shape==(count,2048) and np.isfinite(part).all();parts.append(part)
        if STOP or time.monotonic()-start>10.5*3600:raise SystemExit(75)
    gen=np.concatenate(parts);real=np.load(base/'real_test.npy');train=np.load(base/'real_train.npy')
    scores=metrics(real,gen)
    nearest=((gen[:32,None].astype(np.float64)-train[None].astype(np.float64))**2).sum(-1).argmin(-1)
    data=json.loads(Path(m['data']).read_text());train_paths=Path(data['splits']['train']['path']).read_text().splitlines()
    write(output/'nearest_training_images.json',dict(metric='squared Euclidean distance in frozen Inception features',
          matches=[dict(example=i,training_image=train_paths[j]) for i,j in enumerate(nearest.tolist())]))
    write(output/'result.json',dict(**scores,**row,epoch=epoch,checkpoint=checkpoint,identity=identity,
          extractor=m['evaluation']['extractor'],precision=m['evaluation']['precision'],seconds_this_attempt=time.monotonic()-start))

def smoke(m):
    feature=Features(m);net,_,_=load_source(m['source']);net.cuda().eval().requires_grad_(False)
    a=sample(m,net,1234,0,2);b=sample(m,net,1234,0,2);assert torch.equal(a,b)
    feats=feature(a.float()/255);assert feats.shape==(2,2048) and torch.isfinite(feats).all()
    # Independent covariance implementation checks the small-reference FID formula.
    from scipy.linalg import sqrtm
    rng=np.random.default_rng(3);x=rng.normal(size=(17,8));y=rng.normal(size=(21,8))
    got=metrics(x,y);cx=np.cov(x,rowvar=False);cy=np.cov(y,rowvar=False)
    expected=((x.mean(0)-y.mean(0))**2).sum()+np.trace(cx+cy-2*sqrtm(cx@cy)).real
    assert abs(got['fid']-expected)<1e-9
    kxx=(x@x.T/8+1)**3;kyy=(y@y.T/8+1)**3;kxy=(x@y.T/8+1)**3
    expected_kid=(kxx.sum()-np.trace(kxx))/(17*16)+(kyy.sum()-np.trace(kyy))/(21*20)-2*kxy.mean()
    assert abs(got['kid']-expected_kid)<1e-10
    write(Path(m['report'])/'EVAL_SMOKE_PASS.json',dict(sampler_repeat_exact=True,sampler_steps=1000,
          feature_shape=list(feats.shape),fid_identity_error=abs(got['fid']-expected),kid_identity_error=abs(got['kid']-expected_kid)))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['real','evaluate','smoke']);p.add_argument('--manifest',required=True);p.add_argument('--task',type=int,default=0);a=p.parse_args()
    m=json.loads(Path(a.manifest).read_text())
    for f,h in m['code_sha256'].items():assert digest(Path(m['code'])/f)==h,f
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop)
    if a.stage=='smoke':smoke(m)
    else:
        assert (Path(m['report'])/'EVAL_SMOKE_PASS.json').exists()
        if a.stage=='real':real_features(m)
        else:evaluate(m,a.task)

if __name__=='__main__':main()
