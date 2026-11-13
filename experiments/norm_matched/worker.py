"""One-GPU, atomic-shard, resumable norm-matched transplant evaluation."""
from pathlib import Path
import argparse,fcntl,gc,hashlib,json,math,os,signal,subprocess,sys,time
import torch
import numpy as np
HERE=Path(__file__).resolve().parent
STOP=False
def stop(*_):
    global STOP
    STOP=True
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp')
    q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
def save(p,**data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp')
    with q.open('wb') as f:np.savez_compressed(f,**data)
    q.replace(p)
def check_stop():
    if STOP:raise SystemExit(75)

def identity(m,case,out):
    path=out/'identity.json'
    value=dict(manifest_sha256=sha(HERE/'manifest.json'),case=case['id'],
               worker_sha256=sha(__file__))
    if path.exists():assert read(path)==value,'Refusing to mix cached evaluation identities'
    else:write(path,value)

def norm(state,source):
    return math.sqrt(sum(float((state[k].double()-v.double()).square().sum())
                         for k,v in source.items() if torch.is_floating_point(v)))

def matched(source,adapted,bits,hybrid,radius=None):
    raw=hybrid(source,adapted,bits)
    if bits in ['000','111']:return raw,dict(scale=1.,norm=norm(raw,source))
    if radius is None:radius=norm(hybrid(source,adapted,'100'),source)
    before=norm(raw,source);assert radius>0 and before>0
    scale=radius/before;assert scale<=1.00001,(bits,scale)
    if bits=='100':assert abs(scale-1)<1e-12;state=raw
    else:
        state={k:(v if torch.equal(v,source[k]) else source[k]+(v-source[k])*scale)
               for k,v in raw.items()}
    actual=norm(state,source);assert abs(actual-radius)/radius<3e-6,(actual,radius)
    return state,dict(scale=scale,original_norm=before,norm=actual,radius=radius)

class Net:
    def __init__(self,m,case):
        from experiments.component_transfer_scale.model import load_source,hybrid_state,uvit_module
        from experiments.component_transfer_scale.runner import vae_for
        self.m,self.case=m,case;self.family=case['family'];self.hybrid=hybrid_state
        self.net,self.source,self.config=load_source(case['source'])
        self.net.cuda().eval().requires_grad_(False)
        self.vae=vae_for(case['source']) if self.family=='lfm256' else None
        self.sde=uvit_module(case['source'],'sde') if self.family=='uvit64' else None
        if self.family=='dit128':
            from edm import vp_heun_sampler
            self.heun=vp_heun_sampler
        self.load(case['checkpoint'],case['epoch'],case['checkpoint_sha256'])
    def load(self,path,epoch,expected):
        assert sha(path)==expected
        p=torch.load(path,map_location='cpu',weights_only=False)
        assert p['format']=='matched-scale-effective-ema-v1' and p['reference_epoch']==epoch
        assert p['source']==self.case['source'] and p['model_cfg']==self.config
        self.adapted=p['ema'];self.radius=norm(self.hybrid(self.source,self.adapted,'100'),self.source)
        self.epoch=epoch;self.checkpoint_sha256=expected
        assert all(torch.isfinite(v).all() for v in self.adapted.values())
    def set(self,bits):
        state,info=matched(self.source,self.adapted,bits,self.hybrid,self.radius)
        self.net.load_state_dict(state,strict=True)
        if bits=='000':assert all(torch.equal(v,self.source[k]) for k,v in state.items())
        if bits=='111':assert all(torch.equal(v,self.adapted[k]) for k,v in state.items())
        if bits=='011':
            source_qk=self.hybrid(self.source,state,'100')
            assert all(torch.equal(v,self.source[k]) for k,v in source_qk.items())
        return info
    def predict(self,x,t,bf16=True):
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):
            return (self.net(t,x,y=None) if self.family=='lfm256' else self.net(x,t)).float()
    def sample(self,shard,count,qual=None):
        family=self.family;batch=self.case['batch'];bf16=qual is None
        if qual:
            if family=='uvit64':
                high=torch.randn(64,3,128,128,generator=torch.Generator().manual_seed(qual['seed']))*.5
                z=(high[:4]*2).reshape(4,3,64,2,64,2).sum((3,5))/2
            else:z=torch.randn(4,4,32,32,generator=torch.Generator().manual_seed(qual['seed']))
            assert hashlib.sha256(z.numpy().tobytes()).hexdigest()==qual['initial_noise_sha256']
            batch=count=4
        else:
            shape=(count,4,32,32) if family=='lfm256' else (count,3,self.case['model_spec']['resolution'],self.case['model_spec']['resolution'])
            z=torch.randn(shape,generator=torch.Generator().manual_seed(self.case['sampling_seed']+1000003*shard))
            if family=='dit128':z=z*.5
        parts=[]
        for first in range(0,count,batch):
            x=z[first:first+batch].cuda()
            if family=='uvit64':
                parent=self
                class Wrapper(torch.nn.Module):
                    def forward(self,x,t):return parent.predict(x,t,bf16)
                reverse=self.sde.ReverseSDE(self.sde.ScoreModel(Wrapper(),pred='noise_pred',sde=self.sde.VPSDE()))
                if qual:
                    rng=torch.Generator().manual_seed(qual['path_seed'])
                    times=torch.tensor(np.append(0.,np.linspace(.001,1,1000)),device='cuda',dtype=x.dtype)
                    for s,t in list(zip(times[:-1],times[1:]))[::-1]:
                        dt=s-t;mean=x+reverse.drift(x,t)*dt
                        sigma=reverse.diffusion(t)*(-dt).sqrt()
                        x=mean+self.sde.stp(sigma,torch.randn(x.shape,generator=rng).cuda()) if s!=0 else mean
                else:
                    torch.cuda.manual_seed_all(self.case['path_seed']+1000003*shard+first)
                    x=self.sde.euler_maruyama(reverse,x,sample_steps=1000)
            elif family=='dit128':
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):
                    x=self.heun(self.net,x,num_steps=50,denominator_floor=.05)
            else:
                times=torch.linspace(1,0,51,device='cuda')
                for begin,end in zip(times[:-1],times[1:]):
                    dt=end-begin;v0=self.predict(x,begin.expand(len(x)),bf16)
                    v1=self.predict(x+dt*v0,end.expand(len(x)),bf16);x=x+dt*(v0+v1)*.5
                x=torch.cat([self.vae.decode(chunk/.18215).sample for chunk in x.split(1 if qual else 2)])
            assert torch.isfinite(x).all()
            parts.append(((x.float()+1)*127.5).round().clamp(0,255).byte().permute(0,2,3,1).cpu().numpy())
        return np.concatenate(parts)

class Dino:
    def __init__(self,m):
        from torchvision.transforms import Compose,Resize,ToTensor,Normalize,InterpolationMode
        d=m['dino'];assert sha(d['feature_weights'])==d['feature_weights_sha256']
        assert subprocess.check_output(['git','-C',d['feature_repo'],'rev-parse','HEAD'],text=True).strip()==d['feature_repo_commit']
        self.net=torch.hub.load(d['feature_repo'],d['feature_model'],source='local',pretrained=False)
        self.net.load_state_dict(torch.load(d['feature_weights'],map_location='cpu',weights_only=True))
        self.net.cuda().eval().requires_grad_(False)
        self.transform=Compose([Resize((224,224),interpolation=InterpolationMode.BICUBIC,antialias=True),ToTensor(),Normalize([.485,.456,.406],[.229,.224,.225])])
    def __call__(self,pixels):
        from PIL import Image
        result=[]
        for start in range(0,len(pixels),16):
            x=torch.stack([self.transform(Image.fromarray(im)) for im in pixels[start:start+16]]).cuda()
            y=self.net.forward_features(x)['x_norm_clstoken'].float()
            result.append(torch.nn.functional.normalize(y,dim=-1).cpu().numpy())
        a=np.concatenate(result);assert a.shape==(len(pixels),768) and np.isfinite(a).all();return a

def evaluate(m,case,net,dino,out,qual_only=False):
    diagnostics={bits:net.set(bits) for bits in m['states']}
    write(out/'NORMS.json',dict(epoch=case['epoch'],groups=diagnostics,checkpoint_sha256=net.checkpoint_sha256))
    probe=torch.zeros((2,4,32,32) if net.family=='lfm256' else (2,3,case['model_spec']['resolution'],case['model_spec']['resolution']),device='cuda')
    t=torch.full((2,),499.5 if net.family=='uvit64' else .5,device='cuda')
    for bits in m['states']:
        net.set(bits);assert torch.isfinite(net.predict(probe,t)).all()
    # Exercise the frozen production microbatch and compare BF16 to FP32.
    net.set('000')
    bf=net.predict(probe,t);fp=net.predict(probe,t,bf16=False)
    relative=float((bf-fp).square().mean().sqrt()/fp.square().mean().sqrt().clamp_min(1e-6))
    assert math.isfinite(relative) and relative<.15,relative
    big=probe[:1].expand(case['batch'],*probe.shape[1:]).contiguous()
    assert torch.isfinite(net.predict(big,t[:1].expand(case['batch']))).all()
    del big
    write(out/'GPU_PREFLIGHT.json',dict(passed=True,gpu=torch.cuda.get_device_name(),native_bf16=True,
        endpoints_exact=True,norms_checked=True,bf16_relative_rms=relative,
        batch=case['batch'],at=time.time(),manifest_sha256=sha(HERE/'manifest.json')))
    for qual in [q for q in m['qualitative'] if q['case']==case['id']]:
        net.load(qual['checkpoint'],qual['epoch'],qual['checkpoint_sha256'])
        for bits in m['states']:
            path=out/'qualitative'/f'epoch{qual["epoch"]:04d}_{bits}.npz'
            if path.exists():continue
            info=net.set(bits);tick=time.monotonic();pixels=net.sample(0,4,qual)
            save(path,images=pixels,global_features=dino(pixels),identity=json.dumps(dict(qual=qual,bits=bits,norms=info),sort_keys=True))
            print('QUAL',case['id'],qual['epoch'],bits,time.monotonic()-tick,flush=True);check_stop()
    if qual_only:return
    net.load(case['checkpoint'],case['epoch'],case['checkpoint_sha256'])
    for bits in m['states']:
        net.set(bits)
        for shard in range(m['paired_size']//m['shard_size']):
            path=out/'quantitative'/bits/f'shard{shard:03d}.npz'
            if path.exists():continue
            tick=time.monotonic();pixels=net.sample(shard,m['shard_size'])
            save(path,images=pixels,global_features=dino(pixels))
            write(out/'progress.json',dict(stage='sampling',state=bits,shard=shard,seconds=time.monotonic()-tick,at=time.time()))
            print('SHARD',case['id'],bits,shard,time.monotonic()-tick,flush=True);check_stop()
    native_loss(m,case,net,out)
    write(out/'COMPLETE.json',dict(passed=True,epoch=case['epoch'],paired_size=m['paired_size'],
        states=m['states'],gpu=torch.cuda.get_device_name(),manifest_sha256=sha(HERE/'manifest.json')))

def native_loss(m,case,net,out):
    from experiments.transplant_epoch100.common import images,view_rows
    paths=view_rows(case['view'],'denoising')
    for bits in m['states']:
        net.set(bits);batch=2 if net.family=='lfm256' else 32
        for start in range(0,len(paths),batch):
            path=out/'loss'/bits/f'batch{start:04d}.npz'
            if path.exists():continue
            x=torch.from_numpy(images(case['view'],paths[start:start+batch])).permute(0,3,1,2).cuda().float()/127.5-1
            posterior=net.vae.encode(x).latent_dist if net.family=='lfm256' else None
            values=[]
            for rep in range(case['loss_replicates']):
                gen=torch.Generator().manual_seed(case['loss_seed']+start*1009+rep)
                if posterior is not None:
                    z0=(posterior.mean+posterior.std*torch.randn(posterior.mean.shape,generator=gen).cuda())*.18215
                    t=torch.rand(len(x),generator=gen);eps=torch.randn(z0.shape,generator=gen).cuda();t=t.cuda();tt=t[:,None,None,None]
                    z=(1-tt)*z0+(1e-5+(1-1e-5)*tt)*eps;truth=(1-1e-5)*eps-z0
                    err=net.predict(z,t)-truth
                else:
                    t=torch.rand(len(x),generator=gen) if net.family=='uvit64' else (torch.randn(len(x),generator=gen)*.8-.8).sigmoid()
                    eps=torch.randn(x.shape,generator=gen).cuda();t=t.cuda();tt=t[:,None,None,None]
                    if net.family=='uvit64':
                        ab=torch.exp(-.1*t-9.95*t*t)[:,None,None,None]
                        err=net.predict(ab.sqrt()*x+(1-ab).sqrt()*eps,999*t)-eps
                    else:err=(net.predict(tt*x+(1-tt)*eps*.5,t)-x)/(1-tt).clamp_min(.05)
                loss=err.square().flatten(1).mean(1);assert torch.isfinite(loss).all();values.append(loss.cpu().numpy())
            save(path,losses=np.stack(values,axis=1));check_stop()

def main():
    global torch,np
    import torch,numpy as np
    m=read(HERE/'manifest.json');sys.path.insert(0,m['code'])
    for name,h in m['own_code_sha256'].items():assert sha(HERE/name)==h,name
    assert torch.cuda.is_available()
    assert torch.cuda.is_bf16_supported(including_emulation=False)
    for rel,h in m['driver_sha256'].items():assert sha(Path(m['code'])/rel)==h,rel
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_grad_enabled(False)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for sig in [signal.SIGUSR1,signal.SIGTERM]:signal.signal(sig,stop)
    lock=(HERE/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with torch.inference_mode():
        # Finish both small visual trajectories before the larger paired banks.
        for case in m['cases']:
            qs=[q for q in m['qualitative'] if q['case']==case['id']]
            if not qs:continue
            out=Path(m['cache'])/case['id'];out.mkdir(parents=True,exist_ok=True)
            identity(m,case,out)
            if all((out/'qualitative'/f'epoch{q["epoch"]:04d}_{bits}.npz').exists()
                   for q in qs for bits in m['states']):continue
            net=Net(m,case);dino=Dino(m);evaluate(m,case,net,dino,out,qual_only=True)
            del net,dino;gc.collect();torch.cuda.empty_cache();check_stop()
        for case in m['cases']:
            out=Path(m['cache'])/case['id'];out.mkdir(parents=True,exist_ok=True)
            identity(m,case,out)
            if (out/'COMPLETE.json').exists():
                assert read(out/'COMPLETE.json')['manifest_sha256']==sha(HERE/'manifest.json');continue
            net=Net(m,case);dino=Dino(m)
            evaluate(m,case,net,dino,out)
            del net,dino;gc.collect();torch.cuda.empty_cache();check_stop()
    write(HERE/'GPU_COMPLETE.json',dict(passed=True,cases=9,manifest_sha256=sha(HERE/'manifest.json'),at=time.time()))
if __name__=='__main__':main()
