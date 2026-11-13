"""Native hybrid samplers, frozen features, and paired held-out losses."""
import argparse,gc,json,math,os,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .common import digest,read,write,save_npz,images,view_rows
STOP=False
def stop(*_):
    global STOP
    STOP=True

def setup():
    assert torch.cuda.is_available(), "CUDA GPU required"
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_grad_enabled(False)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for sig in (signal.SIGUSR1,signal.SIGTERM):signal.signal(sig,stop)

class Features:
    def __init__(self,m):
        from torchvision.models import inception_v3
        assert digest(m['inception_weights'])==m['inception_sha256']
        self.net=inception_v3(weights=None,aux_logits=True,transform_input=False,init_weights=False)
        self.net.load_state_dict(torch.load(m['inception_weights'],map_location='cpu',weights_only=True));self.net.fc=torch.nn.Identity();self.net.cuda().eval().requires_grad_(False)
        self.m=m;self.dino=None
    def inception(self,pixels):
        parts=[]
        for start in range(0,len(pixels),32):
            x=torch.from_numpy(pixels[start:start+32]).permute(0,3,1,2).cuda().float()/255
            x=torch.nn.functional.interpolate(x,(299,299),mode='bilinear',align_corners=False)
            x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
            parts.append(self.net(x).float().cpu().numpy())
        a=np.concatenate(parts);assert a.shape==(len(pixels),2048) and np.isfinite(a).all();return a
    def descriptors(self,pixels):
        from torchvision.transforms import Compose,Resize,ToTensor,Normalize,InterpolationMode
        if self.dino is None:
            e=self.m['dino'];assert digest(e['feature_weights'])==e['feature_weights_sha256']
            assert subprocess.check_output(['git','-C',e['feature_repo'],'rev-parse','HEAD'],text=True).strip()==e['feature_repo_commit']
            self.dino=torch.hub.load(e['feature_repo'],e['feature_model'],source='local',pretrained=False)
            self.dino.load_state_dict(torch.load(e['feature_weights'],map_location='cpu',weights_only=True));self.dino.cuda().eval().requires_grad_(False)
        transform=Compose([Resize((224,224),interpolation=InterpolationMode.BICUBIC,antialias=True),ToTensor(),Normalize([.485,.456,.406],[.229,.224,.225])])
        glob=[];spatial=[]
        for i in range(0,len(pixels),16):
            x=torch.stack([transform(Image.fromarray(im)) for im in pixels[i:i+16]]).cuda();f=self.dino.forward_features(x)
            g=torch.nn.functional.normalize(f['x_norm_clstoken'].float(),dim=-1)
            t=f['x_norm_patchtokens'].float();assert t.shape[1:]==(256,768)
            t=(t-t.mean(1,keepdim=True))/(t.std(1,keepdim=True)+1e-6);t=torch.nn.functional.normalize(t,dim=-1)
            k=t@t.transpose(1,2);tri=torch.triu_indices(256,256,offset=1,device='cuda');v=k[:,tri[0],tri[1]]
            glob.append(g.cpu().numpy());spatial.append(v.cpu().numpy())
        g,v=np.concatenate(glob),np.concatenate(spatial);assert np.isfinite(g).all() and np.isfinite(v).all();return g,v
    def release_dino(self):
        self.dino=None;gc.collect();torch.cuda.empty_cache()

class Model:
    def __init__(self,m,family):
        self.m=m;self.family=family;self.spec=m['models'][family];self.resolution=self.spec['resolution'];self.noise=self.spec['noise_scale']
        sys.path.insert(0,self.spec['code'])
        if family=='dit64':
            from experiments.parameter_matched.model import load_source,hybrid_state
            self.net,self.metadata,self.source=load_source(self.spec['source'],device='cpu');self.config=self.metadata['backbone']
        else:
            from experiments.component_transfer_scale.model import load_source,hybrid_state
            self.net,self.source,self.config=load_source(self.spec['source'])
        self.hybrid=hybrid_state;self.net.cuda().eval().requires_grad_(False)
        if family=='uvit64':
            from experiments.component_transfer_scale.model import uvit_module
            self.sde=uvit_module(self.spec['source'],'sde')
        else:
            from edm import vp_heun_sampler
            self.sampler=vp_heun_sampler
    def endpoint(self,row):
        assert digest(row['checkpoint'])==row['sha256'];p=torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
        assert p['step']==row['step']
        if self.family=='dit64':
            meta=p['parameter_matched'];assert meta['source']==self.spec['source'] and meta['run']['run_id']==row['run_id']
            assert meta['run']['seed']==0 and meta['rank']==meta['alpha']==64
            assert meta['counts']==dict(QK=2048000,VO=2048000,MLP=2048000) and meta['ema_updates']==row['step']
            assert p['backbone']==self.metadata['backbone']
        else:
            assert p['format']=='matched-scale-effective-ema-v1' and p['reference_epoch']==100
            assert p['source']==self.spec['source'] and p['model_cfg']==self.config
            assert p['identity']['run_id']==row['run_id']
            if row['checkpoint_identity']:assert p['identity']==row['checkpoint_identity']
        s=p['ema'];assert self.source.keys()==s.keys() and all(torch.isfinite(v).all() for v in s.values())
        return s
    def set_task(self,task):
        self.task=task;self.adapted=self.source if task['transfer'] is None else self.endpoint(next(r for r in self.m['transfers'] if r['id']==task['transfer']))
        self.state=self.hybrid(self.source,self.adapted,task['bits']);self.net.load_state_dict(self.state,strict=True)
    def predict(self,x,t,bf16=True):
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):return self.net(x,t).float()
    def sampling(self,shard,count,batch):
        generator=torch.Generator().manual_seed(self.m['sampling_seed']+1000003*shard)
        z=torch.randn(count,3,self.resolution,self.resolution,generator=generator)*self.noise;parts=[]
        # Fixed per-microbatch path stream; batch is frozen within each family.
        for first in range(0,count,batch):
            torch.cuda.manual_seed_all(self.m['path_seed']+1000003*shard+first)
            x=z[first:first+batch].cuda()
            if self.family=='uvit64':
                parent=self
                class Wrapper(torch.nn.Module):
                    def forward(self,x,t):return parent.predict(x,t)
                reverse=self.sde.ReverseSDE(self.sde.ScoreModel(Wrapper(),pred='noise_pred',sde=self.sde.VPSDE()))
                out=self.sde.euler_maruyama(reverse,x,sample_steps=1000)
            else:
                with torch.autocast('cuda',dtype=torch.bfloat16):out=self.sampler(self.net,x,num_steps=50,denominator_floor=.05)
            assert torch.isfinite(out).all()
            parts.append(((out.float()+1)*127.5).clamp(0,255).round().byte().permute(0,2,3,1).cpu().numpy())
        return np.concatenate(parts)


def real(m,key):
    view=m['views'][key];out=Path(m['cache'])/'real'/key;out.mkdir(parents=True,exist_ok=True)
    if (out/'READY.json').exists():assert read(out/'READY.json')['view']==view;return
    f=Features(m)
    for role in ('test','calibration','spatial'):
        rows=view_rows(view,role);path=out/(role+'.npz')
        if path.exists():continue
        values=[];globals=[];spatials=[]
        for start in range(0,len(rows),64):
            a=images(view,rows[start:start+64])
            if role=='test':values.append(f.inception(a))
            else:
                g,v=f.descriptors(a);globals.append(g);spatials.append(v)
        if role=='test':save_npz(path,features=np.concatenate(values))
        else:save_npz(path,global_features=np.concatenate(globals),spatial=np.concatenate(spatials))
        if STOP:raise SystemExit(75)
    write(out/'READY.json',dict(view=view,inception=m['inception_sha256'],dino=m['dino']['feature_weights_sha256'],sha256={role:digest(out/(role+'.npz')) for role in ('test','calibration','spatial')}))


def losses(m,key):
    row=next(r for r in m['transfers'] if r['id']==key);family=row['model'];net=Model(m,family);adapted=net.endpoint(row)
    out=Path(m['cache'])/'loss'/key;out.mkdir(parents=True,exist_ok=True)
    strata=['native',*m['logsnr']];checks=[]
    for domain in ('source',row['target']):
        view=m['views'][family+'_'+domain];paths=view_rows(view,'denoising');size=len(paths)
        for bits in ('000','100','001','011','111'):
            path=out/f'{domain}_{bits}.npz'
            if path.exists():continue
            net.net.load_state_dict(net.hybrid(net.source,adapted,bits));values=np.zeros((len(strata),size,m['loss_replicates']),np.float32)
            for si,stratum in enumerate(strata):
                for start in range(0,size,32):
                    x=torch.from_numpy(images(view,paths[start:start+32])).permute(0,3,1,2).cuda().float()/127.5-1
                    for rep in range(m['loss_replicates']):
                        gen=torch.Generator().manual_seed(m['loss_seed']+si*10000019+start*1009+rep)
                        if stratum=='native':
                            t=torch.rand(len(x),generator=gen) if family=='uvit64' else (torch.randn(len(x),generator=gen)*.8-.8).sigmoid()
                        elif family=='uvit64':
                            cumulative=math.log1p(math.exp(-stratum));tt=(-.1+math.sqrt(.01+39.8*cumulative))/19.9;t=torch.full((len(x),),tt)
                        else:t=torch.full((len(x),),1/(1+math.exp(-.5*stratum)/net.noise))
                        eps=torch.randn(x.shape,generator=gen).cuda();t=t.cuda()
                        if family=='uvit64':
                            ab=torch.exp(-.1*t-9.95*t*t);z=ab.sqrt()[:,None,None,None]*x+(1-ab).sqrt()[:,None,None,None]*eps
                            pred=net.predict(z,999*t);err=(pred-eps).square()
                        else:
                            z=t[:,None,None,None]*x+(1-t[:,None,None,None])*(eps*net.noise);pred=net.predict(z,t)
                            err=((pred-x)/(1-t[:,None,None,None]).clamp_min(.05)).square()
                        v=err.flatten(1).mean(1);assert torch.isfinite(v).all();values[si,start:start+len(x),rep]=v.cpu().numpy()
                    if STOP:raise SystemExit(75)
            save_npz(path,losses=values,strata=np.array([str(s) for s in strata]),paths=np.array(paths));checks.append(str(path))
    write(out/'READY.json',dict(transfer=key,epoch=100,noise_scale=net.noise,states=['000','100','001','011','111'],replicates=m['loss_replicates']))


def preflight(m,family):
    net=Model(m,family);f=Features(m);results=[]
    x=torch.randn(2,3,net.resolution,net.resolution,device='cuda');t=torch.full((2,),499.5 if family=='uvit64' else .5,device='cuda')
    for row in [r for r in m['transfers'] if r['model']==family]:
        target=net.endpoint(row)
        for bits,expected in [('000',net.source),('111',target)]:
            s=net.hybrid(net.source,target,bits);assert all(torch.equal(s[k],expected[k]) for k in s)
        source_qk=net.hybrid(net.source,net.hybrid(net.source,target,'011'),'100');assert all(torch.equal(source_qk[k],v) for k,v in net.source.items())
        for bits in ('000','100','001','011','111'):
            net.net.load_state_dict(net.hybrid(net.source,target,bits));fp=net.predict(x,t,False);bf=net.predict(x,t,True)
            error=float((fp-bf).square().mean().sqrt()/fp.square().mean().sqrt().clamp_min(1e-6));assert math.isfinite(error) and error<.10,(family,bits,error)
            results.append(dict(transfer=row['id'],bits=bits,bf16_relative_rms_error=error))
    task=next(t for t in m['tasks'] if t['model']==family and t['bits']=='000');net.set_task(task)
    a=net.sampling(900000,2,2);b=net.sampling(900000,2,2);assert np.array_equal(a,b)
    g,v=f.descriptors(a);inc=f.inception(a);assert v.shape==(2,32640) and g.shape==(2,768)
    # Identical images must give exactly zero paired distance, and identical inputs identical features.
    g2,v2=f.descriptors(a);assert np.array_equal(g,g2) and np.array_equal(v,v2)
    trials=[]
    for batch in (64,128,256):
        try:
            torch.cuda.reset_peak_memory_stats();begin=time.monotonic();a=net.sampling(910000+batch,256,batch);inc=f.inception(a);g,v=f.descriptors(a)
            torch.cuda.synchronize();seconds=time.monotonic()-begin;free,total=torch.cuda.mem_get_info()
            trials.append(dict(batch=batch,images=256,seconds=seconds,rate=256/seconds,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,free_gib=free/2**30,total_gib=total/2**30))
        except torch.cuda.OutOfMemoryError:
            gc.collect();torch.cuda.empty_cache();trials.append(dict(batch=batch,oom=True));break
        if STOP:raise SystemExit(75)
    ok=[v for v in trials if not v.get('oom') and v['free_gib']>=v['total_gib']*.2];assert ok
    selected=max(ok,key=lambda x:x['rate'])
    write(Path(m['report'])/f'preflight_{family}.json',dict(model=family,torch_version=torch.__version__,torchvision_version=__import__('torchvision').__version__,endpoint_identities_exact=True,source_qk_exact_in_011=True,sampling_repeat_exact=True,checks=results,trials=trials,batch=selected['batch'],selected=selected))


def generate(m,task):
    out=Path(m['cache'])/'generated'/task['id'];out.mkdir(parents=True,exist_ok=True);net=Model(m,task['model']);net.set_task(task);features=Features(m)
    config=read(Path(m['report'])/f'preflight_{task["model"]}.json');batch=config['batch'];state=out/'worker.json';command=out/'command.json'
    identity=dict(task=task,model=m['models'][task['model']]['source'],transfer=next((r for r in m['transfers'] if r['id']==task['transfer']),None),batch=batch,seed=m['sampling_seed'],path_seed=m['path_seed'],manifest_sha256=m['_sha256'])
    if (out/'identity.json').exists():assert read(out/'identity.json')==identity
    else:write(out/'identity.json',identity)
    def status(label,**kwargs):
        free,total=torch.cuda.mem_get_info();write(state,dict(status=label,pid=os.getpid(),task=task['id'],free_gib=free/2**30,total_gib=total/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,time=time.time(),**kwargs))
    seq=-1;status('ready',seq=seq)
    while not STOP:
        if not command.exists() or read(command)['seq']<=seq:time.sleep(.2);continue
        cmd=read(command);seq=cmd['seq'];start=time.monotonic();count=0;status('working',seq=seq)
        for shard in cmd['shards']:
            path=out/f'shard{shard:03d}.npz';n=min(m['shard_size'],m['bank_size']-shard*m['shard_size']);assert n>0
            if not path.exists():
                pixels=net.sampling(shard,n,batch);inc=features.inception(pixels)
                if shard*m['shard_size']<m['paired_size']:
                    g,v=features.descriptors(pixels);save_npz(path,features=inc,images=pixels,global_features=g,spatial=v)
                else:save_npz(path,features=inc)
                count+=n
            if STOP:raise SystemExit(75)
        status('ready',seq=seq,images=count,seconds=time.monotonic()-start)
        if cmd.get('exit'):return
    raise SystemExit(75)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','real','loss','generate']);p.add_argument('--manifest',required=True);p.add_argument('--key',required=True);a=p.parse_args();m=read(a.manifest);m['_sha256']=digest(a.manifest)
    for rel,h in m['driver_sha256'].items():assert digest(Path(m['code'])/rel)==h,rel
    setup()
    if a.stage=='preflight':preflight(m,a.key)
    elif a.stage=='real':real(m,a.key)
    elif a.stage=='loss':losses(m,a.key)
    else:generate(m,next(t for t in m['tasks'] if t['id']==a.key))
if __name__=='__main__':main()
