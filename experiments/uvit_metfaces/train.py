import argparse,copy,json,os,signal,time
from pathlib import Path
import numpy as np
import torch
from experiments.component_transfer_scale.model import load_source,plain_model,uvit_module,write,save,digest,seed_all,rng_state,restore_rng
from experiments.component_transfer_scale.runner import Images,loader
from .model import attach,effective,frozen,EMA

STOP=False
def stop(*_):
    global STOP
    STOP=True

class Training:
    def __init__(self,m,arm):
        self.m=m;self.arm=arm;self.output=Path(m['runs'][arm]);self.output.mkdir(parents=True,exist_ok=True)
        net,source,_=load_source(m['source']);self.net=net.cuda();self.source={n:v.cuda() for n,v in source.items()}
        self.counts=attach(self.net,arm,m.get('rank_multiplier',1),training_seed=m['recipe']['training_seed']);self.ema=EMA(self.source,self.counts)
        self.params=[p for p in self.net.parameters() if p.requires_grad]
        self.opt=torch.optim.AdamW(self.params,lr=1e-4,betas=(.99,.99),weight_decay=0)
        self.sde=uvit_module(m['source'],'sde').VPSDE();self.step=0;self.best=float('inf');self.best_step=None
        self.identity=dict(arm=arm,source=m['source']['checkpoint_sha256'],data=digest(m['data']),recipe=m['recipe'],code_sha256=m['code_sha256'])
        seed_all(m['recipe']['corruption_seed'])
    def update(self,images):
        self.net.train();self.opt.zero_grad(set_to_none=True)
        x=images.cuda().float()/127.5-1
        flips=torch.rand(len(x),device='cuda')<.5;x[flips]=x[flips].flip(-1)
        t,eps,z=self.sde.sample(x)
        lr=1e-4*min(1.,(self.step+1)/(5*self.m['steps_per_epoch']))
        for pg in self.opt.param_groups:pg['lr']=lr
        loss=0.
        for start in range(0,len(x),8):
            with torch.autocast('cuda',dtype=torch.bfloat16):out=self.net(z[start:start+8],999*t[start:start+8])
            part=(out.float()-eps[start:start+8]).square().flatten(1).mean(1).sum()/len(x)
            assert torch.isfinite(part);part.backward();loss+=float(part.detach())
        norm=torch.nn.utils.clip_grad_norm_(self.params,1.,error_if_nonfinite=True)
        self.opt.step();self.ema.update(self.net,self.source);self.step+=1
        return dict(loss=loss,gradient_norm=float(norm),lr=lr)
    def payload(self):
        return dict(identity=self.identity,step=self.step,net=self.net.state_dict(),optimizer=self.opt.state_dict(),ema=self.ema.shadow,
                    ema_updates=self.ema.updates,rng=rng_state(),best=self.best,best_step=self.best_step)
    def restore(self,p):
        assert p['identity']==self.identity;self.net.load_state_dict(p['net']);self.opt.load_state_dict(p['optimizer'])
        self.ema.shadow={n:v.cuda() for n,v in p['ema'].items()};self.ema.updates=p['ema_updates'];self.step=p['step']
        self.best=p['best'];self.best_step=p['best_step'];restore_rng(p['rng']);assert self.step==self.ema.updates
    def checkpoint(self):
        frozen(self.net,self.source,self.counts)
        save(self.output/'online_next.pt',self.payload());latest=self.output/'online_latest.pt'
        if latest.exists():latest.replace(self.output/'online_previous.pt')
        (self.output/'online_next.pt').replace(latest)
    def export(self,path):
        save(path,dict(identity=self.identity,step=self.step,epoch=self.step/self.m['steps_per_epoch'],ema=self.ema.shadow,counts=self.counts))
    @torch.no_grad()
    def validate(self,data):
        keep=rng_state();net=plain_model(self.m['source'],self.ema.shadow).cuda().eval();seed_all(2026110901)
        values=[]
        for i in range(0,len(data),8):
            x=torch.stack([data[j] for j in range(i,min(i+8,len(data)))]).cuda().float()/127.5-1
            t,eps,z=self.sde.sample(x)
            with torch.autocast('cuda',dtype=torch.bfloat16):out=net(z,999*t)
            values.extend((out.float()-eps).square().flatten(1).mean(1).cpu().tolist())
        del net;restore_rng(keep);return dict(mean=float(np.mean(values)),per_image=values,count=len(values))

def run(m,arm):
    tr=Training(m,arm);data=json.loads(Path(m['data']).read_text());train=Images(data,'train');valid=Images(data,'valid')
    if (tr.output/'COMPLETE.json').exists():return
    latest=tr.output/'online_latest.pt'
    if latest.exists():tr.restore(torch.load(latest,map_location='cpu',weights_only=False))
    start=time.monotonic();last=start;total=m['steps_per_epoch']*1000
    for images in loader(train,m['recipe']['training_seed'],tr.step,total,workers=2):
        result=tr.update(images)
        if tr.step%100==0:
            p=dict(result,step=tr.step,epoch=tr.step/m['steps_per_epoch'],total_updates=total,seconds_per_update=(time.monotonic()-last)/100,job_id=os.getenv('SLURM_JOB_ID'))
            write(tr.output/'progress.json',p);print(json.dumps(p),flush=True);last=time.monotonic()
        if tr.step%(25*m['steps_per_epoch'])==0:
            epoch=tr.step//m['steps_per_epoch'];v=tr.validate(valid);write(tr.output/f'validation_epoch{epoch:04d}.json',dict(step=tr.step,epoch=epoch,**v))
            tr.export(tr.output/f'ema_epoch{epoch:04d}.pt')
            if v['mean']<tr.best:
                tr.best=v['mean'];tr.best_step=tr.step;tr.export(tr.output/'best.pt')
                write(tr.output/'selection.json',dict(step=tr.step,epoch=epoch,validation_loss=tr.best,rule='minimum fixed-noise validation epsilon MSE; earlier exact ties'))
        if tr.step%500==0 or tr.step%(25*m['steps_per_epoch'])==0 or STOP or tr.step==total or time.monotonic()-start>10.5*3600:tr.checkpoint()
        if (STOP or time.monotonic()-start>10.5*3600) and tr.step<total:raise SystemExit(75)
    tr.export(tr.output/'final.pt');write(tr.output/'COMPLETE.json',dict(step=tr.step,epoch=1000,best_step=tr.best_step,trainable=tr.counts['parameters'],identity=tr.identity))

def smoke(m):
    data=json.loads(Path(m['data']).read_text());dataset=Images(data,'train');images=torch.stack([dataset[i] for i in range(32)])
    checks=[]
    for arm in m['arms']:
        tr=Training(m,arm);tr.net.eval();x=images[:2].cuda().float()/127.5-1;t=torch.full((2,),499.5,device='cuda')
        plain=plain_model(m['source'],tr.source).cuda().eval()
        with torch.no_grad():
            assert torch.equal(tr.net(x,t),plain(x,t)),'Nonzero initial adapter'
        del plain
        result=tr.update(images);frozen(tr.net,tr.source,tr.counts)
        if not tr.counts['full']:
            modules=dict(tr.net.named_modules())
            assert all(modules[r['adapter']].lora_B.detach().abs().max()>0 for r in tr.counts['matrices']), 'Inactive adapter'
        # A resumed update must exactly replay parameters and effective EMA.
        snapshot=copy.deepcopy(tr.payload());tr.update(images)
        expected={n:p.detach().clone() for n,p in tr.net.named_parameters() if p.requires_grad};ema={n:tr.ema.shadow[n].clone() for n in tr.ema.names}
        tr.restore(snapshot);tr.update(images)
        assert all(torch.equal(dict(tr.net.named_parameters())[n],p) for n,p in expected.items())
        assert all(torch.equal(tr.ema.shadow[n],v) for n,v in ema.items())
        merged=plain_model(m['source'],effective(tr.net,tr.source,tr.counts)).cuda().eval();tr.net.eval()
        with torch.no_grad():
            error=(merged(x,t)-tr.net(x,t)).abs().max().item();assert error<2e-5,error
        validation=tr.validate(Images(data,'valid'));assert np.isfinite(validation['mean'])
        checks.append(dict(arm=arm,parameters=tr.counts['parameters'],initial_exact=True,resume_exact=True,frozen_exact=True,merged_max_error=error,finite_update=result))
        del tr,merged,snapshot,expected,ema;torch.cuda.empty_cache()
    write(Path(m['report'])/'SMOKE_PASS.json',dict(checks=checks,gpu=torch.cuda.get_device_name(),data_sha256=digest(m['data'])))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','smoke']);p.add_argument('--manifest',required=True);p.add_argument('--task',type=int,default=0);a=p.parse_args()
    m=json.loads(Path(a.manifest).read_text())
    for f,h in m['code_sha256'].items():assert digest(Path(m['code'])/f)==h,f
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported();torch.ones(1,device='cuda').sum().item()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop)
    if a.stage=='smoke':smoke(m)
    else:
        assert (Path(m['report'])/'SMOKE_PASS.json').exists();run(m,m['arms'][a.task])

if __name__=='__main__':main()
