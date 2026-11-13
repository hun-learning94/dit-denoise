"""Independent LFM MetFaces adapters, compact exact resume, effective-weight EMA."""
import argparse,gc,json,math,os,time
from pathlib import Path
import torch
from experiments.component_transfer_scale.model import load_source,plain_model
from . import common as c
from .cache import Cache
from .model import attach,EMA,frozen,effective

class Training:
 def __init__(self,m,arm):
  self.m=m;self.arm=arm;self.net,self.source,_=load_source(m['source']);self.net.cuda().train()
  self.counts=attach(self.net,arm,m['recipe']['seed']);self.params={n:p for n,p in self.net.named_parameters() if p.requires_grad}
  self.opt=torch.optim.AdamW(self.params.values(),lr=m['recipe']['lr'],betas=(.9,.999),weight_decay=0,foreach=False)
  self.ema=EMA(self.source,self.counts);self.cache=Cache(m);self.step=0;self.best=float('inf');self.best_step=0;self.history=[]
  self.out=Path(m['runs'][arm]);self.out.mkdir(parents=True,exist_ok=True);self.last_save=time.monotonic()
  c.seed_all(m['recipe']['noise_seed']);self.permutation=None;self.permutation_epoch=-1
 def indices(self):
  epoch,offset=divmod(self.step,33)
  if epoch!=self.permutation_epoch:
   self.permutation=torch.randperm(1057,generator=torch.Generator().manual_seed(self.m['recipe']['seed']+1000003*epoch)).tolist();self.permutation_epoch=epoch
  return self.permutation[offset*32:(offset+1)*32]
 def batch(self):
  # Draw an entire effective batch before accumulation, pairing all arms exactly.
  flips=torch.rand(32,device='cuda')<.5;z=self.cache.draw(self.indices(),flips=flips)
  t=torch.rand(32,device='cuda');noise=torch.randn_like(z);sigma=1e-5+(1-1e-5)*t
  return (1-t[:,None,None,None])*z+sigma[:,None,None,None]*noise,t,(1-1e-5)*noise-z
 def update(self):
  self.net.train();x,t,target=self.batch();self.opt.zero_grad(set_to_none=True);loss_sum=0.
  lr=self.m['recipe']['lr']*min(1.,(self.step+1)/165)
  for pg in self.opt.param_groups:pg['lr']=lr
  for first in range(0,32,2):
   with torch.autocast('cuda',dtype=torch.bfloat16):pred=self.net(t[first:first+2],x[first:first+2])
   loss=(pred.float()-target[first:first+2]).square().mean();assert torch.isfinite(loss),'Nonfinite loss'
   (loss/16).backward();loss_sum+=float(loss.detach())/16
  norm=torch.nn.utils.clip_grad_norm_(list(self.params.values()),1.,error_if_nonfinite=True)
  self.opt.step();self.ema.update(self.net);self.step+=1
  return dict(step=self.step,epoch=self.step/33,loss=loss_sum,gradient_norm=float(norm),lr=lr)
 def payload(self):
  return dict(identity=c.identity(self.m,self.arm),step=self.step,online=c.cpu(self.params),optimizer=c.cpu(self.opt.state_dict()),
              ema=c.cpu(self.ema.shadow),ema_updates=self.ema.updates,rng=c.rng_state(),best=self.best,best_step=self.best_step,history=self.history)
 def restore(self,p):
  assert p['identity']==c.identity(self.m,self.arm);assert set(p['online'])==set(self.params);assert set(p['ema'])==set(self.ema.shadow)
  assert 0<=p['step']<=33000 and p['ema_updates']==p['step']
  for key,values,expected in [('online',p['online'],self.params),('ema',p['ema'],self.ema.shadow)]:
   for n,v in values.items():assert v.shape==expected[n].shape and torch.isfinite(v).all(),(key,n)
  with torch.no_grad():
   for n,pv in self.params.items():pv.copy_(p['online'][n])
   for n,v in self.ema.shadow.items():v.copy_(p['ema'][n])
  self.opt.load_state_dict(p['optimizer']);self.step=p['step'];self.ema.updates=p['ema_updates'];self.best=p['best'];self.best_step=p['best_step'];self.history=p['history'];self.permutation_epoch=-1
  c.restore_rng(p['rng']);self.last_save=time.monotonic()
 def resume(self):
  errors=[]
  for name in ('latest.pt','previous.pt'):
   p=self.out/name
   if p.exists():
    try:
     self.restore(torch.load(p,map_location='cpu',weights_only=False));print('RESUMED',self.arm,name,self.step,flush=True);return
    except Exception as e:errors.append(f'{name}: {type(e).__name__}: {e}');print('RESUME_REJECT',errors[-1],flush=True)
  if errors:raise RuntimeError('No valid checkpoint: '+repr(errors))
 def checkpoint(self):
  latest=self.out/'latest.pt';previous=self.out/'previous.pt'
  # Link the old durable inode before atomically replacing latest.
  if latest.exists():
   temp=self.out/f'previous.tmp-{os.getpid()}'
   if temp.exists():temp.unlink()
   os.link(latest,temp);temp.replace(previous)
  c.save(latest,self.payload());c.write(self.out/'progress.json',dict(arm=self.arm,step=self.step,epoch=self.step/33,best_epoch=self.best_step/33,best_loss=None if not math.isfinite(self.best) else self.best,history=self.history[-3:],saved_unix=time.time()))
  c.write(self.out/'history.json',self.history);self.last_save=time.monotonic();print('CHECKPOINT',self.arm,self.step,flush=True)
 def export(self,name):
  c.save(self.out/name,dict(identity=c.identity(self.m,self.arm),step=self.step,epoch=self.step/33,ema_is_partial=not self.counts['full'],ema=c.cpu(self.ema.shadow),source_sha256=self.m['source']['checkpoint_sha256']))
 @torch.no_grad()
 def validation(self):
  rng=c.rng_state();net=plain_model(self.m['source'],self.ema.dense(self.source)).cuda().eval();value=denoise(net,self.cache,self.cache.splits['valid'],2026111001,1)
  del net;gc.collect();torch.cuda.empty_cache();c.restore_rng(rng);return value

@torch.no_grad()
def denoise(net,cache,indices,seed,repeats=1,group='target'):
 generator=torch.Generator(device='cuda').manual_seed(seed);values=[]
 for repeat in range(repeats):
  total=0.;count=0
  for first in range(0,len(indices),2):
   selected=indices[first:first+2];z=cache.draw(selected,group=group,generator=generator)
   t=torch.rand(len(z),device='cuda',generator=generator);noise=torch.randn(z.shape,device='cuda',generator=generator)
   x=(1-t[:,None,None,None])*z+(1e-5+(1-1e-5)*t[:,None,None,None])*noise
   with torch.autocast('cuda',dtype=torch.bfloat16):pred=net(t,x)
   loss=(pred.float()-((1-1e-5)*noise-z)).square().flatten(1).mean(1);assert torch.isfinite(loss).all()
   total+=float(loss.sum());count+=len(loss)
  values.append(total/count)
 return dict(mean=sum(values)/len(values),replicates=values,images=len(indices),seed=seed,posterior='fresh sampled, fixed corruption RNG',objective='native flow velocity MSE')

def assert_state(a,b):
 if isinstance(a,torch.Tensor):assert torch.equal(a.cpu(),b.cpu()),'Exact resume tensor mismatch'
 elif isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:assert_state(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b)
  for x,y in zip(a,b):assert_state(x,y)
 else:assert a==b,(a,b)

def smoke(m,arm):
 t=Training(m,arm);torch.cuda.reset_peak_memory_stats();x=torch.randn(2,4,32,32,device='cuda');ts=torch.tensor([.25,.75],device='cuda')
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  plain=plain_model(m['source'],t.source).cuda().eval();expected=plain(ts,x);actual=t.net.eval()(ts,x);torch.testing.assert_close(actual,expected,rtol=0,atol=0)
 del plain;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
 c.seed_all(m['recipe']['noise_seed']);start=time.monotonic();first=t.update();snapshot=t.payload();second=t.update();torch.cuda.synchronize();seconds=(time.monotonic()-start)/2
 # A has zero gradient at zero B, hence inspect after the second complete update.
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in t.params.values())
 if arm!='full':assert all(float(p.grad.norm())>0 for p in t.params.values()),'Inactive LoRA factor'
 frozen(t.net,t.source,t.counts);wanted={k:c.cpu(v) for k,v in [('online',t.params),('optimizer',t.opt.state_dict()),('ema',t.ema.shadow)]}
 tmp=Path(m['report'])/'preflight'/f'{arm}_resume.pt';c.save(tmp,snapshot);del snapshot
 t.restore(torch.load(tmp,map_location='cpu',weights_only=False));repeat=t.update();assert second==repeat
 for key,actual in [('online',t.params),('optimizer',t.opt.state_dict()),('ema',t.ema.shadow)]:assert_state(wanted[key],actual)
 tmp.unlink();del wanted
 with torch.no_grad():
  merged=plain_model(m['source'],effective(t.net,t.source,t.counts)).cuda().eval()
  # Addition before vs after BF16 linear has different rounding; also check FP32.
  a=t.net.eval()(ts,x);b=merged(ts,x);torch.testing.assert_close(a,b,rtol=1e-4,atol=5e-5)
  with torch.autocast('cuda',dtype=torch.bfloat16):a=t.net(ts,x);b=merged(ts,x)
  relative=float((a.float()-b.float()).norm()/a.float().norm());assert relative<.03,relative
 del merged;gc.collect();torch.cuda.empty_cache();validation=t.validation();assert math.isfinite(validation['mean'])
 result=dict(passed=True,arm=arm,manifest_sha256=c.digest(m['_manifest']),parameters=t.counts['parameters'],update_seconds=seconds,peak_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),zero_update_identity=True,resume_exact=True,frozen_identity=True,merge_bf16_relative_error=relative,validation=validation,gpu=torch.cuda.get_device_name())
 c.write(Path(m['report'])/'preflight'/f'{arm}.json',result);print(json.dumps(result),flush=True)

def train(m,arm):
 c.check_pass(m);out=Path(m['runs'][arm])
 if (out/'COMPLETE.json').exists():return
 t=Training(m,arm);t.resume()
 while t.step<33000:
  start=time.monotonic();row=t.update();row['seconds']=time.monotonic()-start
  if t.step%50==0:print(json.dumps(row),flush=True)
  if t.step%825==0:
   row['validation']=t.validation();t.history.append(row)
   if row['validation']['mean']<t.best:t.best=row['validation']['mean'];t.best_step=t.step;t.export('best.pt')
   if t.step//33 in [25,50,100,200,500,1000]:t.export(f'ema_epoch{t.step//33:04d}.pt')
  if t.step%250==0 or time.monotonic()-t.last_save>=300 or t.step%825==0 or c.STOP:t.checkpoint()
  if c.STOP:raise SystemExit(75)
 frozen(t.net,t.source,t.counts);t.export('final.pt');t.checkpoint()
 c.write(out/'COMPLETE.json',dict(identity=c.identity(m,arm),step=t.step,epoch=1000,best_epoch=t.best_step/33,best_loss=t.best,parameters=t.counts['parameters'],final_sha256=c.digest(out/'final.pt'),best_sha256=c.digest(out/'best.pt')))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--arm',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();m=c.setup(a.manifest);m['_manifest']=a.manifest
 (smoke if a.smoke else train)(m,a.arm)
