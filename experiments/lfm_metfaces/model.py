"""Independent full-matrix LoRA factors and EMA of effective dense weights."""
import hashlib,math
import torch
from torch import nn
from torch.nn import functional as F
RANKS={'qk':(60,0,0),'vo':(0,60,0),'mlp':(0,0,24),'vo_mlp':(0,30,12),'qk_vo_mlp':(20,20,8)}
class LowRank(nn.Module):
 def __init__(self,di,do,r,seed,device):
  super().__init__();self.lora_A=nn.Parameter(torch.empty(r,di,device=device));self.lora_B=nn.Parameter(torch.zeros(do,r,device=device))
  self.lora_A.data.uniform_(-1/math.sqrt(di),1/math.sqrt(di),generator=torch.Generator(device=device).manual_seed(seed))
 def forward(self,x):return F.linear(F.linear(x,self.lora_A),self.lora_B)
 def delta(self):return self.lora_B@self.lora_A
class Linear(nn.Module):
 def __init__(self,old,r,seed):
  super().__init__();self.weight=old.weight;self.bias=old.bias;self.adapter=LowRank(old.in_features,old.out_features,r,seed,old.weight.device)
 def forward(self,x):return F.linear(x,self.weight,self.bias)+self.adapter(x)
class QKV(nn.Module):
 def __init__(self,old,ranks,seeds):
  super().__init__();self.weight=old.weight;self.bias=old.bias;self.dim=old.in_features
  self.adapters=nn.ModuleDict({k:LowRank(self.dim,self.dim,r,seeds[k],old.weight.device) for k,r in ranks.items() if r})
 def forward(self,x):
  return torch.cat([v+self.adapters[k](x) if k in self.adapters else v for k,v in zip(['q','k','v'],F.linear(x,self.weight,self.bias).split(self.dim,-1))],-1)
def attach(net,arm,seed):
 assert len(net.blocks)==24
 if arm=='full':
  net.requires_grad_(True);assert sum(p.numel() for p in net.parameters())==457062416
  return dict(full=True,parameters=457062416,matrices=[])
 net.requires_grad_(False);qk,vo,mlp=RANKS[arm];records=[]
 def sd(name):return int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:8],'little')%(2**63-1)
 for i,b in enumerate(net.blocks):
  prefix=f'blocks.{i}';d=b.attn.qkv.in_features;assert d==1024
  ranks={k:r for k,r in [('q',qk),('k',qk),('v',vo)] if r}
  if ranks:
   name=prefix+'.attn.qkv';b.attn.qkv=QKV(b.attn.qkv,ranks,{k:sd(name+'.'+k) for k in ranks})
   for j,k in enumerate(['q','k','v']):
    if k in ranks:records.append(dict(weight=name+'.weight',start=j*d,stop=(j+1)*d,adapter=name+'.adapters.'+k))
  for owner,attr,r,suffix in [(b.attn,'proj',vo,'.attn.proj'),(b.mlp,'fc1',mlp,'.mlp.fc1'),(b.mlp,'fc2',mlp,'.mlp.fc2')]:
   if r:
    name=prefix+suffix;old=getattr(owner,attr);setattr(owner,attr,Linear(old,r,sd(name)))
    records.append(dict(weight=name+'.weight',start=0,stop=old.out_features,adapter=name+'.adapter'))
 count=sum(p.numel() for p in net.parameters() if p.requires_grad);assert count==5898240
 assert all(n.endswith(('.lora_A','.lora_B')) for n,p in net.named_parameters() if p.requires_grad)
 return dict(full=False,parameters=count,matrices=records)
@torch.no_grad()
def frozen(net,source,counts):
 if not counts['full']:
  current=net.state_dict()
  for n,v in source.items():assert torch.equal(current[n].cpu(),v),n
@torch.no_grad()
def effective(net,source,counts):
 if counts['full']:return {n:v.detach() for n,v in net.state_dict().items()}
 result={n:v.cuda().clone() for n,v in source.items()};mods=dict(net.named_modules())
 for r in counts['matrices']:result[r['weight']][r['start']:r['stop']]+=mods[r['adapter']].delta()
 return result
class EMA:
 def __init__(self,source,counts):
  self.counts=counts;self.names=set(source) if counts['full'] else {r['weight'] for r in counts['matrices']}
  self.shadow={n:source[n].cuda().clone() for n in self.names};self.updates=0
 @torch.no_grad()
 def update(self,net):
  state=net.state_dict()
  if self.counts['full']:
   for n in self.names:self.shadow[n].lerp_(state[n].float(),.0001)
  else:
   mods=dict(net.named_modules())
   for r in self.counts['matrices']:
    n=r['weight'];sl=slice(r['start'],r['stop']);self.shadow[n][sl].lerp_(state[n][sl]+mods[r['adapter']].delta(),.0001)
  self.updates+=1
 def dense(self,source):return {n:self.shadow[n] if n in self.shadow else v for n,v in source.items()}
