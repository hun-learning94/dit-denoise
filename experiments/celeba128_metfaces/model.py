import hashlib
import torch
from experiments.lora_metfaces.model import LoRALinear, SplitQKVLoRA, merged_state

RANKS={'qk':(60,0,0),'vo':(0,60,0),'mlp':(0,0,24),'vo_mlp':(0,30,12),'qk_vo_mlp':(20,20,8)}

def attach(net,arm,rank_multiplier=1,training_seed=0):
    assert isinstance(rank_multiplier,int) and rank_multiplier>=1
    if arm=='full':
        net.requires_grad_(True)
        count=sum(p.numel() for p in net.parameters());assert count==133936320;return dict(parameters=count,matrices=[],full=True)
    qk,vo,mlp=(rank_multiplier*r for r in RANKS[arm]);net.requires_grad_(False);records=[]
    seed=lambda s:int.from_bytes(hashlib.sha256((str(training_seed)+':'+s).encode()).digest()[:8],'little')%(2**63-1)
    blocks=[(n,m) for n,m in net.named_modules() if hasattr(m,'attn') and hasattr(m,'mlp')]
    assert len(blocks)==12
    for prefix,b in blocks:
        ranks={p:r for p,r in [('q',qk),('k',qk),('v',vo)] if r}
        if ranks:
            name=prefix+'.attn.qkv';b.attn.qkv=SplitQKVLoRA(b.attn.qkv,ranks,ranks,{p:seed(name+'.'+p) for p in ranks})
            for i,p in enumerate(('q','k','v')):
                if p in ranks:records.append(dict(weight=name+'.weight',start=i*768,stop=(i+1)*768,adapter=name+'.adapters.'+p))
        for owner,attr,r,suffix in [(b.attn,'proj',vo,'.attn.proj'),(b.mlp,'fc1',mlp,'.mlp.fc1'),(b.mlp,'fc2',mlp,'.mlp.fc2')]:
            if not r:continue
            name=prefix+suffix;original=getattr(owner,attr);setattr(owner,attr,LoRALinear(original,r,seed(name)))
            records.append(dict(weight=name+'.weight',start=0,stop=original.out_features,adapter=name+'.adapter'))
    count=sum(p.numel() for p in net.parameters() if p.requires_grad)
    assert count==2211840*rank_multiplier,(arm,count,rank_multiplier)
    return dict(parameters=count,matrices=records,full=False)

def effective(net,source,counts):
    return net.state_dict() if counts['full'] else merged_state(net,source,counts)

def frozen(net,source,counts):
    if not counts['full']:
        s=net.state_dict();assert all(torch.equal(s[n],v) for n,v in source.items())

class EMA:
    def __init__(self,source,counts):
        self.shadow={n:v.clone() for n,v in source.items()};self.counts=counts;self.updates=0
        self.names=set(source) if counts['full'] else {r['weight'] for r in counts['matrices']}
    @torch.no_grad()
    def update(self,net,source):
        state=effective(net,source,self.counts)
        for n in self.names:self.shadow[n].lerp_(state[n].float(),.0001)
        self.updates+=1
