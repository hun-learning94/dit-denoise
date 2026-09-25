#!/usr/bin/env python3
"""Build one source-based weight-only endpoint hybrid with AdaLN and globals frozen."""
import argparse, hashlib, json
from pathlib import Path
import torch

def main():
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True); p.add_argument('--adapted',type=Path,required=True)
    p.add_argument('--code',choices=[f'{i:03b}' for i in range(8)],required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    s=torch.load(a.source,map_location='cpu',weights_only=False); t=torch.load(a.adapted,map_location='cpu',weights_only=False)
    ss=s['ema']; ts=t['ema']; assert ss.keys()==ts.keys(); r,v,m=map(int,a.code)
    out={}; counts={'QK':0,'VO':0,'MLP':0,'source':0}
    for name,x in ss.items():
        y=ts[name]; z=x.clone(); factor=None
        if name.startswith('model.blocks.') and '.attn.qkv.' in name:
            d=x.shape[0]//3; z[:2*d]=y[:2*d] if r else x[:2*d]; z[2*d:]=y[2*d:] if v else x[2*d:]; counts['QK']+=x[:2*d].numel(); counts['VO']+=x[2*d:].numel()
        elif name.startswith('model.blocks.') and '.attn.proj.' in name: z=y.clone() if v else x.clone(); counts['VO']+=x.numel()
        elif name.startswith('model.blocks.') and '.mlp.' in name: z=y.clone() if m else x.clone(); counts['MLP']+=x.numel()
        else:
            # The ordinary trained network must retain the source EMA exactly.
            assert torch.equal(t['net'][name], x), f'Frozen coordinate changed: {name}'
            counts['source']+=x.numel()
        out[name]=z
    payload={k:v for k,v in t.items() if k not in {'net','ema','opt'}}; payload['net']=out; payload['ema']=out
    payload['experiment3_transplant']={'code':a.code,'source':str(a.source.resolve()),'adapted':str(a.adapted.resolve()),'factor_counts':counts,'global_coordinates':'source','adaln_coordinates':'source','scope':'block_weights'}
    a.output.parent.mkdir(parents=True,exist_ok=True); torch.save(payload,a.output)
    print(json.dumps(payload['experiment3_transplant'],indent=2))
if __name__=='__main__': main()
