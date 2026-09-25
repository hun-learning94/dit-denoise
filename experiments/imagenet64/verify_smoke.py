import argparse
import json
from pathlib import Path
import torch
from eval_model_comparison import load_model
from edm import vp_heun_sampler

def main():
    p=argparse.ArgumentParser();p.add_argument('--outdir',type=Path,required=True);args=p.parse_args()
    pointer=json.loads((args.outdir/'resume.json').read_text())
    model,ckpt,_,_,used_ema,_=load_model(Path(pointer['checkpoint']),torch.device('cuda'),True)
    assert ckpt['step']==60 and used_ema
    assert ckpt['distributed_protocol']['effective_batch']==256
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=ckpt['args']['precision']=='bf16'):
        output=vp_heun_sampler(model,torch.randn(2,3,64,64,device='cuda'),num_steps=2,denominator_floor=.05)
    assert torch.isfinite(output).all()
    telemetry=[json.loads(p.read_text()) for p in sorted(args.outdir.glob('telemetry_rank*.json'))]
    assert len(telemetry)==4 and all(t['start_step']==50 and t['final_step']==60 for t in telemetry)
    result=dict(passed=True,checkpoint=pointer['checkpoint'],precision=ckpt['args']['precision'],
                parameters=sum(p.numel() for p in model.parameters()),telemetry=telemetry,
                resumed_from_step=50,final_step=60,effective_batch=256,ema_sampling_finite=True)
    (args.outdir/'smoke_passed.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
