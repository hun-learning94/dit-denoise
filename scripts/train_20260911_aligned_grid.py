"""Launch one immutable aligned-grid transfer or target scratch run."""
from pathlib import Path
import json,os,subprocess,sys
r=json.loads(Path(sys.argv[1]).read_text())[int(os.environ['SLURM_ARRAY_TASK_ID'])]
run=Path(r['run_dir']);latest=max((int(p.stem[9:]),p) for p in (run/'checkpoints').glob('ckpt_step*.pt')) if (run/'checkpoints').exists() and list((run/'checkpoints').glob('ckpt_step*.pt')) else None
if latest and latest[0]>=r['updates']:
 print('Already complete',latest,flush=True);sys.exit(0)
init=['--resume',str(latest[1])] if latest else ['--init_from',r['source_checkpoint'],'--init_weights','ema'] if r['mode']=='transfer' else []
import torch
assert torch.cuda.is_available()
name=torch.cuda.get_device_name(0)
precision='fp32' if '2080 ti' in name.lower() else 'bf16' if torch.cuda.is_bf16_supported() else 'fp16'
print(json.dumps(dict(r,gpu=name,precision=precision,initialization=init)),flush=True)
dim,depth,heads=(256,8,8) if r['scale']=='10m' else (320,10,10)
cmd=[sys.executable,'-u','train.py','--dataset','folder','--data_root',f"datasets/{r['target']}/train",'--resolution','64','--channels','3','--arch','dit','--patch','4','--dim',str(dim),'--depth',str(depth),'--heads',str(heads),'--mlp_ratio','4','--prediction_type','x','--noise_process','vp','--vp_logit_mean','-0.8','--vp_logit_std','0.8','--vp_denominator_floor','0.05','--sample_steps','50','--batch','32','--adapt_steps' if r['mode']=='transfer' else '--steps',str(r['updates']),'--lr','0.0002','--weight_decay','0','--grad_clip','1','--ema','0.999','--precision',precision,'--seed',str(r['seed']),'--num_workers','4','--trainable_groups','block_weights' if r['mode']=='transfer' else 'all','--ckpt_every',str(r['regular_interval']),'--sample_every',str(r['regular_interval']),'--checkpoint_steps',*[str(x) for x in r['checkpoint_steps']],'--outdir',str(run),*init]
# Replace the launcher so the sbatch requeue handler signals the training process itself.
os.execv(sys.executable,cmd)
