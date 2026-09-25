import sys
"""Map each immutable race endpoint to the established three evaluation pipelines."""
import json
import os
from pathlib import Path
import subprocess
import torch

def main():
    c=json.loads(Path(os.environ.get('IMAGENET_CAMPAIGN', 'reports/2026-09-12/imagenet64_race/campaign.json')).read_text())
    r=next(r for r in c['runs'] if r['mode']==os.environ['IMAGENET_RUN'])
    task=int(os.environ['SLURM_ARRAY_TASK_ID'])
    endpoint=c['endpoints'][task//3]
    kind=['vp','denoising','fid'][task%3]
    checkpoint=Path(r['outdir'])/'checkpoints'/f"ckpt_step{endpoint['step']}.pt"
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    if not torch.cuda.is_available(): raise RuntimeError('Evaluation requires CUDA')
    label=f"{r['mode']}_epoch{endpoint['epoch']:03d}"
    folder={'vp':'vp_velocity_loss','denoising':'denoising','fid':'fid50k'}[kind]
    out=Path(c.get('eval_outdir', 'evals/imagenet64_20m'))/folder
    result=out/label/('result.json' if kind=='fid' else 'summary.json')
    if result.exists():
        print('Already complete:',result);return
    command=[sys.executable,'-u','-m','experiments.imagenet64.evaluate',kind]
    if kind=='fid': command+=['model']
    command+=['--checkpoint',str(checkpoint),'--label',label,'--data_root','datasets/imagenet64/prepared',
              '--outdir',str(out),'--batch','32','--num_workers','0','--precision','fp32',
              '--resolution','64','--channels','3']
    if kind=='fid':
        command+=['--num_images','50000','--shard_size','500','--sampler_steps','50',
                  '--real_stats',c.get('real_stats', 'evals/imagenet64_20m/real_stats/validation_stats.npz')]
    elif kind=='denoising':
        command+=['--time_min','.05','--time_max','4','--auc_steps','32']
    subprocess.run(command,check=True)

if __name__=='__main__':main()
