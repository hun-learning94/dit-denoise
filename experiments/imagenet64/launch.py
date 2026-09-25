"""Slurm supervisor: launch chosen topology, resume atomically, checkpoint before requeue."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import shutil
import time

ROOT = Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode', choices=['ddp4','h200'], required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--campaign', type=Path, default=ROOT/'reports/2026-09-12/imagenet64_race/campaign.json')
    p.add_argument('--target_steps', type=int, default=None,
                   help='Override the campaign endpoint for a continuation run')
    args=p.parse_args()
    campaign=args.campaign.resolve()
    config=json.loads(campaign.read_text())
    row=next(r for r in config['runs'] if r['mode']==args.mode)
    out=ROOT/(row['smoke_outdir'] if args.smoke else row['outdir'])
    out.mkdir(parents=True,exist_ok=True)
    stop_file=out/f"stop_{os.environ.get('SLURM_JOB_ID','local')}"
    stop_file.unlink(missing_ok=True)
    stop_requested=False
    def stop(signum, frame):
        nonlocal stop_requested
        stop_requested=True
        stop_file.touch()
    signal.signal(signal.SIGUSR1, stop)
    signal.signal(signal.SIGTERM, stop)
    topology=[shutil.which('torchrun') or 'torchrun','--standalone','--nnodes=1','--nproc_per_node=4'] if args.mode=='ddp4' else [sys.executable]
    base=topology+['-m','experiments.imagenet64.train','--data_root','datasets/imagenet64/prepared',
        '--outdir',str(out),'--batch',str(row['batch']),'--accumulation',str(row['accumulation']),
        '--num_workers',str(row['workers_per_rank']),'--precision','auto','--stop_file',str(stop_file),
        '--ckpt_every','5000','--sample_every','0','--rolling_recovery','--log_every','10' if args.smoke else '100']
    # Require a complete manifest/count check before touching either training run.
    for key, value in config.get('training_args', {}).items():
        base += ['--'+key, str(value)]
    manifest=json.loads((ROOT/'datasets/imagenet64/prepared/manifest.json').read_text())
    assert sum(s['count'] for s in manifest['splits']['train'])==1281167
    assert sum(s['count'] for s in manifest['splits']['validation'])==50000
    target_steps = args.target_steps or config['endpoints'][-1]['step']
    if target_steps <= 0:
        p.error('target_steps must be positive')
    stages=[50,60] if args.smoke else [target_steps]
    for target in stages:
        command=base+['--steps',str(target)]
        pointer=out/'resume.json'
        if pointer.exists():
            saved=json.loads(pointer.read_text())
            if saved['step']>=target: continue
            command+=['--resume',saved['checkpoint']]
        if not args.smoke:
            command+=['--checkpoint_steps']+[str(e['step']) for e in config['endpoints']]
            command+=['--eval_campaign',str(campaign)]
        print('Launch:',command,flush=True)
        child=subprocess.Popen(command,cwd=ROOT)
        while child.poll() is None:
            time.sleep(1)
        if stop_requested:
            if not args.smoke:
                subprocess.run(['scontrol','requeue',os.environ['SLURM_JOB_ID']],check=True)
                return
            raise SystemExit('Smoke interrupted')
        if child.returncode:
            raise SystemExit(child.returncode)
    if args.smoke:
        # Validate checkpoint/sampler with an actual resume before marking the smoke successful.
        subprocess.run([str(executable/'python'),'-m','experiments.imagenet64.verify_smoke','--outdir',str(out)],check=True,cwd=ROOT)

if __name__=='__main__': main()
