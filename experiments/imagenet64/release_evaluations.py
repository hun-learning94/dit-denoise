"""Release pre-submitted evaluations only after an atomically saved epoch checkpoint."""
import fcntl
import json
from pathlib import Path
import subprocess

def release(manifest):
    manifest = Path(manifest)
    with manifest.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        campaign = json.loads(manifest.read_text())
        record = manifest.with_name('released_evaluations.json')
        released = json.loads(record.read_text()) if record.exists() else {}
        for run in campaign['runs']:
            if not run.get('evaluation_array'): continue
            for index, endpoint in enumerate(campaign['endpoints']):
                path = Path(run['outdir']) / 'checkpoints' / f"ckpt_step{endpoint['step']}.pt"
                if not path.is_file(): continue
                for metric in range(3):
                    task = f"{run['evaluation_array']}_{index*3+metric}"
                    if task in released: continue
                    # Retain shared real-stat dependency; checkpoint dependency is now satisfied.
                    if campaign.get('stats_job'):
                        dependency = f"afterok:{campaign['stats_job']}"
                    else:
                        if not Path(campaign['real_stats']).is_file():
                            raise FileNotFoundError(campaign['real_stats'])
                        dependency = ''
                    subprocess.run(['scontrol', 'update', f'JobId={task}',
                                    f"Dependency={dependency}"], check=True, timeout=30,
                                   capture_output=True, text=True)
                    released[task] = dict(checkpoint=str(path), epoch=endpoint['epoch'])
                    temporary = record.with_suffix('.tmp')
                    temporary.write_text(json.dumps(released, indent=2)+'\n')
                    temporary.replace(record)
                    print('Released evaluation', task, flush=True)
