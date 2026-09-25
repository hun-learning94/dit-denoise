"""Evaluate one immutable manifest checkpoint, sequentially, in one allocation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys


def commands(task, precision):
    root = Path(task['output_root'])
    common = ['--checkpoint', task['checkpoint'], '--label', task['label'],
              '--data_root', task['data_root'], '--batch', '32', '--num_workers', '0',
              '--resolution', '64', '--channels', '3']
    for metric, script, filename, extra in [
        ('denoising', 'eval_denoising_paired.py', 'summary.json',
         ['--precision', 'fp32', '--time_min', '.05', '--time_max', '4', '--auc_steps', '32']),
        ('vp_velocity_loss', 'eval_vp_velocity_loss.py', 'summary.json', ['--precision', 'fp32']),
        ('fid50k', 'eval_fid50k_resumable.py', 'result.json',
         ['--precision', precision, '--real_stats', task['real_stats'], '--num_images', '50000',
          '--shard_size', '500', '--sampler_steps', '50']),
    ]:
        command = [sys.executable, '-u', script]
        if metric == 'fid50k':
            command.append('model')
        yield root / metric / task['label'] / filename, command + common + [
            '--outdir', str(root / metric)] + extra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    tasks = json.loads(Path(args.manifest).read_text())
    if args.validate_only:
        seen = set()
        for task in tasks:
            for key in ('checkpoint', 'real_stats'):
                path = Path(task[key])
                assert path.is_file() and path.stat().st_size, path
            assert Path(task['data_root']).is_dir()
            for result, command in commands(task, 'fp32'):
                assert str(result) not in seen, result
                seen.add(str(result))
        print(f'Validated {len(tasks)} checkpoints and {len(seen)} distinct output paths')
        return
    task = tasks[int(os.environ['SLURM_ARRAY_TASK_ID'])]
    checkpoint = Path(task['checkpoint'])
    assert checkpoint.stat().st_size == task['checkpoint_bytes'], checkpoint
    lock = Path(task['output_root']) / '.snapshot_locks' / (task['label'] + '.lock')
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Separate process releases CUDA memory before the metric subprocesses.
        precision = subprocess.check_output([sys.executable, '-c',
            'import torch; assert torch.cuda.is_available(), "GPU unavailable"; '
            'x=torch.ones(1,device="cuda"); torch.cuda.synchronize(); '
            'print("fp32" if "2080 ti" in torch.cuda.get_device_name(0).lower() else "fp16")'
        ], text=True).strip()
        print('Checkpoint:', json.dumps(task), flush=True)
        for result, command in commands(task, precision):
            if result.is_file():
                json.loads(result.read_text())
                print('Already complete:', result, flush=True)
                continue
            print('Starting:', ' '.join(command), flush=True)
            subprocess.run(command, check=True)
            json.loads(result.read_text())
            print('Finished:', result, flush=True)


if __name__ == '__main__':
    main()
