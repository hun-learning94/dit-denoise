"""Run missing trajectory metrics sequentially with a lock per checkpoint."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
from run_snapshot_sequential import commands

def main():
    task=json.loads(Path(sys.argv[1]).read_text())['rows'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
    subprocess.run([sys.executable, '-c', "import torch; assert torch.cuda.is_available(), 'CUDA unavailable; refusing CPU fallback'; torch.ones(1,device='cuda').sum().item(); torch.cuda.synchronize(); print('CUDA:',torch.cuda.get_device_name())"], check=True)
    checkpoint=Path(task['checkpoint'])
    before=checkpoint.stat()
    if task.get('checkpoint_bytes') is not None:assert before.st_size==task['checkpoint_bytes']
    with zipfile.ZipFile(checkpoint) as archive:assert any(n.endswith('/data.pkl') for n in archive.namelist())
    after=checkpoint.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    lock=Path(task['output_root'])/'.snapshot_locks'/(task['label']+'.lock')
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        precision=task['fid_precision']
        if precision=='hardware':
            precision=subprocess.check_output([sys.executable,'-c','import torch; assert torch.cuda.is_available(); print("fp32" if "2080 ti" in torch.cuda.get_device_name(0).lower() else "fp16")'],text=True).strip()
        print('CHECKPOINT',json.dumps(task),flush=True)
        for result,command in commands(task,precision):
            if result.is_file():
                json.loads(result.read_text());print('Already complete',result,flush=True);continue
            # Preserve the existing restricted evaluation batches and precision.
            if 'fid50k' not in str(result) and task['series'] in ['qk_weights','attn_weights']:
                command[command.index('--batch')+1]='64'
            print('Starting',str(result),flush=True)
            subprocess.run(command,check=True)
            json.loads(result.read_text());print('Finished',str(result),flush=True)
        print('All three metrics complete',flush=True)

if __name__=='__main__':main()
