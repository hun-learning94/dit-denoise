"""CPU two-rank integration: tiny DiT, data boundaries, EMA/checkpoint, exact resume."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import shutil
import tempfile

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', type=Path)
    args = parser.parse_args()
    root = args.workdir or Path(tempfile.mkdtemp(prefix='imagenet64-ddp-smoke-'))
    root.mkdir(parents=True, exist_ok=True)
    data = root / 'data'; data.mkdir(exist_ok=True)
    a = np.random.default_rng(7).integers(0, 256, (24, 12288), dtype=np.uint8)
    np.save(data / 'train.npy', a)
    (data / 'manifest.json').write_text(json.dumps({'splits': {'train': [{'path': 'train.npy', 'count': len(a)}]}}))
    torchrun = shutil.which('torchrun') or 'torchrun'
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    launch = [torchrun, '--standalone', '--nnodes=1', '--nproc_per_node=2', '-m']
    subprocess.run(launch + ['experiments.imagenet64.check_distributed'], env=env, check=True)
    common = launch + ['experiments.imagenet64.train', '--device', 'cpu', '--data_root', str(data),
        '--dim', '16', '--depth', '1', '--heads', '2', '--patch', '16',
        '--batch', '2', '--accumulation', '2', '--num_workers', '0',
        '--ckpt_every', '1', '--sample_every', '2', '--sample_n', '2', '--sample_steps', '2', '--log_every', '1']
    # Four updates cross a data-pass boundary (six microbatches per rank per pass).
    subprocess.run(common + ['--outdir', str(root/'full'), '--steps', '4'], env=env, check=True)
    subprocess.run(common + ['--outdir', str(root/'resume'), '--steps', '2'], env=env, check=True)
    subprocess.run(common + ['--outdir', str(root/'resume'), '--steps', '4', '--resume',
        str(root/'resume/checkpoints/ckpt_step2.pt')], env=env, check=True)
    full = torch.load(root/'full/checkpoints/ckpt_step4.pt', map_location='cpu', weights_only=False)
    resumed = torch.load(root/'resume/checkpoints/ckpt_step4.pt', map_location='cpu', weights_only=False)
    for group in ['net', 'ema']:
        for k in full[group]:
            torch.testing.assert_close(full[group][k], resumed[group][k], rtol=0, atol=0)
    assert resumed['step'] == 4 and resumed['images_seen'] == 32
    assert resumed['distributed_protocol']['effective_batch'] == 8
    assert not any(k.startswith('module.') for k in resumed['net'])
    assert (root/'resume/samples/sample_step4.png').is_file()
    print('PASS: two-rank VP DiT training, sampling, atomic checkpoints and bit-exact resume across a data-pass boundary.', root)


if __name__ == '__main__': main()
