#!/usr/bin/env python3
"""Train/evaluate a frozen paired JiT-based scale run, with no pilot gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--phase', choices=('train', 'evaluate'), required=True)
    cli = parser.parse_args()
    manifest = json.loads(cli.manifest.read_text())
    assert Path(manifest['code']).resolve() == CODE
    row = next(r for r in manifest['runs'] if r['run_id'] == cli.run_id)
    for name, expected in manifest['source_sha256'].items():
        assert digest(CODE/name) == expected, name
    reference = manifest['evaluation_references'][row['dataset']]
    assert digest(reference['path']) == reference['sha256']
    assert digest(CODE/'fused_grouped_gelu_wide.py') == manifest['kernel_sha256']
    import torch
    from scripts import train_complementary as original
    from experiments.complementary_training.jit_campaign import JiTTraining
    assert Path(original.__file__).resolve().is_relative_to(CODE)
    assert torch.cuda.is_available(), "CUDA GPU required"
    assert torch.cuda.is_bf16_supported()
    torch.ones(1, device='cuda').sum().item()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch._dynamo.config.cache_size_limit = 64
    print('JIT CAMPAIGN', json.dumps(dict(run_id=row['run_id'], phase=cli.phase,
        recipe=manifest['recipe'], physical_batch=row['batch'], start='continue online checkpoint with optimizer/EMA/RNG/data position',
        source_snapshot=str(CODE), job_id=os.environ.get('SLURM_JOB_ID'),
        numerical_policy=manifest['numerical_policy'])), flush=True)
    if cli.phase == 'evaluate':
        done = json.loads((Path(row['outdir'])/'DONE.json').read_text())
        assert done['epochs'] == 400 and done['step'] == row['total_updates']
        from scripts.evaluate_complementary import main as evaluate
        sys.argv = [sys.argv[0], '--run-dir', row['outdir'], '--reference', reference['path'],
            '--outdir', row['evaluation_outdir']]
        evaluate()
        return 0
    original.Training = JiTTraining
    opts = argparse.Namespace(dataset=row['dataset'], arch=row['arch'], outdir=row['outdir'],
        seed=row['seed'], epochs=row['epochs'], batch=row['batch'], expected_batch=1024,
        lr=manifest['recipe']['lr'], workers=8, cache_workers=16,
        hflip=True, diagnostic_every=10*row['steps_per_epoch'],
        checkpoint_every=5*row['steps_per_epoch'],
        projection_fix=manifest['projection_fix'] if row['arch']=='smlp' else None,
        resume_checkpoint=None, allow_epoch_extension=True)
    return original.run(opts) or 0


if __name__ == '__main__':
    raise SystemExit(main())
