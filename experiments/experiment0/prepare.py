"""Prepare an explicit, read-only checkpoint inventory and task manifest. Never submit."""
import argparse
import json
from pathlib import Path
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {'celeba64':5086, 'ffhq64':1875, 'aahq64':480,
            'stl10_64':3265, 'afhq_v2_64':447, 'icartoonface64':12147}


def config(stage):
    main = stage == 'main'
    return dict(stage=stage, resolution=64, patch=4, epoch=400,
                num_images=256 if main else 16, probes=4 if main else 2,
                sigmas=[.02,.05,.1,.2,.4,.75,1,2,4,8,16,49] if main else [.05,1,16],
                targets=[0,8,128,255,68,75,180,136] if main else [0,136],
                radii=[0,1,2,4,8,15] if main else [0,2,4,8,15],
                cs=[.99,.9], refresh_draws=4 if main else 2,
                qs=[.75,.9,.95], floor=1e-12, bootstrap=1000 if main else 200,
                image_seed=20260910, noise_seed=104729, probe_seed=130363,
                refresh_seed=155921, bootstrap_seed=196613, shard_images=16)


def prepare(stage):
    cfg = config(stage)
    models, tasks = [], []
    for dataset, spe in DATASETS.items():
        # ImageFolderFlat sorts these same extensions recursively.
        data = ROOT/'datasets'/dataset/'test'
        for scale in ['10m','20m']:
            run = ROOT/'runs'/f'experiment3_{scale}'/dataset/'seed0'/'scratch'
            if dataset == 'celeba64' and scale == '10m':
                run = ROOT/'runs/celeba64_dit_p4_xpred_vp_vloss_f1024_matched_b32'
            candidates = [p for p in (run/'checkpoints').glob('ckpt_step*.pt')
                          if (m := re.search(r'step(\d+)(?:_final)?\.pt$', p.name))
                          and int(m[1]) == 400*spe]
            checkpoint = sorted(candidates)[-1] if candidates else None
            model = dict(label=f'{dataset}_{scale}', dataset=dataset, scale=scale,
                         run=str(run.relative_to(ROOT)), data_root=str(data.relative_to(ROOT)),
                         checkpoint=str(checkpoint.relative_to(ROOT)) if checkpoint else None,
                         expected_step=400*spe, status='ready' if checkpoint and data.is_dir() else 'missing')
            models.append(model)
            if model['status'] != 'ready':
                continue
            for si in range(len(cfg['sigmas'])):
                for start in range(0, cfg['num_images'], cfg['shard_images']):
                    tasks.append(dict(model=model['label'], sigma_index=si,
                                      start=start, stop=min(start+cfg['shard_images'],cfg['num_images'])))
    result = dict(schema=1, config=cfg, models=models, tasks=tasks)
    path = ROOT/'reports/experiment0'/f'{stage}_manifest.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(f'{path}: {len(tasks)} tasks; no jobs submitted')
    for model in models:
        print(model['label'], model['status'])
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--stage', choices=['pilot','main'],default='pilot')
    prepare(p.parse_args().stage)
