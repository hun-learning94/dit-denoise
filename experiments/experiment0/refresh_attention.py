"""Stream saved attention shards to refresh 20M summaries and matched plots."""
import json
from pathlib import Path
import numpy as np
from experiments.experiment0.core import coverage,distances
from experiments.experiment0.plot import attention_figures,matched_figures


def main():
    manifest=json.loads(Path('reports/experiment0/main_manifest.json').read_text())
    cfg=manifest['config']; out=Path('reports/experiment0/figures/main')
    d=distances(16,cfg['targets'])
    for model in manifest['models']:
        label=model['label']
        if model['status']!='ready' or not label.endswith('_20m'): continue
        maps=[]; samples={k:[] for k in ['images','indices','influence','attention']}
        reference_ids=None; reference_meta=None
        for si in range(len(cfg['sigmas'])):
            total=None; ids=[]
            tasks=[t for t in manifest['tasks'] if t['model']==label and t['sigma_index']==si]
            for task in tasks:
                path=Path('evals/experiment0/main')/label/f"sigma{si:02d}_images{task['start']:04d}.npz"
                with np.load(path) as z:
                    meta=json.loads(str(z['metadata']))
                    if meta['config']!=cfg or meta['task']!=task or meta['model']!=model:
                        raise ValueError(f'Manifest mismatch: {path}')
                    if reference_meta is None: reference_meta=meta
                    if any(meta[k]!=reference_meta[k] for k in ['checkpoint_sha256','source_sha256']):
                        raise ValueError(f'Mixed evaluation provenance: {path}')
                    if float(z['sigma'])!=cfg['sigmas'][si]: raise ValueError('Noise mismatch')
                    attn=z['attention']
                    if len(attn)!=task['stop']-task['start'] or not np.isfinite(attn).all():
                        raise ValueError(f'Invalid shard {path}')
                    np.testing.assert_allclose(attn.sum(-1),1,atol=2e-6)
                    chunk=attn.sum(0,dtype=np.float64)
                    total=chunk if total is None else total+chunk
                    ids.extend(z['indices'].tolist())
                    if task['start']==0:
                        for k in samples: samples[k].append(z[k][:1])
            if len(ids)!=cfg['num_images'] or len(set(ids))!=len(ids):
                raise ValueError('Incomplete or duplicated image coverage')
            if reference_ids is None: reference_ids=ids
            if ids!=reference_ids: raise ValueError('Unpaired images across noise levels')
            maps.append(total/len(ids))
        maps=np.array(maps)
        layer=coverage(maps,d,cfg['qs'],cfg['floor'])
        pooled=coverage(maps.mean(1),d,cfg['qs'],cfg['floor'])
        path=out/label/'summary.npz'
        with np.load(path) as z: result=dict(z)
        result.update(attention_radius=layer['radius'],attention_count=layer['count'],
                      mean_attention_radius=pooled['radius'],mean_attention_count=pooled['count'])
        np.savez_compressed(path,**result)
        attention_figures(result,cfg,label,out)
        matched_figures({k:np.array(v) for k,v in samples.items()},cfg,label,out)
        print('Refreshed',label,flush=True)


if __name__=='__main__': main()
