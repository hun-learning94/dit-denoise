"""Prepare all-patch evaluations on min(test size, 1024) held-out images."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from diagnose_dpd import build_folder_dataset
from experiments.experiment0.run import digest


def main():
    report=Path('reports/experiment0')
    base=json.loads((report/'main_manifest.json').read_text())
    models=[dict(m,epoch=400) for m in base['models'] if m['label'].endswith('_20m') and m['dataset']!='stl10_64' and m['status']=='ready']
    for name,epoch in [('icartoon20m',400),('stl10_800',800),('imagenet600',600)]:
        models.extend(dict(s,epoch=epoch) for s in json.loads((report/f'{name}_manifest.json').read_text())['models'])
    assert len(models)==7
    old=json.loads((report/'imagenet600_validation_subset.json').read_text())
    source=Path(old['source']);assert digest(source)==old['source_sha256']
    pixels=np.load(source,mmap_mode='r');assert pixels.shape==(50000,12288) and pixels.dtype==np.uint8
    ids=np.random.default_rng(base['config']['image_seed']).permutation(len(pixels))[:1024]
    output=Path('datasets/imagenet64/experiment0_val1024_seed20260910');output.mkdir(parents=True,exist_ok=True)
    records=[]
    for idx in sorted(ids):
        arr=pixels[idx].reshape(3,64,64).transpose(1,2,0);dest=output/f'val_{int(idx):05d}.png'
        if not dest.exists():Image.fromarray(arr).save(dest)
        np.testing.assert_array_equal(np.asarray(Image.open(dest)),arr)
        records.append(dict(source_index=int(idx),path=str(dest),sha256=digest(dest)))
    assert len(list(output.glob('*.png')))==1024
    (report/'allpatch_imagenet_validation_subset.json').write_text(json.dumps(dict(source=str(source),source_sha256=old['source_sha256'],source_count=len(pixels),sample_seed=base['config']['image_seed'],images=records),indent=2)+'\n')
    cfg={k:v for k,v in base['config'].items() if k not in ['epoch','num_images','refresh_seed','refresh_draws','cs','radii','bootstrap','bootstrap_seed']}
    cfg.update(stage='allpatch1024',targets=list(range(256)),max_images=1024,shard_images=16)
    tasks=[]
    for spec in models:
        assert Path(spec['checkpoint']).is_file()
        if spec['dataset']=='imagenet64':spec['data_root']=str(output)
        size=len(build_folder_dataset(spec['data_root'],64,3))
        spec['test_set_size']=50000 if spec['dataset']=='imagenet64' else size
        spec['num_images']=min(spec['test_set_size'],cfg['max_images'])
        assert size>=spec['num_images']
        tasks.extend(dict(model=spec['label'],sigma_index=si,start=i,stop=min(i+16,spec['num_images'])) for si in range(len(cfg['sigmas'])) for i in range(0,spec['num_images'],16))
    (report/'allpatch_manifest.json').write_text(json.dumps(dict(schema=1,config=cfg,models=models,tasks=tasks),indent=2)+'\n')
    print(json.dumps(dict(models=[{k:s[k] for k in ['label','epoch','test_set_size','num_images']} for s in models],tasks=len(tasks)),indent=2))

if __name__=='__main__':main()
