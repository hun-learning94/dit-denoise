"""Provisional near-200 STL-to-CelebA extension; reuse the pinned feature protocol."""
import argparse, fcntl, json, subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from . import pipeline as p
from .core import CODES, density_per_image, unit
PARENT=p.ROOT/'reports/experiment3/primary_metrics_1024'
BASE=PARENT/'stl_to_celeba_near200'
OUT=BASE/'evaluation'

def analyze(config, cache, row, manifest):
    index=json.loads((cache/'features/index.json').read_text())
    cube=json.loads((cache/'cubes'/f"{row['id']}.json").read_text())
    def features(path):
        with np.load(path) as z: return z['features']
    fs={c:features(index['generated'][cube['images'][c]]) for c in CODES}
    ref=features(index['real'][row['target']]['reference'])
    assert ref.shape==(1000,768)
    density=np.stack([density_per_image(fs[c],ref,50) for c in CODES])
    distances=np.stack([np.sum((unit(fs[c])-unit(fs['111']))**2,axis=1) for c in CODES])
    assert density.shape==distances.shape==(8,1024)
    assert np.isfinite(density).all() and np.isfinite(distances).all()
    p.save_npz(cache/'measurements'/f"{row['id']}.npz",densities=density,distances=distances,codes=np.array(CODES),latent_hash=cube['latent_hash'],protocol_hash=p.signature(config))
    rng=np.random.default_rng(config['latent_seed']);dd=[];ee=[]
    for begin in range(0,config['bootstrap_repeats'],128):
        idx=rng.integers(0,1024,(min(128,config['bootstrap_repeats']-begin),1024))
        dd.append(density[:,idx].mean(axis=-1).T);ee.append(distances[:,idx].mean(axis=-1).T)
    dd=np.concatenate(dd);ee=np.concatenate(ee)
    dm=density.mean(axis=1);em=distances.mean(axis=1)
    j=list(CODES).index('111');s=list(CODES).index('000')
    dv=dd[:,j]>=.01;ev=ee[:,s]>=config['min_endpoint_distance']
    metrics={'D_raw':(dm,dd,np.ones(len(dd),dtype=bool)), 'E':(em,ee,np.ones(len(ee),dtype=bool))}
    if dm[j]>=.01 and dv.mean()>=.95: metrics['D_rel']=(dm/dm[j],dd[dv]/dd[dv,j,None],dv)
    if em[s]>=config['min_endpoint_distance'] and ev.mean()>=.95: metrics['P']=(1-em/em[s],1-ee[ev]/ee[ev,s,None],ev)
    summary=dict(metrics={},contrasts={},intervals=dict(metrics={},contrasts={}))
    for metric,(means,draws,valid) in metrics.items():
        summary['metrics'][metric]={c:float(means[i]) for i,c in enumerate(CODES)}
        summary['intervals']['metrics'][metric]={c:dict(ci95=np.quantile(draws[:,i],[.025,.975]).tolist(),valid_fraction=float(valid.mean())) for i,c in enumerate(CODES)}
        summary['contrasts'][metric]={};summary['intervals']['contrasts'][metric]={}
        for name,weights in manifest['contrasts'].items():
            w=np.array([weights.get(c,0) for c in CODES])
            summary['contrasts'][metric][name]=float(means@w)
            summary['intervals']['contrasts'][metric][name]=dict(ci95=np.quantile(draws@w,[.025,.975]).tolist(),valid_fraction=float(valid.mean()))
    p.write_json(cache/'results'/f"{row['id']}.json",dict(row=row,summary=summary,density_k=50,reference_n=1000,provisional=True,normalization_available={m:m in metrics for m in ['P','D_rel']}))

def run(task):
    import torch
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
    torch.set_num_threads(1)
    manifest = json.loads((OUT / 'manifest.json').read_text())
    config = manifest['protocol']
    assert p.digest(BASE / 'manifest.json') == manifest['generation_manifest_sha256']
    assert p.implementation_stamp() == manifest['implementation']
    assert p.digest(__file__) == manifest['runner_sha256']
    assert p.digest(OUT / 'real_manifest.json') == manifest['real_manifest_sha256']
    repo = config['feature_repo']
    assert subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip() == config['feature_repo_commit']
    assert not subprocess.check_output(['git','-C',repo,'status','--porcelain'],text=True).strip()
    assert p.digest(config['feature_weights']) == config['feature_weights_sha256']
    row = manifest['rows'][task]
    cube = json.loads((BASE / 'cache/cubes' / (row['id'] + '.json')).read_text())
    assert cube['row'] == row and cube['protocol_hash'] == manifest['generation_protocol_hash']
    model = torch.hub.load(repo, config['feature_model'], source='local', pretrained=False)
    model.load_state_dict(torch.load(config['feature_weights'], map_location='cpu', weights_only=True), strict=True)
    model.eval().requires_grad_(False).cuda()
    transform = Compose([Resize((224,224), interpolation=InterpolationMode.BICUBIC, antialias=True),
                         ToTensor(), Normalize([.485,.456,.406],[.229,.224,.225])])
    feature_protocol = dict(model=config['feature_model'], commit=config['feature_repo_commit'],
        weights=config['feature_weights_sha256'], preprocess='RGB-full224-bicubic-ImageNet-L2-CLS-fp32',
        runner=manifest['feature_runner_sha256'], packages=manifest['implementation']['packages'])
    def compute(key, count, load):
        dest = PARENT / 'evaluation/shared_features' / (p.signature([key, feature_protocol]) + '.npz')
        dest.parent.mkdir(exist_ok=True)
        with dest.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if dest.exists():
                with np.load(dest) as a:
                    assert a['features'].shape == (count, 768) and np.isfinite(a['features']).all()
                return str(dest)
            features = []
            with torch.inference_mode():
                for start in range(0, count, config['feature_batch']):
                    x = torch.stack([transform(load(i)) for i in range(start, min(count, start + config['feature_batch']))]).cuda()
                    features.append(model(x).float().cpu().numpy())
            p.save_npz(dest, features=p.unit(np.concatenate(features)).astype('float32'), encoder_sha256=config['feature_weights_sha256'])
            print('Cached', dest.name, count, flush=True)
        return str(dest)
    real = json.loads((OUT / 'real_manifest.json').read_text())
    index = dict(real={}, generated={}, protocol_hash=p.signature(config))
    for domain in [row['source'], row['target']]:
        index['real'][domain] = {}
        for role, items in real['domains'][domain].items():
            def load(i):
                item = items[i]
                assert p.digest(item['path']) == item['sha256']
                with Image.open(item['path']) as im:
                    return im.convert('RGB').copy()
            index['real'][domain][role] = compute(items, len(items), load)
    for code, path in cube['images'].items():
        with np.load(path) as a:
            images = a['images']
            assert len(images) == config['images'] and str(a['latent_hash']) == cube['latent_hash']
            assert str(a['checkpoint_sha256']) == row['checkpoints'][code]['sha256']
        index['generated'][path] = compute(p.digest(path), len(images), lambda i: Image.fromarray(images[i].transpose(1,2,0)))
    del model
    torch.cuda.empty_cache()
    cache = OUT / 'tasks' / str(task)
    cube['generation_protocol_hash'] = cube['protocol_hash']
    cube['protocol_hash'] = p.signature(config)
    p.write_json(cache / 'cubes' / (row['id'] + '.json'), cube)
    p.write_json(cache / 'features/index.json', index)
    analyze(config, cache, row, manifest)
    print('COMPLETE', row['id'], flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',type=int,required=True);args=parser.parse_args()
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert p.digest(__file__)==manifest['runner_sha256']
    gen=json.loads((BASE/'manifest.json').read_text())
    assert gen['implementation']==p.implementation_stamp()
    p.generate(gen['protocol'],gen['rows'][args.task],BASE/'cache')
    run(args.task)
