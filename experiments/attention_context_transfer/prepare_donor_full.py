"""Freeze donor-only full campaigns; CPU-only native-resolution data preparation."""
from revision.runtime import path as _release_path
from pathlib import Path
import argparse, hashlib, json
import numpy as np
from PIL import Image
from .prepare import atomic_json, sha256
from .protocol import seed_for
from scripts.imagenet_center_crop import center_crop_arr

ROOT = Path(_release_path('.'))
REPORT = ROOT/'reports/2026-11-08/attention_donor_full'
WORK = Path(_release_path('assets/datasets/attention_donor_full_20261108'))
BUDGETS = [9,12,16,24,32,48,64,96,128,192,256]
METHODS = ['kamb_geometry','niedoba_donor','lukoianov_covariance','attention_calibrated','attention_current']

def read(path): return json.loads(Path(path).read_text())

def common(dataset):
    return dict(created='2026-11-08', dataset=dataset, work_root=str(WORK/dataset),
        budgets=BUDGETS, methods=METHODS, evaluation_draws=2, attention='all heads and all layers; donor only',
        targets_per_image=8, input_hashes={}, primary_times=[.1,.5,.9], bootstrap_replicates=2000,
        metric='masked clean target-patch MSE minus the same recipient dense MSE',
        selection='nine nearest-token core, then ranked remaining tokens; nested exact budgets',
        current_selector_scope='Donor sees full noisy input; fixed-index recipient closure only.')

def celeba():
    out=REPORT/'celeba128';out.mkdir(parents=True,exist_ok=True)
    if (out/'design.json').exists(): return
    old=ROOT/'reports/2026-11-07/attention_locality_smlp_calibration'
    prior=read(old/'design.json'); full=read(ROOT/'reports/2026-11-07/attention_noise_context_full/design.json')
    summary=read(old/'calibration_summary.json')
    assert sha256(summary['calibration_fields'])==summary['calibration_fields_sha256']
    d=common('celeba128');d.update(times=[i/10 for i in range(1,10)], noise_scale=.5,
        checkpoints=prior['checkpoints'], reference_checkpoint=prior['reference_checkpoint'],
        evaluation=read(full['evaluation_manifest']), calibration=read(prior['data']['template']),
        calibration_draws=2, models=['mlp','smlp'], donor_for={'mlp':'smlp','smlp':'mlp'},
        reused_fields=summary['calibration_fields'], reused_fields_sha256=summary['calibration_fields_sha256'],
        evaluation_history='Existing 256-image heldout set already inspected in earlier self-MLP depth-group study; not fresh confirmation.',
        shared_initialization='Separate trained predictors share seed-zero common-backbone initialization; not independently initialized.',
        gradient_protocol='1024 observations/target/time; exact original SMLP image-major draw ordering and output offsets')
    for p in [old/'design.json',old/'calibration_summary.json',Path(full['evaluation_manifest']),Path(prior['data']['template'])]: d['input_hashes'][str(p)]=sha256(p)
    ev={r['pixel_sha256'] for r in d['evaluation']};ca={r['pixel_sha256'] for r in d['calibration']}
    assert len(ev)==256 and len(ca)==512 and not ev&ca
    assert np.array_equal(np.bincount(np.array([r['targets'] for r in d['evaluation']]).flatten(),minlength=256),np.full(256,8))
    (WORK/'celeba128').mkdir(parents=True,exist_ok=True)
    atomic_json(out/'design.json',d)
    atomic_json(out/'DATA_READY.json',dict(passed=True,evaluation_images=256,calibration_images=512,targets_balanced=True,pixel_disjoint=True))

def imagenet():
    out=REPORT/'imagenet256';out.mkdir(parents=True,exist_ok=True)
    if (out/'design.json').exists():return
    old=read(ROOT/'reports/2026-11-08/attention_jit_transfer_pilot/design.json')
    reserved=[]
    for p in [ROOT/'reports/2026-11-08/attention_imagenet_score_gallery/images.json',ROOT/'reports/2026-11-08/attention_jit_transfer_pilot/images.json']:
        reserved+=read(p)
    for name in ['pilot','calibration','test']:reserved+=read(ROOT/f'reports/2026-11-06/attention_context_pilot/data/{name}.json')
    seen=set();ids=set()
    for r in reserved:
        ids.add(Path(r['image_id']).stem)
        seen.add(r.get('pixel_sha256') or hashlib.sha256(np.array(Image.open(r['crop']).convert('RGB')).tobytes()).hexdigest())
    source=Path(_release_path('assets/external/imagenet-val-2012'))
    labelpath=source/'validation_synsets.txt'
    assert sha256(labelpath)==read(source/'complete.json')['labels_sha256']
    labels=labelpath.read_text().splitlines();classmap=read(ROOT/'datasets/imagenet_source/classes.json')
    exclude={x['wnid'] for x in old['classes']}
    rng=np.random.default_rng(seed_for('donor_full_class_selection_v1'))
    candidates=rng.permutation(sorted(set(labels)-exclude));classes=[];evaluation=[];calibration=[];rejected=[]
    # Selection depends only on fixed random order, native size, and duplicate checks.
    for wn in candidates:
        selected=[];local_seen=set()
        rr=np.random.default_rng(seed_for('donor_full_images_v1',str(wn)))
        for ii in rr.permutation([i+1 for i,w in enumerate(labels) if w==wn]):
            ident=f'ILSVRC2012_val_{ii:08d}.JPEG';p=source/'images'/ident
            if p.stem in ids:continue
            with Image.open(p) as im:
                native=list(im.size)
                if min(native)<256:continue
                arr=np.asarray(center_crop_arr(im.convert('RGB'),256))
            ph=hashlib.sha256(arr.tobytes()).hexdigest()
            if ph in seen or ph in local_seen:continue
            local_seen.add(ph);selected.append((p,native,arr,ph))
            if len(selected)==40:break
        if len(selected)<40:
            rejected.append(dict(wnid=str(wn),eligible=len(selected)));continue
        info=dict(wnid=str(wn),class_id=int(classmap[wn]),bank=str(WORK/'imagenet256'/f'{wn}_training.npy'))
        classes.append(info);seen.update(local_seen)
        for j,(p,native,arr,ph) in enumerate(selected):
            dest=WORK/'imagenet256'/'crops'/str(wn)/(p.stem+'.png');dest.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(arr).save(dest)
            rec=dict(image_id=p.name,wnid=str(wn),class_id=info['class_id'],original=str(p),original_sha256=sha256(p),native_size=native,crop=str(dest),crop_sha256=sha256(dest),pixel_sha256=ph,role='evaluation' if j<8 else 'calibration')
            (evaluation if j<8 else calibration).append(rec)
        print('selected',len(classes),wn,flush=True)
        if len(classes)==32:break
    assert len(classes)==32
    # Eight independent permutations distributed in eight-token image blocks:
    # each position appears exactly eight times; no within-image duplication.
    rng=np.random.default_rng(seed_for('donor_full_target_positions_v1'))
    targets=np.concatenate([rng.permutation(256).reshape(32,8) for _ in range(8)])
    targets=targets[rng.permutation(256)]
    for rec,q in zip(evaluation,targets):rec['targets']=[int(x) for x in q]
    for info in classes:info['queries']=sorted({q for r in evaluation if r['wnid']==info['wnid'] for q in r['targets']})
    d=common('imagenet256');d.update(times=[.1,.3,.5,.7,.9],noise_scale=1.,classes=classes,
        evaluation=evaluation,calibration=calibration,calibration_draws=4,models=['jit_b','jit_l'],donor_for={'jit_b':'jit_l','jit_l':'jit_b'},
        checkpoints={k:old['checkpoints'][k] for k in ['jit_b','jit_l']},jit_upstream=old['jit_upstream'],jit_ema=old['jit_ema'],
        class_selection='32 fixed-random-order eligible new classes, >=40 unique native-256 validation crops; no score/loss selection',
        gradient_protocol='128 observations/target/time, same donor and same calibration corruptions as attention; 2x2-stratified output pixels in each half',
        covariance_protocol='512 same-class training crops, all empirical modes; native-coordinate row-relative Wiener maximum then token maximum',
        evaluation_history='Fresh classes and images relative to earlier pilot/gallery; dataset-level architecture choice informed by pilot.')
    d['input_hashes']={p:h for p,h in old['input_hashes'].items() if '/upstream/jit/' in p}
    d['input_hashes'][str(labelpath)]=sha256(labelpath)
    atomic_json(out/'design.json',d)
    atomic_json(out/'DATA_READY.json',dict(passed=True,evaluation_images=256,calibration_images=1024,classes=32,excluded_prior_classes=sorted(exclude),rejected_classes=rejected,no_upscaling=True,unique_crops=len(evaluation+calibration),training_banks_pending=True))

def bank(index):
    out=REPORT/'imagenet256';d=read(out/'design.json');info=d['classes'][index];wn=info['wnid'];receipt=out/'banks'/f'{wn}.json'
    if receipt.exists():
        r=read(receipt);assert sha256(r['path'])==r['sha256'];return
    forbidden={r['pixel_sha256'] for r in d['evaluation']+d['calibration']};seen=set(forbidden);rows=[];records=[];excluded=0
    paths=sorted((Path(_release_path('assets/external/train'))/wn).iterdir());rng=np.random.default_rng(seed_for('donor_full_covariance_bank_v1',wn))
    for ix in rng.permutation(len(paths)):
        p=paths[ix]
        try:
            with Image.open(p) as im:
                native=list(im.size)
                if min(native)<256:excluded+=1;continue
                arr=np.asarray(center_crop_arr(im.convert('RGB'),256))
        except (OSError,ValueError):excluded+=1;continue
        ph=hashlib.sha256(arr.tobytes()).hexdigest()
        if ph in seen:excluded+=1;continue
        seen.add(ph);rows.append(arr);records.append(dict(original=str(p),original_sha256=sha256(p),native_size=native,pixel_sha256=ph))
        if len(rows)==512:break
    assert len(rows)==512
    p=Path(info['bank']);temp=p.with_suffix('.tmp')
    with temp.open('wb') as f:np.save(f,np.stack(rows))
    temp.replace(p);atomic_json(out/'banks'/f'{wn}_manifest.json',records)
    atomic_json(receipt,dict(passed=True,path=str(p),sha256=sha256(p),images=512,excluded=excluded,no_upscaling=True,design_sha256=sha256(out/'design.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bank',type=int);a=p.parse_args()
    if a.bank is None:
        celeba();imagenet()
        link=ROOT/'datasets/attention_donor_full_20261108'
        if not link.exists():link.symlink_to(WORK,target_is_directory=True)
    else:bank(a.bank)
