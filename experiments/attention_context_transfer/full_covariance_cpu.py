"""Full eligible training-set Wiener estimates; CPU-only, versioned artifacts.

ImageNet uses exact sample-space algebra within each class. CelebA accumulates
the full pixel covariance in batches. No new rank cutoff or shrinkage is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gc
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image
from scipy import linalg
from scripts.imagenet_center_crop import center_crop_arr


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(path)


def save_npy(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.save(f, value)
    tmp.replace(path)


def save_npz(path, **values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **values)
    tmp.replace(path)


def stamp(root, receipt_path, **values):
    write_json(receipt_path, dict(design_sha256=digest(root/'design.json'),
        source_sha256=digest(Path(__file__)),
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
        job_id=os.environ.get('SLURM_JOB_ID'), cpu_only=True, **values))


def normalized(pixels):
    return np.ascontiguousarray(pixels.transpose(0, 3, 1, 2)).reshape(len(pixels), -1).astype(np.float64)/127.5-1


def sample_basis(x):
    x = x - x.mean(0); n = len(x)
    lam, v = linalg.eigh(x @ x.T/n, driver='evr', check_finite=False)
    assert lam[0] >= -max(1., lam[-1])*1e-10
    keep = lam > lam[-1]*1e-12
    lam = lam[keep]; u = (x.T @ v[:, keep])/np.sqrt(n*lam)[None]
    trace_error = abs(lam.sum()-np.sum(x*x)/n)/(np.sum(x*x)/n)
    jj = np.linspace(0, len(lam)-1, min(32, len(lam)), dtype=int)
    ortho = np.max(np.abs(u[:, jj].T @ u[:, jj]-np.eye(len(jj))))
    assert trace_error < 1e-10 and ortho < 1e-6, (trace_error, ortho)
    return u, lam, dict(trace_relative_error=float(trace_error), orthogonality_max_abs=float(ortho))


def patch_rows(q, resolution):
    p = resolution//16; rr, cc = divmod(int(q), 16)
    spatial = (np.arange(rr*p, (rr+1)*p)[:, None]*resolution + np.arange(cc*p, (cc+1)*p)).ravel()
    return (np.arange(3)[:, None]*resolution**2 + spatial).ravel()


def field(u, lam, t, q, resolution, noise_scale):
    gain = t*lam/(t*t*lam+(noise_scale*(1-t))**2)
    rows = patch_rows(q, resolution); score = np.zeros((resolution, resolution), np.float64)
    for start in range(0, len(rows), 32):
        w = (u[rows[start:start+32]]*gain) @ u.T
        np.abs(w, out=w); w /= w.max(1, keepdims=True)
        score = np.maximum(score, w.reshape(-1, 3, resolution, resolution).max((0, 1)))
    return score.astype(np.float32)


def token_max(score):
    r = score.shape[-1]; p = r//16
    return score.reshape(16, p, 16, p).max((1, 3)).ravel()


def add_scatter(c, x):
    # x.T is Fortran-contiguous; dsyrk writes only the lower triangle in place.
    assert c.flags.f_contiguous and x.flags.c_contiguous
    out = linalg.blas.dsyrk(1., x.T, beta=1., c=c, lower=1, overwrite_c=1)
    assert np.shares_memory(c, out), 'BLAS unexpectedly copied the dense covariance'


def symmetric(lower):
    return np.tril(lower)+np.tril(lower, -1).T


def self_check():
    rng = np.random.default_rng(61)
    x = rng.normal(size=(19, 48)); x -= x.mean(0)
    c = np.zeros((48, 48), order='F')
    for block in np.array_split(x, 4):
        add_scatter(c, np.ascontiguousarray(block))
    c = symmetric(c/len(x)); direct = x.T @ x/len(x)
    err = float(np.max(np.abs(c-direct))); assert err < 1e-12
    u, lam, _ = sample_basis(x)
    errors = []
    for noise_scale in [.5, 1.]:
        for t in [.1, .5, .9]:
            w = t*c @ np.linalg.inv(t*t*c+(noise_scale*(1-t))**2*np.eye(48))
            got = (u*(t*lam/(t*t*lam+(noise_scale*(1-t))**2))) @ u.T
            errors.append(float(np.max(np.abs(w-got))))
    assert max(errors) < 1e-10
    # Verify native-coordinate pooling and CHW target-row identities.
    probe = np.arange(3*256*256).reshape(3, 256, 256)
    assert np.array_equal(probe[:, 80:96, 80:96].ravel(), patch_rows(85, 256))
    # End-to-end patch readout against a directly materialized Wiener matrix.
    xx = rng.normal(size=(23, 3*16*16)); xx -= xx.mean(0)
    uu, ll, _ = sample_basis(xx); t = .5
    ww = (uu*(t*ll/(t*t*ll+(.5*(1-t))**2))) @ uu.T
    rows = np.abs(ww[patch_rows(85, 16)]); rows /= rows.max(1, keepdims=True)
    expected = rows.reshape(3, 3, 16, 16).max((0, 1))
    field_error = float(np.max(np.abs(field(uu, ll, t, 85, 16, .5)-expected)))
    assert field_error < 1e-6 and np.array_equal(token_max(expected), expected.ravel())
    return dict(passed=True, scatter_max_abs=err, wiener_max_abs=max(errors), field_max_abs=field_error)


def load_crop(item, resolution):
    path = Path(item['path'])
    try:
        data = path.read_bytes()
        with Image.open(io.BytesIO(data)) as im:
            size = list(im.size)
            if min(size) < resolution:
                return dict(**item, reason='native_below_resolution', native_size=size), None
            if resolution == 128:
                assert size == [128, 128], (path, size)
                arr = np.asarray(im.convert('RGB')).copy()
            else:
                arr = np.asarray(center_crop_arr(im.convert('RGB'), resolution)).copy()
        return dict(**item, native_size=size, original_sha256=hashlib.sha256(data).hexdigest(),
            pixel_sha256=hashlib.sha256(arr.tobytes()).hexdigest()), arr
    except (OSError, ValueError) as e:
        return dict(**item, reason='decode_failure', error=str(e)), None


def prepare_bank(root, key, items, resolution, forbidden):
    d = read(root/'design.json'); work = Path(d['work_root'])/key
    ready = root/key/'BANK.json'
    if ready.exists():
        r = read(ready); assert r['design_sha256'] == digest(root/'design.json')
        assert digest(r['path']) == r['sha256']; return r
    work.mkdir(parents=True, exist_ok=True)
    path = work/'training.npy'; temporary = work/'training.tmp.npy'
    pixels = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.uint8,
        shape=(len(items), resolution, resolution, 3))
    manifest = work/'training_manifest.jsonl.gz'; excluded = work/'excluded.jsonl.gz'
    seen = set(forbidden); counts = Counter(); n = 0; sums = np.zeros((resolution, resolution, 3), np.uint64)
    started = time.monotonic()
    with gzip.open(manifest.with_suffix('.tmp'), 'wt') as mf, gzip.open(excluded.with_suffix('.tmp'), 'wt') as ef:
        with ThreadPoolExecutor(min(16, int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))) as pool:
            for start in range(0, len(items), 256):
                for rec, arr in pool.map(lambda item: load_crop(item, resolution), items[start:start+256]):
                    reason = rec.get('reason')
                    if reason is None and rec['pixel_sha256'] in seen:
                        reason = 'duplicate_calibration_evaluation' if rec['pixel_sha256'] in forbidden else 'duplicate_training'
                    if reason:
                        counts[reason] += 1; ef.write(json.dumps(dict(rec, reason=reason))+'\n'); continue
                    seen.add(rec['pixel_sha256']); pixels[n] = arr; sums += arr; n += 1
                    mf.write(json.dumps(dict(rec, bank_index=n-1))+'\n')
                if start % 8192 == 0:
                    pixels.flush(); print(key, 'scanned', start, 'accepted', n, 'seconds', time.monotonic()-started, flush=True)
    pixels.flush(); del pixels; temporary.replace(path)
    manifest.with_suffix('.tmp').replace(manifest); excluded.with_suffix('.tmp').replace(excluded)
    assert n >= 2 and sum(counts.values())+n == len(items)
    # File reserves capacity for all candidates; only [:images] are observations.
    mean = (sums.astype(np.float64)/n/127.5-1).transpose(2, 0, 1).ravel()
    mean_path = work/'mean.npy'; save_npy(mean_path, mean)
    stamp(root, ready, passed=True, path=str(path), sha256=digest(path), images=n,
        candidates=len(items), exclusions=dict(counts), reserved_rows=len(items),
        manifest=str(manifest), manifest_sha256=digest(manifest),
        excluded_manifest=str(excluded), excluded_sha256=digest(excluded),
        mean_path=str(mean_path), mean_sha256=digest(mean_path), seconds=time.monotonic()-started)
    return read(ready)


def image_fields(root, key, u, lam, queries, times, resolution, noise_scale, gallery_queries):
    d = read(root/'design.json'); work = Path(d['work_root'])/key/'fields'; scores = []
    for ti, t in enumerate(times):
        part = work/f't{ti}.npz'; recpath = root/key/f'fields_t{ti}.json'
        if recpath.exists():
            r = read(recpath); assert r['design_sha256'] == digest(root/'design.json') and digest(part) == r['sha256']
            with np.load(part) as z: scores.append(z['score'])
            continue
        started = time.monotonic(); ss = []; native = []
        for j, q in enumerate(queries):
            f = field(u, lam, t, q, resolution, noise_scale); ss.append(token_max(f))
            if q in gallery_queries: native.append(f)
            if j % 32 == 0: print(key, 't', t, 'query', j, 'seconds', time.monotonic()-started, flush=True)
        score = np.stack(ss); assert np.isfinite(score).all() and (score >= 0).all()
        save_npz(part, score=score, queries=queries, native=np.stack(native) if native else np.empty(0),
            native_queries=[q for q in queries if q in gallery_queries], time=t)
        stamp(root, recpath, passed=True, path=str(part), sha256=digest(part), seconds=time.monotonic()-started)
        scores.append(score)
    out = work/'scores.npz'; save_npz(out, score=np.stack(scores), queries=queries, times=times)
    stamp(root, root/key/'COMPLETE.json', passed=True, path=str(out), sha256=digest(out),
        images=read(root/key/'BANK.json')['images'], rank=int(np.sum(lam > 0)),
        retained_modes=len(lam), normalized_covariance='1/N')


def imagenet(root, index):
    d = read(root/'design.json'); info = d['imagenet']['classes'][index]; wn = info['wnid']; key = 'imagenet256/'+wn
    files = sorted((Path(d['imagenet']['train_root'])/wn).iterdir())
    prior = read(info['old_manifest']) if info.get('old_manifest') else []
    old_paths = [r['original'] for r in prior]; paths = old_paths+sorted(set(map(str, files))-set(old_paths))
    items = [dict(path=p, image_id=Path(p).name, wnid=wn, official_split='train') for p in paths]
    r = prepare_bank(root, key, items, 256, set(d['imagenet']['forbidden_pixels']))
    x = normalized(np.load(r['path'], mmap_mode='r')[:r['images']])
    # Agreement with the already published 512-image score, before changing N.
    if info.get('old_bank'):
        assert digest(info['old_bank']) == info['old_bank_sha256']
        assert digest(info['old_scores']) == info['old_scores_sha256']
        oldx = normalized(np.load(info['old_bank']))
        ou, ol, _ = sample_basis(oldx); del oldx
        with np.load(info['old_scores']) as z:
            q = int(z['queries'][0]); expected = z['score'][2, 0]
        got = token_max(field(ou, ol, .5, q, 256, 1.))
        err = float(np.max(np.abs(got-expected))); assert err < 2e-5, err
        del ou, ol; gc.collect()
    else: err = None
    started = time.monotonic(); u, lam, checks = sample_basis(x); del x; gc.collect()
    stamp(root, root/key/'BASIS.json', passed=True, images=r['images'], rank=len(lam),
        legacy_score_max_abs=err, seconds=time.monotonic()-started, **checks)
    image_fields(root, key, u, lam, info['queries'], d['imagenet']['times'], 256, 1., info['gallery_queries'])


def celeba_prepare(root):
    d = read(root/'design.json'); c = d['celeba']; names = Path(c['train_index']).read_text().splitlines()
    assert digest(c['train_index']) == c['train_index_sha256'] and len(names) == 162770
    items = [dict(path=str(Path(c['train_root'])/f'{i:06d}.png'), image_id=name,
        source_index=i, official_split='train') for i, name in enumerate(names)]
    prepare_bank(root, 'celeba128', items, 128, set(c['forbidden_pixels']))


def bank_arrays(root):
    r = read(root/'celeba128/BANK.json'); assert r['design_sha256'] == digest(root/'design.json')
    assert digest(r['path']) == r['sha256'] and digest(r['mean_path']) == r['mean_sha256']
    return r, np.load(r['path'], mmap_mode='r')[:r['images']], np.load(r['mean_path'])


def celeba_benchmark(root):
    r, pix, mean = bank_arrays(root); started = time.monotonic(); dim = len(mean); work = Path(read(root/'design.json')['work_root'])/'celeba128'
    c = np.zeros((dim, dim), dtype=np.float64, order='F'); batches = []; n = min(2048, len(pix))
    for start in range(0, n, 512):
        x = normalized(pix[start:start+512]); x -= mean
        tick = time.monotonic(); add_scatter(c, x); batches.append(time.monotonic()-tick)
        print('full-dimension scatter benchmark', start, batches[-1], flush=True)
    save_npy(work/'scatter_first2048.npy', c)
    checks = self_check(); rows = np.random.default_rng(71).choice(dim, 64, replace=False)
    xx = normalized(pix[:n])[:, rows]-mean[rows]
    small = c[np.ix_(rows, rows)]; small = np.where(rows[:, None] >= rows[None], small, small.T)
    err = float(np.max(np.abs(small-xx.T @ xx))); assert err < 1e-8
    eig = []
    for size in [4096, 8192]:
        a = np.array(c[:size, :size]/n, order='F'); tick = time.monotonic()
        vals, vectors = linalg.eigh(a, lower=True, driver='evr', overwrite_a=True, check_finite=False)
        seconds = time.monotonic()-tick; assert np.isfinite(vals).all() and np.isfinite(vectors).all()
        eig.append(dict(dimension=size, seconds=seconds, cubic_projection_seconds=seconds*(dim/size)**3))
        print('eigen benchmark', eig[-1], flush=True); del a, vals, vectors; gc.collect()
    # Full-size matrix times a small row block estimates the native readout cost.
    row_probe = np.random.default_rng(72).normal(size=(32, dim))
    tick = time.monotonic(); probe = row_probe @ c.T; multiply_seconds = time.monotonic()-tick
    assert np.isfinite(probe).all(); del probe
    accumulation = float(np.median(batches))*r['images']/512
    factorization = max(v['cubic_projection_seconds'] for v in eig)
    fields = multiply_seconds*(dim/32)*len(read(root/'design.json')['celeba']['times'])
    total = accumulation+factorization+fields
    # A conservative scheduling guard, not an asserted runtime prediction.
    proceed = 2.5*total < 10*3600
    stamp(root, root/'celeba128/BENCHMARK.json', passed=True, proceed_cpu_full=proceed,
        bank_sha256=r['sha256'], first_samples=n, partial_scatter=str(work/'scatter_first2048.npy'),
        partial_scatter_sha256=digest(work/'scatter_first2048.npy'), batch_seconds=batches,
        eig_benchmarks=eig, row_block_seconds=multiply_seconds, accumulation_estimate_seconds=accumulation,
        eig_estimate_seconds=factorization, fields_estimate_seconds=fields,
        estimated_total_seconds=total, scheduling_guard_multiplier=2.5,
        scatter_submatrix_max_abs=err, algebra=checks, seconds=time.monotonic()-started)


def celeba_full(root):
    d = read(root/'design.json'); b = read(root/'celeba128/BENCHMARK.json')
    assert b['passed'] and b['design_sha256'] == digest(root/'design.json')
    if not b['proceed_cpu_full']:
        stamp(root, root/'celeba128/CPU_DEFERRED.json', reason='Conservative benchmark estimate exceeds 10-hour compute guard', benchmark=b)
        print('CPU full calculation deferred; see BENCHMARK.json', flush=True); return
    r, pix, mean = bank_arrays(root); work = Path(d['work_root'])/'celeba128'
    assert r['sha256'] == b['bank_sha256']; n = r['images']; dim = len(mean)
    covpath = work/'covariance_lower.npy'; basispath = work/'basis.npy'; lampath = work/'eigenvalues.npy'
    if not (root/'celeba128/COVARIANCE.json').exists():
        progress = root/'celeba128/ACCUMULATION.json'; partial = work/'scatter_progress.npy'
        if progress.exists():
            pr = read(progress); assert pr['design_sha256'] == digest(root/'design.json') and digest(partial) == pr['sha256']
            c = np.array(np.load(partial), order='F'); first = pr['samples']
        else:
            assert digest(b['partial_scatter']) == b['partial_scatter_sha256']
            c = np.array(np.load(b['partial_scatter']), order='F'); first = b['first_samples']
        started = time.monotonic()
        for start in range(first, n, 512):
            stop = min(start+512, n); x = normalized(pix[start:stop]); x -= mean
            add_scatter(c, x)
            if stop % 16384 == 0 or stop == n:
                save_npy(partial, c)
                stamp(root, progress, passed=True, samples=stop, sha256=digest(partial))
                print('CelebA accumulated', stop, '/', n, 'seconds', time.monotonic()-started, flush=True)
        c /= n; save_npy(covpath, c)
        stamp(root, root/'celeba128/COVARIANCE.json', passed=True, path=str(covpath), sha256=digest(covpath),
            images=n, dimension=dim, lower_triangle_only=True, trace=float(np.trace(c)), seconds=time.monotonic()-started)
        del c; gc.collect()
    if not (root/'celeba128/BASIS.json').exists():
        cr = read(root/'celeba128/COVARIANCE.json'); assert digest(covpath) == cr['sha256']
        c = np.array(np.load(covpath), order='F'); jj = np.random.default_rng(79).choice(dim, 64, replace=False)
        ref = c[np.ix_(jj, jj)]; ref = np.where(jj[:, None] >= jj[None], ref, ref.T)
        started = time.monotonic()
        # EVR avoids EVD's quadratic floating workspace / LP64 lwork overflow.
        lam, u = linalg.eigh(c, lower=True, driver='evr', overwrite_a=True, check_finite=False)
        del c; gc.collect()
        assert np.isfinite(lam).all() and np.isfinite(u).all() and lam[0] >= -max(1., lam[-1])*1e-10
        negative = int(np.sum(lam < 0)); lam = np.maximum(lam, 0.)
        err = float(np.max(np.abs((u[jj]*lam) @ u[jj].T-ref)))
        trace_error = float(abs(lam.sum()-cr['trace'])/cr['trace'])
        assert err < 1e-7 and trace_error < 1e-9, (err, trace_error)
        save_npy(basispath, u); save_npy(lampath, lam)
        stamp(root, root/'celeba128/BASIS.json', passed=True, rank=int(np.sum(lam > 0)), retained_modes=len(lam),
            clipped_negative_roundoff_modes=negative, reconstruction_max_abs=err, trace_relative_error=trace_error,
            basis=str(basispath), basis_sha256=digest(basispath), eigenvalues=str(lampath), eigenvalues_sha256=digest(lampath),
            seconds=time.monotonic()-started)
    else:
        br = read(root/'celeba128/BASIS.json')
        assert br['design_sha256'] == digest(root/'design.json') and digest(basispath) == br['basis_sha256'] and digest(lampath) == br['eigenvalues_sha256']
        u = np.load(basispath); lam = np.load(lampath)
    image_fields(root, 'celeba128', u, lam, list(range(256)), d['celeba']['times'], 128, .5, [85, 136, 187])


def main():
    p = argparse.ArgumentParser(); p.add_argument('--report', type=Path, required=True)
    p.add_argument('--stage', required=True, choices=['check', 'imagenet', 'celeba-prepare', 'celeba-benchmark', 'celeba-full'])
    p.add_argument('--index', type=int); a = p.parse_args(); root = a.report.resolve()
    code = Path(__file__).resolve().parents[2]
    if (code/'MANIFEST.json').exists():
        for rel, h in read(code/'MANIFEST.json').items(): assert digest(code/rel) == h
    checks = self_check()
    if a.stage == 'check': stamp(root, root/'CHECKS.json', **checks)
    elif a.stage == 'imagenet': imagenet(root, a.index if a.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID']))
    elif a.stage == 'celeba-prepare': celeba_prepare(root)
    elif a.stage == 'celeba-benchmark': celeba_benchmark(root)
    else: celeba_full(root)


if __name__ == '__main__':
    main()
