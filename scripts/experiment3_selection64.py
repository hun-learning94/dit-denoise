#!/usr/bin/env python3
"""Freeze and render 64 matched latents at the currently reported transfer endpoints."""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import html
import io
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CAMPAIGN = ROOT / 'reports/2026-09-19/experiment3_selection64'
GALLERY = ROOT / 'reports/experiment3_selection64'
LEGACY_GALLERY = CAMPAIGN / 'archive/transfer_gallery'
if LEGACY_GALLERY.exists():
    # Keep frozen-bank generation/index commands out of the new source gallery.
    GALLERY = LEGACY_GALLERY
CODES = ('000', '100', '010', '110', '001', '101', '011', '111')
TITLES = ('Source', 'QK', 'VO', 'QK+VO', 'MLP', 'QK+MLP', 'VO+MLP', 'Joint')


def read(path):
    return json.loads(Path(path).read_text())


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(text)
    temp.replace(path)


def write(path, obj):
    atomic_text(path, json.dumps(obj, indent=2) + '\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_npz(path, **arrays):
    import numpy as np
    path = Path(path)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    temp.replace(path)


def pixels(array):
    import numpy as np
    if array.dtype == np.uint8:
        return array
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
        raise ValueError('Invalid image range')
    return np.rint(array * 255).astype(np.uint8)


def prepare():
    manifest = CAMPAIGN / 'manifest.json'
    if manifest.exists():
        raise FileExistsError('Frozen manifest already exists; run tasks or index instead')
    endpoints = read(ROOT / 'reports/experiment3/tables/endpoint_selections.json')
    groups = ['reports/experiment3/metrics/cube_results',
              'reports/experiment3/metrics/extension_results',
              'reports/experiment3/metrics/imagenet600_results',
              'reports/2026-09-16/stl10_source800_hybrid_metrics/reporting_results']
    rows = []
    for group in groups:
        for path in sorted((ROOT / group).glob('*.json')):
            row = read(path)['row']
            if group.endswith('/cube_results') and row['source'] == 'stl10_64':
                continue
            rows.append((row, path))
    assert len(rows) == 32
    tasks = []
    for row, result_path in rows:
        endpoint, = [e for e in endpoints if (e['source'], e['target']) == (row['source'], row['target'])]
        seed = row['adaptation_seed']
        step = endpoint['step']
        if 'adaptation_updates' in row:
            assert row['adaptation_updates'] == step
        elif 'adapted_checkpoint' in row:
            assert Path(row['adapted_checkpoint']).stem == f'ckpt_step{step}'
        else:
            assert row['selected_epoch'] == endpoint['target_epoch']
        stem = f"primary_20m_{row['source']}_to_{row['target']}_"
        suffix = f'_generation_seed{seed}'
        base = ROOT / 'reports/experiment3/figures'
        visual = base / f'{stem}step{step}{suffix}.json'
        if not visual.exists() and row['source'] == 'celeba64' and row['target'] != 'icartoonface64':
            visual = base / f"{stem}epoch{int(endpoint['target_epoch'])}{suffix}.json"
        task = dict(id=row['id'], row=row, endpoint=endpoint, metric_result=str(result_path),
                    checkpoints=row['checkpoints'], latent_seed=20260904, sampler_steps=50,
                    images=64, generation_batch=16, precision='fp32', mode='generate')
        if visual.exists():
            meta = read(visual)
            snap = CAMPAIGN / 'prefix16' / (row['id'] + '.npz')
            snap.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(visual.with_suffix('.npz'), snap)
            shutil.copy2(visual, snap.with_suffix('.json'))
            task.update(prefix16=str(snap), prefix16_sha256=digest(snap), original_visual=str(visual))
            if 'cube_dir' in meta:
                assert meta['images'] == 16 and meta['sampler_steps'] == 50
                task['latent_seed'] = meta['latent_seed']
                task['checkpoints'] = {c: dict(path=str((ROOT / meta['cube_dir'] / f'ckpt_{c}.pt').resolve()),
                                             sha256=meta['checkpoint_sha256'][c]) for c in CODES}
            else:
                assert row['source'] == 'imagenet64' and row['source_training_epoch'] == 600
                assert meta['row'] == row
                assert meta['latent_indices'] == list(range(16))
                task.update(mode='export', cached_images=meta['images'], latent_seed=20260911,
                            cached_latent_hash=meta['latent_hash'])
        for ck in task['checkpoints'].values():
            assert Path(ck['path']).is_file(), ck['path']
        for p in task.get('cached_images', {}).values():
            assert Path(p).is_file(), p
        tasks.append(task)
    assert len({(t['row']['source'], t['row']['target'], t['row']['adaptation_seed']) for t in tasks}) == 32
    write(manifest, dict(purpose='Candidate gallery for first-page figure; appendix grids unchanged',
                        frozen_date='2026-09-19', tasks=tasks))
    write(GALLERY / 'manifest.json', read(manifest))
    atomic_text(GALLERY / 'README.md', '''# Experiment 3 figure selection gallery

Open `index.html` for all 16 transfer directions and adaptation seeds 0 and 1.
Each completed bank has 64 fixed latents, four 16-row contact sheets, and 64
individual row strips. Every row compares all eight hybrid states on the same latent.
Rows are labeled by adaptation seed and **one-based latent index (001–064)**.
The RNG seed is recorded separately in `provenance.json` and `index.csv`.
The existing 16 latents are retained as rows 001–016 wherever a grid already exists.
ImageNet uses the currently reported **20M epoch-600 source**, exporting the first
64 existing evaluation latents. Other directions extend the existing visual latent
sequence to 64. Source/checkpoint hashes and endpoint selections are frozen in
`manifest.json`; this gallery will not silently switch to a later report version.

The gallery is for choosing illustrative examples for the first-page figure.
No examples have been selected yet. Record the transfer direction, adaptation seed,
and latent index when choosing a row. The fixed-latent appendix grids and all
quantitative evaluations remain unchanged. No gallery is added to `_images` Markdown.

Each bank includes `images.npz` (uint8 RGB arrays, codes, raw latents and indices),
`provenance.json`, `index.csv`, and `index.html`. Contact sheets and strips use
nearest-neighbor display enlargement of the original 64×64 samples.
''')
    refresh_index()
    print(json.dumps({mode: [i for i, t in enumerate(tasks) if t['mode'] == mode]
                      for mode in ('generate', 'export')}, indent=2))


def page(title, body):
    return ('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            '<style>body{font:17px system-ui;max-width:1500px;margin:32px auto;padding:0 20px;'
            'background:#fafafa;color:#222}img{max-width:100%;height:auto}td,th{padding:8px;text-align:left}'
            'table{border-collapse:collapse}tr{border-bottom:1px solid #ddd}a{color:#075cba}</style>'
            f'<title>{html.escape(title)}</title><h1>{html.escape(title)}</h1>{body}')


def refresh_index():
    GALLERY.mkdir(parents=True, exist_ok=True)
    with (GALLERY / '.index.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        tasks = read(CAMPAIGN / 'manifest.json')['tasks']
        body = ('<p>64 matched latents per adaptation seed. Choose a complete row across the eight states. '
                'ImageNet: 20M, source epoch 600. Latent indices are one-based.</p>'
                '<p><a href="README.md">Guide</a> · <a href="index.csv">Row index (CSV)</a></p>'
                '<table><tr><th>Transfer</th><th>Seed</th><th>Reference epoch</th><th>Gallery</th></tr>')
        if (GALLERY / 'candidates/index.html').exists():
            body = '<p><a href="candidates/index.html">Reviewed shortlist: five candidates per run</a></p>' + body
        csvout = io.StringIO()
        writer = csv.writer(csvout, lineterminator='\n')
        writer.writerow(['bank', 'source', 'target', 'adaptation_seed', 'latent_seed', 'latent_index', 'row_png', 'status'])
        for task in tasks:
            row = task['row']; bank = task['id']; seed = row['adaptation_seed']
            complete = (GALLERY / bank / 'complete.json').exists()
            link = f'<a href="{bank}/index.html">64 rows</a>' if complete else 'Queued / generating'
            body += f'<tr><td>{row["source"]} → {row["target"]}</td><td>{seed}</td><td>{task["endpoint"]["reference_epoch"]:g}</td><td>{link}</td></tr>'
            for i in range(1, 65):
                writer.writerow([bank, row['source'], row['target'], seed, task['latent_seed'], i,
                                 f'{bank}/rows/seed{seed}_latent{i:03d}.png', 'complete' if complete else 'pending'])
        atomic_text(GALLERY / 'index.html', page('Experiment 3 — figure candidates', body + '</table>'))
        atomic_text(GALLERY / 'index.csv', csvout.getvalue())


def render(task, images, dest):
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib
    font_path = Path(matplotlib.get_data_path()) / 'fonts/ttf/DejaVuSans.ttf'
    font = ImageFont.truetype(str(font_path), 22)
    small = ImageFont.truetype(str(font_path), 19)
    tile, gap, left, header = 128, 8, 172, 42
    width = left + 8 * (tile + gap)
    seed = task['row']['adaptation_seed']
    (dest / 'rows').mkdir(exist_ok=True)
    def sheet(indices):
        image = Image.new('RGB', (width, header + len(indices) * (tile + gap)), 'white')
        draw = ImageDraw.Draw(image)
        draw.text((12, 8), '(seed, latent)', font=small, fill='black')
        for j, title in enumerate(TITLES):
            draw.text((left + j * (tile + gap) + tile / 2, 8), title, font=font, fill='black', anchor='mt')
        for n, i in enumerate(indices):
            y = header + n * (tile + gap)
            draw.text((12, y + tile / 2), f'({seed}, {i+1:03d})', font=font, fill='black', anchor='lm')
            for j in range(8):
                im = Image.fromarray(images[j, i].transpose(1, 2, 0)).resize((tile, tile), Image.Resampling.NEAREST)
                image.paste(im, (left + j * (tile + gap), y))
        return image
    for start in range(0, 64, 16):
        sheet(range(start, start+16)).save(dest / f'contact_{start+1:03d}-{start+16:03d}.png')
    body = '<p><a href="../index.html">All transfers</a> · <a href="index.csv">Row index</a> · <a href="provenance.json">Provenance</a></p>'
    body += f'<p>Adaptation seed {seed}; latent RNG seed {task["latent_seed"]}. Rows 001–064.</p>'
    for start in range(0, 64, 16):
        name = f'contact_{start+1:03d}-{start+16:03d}.png'
        body += f'<h2>Latents {start+1:03d}–{start+16:03d}</h2><a href="{name}"><img loading="lazy" src="{name}"></a>'
    body += '<h2>Individual comparison rows</h2>'
    csvout = io.StringIO(); writer = csv.writer(csvout, lineterminator='\n')
    writer.writerow(['adaptation_seed', 'latent_seed', 'latent_index', 'row_png'])
    for i in range(64):
        name = f'rows/seed{seed}_latent{i+1:03d}.png'
        sheet([i]).save(dest / name)
        writer.writerow([seed, task['latent_seed'], i+1, name])
        body += f'<p><a href="{name}">({seed}, {i+1:03d})</a></p><img loading="lazy" src="{name}">'
    atomic_text(dest / 'index.csv', csvout.getvalue())
    atomic_text(dest / 'index.html', page(task['id'], body))


def run(index):
    import numpy as np
    import torch
    task = read(CAMPAIGN / 'manifest.json')['tasks'][index]
    dest = GALLERY / task['id']; dest.mkdir(parents=True, exist_ok=True)
    with (dest / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (dest / 'complete.json').exists():
            refresh_index(); return
        torch.set_num_threads(1)
        latents = torch.randn(64, 3, 64, 64, generator=torch.Generator().manual_seed(task['latent_seed']))
        prefix = None
        if 'prefix16' in task:
            assert digest(task['prefix16']) == task['prefix16_sha256']
            with np.load(task['prefix16']) as a:
                order = list(a['codes'])
                prefix = pixels(a['images'][[order.index(c) for c in CODES]])
                assert prefix.shape == (8, 16, 3, 64, 64)
                if 'latents' in a:
                    assert np.array_equal(a['latents'], latents[:16].numpy())
        if task['mode'] == 'generate':
            assert torch.cuda.is_available(), 'CUDA unavailable; refusing CPU fallback'
            torch.ones(1, device='cuda').sum().item(); torch.cuda.synchronize()
            from generate_experiment3_secondary_examples import sample
        else:
            full_latents = torch.randn(1024, 3, 64, 64, generator=torch.Generator().manual_seed(task['latent_seed']))
            assert hashlib.sha256(full_latents.numpy().tobytes()).hexdigest() == task['cached_latent_hash']
            assert torch.equal(full_latents[:64], latents)
        columns = []
        for j, code in enumerate(CODES):
            ck = task['checkpoints'][code]
            assert digest(ck['path']) == ck['sha256'], f'Checkpoint changed: {ck["path"]}'
            if task['mode'] == 'export':
                with np.load(task['cached_images'][code]) as a:
                    assert str(a['checkpoint_sha256']) == ck['sha256']
                    assert str(a['latent_hash']) == task['cached_latent_hash']
                    column = a['images'][:64].copy()
                if prefix is not None:
                    assert np.array_equal(column[:16], prefix[j])
            else:
                cached = dest / f'state_{code}.npz'
                if cached.exists():
                    with np.load(cached) as a:
                        assert str(a['checkpoint_sha256']) == ck['sha256']
                        column = a['images'].copy()
                else:
                    chunks = [prefix[j]] if prefix is not None else []
                    for start in range(16 if prefix is not None else 0, 64, 16):
                        print(f'{task["id"]}: state {code}, latents {start+1}-{start+16}', flush=True)
                        chunks.append(pixels(sample(Path(ck['path']), latents[start:start+16].cuda(), 50).numpy()))
                    column = np.concatenate(chunks)
                    save_npz(cached, images=column, checkpoint_sha256=ck['sha256'])
            assert column.shape == (64, 3, 64, 64) and column.dtype == np.uint8
            columns.append(column)
        images = np.stack(columns)
        save_npz(dest / 'images.npz', images=images, codes=np.asarray(CODES), latents=latents.numpy(),
                 latent_indices=np.arange(1, 65), adaptation_seed=task['row']['adaptation_seed'],
                 latent_seed=task['latent_seed'], sampler_steps=50)
        write(dest / 'provenance.json', task)
        render(task, images, dest)
        write(dest / 'complete.json', dict(images_per_state=64, states=8, rows=64,
                                          first16_preserved=prefix is not None,
                                          images_sha256=digest(dest / 'images.npz')))
    refresh_index()
    print(f'Complete: {dest}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run', 'index'])
    parser.add_argument('--task', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', '0')))
    args = parser.parse_args()
    if args.stage == 'prepare': prepare()
    elif args.stage == 'index': refresh_index()
    else: run(args.task)
