"""Apply the validated real-dataset spatial descriptor to paired hybrid images."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'reports/experiment3'))
from hybrid_analysis import CODES, selected_rows, locate, resolve, read, digest
from scripts import experiment3_dataset_geometry as geometry

BASE = ROOT/'reports/2026-09-20/experiment3_hybrid_geometry'
OUT = ROOT/'reports/experiment3/metrics/geometry_results'
EDGES = [('000', '100'), ('010', '110'), ('001', '101'), ('011', '111')]


def prepare():
    assert read(geometry.BASE/'results.json')['validated'], 'Validate real-dataset geometry first'
    specs = []
    for row in selected_rows()[::32]:
        cube_path = locate(row)[2]
        cube = read(cube_path)
        assert cube['row']['id'] == row['id']
        specs.append(dict(id=row['id'], cube=str(cube_path.relative_to(ROOT)), cube_sha256=digest(cube_path)))
    manifest = dict(rows=specs, code_sha256=digest(__file__),
                    dataset_manifest_sha256=digest(geometry.BASE/'manifest.json'),
                    descriptor='Identical spatially normalized patch cosine descriptor to the validated dataset-first analysis',
                    images_per_state=1024, qk_edges=EDGES, min_source_joint_distance=1e-6)
    path = BASE/'manifest.json'
    if path.exists(): assert read(path) == manifest
    else: geometry.write(path, manifest)
    print(path, flush=True)


def run_cube(cube_path, destination):
    import torch
    from PIL import Image
    cube_path = Path(cube_path); destination = Path(destination)
    cube = read(cube_path)
    assert read(geometry.BASE/'results.json')['validated'], 'Real-dataset controls must pass first'
    signature = dict(id=cube['row']['id'], cube_sha256=digest(cube_path),
                     code_sha256=digest(__file__), dataset_manifest_sha256=digest(geometry.BASE/'manifest.json'))
    meta_path = destination.with_suffix('.json')
    if destination.exists() and meta_path.exists():
        metadata = read(meta_path)
        assert all(metadata[k] == v for k, v in signature.items())
        assert metadata['measurement_sha256'] == digest(destination)
        print('Reusing validated spatial measurement', signature['id'], flush=True)
        return
    _, model, transform = geometry.setup()
    arrays, hashes = {}, {}
    for code in CODES:
        path = resolve(cube['images'][code])
        hashes[code] = dict(path=str(path.relative_to(ROOT)), sha256=digest(path))
        with np.load(path) as a:
            assert str(a['latent_hash']) == cube['latent_hash']
            assert str(a['checkpoint_sha256']) == cube['row']['checkpoints'][code]['sha256']
            arrays[code] = a['images']
            assert arrays[code].shape == (1024, 3, 64, 64) and arrays[code].dtype == np.uint8
    distances, edge_distances = [], []
    with torch.inference_mode():
        for start in range(0, 1024, 16):
            desc = {}
            for code in CODES:
                images = [Image.fromarray(x.transpose(1, 2, 0)) for x in arrays[code][start:start+16]]
                batch = torch.stack([transform(x) for x in images]).cuda()
                desc[code] = geometry.descriptors(model, batch)[0]
            distances.append(np.stack([(desc[c]-desc['111']).double().square().mean(1).cpu().numpy() for c in CODES]))
            edge_distances.append(np.stack([(desc[a]-desc[b]).double().square().mean(1).cpu().numpy() for a, b in EDGES]))
            if start % 256 == 0: print(signature['id'], start+16, '/ 1024 matched latents', flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, distances=np.concatenate(distances, 1), qk_edge_distances=np.concatenate(edge_distances, 1),
                        codes=np.array(CODES), qk_edges=np.array(EDGES), latent_hash=cube['latent_hash'])
    os.replace(tmp, destination)
    geometry.write(meta_path, dict(**signature, images=hashes, latent_hash=cube['latent_hash'],
                                   cube=str(cube_path.relative_to(ROOT)), measurement_sha256=digest(destination),
                                   descriptor_dimension=32640, paired_latents=1024,
                                   note='Mean squared difference of fixed-grid spatial self-similarity descriptors; matches the validated real-dataset descriptor'))
    print('Completed spatial cube', signature['id'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'run'])
    parser.add_argument('--task', type=int, default=0)
    args = parser.parse_args()
    if args.stage == 'prepare': prepare()
    else:
        manifest = read(BASE/'manifest.json')
        assert manifest['code_sha256'] == digest(__file__)
        row = manifest['rows'][args.task]
        cube_path = ROOT/row['cube']; assert digest(cube_path) == row['cube_sha256']
        run_cube(cube_path, OUT/f"{row['id']}.npz")
