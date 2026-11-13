"""Validate prepared native-view pixels and create the recorded artwork splits."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=['uvit64', 'dit128', 'lfm256'], required=True)
    parser.add_argument('--images', type=Path, required=True, help='Already preprocessed native-view RGB PNGs; see docs/data.md.')
    parser.add_argument('--source-images', type=Path, help='Prepared source-validation images; required for DiT/LFM.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    name = 'metfaces_' + args.family
    resolution = {'uvit64': 64, 'dit128': 128, 'lfm256': 256}[args.family]
    output = args.output or ROOT / 'assets/manifests' / (name + '.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    folder = output.parent / name
    folder.mkdir(exist_ok=True)
    membership = json.loads((ROOT / 'revision/data' / (name + '_membership.json')).read_text())
    index = {}
    for path in sorted(args.images.rglob('*.png')):
        assert path.name not in index, f'Ambiguous image basename: {path.name}'
        index[path.name] = path.resolve()
    metadata = dict(kind='files', resolution=resolution, splits={}, counts={}, prepared_view=args.family)
    all_paths = []
    for split in ['train', 'valid', 'test']:
        paths = [index[name] for name in membership[split]]
        for path in paths:
            with Image.open(path) as image:
                assert image.mode == 'RGB' and image.size == (resolution, resolution), path
        start = len(all_paths)
        all_paths += paths
        if args.family == 'dit128':
            p = folder / (split + '.npy')
            np.save(p, np.arange(start, len(all_paths), dtype=np.int64))
        else:
            p = folder / (split + '.txt')
            p.write_text(''.join(str(path) + '\n' for path in paths))
        metadata['splits'][split] = dict(path=str(p.resolve()), count=len(paths), sha256=sha(p))
        metadata['counts'][split] = len(paths)
    if args.family in ['dit128', 'lfm256']:
        assert args.source_images, '--source-images is required for native validation and posterior-cache preparation.'
        source_index = {p.name: p.resolve() for p in args.source_images.rglob('*.png')}
        if args.family == 'lfm256':
            names = json.loads((ROOT / 'revision/data/lfm_source_validation_membership.json').read_text())
            source = [source_index[name] for name in names]
            p = output.parent / 'lfm_source_validation.txt'
            p.write_text(''.join(str(path) + '\n' for path in source))
        else:
            source = sorted(source_index.values())[:1024]
            assert len(source) == 1024
            def pixels(paths):
                arrays = []
                for path in paths:
                    with Image.open(path) as image:
                        assert image.mode == 'RGB' and image.size == (128, 128), path
                        arrays.append(np.array(image, copy=True))
                return np.stack(arrays)
            p = folder / 'images.npy'
            np.save(p, pixels(all_paths))
            metadata.update(kind='mmap', images=str(p.resolve()), images_sha256=sha(p))
            p = folder / 'source_test.npy'
            np.save(p, pixels(source))
            metadata.update(source_test=str(p.resolve()), source_test_sha256=sha(p), source_count=1024)
    output.write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Validated {metadata["counts"]} images at {resolution}px; wrote {output}')


if __name__ == '__main__':
    main()
