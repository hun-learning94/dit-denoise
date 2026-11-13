"""Materialize a portable protocol template against local data and source code."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def materialize(value):
    if isinstance(value, str):
        return value.replace('@ROOT@', str(ROOT))
    if isinstance(value, list):
        return [materialize(x) for x in value]
    if isinstance(value, dict):
        return {k: materialize(v) for k, v in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('template')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--source-checkpoint', type=Path)
    parser.add_argument('--training-seed', type=int)
    parser.add_argument('--dry-run', action='store_true', help='Inspect the materialized template without accessing external assets.')
    args = parser.parse_args()
    template = ROOT / 'revision/configs' / (args.template + '.json')
    m = materialize(json.loads(template.read_text()))
    m['code'] = str(ROOT)
    if args.data:
        m['data'] = str(args.data.resolve())
    if args.source_checkpoint:
        m['source']['checkpoint'] = str(args.source_checkpoint.resolve())
        m['source']['checkpoint_sha256'] = digest(args.source_checkpoint)
        m['source']['release_note'] = 'Reader-supplied checkpoint; this changes the source identity.'
    if args.training_seed is not None:
        seed = args.training_seed
        assert seed >= 0
        protocols = json.loads((ROOT / 'revision/data/training_seed_protocols.json').read_text())
        family = m['source']['family']
        assert str(seed) in protocols[family], 'Recorded training seeds are 0, 1 and 2.'
        m['recipe'] = protocols[family][str(seed)]
        m['runs'] = {key: value + f'_seed{seed}' for key, value in m['runs'].items()}
        m['eval_output'] += f'_seed{seed}'
        m['report'] += f'_seed{seed}'
    m['code_sha256'] = {str(p.relative_to(ROOT)): digest(p) for p in sorted(ROOT.rglob('*.py'))
                        if not any(x in p.relative_to(ROOT).parts for x in ['.git', '.venv', 'external', 'assets', 'outputs'])}
    if args.template == 'functional_context':
        m['source_hashes'] = {str(ROOT / name): value for name, value in m['code_sha256'].items()}
    if args.template == 'norm_control':
        m['own_code_sha256'] = {'worker.py': digest(ROOT / 'experiments/norm_matched/worker.py')}
        m['driver_sha256'] = dict(m['code_sha256'])
    if args.template == 'scale_jit':
        m['source_sha256'] = dict(m['code_sha256'])
        m['kernel_sha256'] = digest(ROOT / 'fused_grouped_gelu_wide.py')
    if not args.dry_run:
        if 'data' in m:
            assert Path(m['data']).is_file(), 'Prepare the data manifest first; see docs/experiments.md.'
            m['data_sha256'] = digest(m['data'])
        if args.template.startswith('transfer_'):
            for row in m['runs']:
                assert Path(row['data']).is_file(), f"Prepare the native-view target manifest: {row['data']}"
                row['data_sha256'] = digest(row['data'])
        if 'source' in m:
            source = m['source']
            assert Path(source['checkpoint']).is_file(), 'Download or train the source checkpoint first.'
            assert digest(source['checkpoint']) == source['checkpoint_sha256'], 'Source checkpoint differs from the selected identity.'
        if 'cache_identity' in m:
            m['cache_identity']['data_sha256'] = m['data_sha256']
            m['cache_identity']['source_view_sha256'] = digest(m['source_view']['path'])
            m['source_view']['sha256'] = m['cache_identity']['source_view_sha256']
        Path(m['report']).mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(m, indent=2) + '\n')
    print(f'Wrote {args.output}' + (' (dry run; assets not validated)' if args.dry_run else ''))


if __name__ == '__main__':
    main()
