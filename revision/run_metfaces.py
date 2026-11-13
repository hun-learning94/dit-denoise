"""Run the revised independent MetFaces experiment from a configured manifest."""
import argparse
import inspect
import json
import signal
from pathlib import Path
import torch


def retain_100_epoch_exports(cls):
    original = cls.export
    def export(self, path):
        name = Path(path).name
        if name == 'best.pt':
            return
        if name.startswith('ema_epoch') and int(name[9:-3]) % 100:
            return
        return original(self, path)
    cls.export = export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['cache', 'smoke', 'train', 'real', 'generate', 'score'])
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--arm', choices=['qk', 'vo', 'mlp', 'vo_mlp', 'qk_vo_mlp', 'full'], default='vo')
    parser.add_argument('--task', type=int, default=0)
    args = parser.parse_args()
    m = json.loads(args.manifest.read_text())
    family = m['source']['family']
    from experiments.component_transfer_scale.model import digest, write
    for name, expected in m['code_sha256'].items():
        assert digest(Path(m['code']) / name) == expected, name
    assert digest(m['data']) == m['data_sha256']
    if args.stage != 'score':
        assert torch.cuda.is_available(), 'CUDA required; no CPU fallback for training/evaluation.'
        assert torch.cuda.is_bf16_supported(including_emulation=False)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if family == 'lfm':
        from experiments.lfm_metfaces import common as c, train as tr, evaluate as ev, cache
        m = c.setup(str(args.manifest), gpu=args.stage != 'score')
        m['_manifest'] = str(args.manifest)
        if args.stage == 'cache':
            cache.prepare(m)
        elif args.stage == 'smoke':
            for arm in m['runs']:
                tr.smoke(m, arm)
            ev.smoke(m)
            write(Path(m['report']) / 'PREFLIGHT_PASS.json', dict(passed=True, manifest_sha256=digest(args.manifest)))
        elif args.stage == 'train':
            source = inspect.getsource(tr.train)
            for before, after in {
                ";t.export('best.pt')": '',
                '[25,50,100,200,500,1000]': '[100,200,500,1000]',
                "best_sha256=c.digest(out/'best.pt')": "best_sha256=None,checkpoint_policy='fixed-epochs-no-selected'",
            }.items():
                assert source.count(before) == 1
                source = source.replace(before, after)
            exec(compile(source, __file__, 'exec'), tr.__dict__)
            tr.train(m, args.arm)
        elif args.stage == 'real':
            ev.real(m)
        elif args.stage == 'generate':
            source = inspect.getsource(ev.generate)
            before = "if row['endpoint']=='final' and"
            assert source.count(before) == 1
            exec(compile(source.replace(before, "if False and row['endpoint']=='final' and"), __file__, 'exec'), ev.__dict__)
            ev.generate(m, args.task)
        elif args.stage == 'score':
            ev.score(m, args.task)
        return
    if family == 'uvit':
        from experiments.uvit_metfaces import train as tr, evaluate as ev
        from experiments.uvit_metfaces.dpm_sampler import sample, identity
        m['sampling_protocol'] = identity(50)
        ev.sample = sample
    else:
        from experiments.celeba128_metfaces import train as tr, evaluate as ev
        from experiments.uvit_metfaces import train as base
        base.Training = tr.Training
    if args.stage == 'score':
        raise SystemExit('U-ViT/DiT generation writes its own exact KID/FID result.json after completing the feature bank.')
    torch.set_num_threads(1)
    for sig in [signal.SIGTERM, signal.SIGUSR1]:
        signal.signal(sig, tr.stop if family == 'uvit' and args.stage == 'train' else
                      (base.stop if family == 'pixel' and args.stage == 'train' else ev.stop))
    if args.stage == 'smoke':
        tr.smoke(m)
        ev.smoke(m)
        if family == 'uvit':
            p = Path(m['report']) / 'EVAL_SMOKE_PASS.json'
            result = json.loads(p.read_text())
            result.update(sampler_steps=50, sampling_protocol=m['sampling_protocol'])
            write(p, result)
    elif args.stage == 'train':
        assert (Path(m['report']) / 'SMOKE_PASS.json').exists(), 'Run the GPU smoke checks first.'
        retain_100_epoch_exports(tr.Training)
        if family == 'pixel':
            base.run(m, args.arm)
        else:
            tr.run(m, args.arm)
    elif args.stage == 'real':
        ev.real_features(m)
    elif args.stage == 'generate':
        ev.evaluate(m, args.task)
    else:
        raise SystemExit('Only LFM uses a VAE posterior cache.')


if __name__ == '__main__':
    main()
