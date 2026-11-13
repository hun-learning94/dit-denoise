"""Native CelebA128 independent restricted adaptation, using audited resume code."""
import argparse
import copy
import gc
import json
import signal
import time
from pathlib import Path
import numpy as np
import torch
from experiments.component_transfer_scale.model import load_source, plain_model, digest, write, seed_all, rng_state, restore_rng
from experiments.component_transfer_scale.runner import Images
from experiments.uvit_metfaces import train as base
from .model import attach, frozen, EMA


class Training(base.Training):
    def __init__(self, m, arm):
        self.m, self.arm = m, arm
        self.output = Path(m['runs'][arm]); self.output.mkdir(parents=True, exist_ok=True)
        net, source, self.config = load_source(m['source'])
        assert self.config == m['source']['model_cfg']
        self.net = net.cuda()
        self.source = {n: v.cuda() for n, v in source.items()}
        self.counts = attach(self.net, arm, training_seed=m['recipe']['training_seed'])
        self.ema = EMA(self.source, self.counts)
        self.params = [p for p in self.net.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=1e-4, betas=(.9, .95), weight_decay=0)
        self.step, self.best, self.best_step = 0, float('inf'), None
        self.identity = dict(arm=arm, source=m['source']['checkpoint_sha256'], data=digest(m['data']),
                             recipe=m['recipe'], code_sha256=m['code_sha256'])
        seed_all(m['recipe']['corruption_seed'])

    def corrupt(self, images, augment=False):
        x = images.cuda().float() / 127.5 - 1
        if augment:
            flip = torch.rand(len(x), device='cuda') < .5
            x[flip] = x[flip].flip(-1)
        t = (torch.randn(len(x), 1, 1, 1, device='cuda') * .8 - .8).sigmoid()
        noise = torch.randn_like(x) * .5
        return x, t, t * x + (1 - t) * noise

    def update(self, images):
        self.net.train(); self.opt.zero_grad(set_to_none=True)
        x, t, z = self.corrupt(images, True)
        lr = 1e-4 * min(1., (self.step + 1) / (5 * self.m['steps_per_epoch']))
        for group in self.opt.param_groups:
            group['lr'] = lr
        loss = 0.
        for start in range(0, len(x), self.m['recipe']['microbatch']):
            end = start + self.m['recipe']['microbatch']
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = self.net(z[start:end], t[start:end].flatten())
            part = ((prediction.float() - x[start:end]) / (1 - t[start:end]).clamp_min(.05)).square().flatten(1).mean(1).sum() / len(x)
            assert torch.isfinite(part)
            part.backward(); loss += float(part.detach())
        norm = torch.nn.utils.clip_grad_norm_(self.params, 1., error_if_nonfinite=True)
        self.opt.step(); self.ema.update(self.net, self.source); self.step += 1
        return dict(loss=loss, gradient_norm=float(norm), lr=lr)

    @torch.no_grad()
    def validate(self, data):
        keep = rng_state()
        net = plain_model(self.m['source'], self.ema.shadow, self.config).cuda().eval()
        seed_all(2026110901)
        values = []
        for start in range(0, len(data), 8):
            images = torch.stack([data[i] for i in range(start, min(start + 8, len(data)))])
            x, t, z = self.corrupt(images)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = net(z, t.flatten())
            values.extend(((prediction.float() - x) / (1 - t).clamp_min(.05)).square().flatten(1).mean(1).cpu().tolist())
        del net; restore_rng(keep)
        return dict(mean=float(np.mean(values)), per_image=values, count=len(values))


def setup(m):
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    for name, expected in m['code_sha256'].items():
        assert digest(Path(m['code']) / name) == expected, name
    assert digest(m['data']) == m['data_sha256']
    assert digest(m['source']['checkpoint']) == m['source']['checkpoint_sha256']


def smoke(m):
    data = json.loads(Path(m['data']).read_text())
    train, valid = Images(data, 'train'), Images(data, 'valid')
    images = torch.stack([train[i] for i in range(32)])
    checks = []; paired = None
    for arm in m['arms']:
        torch.cuda.reset_peak_memory_stats()
        tr = Training(m, arm)
        initial_rng = torch.cuda.get_rng_state().clone()
        if paired is None:
            paired = initial_rng.clone()
        assert torch.equal(paired, initial_rng), 'Training randomness must be paired across arms'
        tr.net.eval(); x = images[:2].cuda().float() / 127.5 - 1; t = torch.full((2,), .4, device='cuda')
        plain = plain_model(m['source'], tr.source, tr.config).cuda().eval()
        with torch.no_grad():
            assert torch.equal(tr.net(x, t), plain(x, t)), 'Initial model changed'
        del plain
        first = tr.update(images); frozen(tr.net, tr.source, tr.counts)
        if not tr.counts['full']:
            modules = dict(tr.net.named_modules())
            assert all(modules[r['adapter']].lora_B.detach().abs().max() > 0 for r in tr.counts['matrices'])
        snapshot = copy.deepcopy(tr.payload())
        tr.update(images)
        expected = {n: p.detach().clone() for n, p in tr.net.named_parameters() if p.requires_grad}
        ema = {n: tr.ema.shadow[n].clone() for n in tr.ema.names}
        tr.restore(snapshot); tr.update(images)
        assert all(torch.equal(dict(tr.net.named_parameters())[n], v) for n, v in expected.items())
        assert all(torch.equal(tr.ema.shadow[n], v) for n, v in ema.items())
        from .model import effective
        merged = plain_model(m['source'], effective(tr.net, tr.source, tr.counts), tr.config).cuda().eval()
        tr.net.eval()
        with torch.no_grad():
            error = float((merged(x, t) - tr.net(x, t)).abs().max())
        assert error < 2e-5, error
        validation = tr.validate(valid); assert np.isfinite(validation['mean'])
        del merged, expected, ema, snapshot
        gc.collect(); torch.cuda.empty_cache()
        # Warm up the optimizer, then measure ordinary production updates.
        torch.cuda.synchronize(); started = time.monotonic()
        for _ in range(10):
            tr.update(images)
        torch.cuda.synchronize()
        checks.append(dict(arm=arm, parameters=tr.counts['parameters'], initial_exact=True,
            frozen_exact=True, resume_exact=True, effective_ema_resume_exact=True,
            paired_corruption=True, merged_max_error=error, first_update=first,
            steady_seconds_per_update=(time.monotonic() - started) / 10,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30))
        del tr; gc.collect(); torch.cuda.empty_cache()
        print(json.dumps(checks[-1]), flush=True)
    write(Path(m['report']) / 'SMOKE_PASS.json', dict(checks=checks,
        gpu=torch.cuda.get_device_name(), microbatch=m['recipe']['microbatch'],
        global_batch=32, data_sha256=m['data_sha256']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['smoke', 'train'])
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--task', type=int, default=0)
    args = parser.parse_args()
    m = json.loads(Path(args.manifest).read_text()); setup(m)
    signal.signal(signal.SIGUSR1, base.stop); signal.signal(signal.SIGTERM, base.stop)
    if args.stage == 'smoke':
        smoke(m)
    else:
        assert (Path(m['report']) / 'SMOKE_PASS.json').exists()
        base.Training = Training
        base.run(m, m['arms'][args.task])


if __name__ == '__main__':
    main()
