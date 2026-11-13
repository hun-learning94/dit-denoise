"""Shared JiT optimization for the existing paired GELU-MLP/SMLP backbones."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from edm import VPVelocityXPredLoss
from experiments.complementary_training.balanced_pilot import PilotTraining, health
from experiments.complementary_training.common import EMA, rng_state
from experiments.complementary_training.corrected_campaign import build_corrected_model, FIX
from scripts import train_complementary as original


BASE_TRAINING = original.Training


def learning_rate_at_step(step, steps_per_epoch, base_lr=2e-4, warmup_epochs=5):
    """Official JiT fractional-epoch schedule, including LR=0 on update zero."""
    assert step >= 0 and steps_per_epoch > 0 and warmup_epochs > 0
    return base_lr * min(1.0, step / (warmup_epochs * steps_per_epoch))


class JiTEMA(EMA):
    @torch.no_grad()
    def update(self, canonical):
        for name, value in canonical.state_dict().items():
            self.shadow[name].mul_(0.9999).add_(value.detach().float(), alpha=0.0001)


class JiTTraining(BASE_TRAINING):
    def __init__(self, args, dataset, cache):
        assert args.batch == args.expected_batch == 1024
        assert args.epochs in (200, 400) and args.lr == 2e-4 and args.hflip
        original.build_model = build_corrected_model
        super().__init__(args, dataset, cache)
        self.optimizer = torch.optim.AdamW(self.canonical.parameters(), lr=0.,
            betas=(0.9, 0.95), eps=1e-8, weight_decay=0.)
        self.ema = JiTEMA(self.canonical)
        # Table 9's image_size/256 rule; 128 is an explicit extension.
        self.loss_fn = VPVelocityXPredLoss(-0.8, 0.8, 0.05,
            noise_scale=self.resolution / 256)
        self.recipe.update(training_recipe='JiT-based, shared between MLP and SMLP',
            betas=[0.9, 0.95], grad_clip=None, ema_decay=0.9999,
            scheduler='linear warmup for five epochs, then constant',
            warmup_epochs=5, warmup_steps=5*self.steps_per_epoch,
            warmup_first_update_lr=0., gradient_accumulation=1,
            noise_scale=self.loss_fn.noise_scale,
            noise_scale_rule='image_size/256; 128-resolution extrapolation',
            augmentation='random horizontal flip, probability0.5',
            output_head_precision='FP32 for both architectures',
            gradient_comparison_policy='recorded historical BF16 warning; no 3% launch or 50k gate',
            conditioning='unconditional; no labels or CFG',
            ema_selection='fixed0.9999; no EMA sweep',
            jit_sources=[
                'https://arxiv.org/html/2511.13720#A1',
                'https://github.com/LTH14/JiT/blob/main/main_jit.py',
                'https://github.com/LTH14/JiT/blob/main/engine_jit.py',
                'https://github.com/LTH14/JiT/blob/main/util/lr_sched.py'])
        if args.arch == 'smlp':
            assert self.recipe['projection_fix'] == FIX
            self.recipe['initialization'] = self.model_info['initialization']
        self.recipe_hash = hashlib.sha256(json.dumps(self.recipe, sort_keys=True).encode()).hexdigest()
        self.last_gradient_norm = None
        self._optimization_window = []

    capture_update = PilotTraining.capture_update

    def metadata(self):
        record = super().metadata()
        record['vp_noise_scale'] = self.loss_fn.noise_scale
        return record

    def update(self, images):
        self.net.train()
        images = images.to('cuda', non_blocking=True)
        assert images.is_contiguous()
        before_rng = rng_state()
        lr = learning_rate_at_step(self.step, self.steps_per_epoch, self.args.lr)
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.last_gradient_norm = None
        self.optimizer.zero_grad(set_to_none=True)
        try:
            mask = torch.rand(len(images), device='cuda') < .5
            augmented = torch.where(mask[:, None, None, None], images.flip(-1), images)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = self.loss_fn(self.net, augmented).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            parameters = [p for p in self.canonical.parameters() if p.requires_grad]
            if any(p.grad is None for p in parameters):
                raise RuntimeError('Missing trainable gradient')
            # Measure without clipping or modifying the JiT optimizer update.
            norms = torch._foreach_norm([p.grad.detach() for p in parameters], 2)
            norm = torch.linalg.vector_norm(torch.stack(norms), 2)
            self.last_gradient_norm = norm.item()
            if not bool(torch.isfinite(norm)):
                raise RuntimeError('Nonfinite training gradient')
            if self.step >= 2000 and (float(loss.detach()) > .2 or self.last_gradient_norm > 5):
                self.capture_update('first_optimization_warning', images, before_rng, loss)
        except RuntimeError as error:
            self.capture_update('failed_update', images, before_rng,
                locals().get('loss'), str(error))
            raise
        self.optimizer.step()
        self.ema.update(self.canonical)
        self.step += 1
        self._optimization_window.append((loss.item(), self.last_gradient_norm))
        if self.step % 100 == 0:
            values, norms = zip(*self._optimization_window)
            record = dict(step=self.step, mean_loss=sum(values)/len(values),
                max_loss=max(values), gradient_norm_mean=sum(norms)/len(norms),
                gradient_norm_max=max(norms), updates=len(values), lr=lr,
                epoch_equivalent=self.step/self.steps_per_epoch)
            output = Path(self.args.outdir)
            output.mkdir(parents=True, exist_ok=True)
            with (output/'optimization.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            print('OPTIMIZATION', json.dumps(record), flush=True)
            self._optimization_window.clear()
        return loss.detach()

    def evaluate(self, panel, output, *, sample=True):
        result = super().evaluate(panel, output, sample=sample)
        result['noise_scale'] = self.loss_fn.noise_scale
        if self.args.arch == 'smlp':
            probe = health(self, panel)
            with (Path(output)/'health.jsonl').open('a') as stream:
                stream.write(json.dumps(probe)+'\n')
            print('HEALTH', json.dumps({k:v for k,v in probe.items() if k!='branch_outputs'}), flush=True)
            assert probe['finite'], 'Nonfinite online health probe'
        return result
