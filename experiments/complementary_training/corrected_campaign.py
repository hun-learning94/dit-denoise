"""Matched LR-1e-4 campaign and explicit continuation of the corrected pilot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from experiments.complementary_training.balanced_pilot import (
    BASE_BUILD_MODEL, PilotTraining, balanced_model, health,
)
from experiments.complementary_training.common import restore_rng
from scripts import train_complementary as original


FIX = 'packed_key_value_partial_and_output_gradients_v2'
BASE_TRAINING = original.Training


def recipe_hash(recipe):
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()


def build_corrected_model(arch, resolution, seed):
    raw, cfg, info = (balanced_model if arch == 'smlp' else BASE_BUILD_MODEL)(arch, resolution, seed)
    raw.cfg.output_head_fp32 = True
    cfg['output_head_fp32'] = True
    info['output_head_precision'] = 'fp32; identical policy for MLP and SMLP'
    signature = hashlib.sha256()
    for name, value in sorted(raw.state_dict().items()):
        if name in {'key_prototypes', 'value_prototypes'} or '.expansion.' in name:
            continue
        if name.startswith('blocks.') and '.mlp.' in name:
            continue
        signature.update(name.encode())
        signature.update(value.detach().cpu().numpy().tobytes())
    info['all_common_initialization_sha256'] = signature.hexdigest()
    return raw, cfg, info


class CorrectedTraining(BASE_TRAINING):
    def __init__(self, args, dataset, cache):
        original.build_model = build_corrected_model
        super().__init__(args, dataset, cache)
        assert self.recipe['lr'] == 1e-4 and not args.hflip
        if args.arch == 'smlp':
            assert self.recipe['projection_fix'] == FIX
            self.recipe['initialization'] = self.model_info['initialization']
        self.recipe_hash = recipe_hash(self.recipe)
        self.last_gradient_norm = None
        self._optimization_window = []

    # Same optimization as the validated pilot, including warning capture.
    capture_update = PilotTraining.capture_update

    def update(self, images):
        loss = PilotTraining.update(self, images)
        self._optimization_window.append((float(loss), self.last_gradient_norm))
        if self.step % 100 == 0:
            values, norms = zip(*self._optimization_window)
            record = dict(step=self.step, mean_loss=sum(values)/len(values),
                max_loss=max(values), gradient_norm_mean=sum(norms)/len(norms),
                gradient_norm_max=max(norms), updates=len(values))
            output = Path(self.args.outdir)
            output.mkdir(parents=True, exist_ok=True)
            with (output/'optimization.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            print('OPTIMIZATION', json.dumps(record), flush=True)
            self._optimization_window.clear()
        return loss

    def load(self, path):
        path = Path(path).resolve()
        pilot = getattr(self.args, 'pilot_continuation', None)
        if not pilot or path != Path(pilot['checkpoint']).resolve():
            return super().load(path)
        # Only the declared 50k pilot may change horizon. Every optimization,
        # initialization, dataset and precision field must remain identical.
        saved = torch.load(path, map_location='cpu', weights_only=False)
        assert saved['step'] == pilot['step'] == 50000
        assert saved['recipe_sha256'] == pilot['recipe_sha256'] == recipe_hash(saved['recipe'])
        assert saved['recipe'] == pilot['recipe']
        previous = dict(saved['recipe'])
        assert previous.pop('planned_full_training_epochs') == self.args.epochs == 200
        assert previous.pop('horizon') == 'fixed number of optimizer updates'
        assert previous['epochs'] is None and previous['total_steps'] == 50000
        previous.update(epochs=self.args.epochs, total_steps=self.target_steps)
        assert previous == self.recipe, 'Pilot continuation changed more than its training horizon'
        assert {'net', 'opt', 'ema', 'rng', 'data_position'} <= saved.keys()
        self.canonical.load_state_dict(saved['net'], strict=True)
        self.optimizer.load_state_dict(saved['opt'])
        assert all(g['lr'] == 1e-4 for g in self.optimizer.param_groups)
        self.ema.shadow = {k: v.to('cuda', dtype=torch.float32) for k, v in saved['ema'].items()}
        self.step = int(saved['step'])
        assert saved['data_position'] == dict(epoch=self.step//self.steps_per_epoch,
            batch_offset=self.step % self.steps_per_epoch)
        restore_rng(saved['rng'])
        audit = dict(checkpoint=str(path), checkpoint_sha256=pilot['checkpoint_sha256'],
            step=self.step, previous_recipe=saved['recipe'], production_recipe=self.recipe,
            previous_recipe_sha256=saved['recipe_sha256'], production_recipe_sha256=self.recipe_hash,
            restored=['online weights', 'AdamW state', 'EMA', 'all training RNG', 'data position'],
            changed_fields=['epochs', 'total_steps', 'horizon metadata'],
            optimizer_schedule='constant; no warmup; unchanged from update zero')
        original.publish(Path(self.args.outdir)/'pilot_continuation.json', audit)
        print('PILOT CONTINUATION', json.dumps(audit), flush=True)

    def evaluate(self, panel, output, *, sample=True):
        record = super().evaluate(panel, output, sample=sample)
        if self.args.arch == 'smlp':
            probe = health(self, panel)
            with (Path(output)/'health.jsonl').open('a') as stream:
                stream.write(json.dumps(probe)+'\n')
            print('HEALTH', json.dumps({k: v for k, v in probe.items() if k != 'branch_outputs'}), flush=True)
            assert probe['finite'], 'Nonfinite online health probe'
            if self.step == 50000:
                from experiments.complementary_training.gradient_check import compare_compiled_gradients
                images = torch.stack([panel[i][0] for i in range(256)])
                try:
                    checked = compare_compiled_gradients(self, images)
                except Exception:
                    self.save(Path(output)/'events/compiled_gradient_failure_step50000.pt')
                    raise
                original.publish(Path(output)/'compiled_gradients_step50000.json', checked)
        return record
