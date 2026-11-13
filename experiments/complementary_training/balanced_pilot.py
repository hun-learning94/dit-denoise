"""Initialization and diagnostics for the ImageNet128 SMLP stability pilot."""
from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset

from experiments.complementary_training.common import rng_state
from experiments.complementary_training.input import CachedImages, EpochBatches
from scripts import train_complementary as original


BASE_BUILD_MODEL = original.build_model


def common_signature(raw, *, baseline_compatible=False):
    signature = hashlib.sha256()
    for name, value in sorted(raw.state_dict().items()):
        if name in {'key_prototypes', 'value_prototypes'} or '.expansion.' in name:
            continue
        # The frozen baseline also excludes time_embed.mlp.*. Preserve its
        # checksum definition while checking all common tensors separately.
        if baseline_compatible and '.mlp.' in name:
            continue
        signature.update(name.encode())
        signature.update(value.detach().cpu().numpy().tobytes())
    return signature.hexdigest()


def balanced_model(arch, resolution, seed):
    assert arch == 'smlp' and resolution in (128, 256)
    raw, cfg, info = BASE_BUILD_MODEL(arch, resolution, seed)
    before_rng = torch.get_rng_state()
    before_common = common_signature(raw)
    assert common_signature(raw, baseline_compatible=True) == info['common_initialization_sha256']
    gk = (3 * cfg['prototype_key_width']) ** (1 / 6)
    gv = (3 * cfg['prototype_value_width'] / cfg['prototype_count']) ** (1 / 6)
    with torch.no_grad():
        raw.key_prototypes.mul_(gk)
        raw.value_prototypes.mul_(gv)
        for block in raw.blocks:
            expansion = block.expansion
            expansion.q_proj.weight.mul_(gk)
            expansion.q_proj.bias.mul_(gk)
            kb = expansion.kb_proj.weight.view(expansion.groups, expansion.group_dim + 1, -1)
            kb[:, :expansion.group_dim].mul_(gk)
            kb[:, expansion.group_dim].div_(gk)
            expansion.v_proj.weight.mul_(gv)
            expansion.out_proj.weight.mul_(gv)
            expansion.out_proj.bias.mul_(0.5)
    assert torch.equal(before_rng, torch.get_rng_state())
    assert common_signature(raw) == before_common, 'Initialization changed a common backbone tensor'
    assert common_signature(raw, baseline_compatible=True) == info['common_initialization_sha256']
    assert sum(p.numel() for p in raw.parameters()) == {128: 134011584, 256: 134896896}[resolution]
    info['initialization'] = dict(name='match effective predictor activation variance',
        key_path_per_factor_gain=gk, value_path_per_factor_gain=gv,
        generated_bias_projection_gain=1/gk, output_bias_gain=0.5,
        key_path_factors=['raw key bank', 'q projection', 'projected keys'],
        value_path_factors=['raw value bank', 'v projection', 'output projection'])
    return raw, cfg, info


@torch.no_grad()
def check_initializer():
    """Exercise the real initializer on CPU before allocating/caching images."""
    with torch.random.fork_rng(devices=[]):
        raw, _, info = balanced_model('smlp', 128, 0)
        ordinary, _, ordinary_info = BASE_BUILD_MODEL('mlp', 128, 0)
        assert info['common_initialization_sha256'] == ordinary_info['common_initialization_sha256']
        source = ordinary.state_dict()
        common = {k: v for k, v in raw.state_dict().items()
                  if k in source and source[k].shape == v.shape}
        assert all(torch.equal(v, source[k]) for k, v in common.items())
        assert any(k.startswith('time_embed.mlp.') for k in common)
        torch.manual_seed(20261020)
        query = torch.nn.functional.layer_norm(torch.randn(2, 64, 768), (768,))
        ratios = []
        for i in (0, 5, 11):
            expansion = raw.blocks[i].expansion
            expansion.fused = False
            a = ordinary.blocks[i].mlp(query)
            b = expansion(query, raw.key_prototypes, raw.value_prototypes)
            ac = (a-a.mean((0, 1), keepdim=True)).square().mean().sqrt()
            bc = (b-b.mean((0, 1), keepdim=True)).square().mean().sqrt()
            ratios.append((bc/ac).item())
        assert all(.95 < ratio < 1.05 for ratio in ratios), ratios
        result = dict(passed=True, parameters=info['parameters'],
            common_initialization_sha256=info['common_initialization_sha256'],
            all_common_tensors_equal_including_time_embedding=True,
            predictor_content_rms_ratios_to_mlp=ratios,
            initialization=info['initialization'])
    return result


class PilotImages(CachedImages):
    def __getstate__(self):
        # Reopen the mmap and recreate the lambda-containing torchvision
        # transform in each spawned worker. Only small metadata is serialized.
        return dict(self.__dict__, array=None, transform=None)

    def __setstate__(self, state):
        from data import build_transform
        self.__dict__.update(state)
        self.array = None
        self.transform = build_transform(self.resolution, hflip=False)


def check_input_pipeline():
    """Check actual spawn serialization and pixels without a full image cache."""
    import pickle
    import tempfile
    import numpy as np
    with tempfile.TemporaryDirectory(prefix='smlp-input-check-') as directory:
        path = Path(directory)/'images.npy'
        pixels = (np.arange(32*128*128*3, dtype=np.uint32) % 251).astype(np.uint8)
        np.save(path, pixels.reshape(32, 128, 128, 3))
        dataset = PilotImages(path)
        expected = torch.stack([dataset[i][0] for i in range(len(dataset))])
        payload = pickle.dumps(dataset)
        assert len(payload) < 4096, 'Serialized transform or mmap instead of metadata'
        restored = pickle.loads(payload)
        assert restored.array is None
        assert torch.equal(restored[0][0], expected[0])
        loader = DataLoader(dataset, batch_size=8, num_workers=2,
            multiprocessing_context='spawn',
            generator=torch.Generator().manual_seed(4000001))
        actual = torch.cat([batch for batch, _ in loader])
        assert actual.is_contiguous() and torch.equal(actual, expected)
        return dict(passed=True, spawn_workers=2, images=32,
            normalized_pixels_and_order_exact=True, serialized_dataset_bytes=len(payload),
            parent_mmap_opened_before_spawn=True)


class PilotTraining(original.Training):
    def __init__(self, args, dataset, cache):
        original.build_model = balanced_model
        super().__init__(args, dataset, cache)
        self.target_steps = args.max_updates
        self.recipe.update(epochs=None, planned_full_training_epochs=args.epochs,
            horizon='fixed number of optimizer updates', total_steps=self.target_steps,
            initialization=self.model_info['initialization'])
        self.recipe_hash = hashlib.sha256(json.dumps(self.recipe, sort_keys=True).encode()).hexdigest()
        self.last_gradient_norm = None

    def loader(self, step=None):
        step = self.step if step is None else step
        epoch, offset = divmod(step, self.steps_per_epoch)
        return DataLoader(self.dataset,
            batch_sampler=EpochBatches(len(self.dataset), self.args.batch, self.args.seed, epoch, offset),
            num_workers=self.args.workers, pin_memory=True,
            persistent_workers=self.args.workers > 0,
            **({'prefetch_factor': 2, 'multiprocessing_context': 'spawn'} if self.args.workers else {}),
            generator=torch.Generator().manual_seed(4000001+self.args.seed+epoch))

    def capture_update(self, reason, images, before_rng, loss=None, error=None):
        output = Path(self.args.outdir) / 'events'
        output.mkdir(parents=True, exist_ok=True)
        path = output / f'{reason}.pt'
        if path.exists():
            return
        # This state precedes the optimizer update. Restoring its RNG reproduces
        # the loss's time/noise draws on the stored, original contiguous batch.
        original.atomic_torch_save(dict(self.metadata(), net=self.canonical.state_dict(),
            opt=self.optimizer.state_dict(), ema=self.ema.shadow, rng=before_rng,
            data_position=dict(epoch=self.step//self.steps_per_epoch,
                               batch_offset=self.step % self.steps_per_epoch),
            input_images=images.detach().cpu(), event=reason, error=error,
            loss=None if loss is None else float(loss.detach()),
            gradient_norm=self.last_gradient_norm), path)

    def update(self, images):
        self.net.train()
        images = images.to('cuda', non_blocking=True)
        assert images.is_contiguous()
        assert not self.args.hflip
        before_rng = rng_state()
        self.last_gradient_norm = None
        self.optimizer.zero_grad(set_to_none=True)
        try:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = self.loss_fn(self.net, images).mean()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.canonical.parameters(), 1.0, error_if_nonfinite=True)
            self.last_gradient_norm = norm.item()
            if self.last_gradient_norm > 1000:
                self.capture_update('first_gradient_warning', images, before_rng, loss)
            if self.step >= 2000 and (float(loss.detach()) > .2 or self.last_gradient_norm > 5):
                self.capture_update('first_optimization_warning', images, before_rng, loss)
        except RuntimeError as error:
            self.capture_update('failed_update', images, before_rng, locals().get('loss'), str(error))
            raise
        self.optimizer.step()
        self.ema.update(self.canonical)
        self.step += 1
        return loss.detach()


@torch.no_grad()
def health(trainer, panel):
    """Online activations on a fixed noise probe, isolated from training RNG."""
    raw = trainer.raw
    branch = {}
    handles = []
    for i, block in enumerate(raw.blocks):
        def record(module, inputs, output, i=i):
            value = output.detach().float()
            branch[str(i)] = dict(rms=value.square().mean().sqrt().item(),
                                  max_abs=value.abs().max().item())
        handles.append(block.expansion.register_forward_hook(record))
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(93017)
            clean = torch.stack([panel[i][0] for i in range(8)]).cuda()
            times = torch.tensor([.05, .2, .35, .5, .65, .8, .9, .95], device='cuda').view(-1,1,1,1)
            noisy = times*clean+(1-times)*torch.randn_like(clean)*getattr(trainer.loss_fn, 'noise_scale', 1.0)
            raw.eval()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                condition = raw.time_embed(times.flatten())
                modulation = [block.adaLN(condition).chunk(6,-1) for block in raw.blocks]
                result = raw(noisy, times.flatten(), return_all=True)
            record = dict(step=trainer.step,
                time_embedding_rms=condition.float().square().mean().sqrt().item(),
                max_attention_gate=max(p[2].float().abs().max().item() for p in modulation),
                max_predictor_gate=max(p[5].float().abs().max().item() for p in modulation),
                max_predictor_scale=max(p[4].float().abs().max().item() for p in modulation),
                max_residual=max(x.float().abs().max().item() for x in result['states']),
                max_predictor_output=max(x['max_abs'] for x in branch.values()),
                branch_outputs=branch,
                raw_key_bank_rms=raw.key_prototypes.float().square().mean().sqrt().item(),
                raw_value_bank_rms=raw.value_prototypes.float().square().mean().sqrt().item(),
                finite=bool(torch.isfinite(result['final']).all()),
                weights='online', noise_seed=93017, noise_times=times.flatten().tolist())
            return record
    finally:
        for handle in handles:
            handle.remove()
        raw.train()


def preflight(args, dataset, cache, panel, report, production_data):
    import argparse
    from scripts.validate_complementary_training import Stream, compare
    args = argparse.Namespace(**dict(vars(args), outdir=str(report/'preflight_work')))
    trainer = PilotTraining(args, dataset, cache)
    checks = dict(parameters=trainer.model_info['parameters'],
        source_snapshot=str(original.CODE),
        initialization=trainer.model_info['initialization'], recipe_sha256=trainer.recipe_hash,
        common_initialization_sha256=trainer.model_info['common_initialization_sha256'],
        validation_scope='4096 real ImageNet images; full batch, model, optimizer, loader, and sampling',
        validation_dataset_count=len(dataset), validation_dataset_identity=cache['identity'])
    production_recipe = dict(trainer.recipe, **production_data)
    checks['production_recipe_sha256'] = hashlib.sha256(
        json.dumps(production_recipe, sort_keys=True).encode()).hexdigest()
    # Compare the actual initializer implementation with ordinary MLP branches.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        ordinary, _, _ = BASE_BUILD_MODEL('mlp', 128, 0)
        torch.manual_seed(20261020)
        query = torch.nn.functional.layer_norm(torch.randn(2,64,768), (768,)).cuda()
        ratios = []
        with torch.no_grad():
            for i in (0,5,11):
                expansion = trainer.raw.blocks[i].expansion
                expansion.fused = False
                a = ordinary.blocks[i].mlp.cuda()(query)
                b = expansion(query, trainer.raw.key_prototypes, trainer.raw.value_prototypes)
                ac = (a-a.mean((0,1),keepdim=True)).square().mean().sqrt()
                bc = (b-b.mean((0,1),keepdim=True)).square().mean().sqrt()
                ratios.append((bc/ac).item())
                expansion.fused = True
        assert all(.95 < ratio < 1.05 for ratio in ratios), ratios
        checks['predictor_content_rms_ratios_to_mlp'] = ratios
        del ordinary, query, a, b, ac, bc
    stream = Stream(trainer)
    for _ in range(8):
        trainer.update(stream.next())
    stream.close()
    checkpoint = report/'resume_probe.pt'
    trainer.save(checkpoint)
    expected_images, expected_losses = [], []
    for _ in range(3):
        images = stream.next()
        expected_images.append(images.clone())
        expected_losses.append(trainer.update(images).item())
    stream.close()
    expected = {k:v.clone() for k,v in trainer.canonical.state_dict().items()}
    expected_ema = {k:v.clone() for k,v in trainer.ema.shadow.items()}
    expected_rng = rng_state()
    trainer.load(checkpoint)
    before_probe = rng_state()
    online = {k:v.clone() for k,v in trainer.canonical.state_dict().items()}
    checks['health_probe'] = health(trainer, panel)
    checks['ema_probe'] = trainer.evaluate(Subset(panel, range(64)), report, sample=True)
    assert all(torch.equal(v,trainer.canonical.state_dict()[k]) for k,v in online.items())
    assert torch.equal(before_probe['cpu'],rng_state()['cpu'])
    assert torch.equal(before_probe['cuda'],rng_state()['cuda'])
    checks['health_and_ema_preserve_online_weights_and_rng'] = True
    del online
    actual_losses = []
    for images in expected_images:
        actual = stream.next()
        assert torch.equal(actual,images), 'Spawn loader changed data order or pixels'
        actual_losses.append(trainer.update(actual).item())
    stream.close()
    online_error = compare(trainer.canonical.state_dict(), expected)
    ema_error = compare(trainer.ema.shadow, expected_ema)
    assert online_error['relative_l2'] < 1e-5 and online_error['max_abs'] < 2e-5, online_error
    assert ema_error['relative_l2'] < 1e-5 and ema_error['max_abs'] < 2e-5, ema_error
    assert torch.equal(expected_rng['cpu'],rng_state()['cpu'])
    assert torch.equal(expected_rng['cuda'],rng_state()['cuda'])
    assert all(abs(a-b) < 1e-4*max(abs(a),abs(b),.01) for a,b in zip(actual_losses,expected_losses))
    checks.update(resume=dict(online=online_error, ema=ema_error,
        data_order_and_pixels_exact=True, rng_exact=True,
        uninterrupted_losses=expected_losses, resumed_losses=actual_losses),
        nonzero_raw_key_gradient=trainer.raw.key_prototypes.grad.norm().item()>0,
        nonzero_raw_value_gradient=trainer.raw.value_prototypes.grad.norm().item()>0,
        fresh_training_phase_starts_separately=True, passed=True)
    assert checks['nonzero_raw_key_gradient'] and checks['nonzero_raw_value_gradient']
    if getattr(args, 'projection_fix', None):
        from experiments.complementary_training.gradient_check import compare_compiled_gradients
        before_gradient_check = {k:v.detach().clone() for k,v in trainer.canonical.state_dict().items()}
        checks['compiled_gradient_check'] = compare_compiled_gradients(trainer, expected_images[0])
        assert all(torch.equal(v,trainer.canonical.state_dict()[k]) for k,v in before_gradient_check.items())
        assert torch.equal(expected_rng['cuda'],rng_state()['cuda'])
        del before_gradient_check
    original.publish(report/'preflight.json', checks)
    checkpoint.unlink()
    print('PREFLIGHT PASS', json.dumps(checks), flush=True)


def checkpoint(trainer, output):
    relative = Path('checkpoints')/f'step{trainer.step:09d}.pt'
    trainer.save(output/relative)
    original.publish(output/'latest.json', dict(checkpoint=str(relative),step=trainer.step,
        recipe_sha256=trainer.recipe_hash))
    # Keep all 5k milestones, initialization, and the two latest intermediate saves.
    intermediate = sorted(p for p in (output/'checkpoints').glob('step*.pt')
                          if int(p.stem[4:]) % 5000 != 0)
    for path in intermediate[:-2]:
        path.unlink()
    print('CHECKPOINT',trainer.step,relative,flush=True)


def train(args, dataset, cache, panel, report, stop):
    output = Path(args.outdir)
    trainer = PilotTraining(args, dataset, cache)
    assert trainer.recipe_hash == json.loads((report/'preflight.json').read_text())['production_recipe_sha256']
    original.publish(output/'config.json',dict(recipe=trainer.recipe,
        model_info=trainer.model_info,source_snapshot=str(original.CODE),
        dataloader_start_method='spawn; same sampler, private loader generator, and pixel transform',
        checkpoint_retention='step0, every5000, last2 intermediate1000-step saves, first warnings/failure',
        probe_thresholds=dict(gradient_norm=1000,predictor_gate=50,residual_max=100000)))
    latest = output/'latest.json'
    if latest.exists():
        trainer.load(output/json.loads(latest.read_text())['checkpoint'])
    def diagnostics():
        record = trainer.evaluate(panel,output)
        with (output/'diagnostics.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        print('DIAGNOSTIC',json.dumps(record),flush=True)
    def probe():
        record = health(trainer,panel)
        with (output/'health.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        print('HEALTH',json.dumps({k:v for k,v in record.items() if k!='branch_outputs'}),flush=True)
        if record['max_predictor_gate'] > 50 or record['max_residual'] > 100000 or not record['finite']:
            path = output/'events/first_activation_warning.pt'
            if not path.exists():trainer.save(path)
    if trainer.step == 0:
        checkpoint(trainer,output)
        probe()
        diagnostics()
    losses, norms = [], []
    previous, logged_step = time.monotonic(), trainer.step
    loader = iterator = None
    try:
        while trainer.step < trainer.target_steps:
            loader = trainer.loader()
            iterator = iter(loader)
            for images,_ in iterator:
                if stop['requested']:
                    checkpoint(trainer,output)
                    return 75
                loss = trainer.update(images)
                losses.append(loss.item());norms.append(trainer.last_gradient_norm)
                if trainer.step % 100 == 0:
                    now = time.monotonic()
                    record = dict(status='training',step=trainer.step,target_steps=trainer.target_steps,
                        epoch_equivalent=trainer.step/trainer.steps_per_epoch,
                        loss=sum(losses)/len(losses),max_loss=max(losses),
                        gradient_norm_mean=sum(norms)/len(norms),gradient_norm_max=max(norms),
                        seconds_per_update=(now-previous)/(trainer.step-logged_step))
                    assert math.isfinite(record['loss'])
                    original.publish(output/'progress.json',record)
                    print('PROGRESS',json.dumps(record),flush=True)
                    losses,norms,previous,logged_step=[],[],now,trainer.step
                if trainer.step % 1000 == 0:probe()
                if trainer.step % 5000 == 0:diagnostics()
                if trainer.step % 1000 == 0 or trainer.step == trainer.target_steps:
                    checkpoint(trainer,output)
                if trainer.step >= trainer.target_steps or stop['requested']:break
            if hasattr(iterator,'_shutdown_workers'):iterator._shutdown_workers()
            iterator = loader = None
            if stop['requested']:
                checkpoint(trainer,output)
                return 75
    except Exception as error:
        original.publish(output/'ERROR.json',dict(step=trainer.step,error=repr(error)))
        raise
    finally:
        if iterator is not None and hasattr(iterator,'_shutdown_workers'):iterator._shutdown_workers()
    original.atomic_torch_save(dict(trainer.metadata(),ema=trainer.ema.shadow,net=trainer.ema.shadow,
        weights='EMA only; diagnostic pilot endpoint'),output/'ema_final.pt')
    original.publish(output/'DONE.json',dict(step=trainer.step,target_steps=trainer.target_steps,
        epoch_equivalent=trainer.step/trainer.steps_per_epoch,recipe_sha256=trainer.recipe_hash,
        final_ema='ema_final.pt',purpose=getattr(args,'pilot_purpose','50k initialization stability pilot')))
    original.publish(output/'progress.json',dict(status='completed',step=trainer.step,
        target_steps=trainer.target_steps))
    return 0
