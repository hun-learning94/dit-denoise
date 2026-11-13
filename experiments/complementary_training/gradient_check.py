"""Full-batch checks at a trained, nonzero-gate pilot state."""
import torch

from experiments.complementary_training.common import rng_state, restore_rng


def compare_compiled_gradients(trainer, images):
    """Compare the real update graphs against FP32 without changing training."""
    state = rng_state()
    images = images.cuda().contiguous()
    results = []
    references = {}
    try:
        for name, model, amp in (("eager_fp32", trainer.canonical, False),
                                 ("eager_bf16", trainer.canonical, True),
                                 ("compiled_bf16", trainer.net, True)):
            restore_rng(state)
            trainer.optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = trainer.loss_fn(model, images).mean()
            loss.backward()
            gradients = {key: p.grad.detach().cpu().clone()
                         for key, p in trainer.canonical.named_parameters() if p.grad is not None}
            required = {key for key, p in trainer.canonical.named_parameters() if p.requires_grad}
            assert set(gradients) == required, dict(missing=sorted(required-set(gradients)))
            assert all(bool(torch.isfinite(g).all()) for g in gradients.values())
            assert float(gradients["model.key_prototypes"].norm()) > 1e-10
            assert float(gradients["model.value_prototypes"].norm()) > 1e-10
            row = dict(mode=name, loss=float(loss), comparisons={},
                       gradient_norm=sum(float(g.double().square().sum()) for g in gradients.values())**.5)
            for reference_name, previous in references.items():
                difference = total = 0.
                banks, parameters = {}, {}
                for key, value in gradients.items():
                    d = float((value.double()-previous[key].double()).square().sum())
                    r = float(previous[key].double().square().sum())
                    difference += d
                    total += r
                    parameters[key] = dict(relative_l2=(d/max(r, 1e-300))**.5,
                        reference_l2=r**.5, difference_l2=d**.5)
                    if key in ("model.key_prototypes", "model.value_prototypes"):
                        banks[key] = (d/max(r, 1e-300))**.5
                row['comparisons'][reference_name] = dict(relative_l2=(difference/total)**.5,
                    shared_bank_relative_l2=banks, parameters=parameters)
            if name != 'compiled_bf16':
                references[name] = gradients
            results.append(row)
        ordinary = results[1]['comparisons']['eager_fp32']
        corrected = results[2]['comparisons']['eager_fp32']
        assert results[2]['comparisons']['eager_bf16']['relative_l2'] < .03, results
        assert corrected['relative_l2'] <= max(.03, 1.25*ordinary['relative_l2']+.01), results
        for name, error in corrected['shared_bank_relative_l2'].items():
            assert error <= 1.25*ordinary['shared_bank_relative_l2'][name]+.01, results
        failures = {}
        for name, error in corrected['parameters'].items():
            if error['reference_l2'] <= 1e-10:
                assert error['difference_l2'] < 1e-8, (name, error)
            elif error['relative_l2'] > max(.03, 1.25*ordinary['parameters'][name]['relative_l2']+.01):
                failures[name] = dict(compiled=error, ordinary_bf16=ordinary['parameters'][name])
        assert not failures, failures
        assert abs(results[2]['loss']/results[1]['loss']-1) < .005, results
    finally:
        trainer.optimizer.zero_grad(set_to_none=True)
        restore_rng(state)
    return dict(passed=True, step=trainer.step, batch=len(images), modes=results,
                nonzero_shared_bank_gradients=True, optimizer_and_ema_unchanged=True,
                rng_restored=True,
                per_parameter_gate='FP32-relative error <= max(3%, 1.25*ordinary_BF16_error+1%); shared-bank bounds unchanged')
