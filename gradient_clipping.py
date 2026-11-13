"""Preserve finite-gradient clipping when an FP32 norm calculation overflows."""
import torch


@torch.no_grad()
def clip_grad_norm_stable_(parameters, max_norm, *, step=None):
    parameters = list(parameters)
    try:
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    except RuntimeError as error:
        if 'non-finite' not in str(error):
            raise
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        # Never turn NaN/Inf gradients into an optimizer update.
        if not gradients or not all(torch.isfinite(gradient).all().item() for gradient in gradients):
            raise
        norm = torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(gradient.double(), 2) for gradient in gradients
        ]), 2)
        if not torch.isfinite(norm).item():
            raise
        coefficient = (float(max_norm) / (norm + 1e-6)).clamp(max=1.0)
        for gradient in gradients:
            gradient.copy_((gradient.double() * coefficient).to(gradient.dtype))
        print(f'[gradient clipping] step={step} finite-gradient norm overflow; '
              f'FP64 norm={norm.item():.9g} coefficient={coefficient.item():.9g}', flush=True)
        return norm
