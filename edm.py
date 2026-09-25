import math
import numpy as np
import torch

"""
EDM sampler, copied from https://github.com/NVlabs/edm/generate.py
"""
def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, # on reverse SDE sampling. by default (S_churn=0), do deterministic Heun
    solver="heun",
):
    if solver not in {"euler", "heun"}:
        raise ValueError(f"solver must be 'euler' or 'heun', got {solver!r}")
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        t_cur_val = float(t_cur)
        gamma = min(S_churn / num_steps, math.sqrt(2) - 1) if (S_min <= t_cur_val <= S_max) else 0.0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction only for Heun.  Euler deliberately uses
        # exactly one denoiser evaluation per discretization interval.
        if solver == "heun" and i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


@torch.no_grad()
def geometric_heun_sampler(
    net, latents, class_labels=None, num_steps=50,
    t_min=0.05, t_max=4.0,
):
    """VE probability-flow sampler on a geometric positive-time grid.

    ``num_steps`` is the number of positive levels, including both endpoints.
    The 49 positive-to-positive intervals use Heun when ``num_steps=50``; the
    appended final interval to zero is Euler and equals the last denoiser.
    """
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")
    if not (0 < t_min < t_max):
        raise ValueError("require 0 < t_min < t_max")
    indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = t_max * (t_min / t_max) ** (indices / (num_steps - 1))
    x_next = latents.to(torch.float64) * t_steps[0]
    for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
        x_cur = x_next
        denoised = net(x_cur, t_cur, class_labels).to(torch.float64)
        d_cur = (x_cur - denoised) / t_cur
        proposal = x_cur + (t_next - t_cur) * d_cur
        denoised_next = net(proposal, t_next, class_labels).to(torch.float64)
        d_next = (proposal - denoised_next) / t_next
        x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_next)
    # Euler from t_min to zero: algebraically the denoiser at t_min.
    return net(x_next, t_steps[-1], class_labels).to(torch.float64)


@torch.no_grad()
def vp_heun_sampler(net, latents, class_labels=None, num_steps=50, denominator_floor=0.05):
    """Li--He VP/interpolant sampler: linear time grid from noise (0) to data (1)."""
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")
    if not (0 < denominator_floor <= 1):
        raise ValueError("denominator_floor must be in (0, 1]")
    times = torch.linspace(0, 1, num_steps + 1, dtype=torch.float64, device=latents.device)
    state = latents.to(torch.float64)
    for index, (t_cur, t_next) in enumerate(zip(times[:-1], times[1:])):
        x_pred = net(state, t_cur, class_labels).to(torch.float64)
        denom = max(1.0 - float(t_cur), denominator_floor)
        velocity = (x_pred - state) / denom
        proposal = state + (t_next - t_cur) * velocity
        # Match JiT: Heun for the first N-1 intervals, final interval Euler.
        if index == num_steps - 1:
            state = proposal
            continue
        x_pred_next = net(proposal, t_next, class_labels).to(torch.float64)
        denom_next = max(1.0 - float(t_next), denominator_floor)
        velocity_next = (x_pred_next - proposal) / denom_next
        state = state + (t_next - t_cur) * (0.5 * velocity + 0.5 * velocity_next)
    return state

"""
Preconditioning scheme, copied from https://github.com/NVlabs/edm/training/networks.py
"""
class EDMPrecond(torch.nn.Module):
    def __init__(self,
        backbone: torch.nn.Module,   # Class name of the underlying model backbone.
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = backbone

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        # sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        B = x.shape[0]
        sigma = sigma.to(torch.float32)
        sigma = sigma.reshape(-1, 1, 1, 1)
        if sigma.shape[0] == 1 and B != 1:
            sigma = sigma.expand(B, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([x.shape[0], self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        F_x = F_x.to(dtype)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


class XPredPrecond(torch.nn.Module):
    """Input-normalized backbone whose raw output is the clean-image estimate."""

    prediction_type = "x"

    def __init__(self, backbone, img_resolution, img_channels, label_dim=0,
                 sigma_data=0.5, t_min=0.05, t_max=4.0):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.sigma_data = float(sigma_data)
        self.sigma_min = float(t_min)
        self.sigma_max = float(t_max)
        self.model = backbone

    def forward(self, x, sigma, class_labels=None, **model_kwargs):
        x = x.to(torch.float32)
        sigma = torch.as_tensor(sigma, device=x.device, dtype=torch.float32).reshape(-1, 1, 1, 1)
        if sigma.shape[0] == 1 and x.shape[0] != 1:
            sigma = sigma.expand(x.shape[0], 1, 1, 1)
        if sigma.shape[0] != x.shape[0]:
            raise ValueError("sigma must be scalar or have one value per image")
        labels = None if self.label_dim == 0 else class_labels
        c_in = (self.sigma_data ** 2 + sigma.square()).rsqrt()
        c_noise = sigma.log().flatten() / 4
        return self.model(c_in * x, c_noise, class_labels=labels, **model_kwargs)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


class VPXPred(torch.nn.Module):
    """Li--He direct-x backbone on z_t = t*x + (1-t)*epsilon."""

    prediction_type = "x"
    noise_process = "vp"
    sigma_min = 0.0
    sigma_max = 1.0

    def __init__(self, backbone, img_resolution, img_channels, label_dim=0):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.model = backbone

    def forward(self, z, t, class_labels=None, **model_kwargs):
        z = z.to(torch.float32)
        t = torch.as_tensor(t, device=z.device, dtype=torch.float32).reshape(-1)
        if t.numel() == 1 and z.shape[0] != 1:
            t = t.expand(z.shape[0])
        if t.numel() != z.shape[0]:
            raise ValueError("t must be scalar or have one value per image")
        labels = None if self.label_dim == 0 else class_labels
        return self.model(z, t, class_labels=labels, **model_kwargs)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

"""
EDM loss function, copied from https://github.com/NVlabs/edm/training/loss.py
"""
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss


class VelocityXPredLoss:
    """Unweighted VE velocity MSE with a direct clean-image prediction."""

    def __init__(self, t_min=0.05, t_max=4.0, time_distribution="uniform_t"):
        if not (0 < t_min < t_max):
            raise ValueError("require 0 < t_min < t_max")
        if time_distribution not in {"uniform_t", "uniform_t2"}:
            raise ValueError("time_distribution must be uniform_t or uniform_t2")
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.time_distribution = time_distribution

    def sample_t(self, batch_size, device):
        u = torch.rand([batch_size, 1, 1, 1], device=device, dtype=torch.float32)
        if self.time_distribution == "uniform_t":
            return self.t_min + u * (self.t_max - self.t_min)
        return (self.t_min ** 2 + u * (self.t_max ** 2 - self.t_min ** 2)).sqrt()

    def __call__(self, net, images, labels=None, augment_pipe=None):
        clean, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        clean_f = clean.float()
        t = self.sample_t(clean.shape[0], clean.device)
        eps = torch.randn_like(clean_f)
        noisy = clean_f + t * eps
        x_pred = net(noisy, t, labels, augment_labels=augment_labels)
        # Keep subtraction/division and loss accumulation in FP32 under AMP.
        v_pred = (noisy - x_pred.float()) / t
        return (v_pred - eps).square()


class VPVelocityXPredLoss:
    """Exact Li--He direct-x velocity loss for z_t=t*x+(1-t)*epsilon."""

    def __init__(self, logit_mean=-0.8, logit_std=0.8, denominator_floor=0.05):
        if logit_std <= 0:
            raise ValueError("logit_std must be positive")
        if not (0 < denominator_floor <= 1):
            raise ValueError("denominator_floor must be in (0, 1]")
        self.logit_mean = float(logit_mean)
        self.logit_std = float(logit_std)
        self.denominator_floor = float(denominator_floor)

    def sample_t(self, batch_size, device):
        logits = torch.randn([batch_size, 1, 1, 1], device=device, dtype=torch.float32)
        return (logits * self.logit_std + self.logit_mean).sigmoid()

    def __call__(self, net, images, labels=None, augment_pipe=None):
        clean, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        clean_f = clean.float()
        t = self.sample_t(clean.shape[0], clean.device)
        eps = torch.randn_like(clean_f)
        z = t * clean_f + (1 - t) * eps
        x_pred = net(z, t.flatten(), labels, augment_labels=augment_labels)
        denom = (1 - t).clamp_min(self.denominator_floor)
        v_target = (clean_f - z) / denom
        v_pred = (x_pred.float() - z) / denom
        return (v_pred - v_target).square()
