"""Half-text-width locality figure, using completed clean-target denoising losses."""

import hashlib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator

from settings import DATA, OUT

EPSILON = 0.003
MAX_NOISE = 32.0
colors = {"jit_b": "#0072B2", "jit_l": "#D55E00"}
labels = {"jit_b": "JiT-B", "jit_l": "JiT-L"}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 6.6,
        "axes.labelsize": 6.6,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6.5,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
    }
)
fig, axes = plt.subplots(1, 2, figsize=(2.8, 1.27))
rows = []
provenance = {}
for name in colors:
    p = DATA / f"{name}_locality.npz"
    with np.load(p) as z:
        d = {k: z[k] for k in z.files}
    k = d["budgets"]
    sigma = d["sigmas"]
    loss = d["mean_clean_mse"]
    dense = d["dense_clean_mse"]
    assert loss.shape == (len(sigma), 256) and np.isfinite(loss).all()
    np.testing.assert_array_equal(loss[:, -1], dense)
    excess = loss - dense[:, None]
    envelope = np.maximum.accumulate(excess[:, ::-1], axis=1)[:, ::-1]
    assert np.all(envelope[:, -1] <= EPSILON)
    needed = np.argmax(envelope <= EPSILON, axis=1) + 1
    middle = int(np.flatnonzero(sigma == 1)[0])
    axes[0].plot(k, excess[middle], color=colors[name], lw=1.2, label=labels[name])
    visible = sigma <= MAX_NOISE
    time = 1 / (1 + sigma[visible])
    axes[1].plot(time, needed[visible], color=colors[name], lw=1.2)
    for i, s in enumerate(sigma):
        rows.append(
            dict(
                model=name,
                sigma=float(s),
                t=float(1 / (1 + s)),
                excess_clean_mse_tolerance=EPSILON,
                required_patches=int(needed[i]),
            )
        )
    provenance[name] = dict(
        aggregate_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        required_counts=needed.tolist(),
        displayed_sigmas=sigma[visible].tolist(),
        displayed_required_counts=needed[visible].tolist(),
        left_sigma=1.0,
        left_t=0.5,
    )
axes[0].set(
    xlabel="Retained patches $k$",
    ylabel="Excess MSE",
    xlim=(1, 256),
    title=r"(a) $t=0.5$",
)
axes[0].set_xscale("log", base=2)
axes[0].set_xticks([1, 16, 256], ["1", "16", "256"])
axes[0].axhline(0, color=".55", lw=0.55, ls="--", zorder=0)
axes[0].yaxis.set_major_locator(MaxNLocator(3))
formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-2, 2))
formatter.set_useOffset(False)
axes[0].yaxis.set_major_formatter(formatter)
axes[1].set(
    xlabel=r"Time $t$",
    ylabel="Context size $k$",
    xlim=(1, 1 / (1 + MAX_NOISE)),
    ylim=(0, 256),
    title=r"(b) $\epsilon=0.003$",
)
axes[1].set_xticks(
    [1, 0.75, 0.5, 0.25, 1 / (1 + MAX_NOISE)], ["1", ".75", ".5", ".25", ".03"]
)
axes[1].set_yticks([0, 128, 256])
for ax in axes:
    ax.set_title(ax.get_title(), y=1.20, pad=0.6)
    ax.grid(alpha=0.16, lw=0.4)
    ax.tick_params(pad=1)
    ax.xaxis.labelpad = 1.4
    ax.yaxis.labelpad = 1.4
fig.subplots_adjust(left=0.14, right=0.99, bottom=0.285, top=0.82, wspace=0.52)
fig.legend(
    *axes[0].get_legend_handles_labels(),
    loc="lower center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.53, 0.005),
    borderaxespad=0,
    handlelength=1.3,
    columnspacing=1,
)
for ext in ["png", "pdf"]:
    fig.savefig(
        OUT / f"locality_summary.{ext}", dpi=350, bbox_inches="tight", pad_inches=0.003
    )
plt.close(fig)
