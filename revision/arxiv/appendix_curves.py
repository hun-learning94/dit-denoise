"""Appendix curve panels from unchanged, saved evaluation arrays."""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import Formatter, MaxNLocator, ScalarFormatter
import numpy as np

from settings import INPUTS, OUT

curves = {}


class ScientificTicks(Formatter):
    def __init__(self, exponent):
        self.exponent = exponent

    def __call__(self, value, pos=None):
        scaled = value / 10.0**self.exponent
        if abs(scaled) < 0.05:
            scaled = 0.0
        return self.fix_minus(f"{scaled:.1f}")

    def get_offset(self):
        return rf"$\times 10^{{{self.exponent}}}$"


# The nine-noise context sweep retains all 256 evaluated budgets.
data = {}
for model in ["jit_b", "jit_l"]:
    path = INPUTS / f"{model}_original.npz"
    with np.load(path, allow_pickle=False) as archive:
        data[model] = {key: archive[key].copy() for key in archive.files}
    assert data[model]["mean_clean_mse"].shape == (9, 256)
    np.testing.assert_array_equal(
        data[model]["mean_clean_mse"][:, -1], data[model]["dense_clean_mse"]
    )
colors, labels = (
    dict(jit_b="#0072B2", jit_l="#D55E00"),
    dict(jit_b="JiT-B", jit_l="JiT-L"),
)
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig, axes = plt.subplots(3, 3, figsize=(10.2, 6.1), sharex=True)
for i, (sigma, ax) in enumerate(zip(data["jit_b"]["sigmas"], axes.flat)):
    for model, values in data.items():
        y = values["mean_clean_mse"][i] - values["dense_clean_mse"][i]
        (line,) = ax.plot(
            values["budgets"], y, color=colors[model], lw=1.8, label=labels[model]
        )
        np.testing.assert_array_equal(line.get_ydata(), y)
        curves[f"locality_{model}_{i}"] = [values["budgets"].tolist(), y.tolist()]
    ax.axhline(0, color=".4", ls="--", lw=0.8, zorder=0)
    ax.set_xscale("log", base=2)
    ax.set_xlim(1, 256)
    ax.set_xticks([1, 4, 16, 64, 256], ["1", "4", "16", "64", "256"])
    ax.set_title(rf"$\sigma={sigma:g}$  ($t={1 / (1 + sigma):.2f}$)")
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 3))
    formatter.set_useOffset(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.grid(alpha=0.16)
    if i // 3 == 2:
        ax.set_xlabel("Retained nearest patches")
axes[0, 0].legend(frameon=False, fontsize=9)
fig.supylabel("Excess clean-image denoising MSE", x=0.007)
fig.tight_layout(pad=0.55, h_pad=0.6, w_pad=0.6, rect=(0.02, 0, 1, 1))
for suffix in ["pdf", "png"]:
    fig.savefig(
        OUT / f"denoising_excess_context.{suffix}",
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.03,
    )
plt.close(fig)

# The six-panel pooled-Wiener control retains the original points and colors.
path = INPUTS / "pooled.csv"
with path.open() as handle:
    rows = list(csv.DictReader(handle))
pooled = {}
for model in ["jit_b", "jit_l"]:
    path = INPUTS / f"{model}_pooled.npz"
    with np.load(path, allow_pickle=False) as archive:
        pooled[model] = {
            key: archive[key].copy()
            for key in ["excess_mean", "times", "budgets", "methods"]
        }
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 11,
        "axes.spines.top": True,
        "axes.spines.right": True,
    }
)
fig, axes = plt.subplots(2, 3, figsize=(12, 7.4), layout="constrained")
methods = [
    "kamb_geometry",
    "niedoba_donor",
    "lukoianov_covariance",
    "attention_calibrated",
    "attention_current",
    "self_attention_calibrated",
    "self_attention_current",
    "wiener_pooled",
]
legend = {}
for ri, recipient in enumerate(["jit_b", "jit_l"]):
    donor = "jit_l" if recipient == "jit_b" else "jit_b"
    for col, t in enumerate([0.1, 0.5, 0.9]):
        ax, shown = axes[ri, col], []
        for index, method in enumerate(methods):
            selected = sorted(
                [
                    r
                    for r in rows
                    if r["recipient"] == recipient
                    and float(r["t"]) == t
                    and r["method"] == method
                    and 12 <= int(r["budget"]) <= 48
                ],
                key=lambda r: int(r["budget"]),
            )
            assert len(selected) == 5
            raw = pooled[recipient]
            mask = (raw["budgets"] >= 12) & (raw["budgets"] <= 48)
            (ti,) = np.flatnonzero(raw["times"] == t)
            (mi,) = np.flatnonzero(raw["methods"] == method)
            x, y = raw["budgets"][mask], raw["excess_mean"][ti, mask, mi]
            np.testing.assert_array_equal(x, [int(r["budget"]) for r in selected])
            np.testing.assert_allclose(
                y, [float(r["excess"]) for r in selected], rtol=1e-7, atol=1e-12
            )
            if index < 3:
                label = ["Distance", "Sensitivity", "Wiener (class-specific)"][index]
                color, style = ["#333333", "#E69F00", "#009E73"][index], "-"
            elif method == "wiener_pooled":
                label, color, style = "Wiener (pooled)", "#D55E00", "-"
            else:
                source = donor if index < 5 else recipient
                average = index in [3, 5]
                label = (
                    labels[source] + " attention" + (" (average)" if average else "")
                )
                color, style = (
                    ("#0072B2" if source == "jit_b" else "#AA4499"),
                    ("--" if average else "-"),
                )
            attention = 3 <= index <= 6
            (line,) = ax.plot(
                x,
                y,
                color=color,
                ls=style,
                lw=2.5 if attention else 1.5,
                label=label,
                zorder=5 if attention else 2,
            )
            np.testing.assert_array_equal(line.get_ydata(), y)
            curves[f"pooled_{recipient}_{t}_{method}"] = [x.tolist(), y.tolist()]
            shown.extend(y)
            legend.setdefault(label, line)
        lo, hi = min(shown), max(shown)
        pad = max((hi - lo) * 0.06, max(abs(lo), abs(hi)) * 1e-6, 1e-12)
        ax.set_ylim(lo - pad, hi + pad)
        locator = MaxNLocator(nbins=4, steps=[1, 2, 5, 10], min_n_ticks=3)
        ax.yaxis.set_major_locator(locator)
        exponent = int(np.floor(np.log10(max(abs(lo), abs(hi), 1e-12))))
        spacing = float(np.diff(locator.tick_values(lo - pad, hi + pad)).min())
        while spacing / 10.0**exponent < 0.1:
            exponent -= 1
        ax.yaxis.set_major_formatter(ScientificTicks(exponent))
        ax.set_title(f"Denoiser: {labels[recipient]}, t={t:g}")
        ax.set_xticks([12, 16, 24, 32, 48])
        ax.margins(x=0.01)
        ax.grid(alpha=0.16)
        if ri == 1:
            ax.set_xlabel("Retained patches")
fig.supylabel("Excess clean-patch MSE", fontsize=13)
order = [
    "Distance",
    "Sensitivity",
    "Wiener (class-specific)",
    "Wiener (pooled)",
    "JiT-B attention (average)",
    "JiT-B attention",
    "JiT-L attention (average)",
    "JiT-L attention",
]
fig.legend(
    [legend[label] for label in order],
    order,
    loc="outside lower center",
    ncol=3,
    frameon=False,
)
for suffix in ["pdf", "png"]:
    fig.savefig(
        OUT / f"attention_pooled.{suffix}",
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.03,
    )
plt.close(fig)
