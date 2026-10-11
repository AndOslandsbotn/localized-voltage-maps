import json
from pathlib import Path

import numpy as np

from experiments.common import style
from experiments.sfv.mnist.distribution_shift.run import COUNTED

HERE = Path(__file__).resolve().parent
BARS = {("umap", None): "UMAP", ("tsne", None): "t-SNE", ("laplacian_eigenmaps", None): "Laplacian eigenmaps",
        ("landmark_isomap", None): "Landmark Isomap", ("lvm_chart", "uniform"): "LVM + chart, uniform masses",
        ("lvm_chart", "balanced"): "LVM + chart, masses from balanced",
        ("lvm_chart", "shifted"): "LVM + chart, masses from shifted", ("lvm_chart", "refit"): "LVM + chart, refit"}
N_SHIFTED = 14_000


def main() -> None:
    rows = json.loads((HERE / "results.json").read_text())["rows"]
    values = lambda method, condition, metric: [r[metric] for r in rows
                                                if r["method"] == method and r["condition"] == condition]
    fig, axes = style.figure("full", aspect=0.4, ncols=2)
    for y, ((method, condition), label) in enumerate(BARS.items()):
        v = values(method, condition, "density_preservation")
        axes[0].barh(y, np.mean(v), xerr=np.std(v), color=style.method_colour(method),
                     alpha=1.0 if condition in (None, "shifted") else 0.55, error_kw={"linewidth": 0.5, "capsize": 1.0})
    axes[0].set_yticks(range(len(BARS)), list(BARS.values()))
    axes[0].invert_yaxis()
    axes[0].set_xlabel("density preservation")

    counts = [*COUNTED, N_SHIFTED]
    for method, label in (("lvm", "LVM"), ("lvm_chart", "LVM + chart")):
        curve = np.array([values(method, f"shifted_{m}" if m != N_SHIFTED else "shifted", "density_preservation")
                          for m in counts])
        colour = style.method_colour(method)
        axes[1].errorbar(counts, curve.mean(axis=1), yerr=curve.std(axis=1), color=colour, linewidth=0.7,
                         marker="o", markersize=2.0, elinewidth=0.5, capsize=1.0, label=f"{label}, masses from shifted")
        axes[1].axhline(np.mean(values(method, "uniform", "density_preservation")), color=colour, linestyle=":",
                        linewidth=0.7, label=f"{label}, uniform masses")
    axes[1].set_xscale("log")
    axes[1].set_xticks(counts, [f"{m // 1000}k" for m in counts])
    axes[1].minorticks_off()
    axes[1].set_xlabel("shifted points counted into the masses")
    axes[1].set_ylabel("density preservation")
    axes[1].legend(loc="upper left", fontsize="small")
    for letter, ax in zip("ab", axes):
        ax.set_title(f"({letter})", loc="left")
    style.save_figure(fig, HERE / "density")


if __name__ == "__main__":
    main()
