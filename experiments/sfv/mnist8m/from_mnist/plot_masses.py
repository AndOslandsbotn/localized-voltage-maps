import json
from pathlib import Path

import numpy as np

from experiments.common import style
from experiments.sfv.mnist8m.from_mnist.run import COUNTED

HERE = Path(__file__).resolve().parent
N_MNIST8M = 8_100_000
PANELS = {"density_preservation": "density preservation", "knn_accuracy": "kNN accuracy",
          "trustworthiness": "trustworthiness"}


def main() -> None:
    rows = json.loads((HERE / "results.json").read_text())["rows"]
    values = lambda method, condition, metric: [r[metric] for r in rows
                                                if r["method"] == method and r["condition"] == condition]
    counts = [*COUNTED, N_MNIST8M]
    conditions = [*(f"mnist8m_{m}" for m in COUNTED), "mnist8m"]
    fig, axes = style.figure("full", aspect=0.32, ncols=len(PANELS))
    for letter, ax, (metric, label_y) in zip("abc", axes, PANELS.items()):
        for method, label in (("lvm", "LVM"), ("lvm_chart", "LVM + chart")):
            colour = style.method_colour(method)
            curve = np.array([values(method, condition, metric) for condition in conditions])
            ax.errorbar(counts, curve.mean(axis=1), yerr=curve.std(axis=1), color=colour, linewidth=0.7, marker="o",
                        markersize=2.0, elinewidth=0.5, capsize=1.0, label=f"{label}, MNIST8M masses")
            ax.axhline(np.mean(values(method, "mnist", metric)), color=colour, linestyle="--", linewidth=0.7,
                       label=f"{label}, MNIST masses")
        ax.set_xscale("log")
        ax.set_xticks(counts, ["10k", "100k", "1M", "8.1M"])
        ax.minorticks_off()
        ax.set_xlabel("MNIST8M points counted")
        ax.set_ylabel(label_y)
        ax.set_title(f"({letter})", loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2)
    style.save_figure(fig, HERE / "masses")


if __name__ == "__main__":
    main()
