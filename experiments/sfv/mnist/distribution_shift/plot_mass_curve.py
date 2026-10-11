import json
from pathlib import Path

import numpy as np

from experiments.common import style
from experiments.sfv.mnist.distribution_shift.run_mass_curve import COUNTED

HERE = Path(__file__).resolve().parent
PANELS = {"density_preservation": "density preservation", "trustworthiness": "trustworthiness",
          "continuity": "continuity", "knn_accuracy": "kNN accuracy"}
METHODS = {"lvm": "LVM", "lvm_chart": "LVM + chart"}
BASELINES = {"umap": "UMAP", "tsne": "t-SNE", "laplacian_eigenmaps": "Laplacian eigenmaps",
             "landmark_isomap": "Landmark Isomap"}
ZERO_AT = 4
LABELLED = {0: "0", 100: "100", 1_000: "1k", 14_000: "14k"}
DENSITY_FLOOR = 0.5
SHORT = {"Laplacian eigenmaps": "LE"}


def main() -> None:
    rows = json.loads((HERE / "mass_curve.json").read_text())["rows"]
    baselines = json.loads((HERE / "results.json").read_text())["rows"]
    x = [ZERO_AT if m == 0 else m for m in COUNTED]
    fig, axes = style.figure("full", aspect=0.32, ncols=len(PANELS))
    for letter, ax, (metric, label_y) in zip("abcd", axes, PANELS.items()):
        for method, label in METHODS.items():
            values = np.array([[r[metric] for r in rows if r["method"] == method and r["counted"] == m]
                               for m in COUNTED])
            ax.errorbar(x, values.mean(axis=1), yerr=values.std(axis=1), color=style.method_colour(method),
                        linewidth=0.7, marker="o", markersize=1.8, elinewidth=0.5, capsize=1.0, label=label)
        for method, label in BASELINES.items():
            ax.axhline(np.mean([r[metric] for r in baselines if r["method"] == method]),
                       color=style.method_colour(method), linestyle="--", linewidth=0.7, label=label)
        ax.set_xscale("log")
        ax.set_xticks(x, [LABELLED.get(m, "") for m in COUNTED])
        ax.minorticks_off()
        ax.set_xlabel("points counted")
        if metric == "density_preservation":
            below = [f"{SHORT.get(label, label)} {np.mean([r[metric] for r in baselines if r['method'] == method]):.2f}"
                     for method, label in BASELINES.items()
                     if np.mean([r[metric] for r in baselines if r["method"] == method]) < DENSITY_FLOOR]
            ax.set_ylim(DENSITY_FLOOR, 0.8)
            ax.text(0.03, 0.97, "off scale:\n" + "\n".join(below), transform=ax.transAxes, fontsize="x-small",
                    va="top")
        ax.set_ylabel(label_y)
        ax.set_title(f"({letter})", loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=6)
    style.save_figure(fig, HERE / "mass_curve")


if __name__ == "__main__":
    main()
