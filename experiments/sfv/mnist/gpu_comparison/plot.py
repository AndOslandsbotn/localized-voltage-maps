import json
from pathlib import Path

import numpy as np

from experiments.common import style

HERE = Path(__file__).resolve().parent
LABELS = {"lvm": "LVM", "lvm_chart": "LVM + local chart", "umap": "UMAP", "tsne": "t-SNE",
          "laplacian_eigenmaps": "Laplacian eigenmaps", "landmark_isomap": "Landmark Isomap"}
LINESTYLES = {"lvm_chart": ":"}
PANELS = {"seconds": "time (s)", "knn_accuracy": "kNN accuracy", "trustworthiness": "trustworthiness",
          "continuity": "continuity", "density_preservation": "density preservation", "density_slope": "density slope"}


def main() -> None:
    results = json.loads((HERE / "results.json").read_text())
    rows, sizes = results["rows"], results["sizes"]
    fig, axes = style.figure("full", aspect=0.28, ncols=len(PANELS))
    for letter, ax, (metric, label_y) in zip("abcdef", axes, PANELS.items()):
        for method, label in LABELS.items():
            values = np.array([[row[metric] for row in rows if row["method"] == method and row["n"] == n]
                               for n in sizes], dtype=float)
            mean, spread = np.nanmean(values, axis=1), np.nanstd(values, axis=1)
            colour = style.method_colour("lvm" if method == "lvm_chart" else method)
            ax.fill_between(sizes, mean - spread, mean + spread, color=colour, alpha=0.2, linewidth=0)
            ax.plot(sizes, mean, color=colour, linestyle=LINESTYLES.get(method, "-"), linewidth=0.8, marker="o",
                    markersize=1.6, label=label)
        ax.set_xscale("log")
        ax.set_xticks(sizes, [f"{n // 1000}k" if n in (5_000, 20_000, 50_000) else "" for n in sizes])
        ax.minorticks_off()
        ax.set_xlabel("n")
        ax.set_ylabel(label_y)
        ax.set_title(f"({letter})", loc="left")
        if metric == "seconds":
            ax.set_ylim(bottom=0)
        if metric == "density_slope":
            ax.axhline(1.0, color="0.6", linewidth=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=len(LABELS))
    style.save_figure(fig, HERE / "gpu_comparison")


if __name__ == "__main__":
    main()
