import json
from pathlib import Path

import numpy as np

from experiments.common import style

HERE = Path(__file__).resolve().parent
LABELS = {"lvm": "LVM", "lvm_chart": "LVM + local chart", "umap": "UMAP", "tsne": "t-SNE",
          "laplacian_eigenmaps": "Laplacian eigenmaps", "landmark_isomap": "Landmark Isomap"}
PANELS = {"seconds": "time (s)", "knn_accuracy": "kNN accuracy", "trustworthiness": "trustworthiness",
          "continuity": "continuity",
          "density_preservation": "density preservation"}
SIZE_LABELS = {5_000: "5k", 20_000: "20k", 100_000: "100k"}


def main() -> None:
    results = json.loads((HERE / "results.json").read_text())
    rows, sizes = results["rows"], results["sizes"]
    fig, axes = style.figure("full", aspect=0.32, ncols=len(PANELS))
    for letter, ax, (metric, label_y) in zip("abcdef", axes, PANELS.items()):
        for method, label in LABELS.items():
            values = np.array([[row[metric] for row in rows if row["method"] == method and row["n"] == n]
                               for n in sizes])
            ax.errorbar(sizes, values.mean(axis=1), yerr=values.std(axis=1), color=style.method_colour(method),
                        linewidth=0.7, marker="o", markersize=1.8, elinewidth=0.5, capsize=1.0, label=label)
        ax.set_xscale("log")
        ax.set_xticks(sizes, [SIZE_LABELS.get(n, "") for n in sizes])
        ax.minorticks_off()
        ax.set_xlabel("n")
        ax.set_ylabel(label_y)
        ax.set_title(f"({letter})", loc="left")
        if metric == "seconds":
            ax.set_yscale("log")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=len(LABELS))
    style.save_figure(fig, HERE / "gpu_comparison")


if __name__ == "__main__":
    main()
