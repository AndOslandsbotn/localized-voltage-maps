import json
from pathlib import Path

import numpy as np

from experiments.common import style

HERE = Path(__file__).resolve().parent
GROUPS = {("kmeans", "uniform"): "k-means\nuniform", ("kmeans", "counted"): "k-means\ncounted",
          ("farthest", "uniform"): "farthest\nuniform", ("farthest", "counted"): "farthest\ncounted"}
METHODS = {"lvm": "LVM", "lvm_chart": "LVM + local chart"}
PANELS = {"trustworthiness": "trustworthiness", "continuity": "continuity", "knn_accuracy": "kNN accuracy",
          "density_preservation": "density preservation"}


def main() -> None:
    rows = json.loads((HERE / "results.json").read_text())["rows"]
    positions = np.arange(len(GROUPS))
    fig, axes = style.figure("full", aspect=0.32, ncols=len(PANELS))
    for letter, ax, (metric, label_y) in zip("abcdef", axes, PANELS.items()):
        for i, (method, label) in enumerate(METHODS.items()):
            values = np.array([[r[metric] for r in rows if r["method"] == method and (r["placement"], r["masses"])
                                == group] for group in GROUPS])
            ax.errorbar(positions + 0.1 * (2 * i - 1), values.mean(axis=1), yerr=values.std(axis=1),
                        color=style.method_colour(method), linestyle="", marker="o", markersize=2.5, elinewidth=0.5,
                        capsize=1.5, label=label)
        ax.set_xticks(positions, list(GROUPS.values()), rotation=45, ha="right")
        ax.set_ylabel(label_y)
        ax.set_title(f"({letter})", loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=len(METHODS))
    style.save_figure(fig, HERE / "cell_placement")


if __name__ == "__main__":
    main()
