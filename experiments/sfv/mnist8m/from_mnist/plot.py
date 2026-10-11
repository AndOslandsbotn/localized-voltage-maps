import json
from pathlib import Path

import numpy as np

from experiments.common import style

HERE = Path(__file__).resolve().parent
ROWS = {("lvm", "uniform"): "LVM, uniform masses", ("lvm", "mnist"): "LVM, MNIST masses",
        ("lvm", "mnist8m"): "LVM, MNIST8M masses", ("lvm", "refit"): "LVM, refit on MNIST8M",
        ("lvm_chart", "uniform"): "LVM + chart, uniform masses", ("lvm_chart", "mnist"): "LVM + chart, MNIST masses",
        ("lvm_chart", "mnist8m"): "LVM + chart, MNIST8M masses", ("lvm_chart", "refit"): "LVM + chart, refit",
        ("umap", None): "UMAP", ("landmark_isomap", None): "Landmark Isomap"}
PANELS = {"transform_seconds": "time to place (s)", "trustworthiness": "trustworthiness", "continuity": "continuity",
          "knn_accuracy": "kNN accuracy", "density_preservation": "density\npreservation"}


def main() -> None:
    rows = json.loads((HERE / "results.json").read_text())["rows"]
    labels = list(ROWS.values())
    fig, axes = style.figure("full", aspect=0.45, ncols=len(PANELS), sharey=True)
    for letter, ax, (metric, label_x) in zip("abcde", axes, PANELS.items()):
        for y, (method, condition) in enumerate(ROWS):
            values = [r[metric] for r in rows if r["method"] == method and r["condition"] == condition]
            ax.barh(y, np.mean(values), xerr=np.std(values), color=style.method_colour(method),
                    alpha=1.0 if condition in (None, "mnist8m") else 0.55, error_kw={"linewidth": 0.5, "capsize": 1.0})
        ax.set_xlabel(label_x)
        ax.set_title(f"({letter})", loc="left")
        if metric == "density_preservation":
            ax.set_xlim(-0.5, 1.0)
        elif metric != "transform_seconds":
            ax.set_xlim(0.0, 1.0)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].invert_yaxis()
    style.save_figure(fig, HERE / "from_mnist")


if __name__ == "__main__":
    main()
