from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from experiments.common import style
from experiments.common.datasets import mnist8m_points

HERE = Path(__file__).resolve().parent
SEED = 0
PANELS = {"lvm_chart_uniform": "LVM + chart, uniform masses", "lvm_chart_mnist": "LVM + chart, MNIST masses",
          "lvm_chart_mnist8m": "LVM + chart, MNIST8M masses", "lvm_chart_refit": "LVM + chart, refit",
          "lvm_mnist": "LVM, MNIST masses", "lvm_mnist8m": "LVM, MNIST8M masses", "umap": "UMAP",
          "landmark_isomap": "Landmark Isomap"}


def main() -> None:
    colours = style.category_colours(10)
    fig, axes = style.figure("full", aspect=0.55, nrows=2, ncols=4)
    for ax, (name, title) in zip(axes.flat, PANELS.items()):
        saved = np.load(HERE / "embeddings" / f"{name}_seed{SEED}.npz")
        labels = mnist8m_points(saved["rows"])[1]
        ax.scatter(saved["Z"][:, 0], saved["Z"][:, 1], c=labels, cmap=colours, vmin=-0.5, vmax=9.5, s=1.35,
                   linewidths=0, rasterized=True)
        ax.set_title(title.replace(", ", "\n"))
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_axis_off()
    digits = [Line2D([], [], marker="o", linestyle="", color=colours(d), label=str(d)) for d in range(10)]
    fig.legend(handles=digits, title="digit", loc="outside lower center", ncols=10)
    style.save_figure(fig, HERE / "embeddings")


if __name__ == "__main__":
    main()
