from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from experiments.common import style
from experiments.common.datasets import mnist8m_labels

HERE = Path(__file__).resolve().parent
PANELS = {"lvm": "(a) LVM", "lvm_chart": "(b) LVM + local chart", "umap": "(c) UMAP",
          "landmark_isomap": "(d) Landmark Isomap", "tsne": "(e) t-SNE, the 50k fitted only",
          "laplacian_eigenmaps": "(f) LE, the 50k fitted only"}
SHOWN = 50_000


def main() -> None:
    labels = mnist8m_labels()
    colours = style.category_colours(10)
    fig, axes = style.figure("full", aspect=0.75, nrows=2, ncols=3)
    for ax, (name, title) in zip(axes.flat, PANELS.items()):
        Z = np.load(HERE / "embeddings" / f"{name}.npz")["Z"]
        shown = np.random.default_rng(0).choice(len(Z), SHOWN, replace=False)
        ax.scatter(Z[shown, 0], Z[shown, 1], c=labels[shown], cmap=colours, vmin=-0.5, vmax=9.5, s=0.15,
                   linewidths=0, rasterized=True)
        ax.set_title(title, loc="left")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_axis_off()
    digits = [Line2D([], [], marker="o", linestyle="", color=colours(d), label=str(d)) for d in range(10)]
    fig.legend(handles=digits, title="digit", loc="outside lower center", ncols=10)
    style.save_figure(fig, HERE / "embedding")


if __name__ == "__main__":
    main()
