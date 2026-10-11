from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from experiments.common import style
from experiments.sfv.mnist.distribution_shift.run import shifted

HERE = Path(__file__).resolve().parent
SEED = 0
PANELS = {"lvm_chart_uniform": "LVM + chart, uniform masses", "lvm_chart_balanced": "masses from balanced data",
          "lvm_chart_shifted": "masses from shifted data", "lvm_chart_refit": "refit on shifted data",
          "umap": "UMAP", "tsne": "t-SNE", "laplacian_eigenmaps": "Laplacian eigenmaps",
          "landmark_isomap": "Landmark Isomap"}


def main() -> None:
    labels = shifted(SEED)[1]
    colours = style.category_colours(10)
    fig, axes = style.figure("full", aspect=0.55, nrows=2, ncols=4)
    for ax, (name, title) in zip(axes.flat, PANELS.items()):
        Z = np.load(HERE / "embeddings" / f"{name}_seed{SEED}.npz")["Z"]
        ax.scatter(Z[:, 0], Z[:, 1], c=labels, cmap=colours, vmin=-0.5, vmax=9.5, s=1.35, linewidths=0,
                   rasterized=True)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_axis_off()
    digits = [Line2D([], [], marker="o", linestyle="", color=colours(d), label=str(d)) for d in range(10)]
    fig.legend(handles=digits, title="digit", loc="outside lower center", ncols=10)
    style.save_figure(fig, HERE / "embeddings")


if __name__ == "__main__":
    main()
