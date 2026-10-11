from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from experiments.common import style
from experiments.common.datasets import load

HERE = Path(__file__).resolve().parent
N, SEED = 50_000, 0
PLACEMENTS = {"kmeans": "k-means cells", "farthest": "farthest-point cells"}
MASSES = {"uniform": "uniform masses", "counted": "counted masses"}


def main() -> None:
    labels = load("eval", n=N, seed=SEED)[1]
    colours = style.category_colours(10)
    fig, axes = style.figure("full", aspect=0.95, nrows=len(PLACEMENTS), ncols=len(MASSES))
    for row, (placement, row_name) in enumerate(PLACEMENTS.items()):
        for column, (masses, column_name) in enumerate(MASSES.items()):
            Z = np.load(HERE / "embeddings" / f"lvm_chart_{placement}_{masses}_seed{SEED}.npz")["Z"]
            ax = axes[row, column]
            ax.scatter(Z[:, 0], Z[:, 1], c=labels, cmap=colours, vmin=-0.5, vmax=9.5, s=1.35, linewidths=0,
                       rasterized=True)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_axis_off()
            if row == 0:
                ax.set_title(column_name)
            if column == 0:
                ax.text(-0.05, 0.5, row_name, transform=ax.transAxes, rotation=90, ha="right", va="center")
    digits = [Line2D([], [], marker="o", linestyle="", color=colours(d), label=str(d)) for d in range(10)]
    fig.legend(handles=digits, title="digit", loc="outside lower center", ncols=10)
    style.save_figure(fig, HERE / "embeddings")


if __name__ == "__main__":
    main()
