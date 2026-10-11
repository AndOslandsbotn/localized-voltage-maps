from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from experiments.common import style
from experiments.common.datasets import load

HERE = Path(__file__).resolve().parent
METHODS = {"umap": "UMAP", "tsne": "t-SNE", "lvm_chart": "LVM + local chart"}
SIZES = (10_000, 50_000, 100_000)
SEED, SHOWN = 0, 5_000


def main() -> None:
    colours = style.category_colours(10)
    fig, axes = style.figure("full", aspect=0.95, nrows=len(METHODS), ncols=len(SIZES))
    for column, n in enumerate(SIZES):
        labels = load("eval", n=n, seed=SEED, dataset="mnist8m")[1]
        shown = np.random.default_rng(SEED).choice(n, SHOWN, replace=False)
        for row, (method, name) in enumerate(METHODS.items()):
            Z = np.load(HERE / "embeddings" / f"{method}_n{n}_seed{SEED}.npz")["Z"]
            ax = axes[row, column]
            ax.scatter(Z[shown, 0], Z[shown, 1], c=labels[shown], cmap=colours, vmin=-0.5, vmax=9.5, s=1.35,
                       linewidths=0, rasterized=True)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_axis_off()
            if row == 0:
                ax.set_title(f"n = {n // 1000}k")
            if column == 0:
                ax.text(-0.08, 0.5, name, transform=ax.transAxes, rotation=90, ha="right", va="center")
    digits = [Line2D([], [], marker="o", linestyle="", color=colours(d), label=str(d)) for d in range(10)]
    fig.legend(handles=digits, title="digit", loc="outside lower center", ncols=10)
    style.save_figure(fig, HERE / "embeddings_by_size")


if __name__ == "__main__":
    main()
