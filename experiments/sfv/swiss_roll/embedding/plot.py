from pathlib import Path

import numpy as np

from experiments.common import style
from experiments.common.datasets import load, unrolled

HERE = Path(__file__).resolve().parent
N, SEED = 20_000, 0
TITLES = {"lvm": "(b) LVM", "lvm_chart": "(c) LVM+chart", "umap": "(d) UMAP", "tsne": "(e) t-SNE",
          "laplacian_eigenmaps": "(f) LE", "landmark_isomap": "(g) L-Isomap"}


def frame(ax, Z):
    """Axes around the central 99% of the points: a handful of far outliers (Laplacian eigenmaps has 22 of 20,000)
    would otherwise squeeze the rest into a line."""
    low, high = np.percentile(Z, [0.5, 99.5], axis=0)
    margin = 0.08 * (high - low)
    ax.set_xlim(low[0] - margin[0], high[0] + margin[0])
    ax.set_ylim(low[1] - margin[1], high[1] + margin[1])
    ax.set_aspect("equal", adjustable="datalim")


def main() -> None:
    sheet = unrolled(load("eval", n=N, seed=SEED, dataset="swiss_roll")[0])
    rows = {"position along the roll": sheet[:, 0], "height": sheet[:, 1]}
    columns = {"(a) Truth": sheet}
    for name, title in TITLES.items():
        columns[title] = np.load(HERE / "embeddings" / f"{name}_seed{SEED}.npz")["Z"]
    fig, axes = style.figure("full", aspect=0.32, nrows=len(rows), ncols=len(columns))
    for row, (name, values) in zip(axes, rows.items()):
        for ax, (title, Z) in zip(row, columns.items()):
            ax.scatter(Z[:, 0], Z[:, 1], c=values, cmap=style.continuous_colours(), s=0.2, linewidths=0,
                       rasterized=True)
            frame(ax, Z)
            ax.set_axis_off()
            if name == "position along the roll":
                ax.set_title(title, loc="left")
        row[0].text(-0.08, 0.5, name.replace(" along", "\nalong"), transform=row[0].transAxes, rotation=90,
                    ha="right", va="center")
    style.save_figure(fig, HERE / "embedding")


if __name__ == "__main__":
    main()
