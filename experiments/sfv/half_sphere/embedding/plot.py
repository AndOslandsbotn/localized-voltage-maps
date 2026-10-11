from pathlib import Path

import numpy as np

from experiments.common import style
from experiments.common.datasets import load

HERE = Path(__file__).resolve().parent
N, SEED = 20_000, 0
TITLES = {"lvm": "(b) LVM", "lvm_chart": "(c) LVM+chart", "umap": "(d) UMAP", "tsne": "(e) t-SNE",
          "laplacian_eigenmaps": "(f) LE", "landmark_isomap": "(g) L-Isomap"}


def main() -> None:
    X = load("eval", n=N, seed=SEED, dataset="half_sphere")[0]
    rows = {"height": (X[:, 2], style.continuous_colours()),
            "angle": (np.arctan2(X[:, 1], X[:, 0]), style.cyclic_colours())}
    columns = {"(a) Truth": X[:, :2]}
    for name, title in TITLES.items():
        columns[title] = np.load(HERE / "embeddings" / f"{name}_seed{SEED}.npz")["Z"]
    fig, axes = style.figure("full", aspect=0.32, nrows=len(rows), ncols=len(columns))
    for row, (name, (values, colours)) in zip(axes, rows.items()):
        for ax, (title, Z) in zip(row, columns.items()):
            ax.scatter(Z[:, 0], Z[:, 1], c=values, cmap=colours, s=0.3, linewidths=0, rasterized=True)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_axis_off()
            if name == "height":
                ax.set_title(title, loc="left")
        row[0].text(-0.08, 0.5, name, transform=row[0].transAxes, rotation=90, ha="right", va="center")
    style.save_figure(fig, HERE / "embedding")


if __name__ == "__main__":
    main()
