from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes

from experiments.common import style
from experiments.common.datasets import load

HERE = Path(__file__).resolve().parent
N, SEED, SHOWN = 20_000, 0, 8000
TITLES = {"lvm": "(b) LVM", "lvm_chart": "(c) LVM+chart", "umap": "(d) UMAP", "tsne": "(e) t-SNE",
          "laplacian_eigenmaps": "(f) LE", "landmark_isomap": "(g) L-Isomap"}


def aligned(Z: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Z turned (or reflected) and scaled onto X by orthogonal Procrustes; its shape is unchanged."""
    Zc, Xc = Z - Z.mean(axis=0), X - X.mean(axis=0)
    rotation, scale = orthogonal_procrustes(Zc, Xc)
    return scale / (Zc ** 2).sum() * Zc @ rotation + X.mean(axis=0)


def main() -> None:
    X = load("eval", n=N, seed=SEED, dataset="half_sphere")[0]
    shown = np.random.default_rng(0).permutation(len(X))[:SHOWN]
    rows = {"height": (X[:, 2], style.continuous_colours()),
            "angle": (np.arctan2(X[:, 1], X[:, 0]), style.cyclic_colours())}
    columns = {"(a) Truth": X}
    for name, title in TITLES.items():
        columns[title] = aligned(np.load(HERE / "embeddings" / f"{name}.npz")["Z"], X)
    fig, axes = style.figure("full", aspect=0.32, nrows=len(rows), ncols=len(columns),
                             subplot_kw={"projection": "3d"})
    for row, (name, (values, colours)) in zip(axes, rows.items()):
        for ax, (title, Z) in zip(row, columns.items()):
            ax.scatter(*Z[shown].T, c=values[shown], cmap=colours, s=0.3, linewidths=0, rasterized=True)
            ax.view_init(elev=25, azim=-60)
            ax.set_box_aspect(np.ptp(X, axis=0), zoom=1.5)
            ax.set_axis_off()
            if name == "height":
                ax.set_title(title, loc="left")
        row[0].text2D(-0.05, 0.5, name, transform=row[0].transAxes, rotation=90, ha="right", va="center")
    style.save_figure(fig, HERE / "embedding_3d")


if __name__ == "__main__":
    main()
