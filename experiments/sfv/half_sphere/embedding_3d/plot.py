from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes

from experiments.common import style

HERE = Path(__file__).resolve().parent
COLUMNS = {"truth": "(a) Truth", "lvm": "(b) LVM", "lvm_chart": "(c) LVM + local chart",
           "lvm_12": "(d) LVM, 12 landmarks"}
SHOWN = 8000                    # points drawn per panel, so the 3-D scatter stays readable


def aligned(Z: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Z turned (or reflected) and scaled onto X by orthogonal Procrustes; its shape is unchanged."""
    Zc, Xc = Z - Z.mean(axis=0), X - X.mean(axis=0)
    rotation, scale = orthogonal_procrustes(Zc, Xc)
    return scale / (Zc ** 2).sum() * Zc @ rotation + X.mean(axis=0)


def main() -> None:
    embedding = np.load(HERE / "embeddings.npz")
    X = embedding["X"]
    shown = np.random.default_rng(0).permutation(len(X))[:SHOWN]
    rows = {"height": (X[:, 2], style.continuous_colours()),
            "angle": (np.arctan2(X[:, 1], X[:, 0]), style.cyclic_colours())}
    fig, axes = style.figure("full", aspect=0.5, nrows=len(rows), ncols=len(COLUMNS), subplot_kw={"projection": "3d"})
    for row, (name, (values, colours)) in zip(axes, rows.items()):
        for ax, (key, title) in zip(row, COLUMNS.items()):
            Z = X if key == "truth" else aligned(embedding[key], X)
            ax.scatter(*Z[shown].T, c=values[shown], cmap=colours, s=0.4, linewidths=0, rasterized=True)
            ax.view_init(elev=25, azim=-60)
            ax.set_box_aspect(np.ptp(X, axis=0), zoom=1.5)                    # the half sphere's true proportions
            ax.set_axis_off()
            if name == "height":
                ax.set_title(title, loc="left")
        row[0].text2D(-0.05, 0.5, f"coloured by {name}", transform=row[0].transAxes, rotation=90, ha="right",
                      va="center")
    style.save_figure(fig, HERE / "embedding_3d")


if __name__ == "__main__":
    main()
