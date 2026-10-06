from pathlib import Path

import numpy as np

from experiments.common import style

HERE = Path(__file__).resolve().parent
COLUMNS = {"truth": "(a) Truth, seen from above", "lvm": "(b) LVM", "lvm_chart": "(c) LVM + local chart"}


def main() -> None:
    embedding = np.load(HERE / "embeddings.npz")
    X = embedding["X"]
    rows = {"height": (X[:, 2], style.continuous_colours()),
            "angle": (np.arctan2(X[:, 1], X[:, 0]), style.cyclic_colours())}
    fig, axes = style.figure("full", aspect=0.7, nrows=len(rows), ncols=len(COLUMNS))
    for row, (name, (values, colours)) in zip(axes, rows.items()):
        for ax, (key, title) in zip(row, COLUMNS.items()):
            Z = X[:, :2] if key == "truth" else embedding[key]
            ax.scatter(Z[:, 0], Z[:, 1], c=values, cmap=colours, s=0.3, linewidths=0, rasterized=True)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_axis_off()
            if name == "height":
                ax.set_title(title, loc="left")
        row[0].text(-0.05, 0.5, f"coloured by {name}", transform=row[0].transAxes, rotation=90, ha="right",
                    va="center")
    style.save_figure(fig, HERE / "embedding")


if __name__ == "__main__":
    main()
