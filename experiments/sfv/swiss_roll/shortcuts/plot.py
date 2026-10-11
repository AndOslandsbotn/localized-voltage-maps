from pathlib import Path

import numpy as np

from experiments.common import style
from experiments.common.datasets import load, unrolled

HERE = Path(__file__).resolve().parent
N, SEED = 20_000, 0
TITLES = {"default": "(b) 300 cells, k = 10", "cells_1000": "(c) 1000 cells, k = 10", "k_5": "(d) 300 cells, k = 5"}


def main() -> None:
    sheet = unrolled(load("eval", n=N, seed=SEED, dataset="swiss_roll")[0])
    rows = {"position along the roll": sheet[:, 0], "height": sheet[:, 1]}
    columns = {"(a) Truth, unrolled": sheet}
    for name, title in TITLES.items():
        columns[title] = np.load(HERE / "embeddings" / f"{name}_seed{SEED}.npz")["Z"]
    fig, axes = style.figure("full", aspect=0.45, nrows=len(rows), ncols=len(columns))
    for row, (name, values) in zip(axes, rows.items()):
        for ax, (title, Z) in zip(row, columns.items()):
            ax.scatter(Z[:, 0], Z[:, 1], c=values, cmap=style.continuous_colours(), s=0.2, linewidths=0,
                       rasterized=True)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_axis_off()
            if name == "position along the roll":
                ax.set_title(title, loc="left")
        row[0].text(-0.05, 0.5, f"coloured by\n{name}", transform=row[0].transAxes, rotation=90, ha="right",
                    va="center")
    style.save_figure(fig, HERE / "shortcuts")


if __name__ == "__main__":
    main()
