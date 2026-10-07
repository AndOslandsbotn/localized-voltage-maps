from pathlib import Path

import numpy as np
from matplotlib import patheffects
from matplotlib.lines import Line2D

from experiments.common import style

HERE = Path(__file__).resolve().parent
ZOOMS = ((7, 9), (3, 5), (2, 5, 8))
COLUMNS = ("(a) Level 0, the region", "(b) Level 0, only the region's points", "(c) Level 1, LVM on the region")


def points(ax, Z, digits, colours, size):
    ax.scatter(Z[:, 0], Z[:, 1], c=digits, cmap=colours, vmin=-0.5, vmax=9.5, s=size, linewidths=0, rasterized=True)


def crosses(ax, P, colour):
    marks = ax.scatter(P[:, 0], P[:, 1], marker="x", s=30, color=colour, linewidths=1.1, zorder=3)
    marks.set_path_effects([patheffects.withStroke(linewidth=2.2, foreground="white")])


def main() -> None:
    embedding = np.load(HERE / "embeddings.npz")
    labels, level0, landmarks0 = embedding["labels"], embedding["level0"], embedding["level0_landmarks"]
    colours = style.category_colours(10)
    fig, axes = style.figure("full", aspect=1.0, nrows=len(ZOOMS), ncols=len(COLUMNS))
    for row, digits in zip(axes, ZOOMS):
        name = "zoom_" + "_".join(map(str, digits))
        rows, own = embedding[f"{name}_rows"], embedding[f"{name}_landmarks"]
        region_digits = labels[rows]
        others = np.setdiff1d(np.arange(len(landmarks0)), own)

        outside = np.setdiff1d(np.arange(len(labels)), rows)
        row[0].scatter(level0[outside, 0], level0[outside, 1], color="0.88", s=0.15, linewidths=0, rasterized=True)
        points(row[0], level0[rows], region_digits, colours, 0.15)
        crosses(row[0], landmarks0[others], "0.55")
        crosses(row[0], landmarks0[own], "black")

        points(row[1], level0[rows], region_digits, colours, 0.4)
        crosses(row[1], landmarks0[own], "black")

        level1, landmarks1 = embedding[f"{name}_level1"], embedding[f"{name}_level1_landmarks"]
        points(row[2], level1, region_digits, colours, 0.4)
        crosses(row[2], landmarks1, "black")

        for ax in row:
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_axis_off()
        named = ", ".join(map(str, digits[:-1])) + f" and {digits[-1]}"
        row[0].text(-0.05, 0.5, f"digits {named}", transform=row[0].transAxes, rotation=90, ha="right", va="center")
    for ax, title in zip(axes[0], COLUMNS):
        ax.set_title(title, loc="left")
    digits = [Line2D([], [], marker="o", linestyle="", color=colours(d), label=str(d)) for d in range(10)]
    fig.legend(handles=digits, title="digit", loc="outside lower center", ncols=10)
    style.save_figure(fig, HERE / "zoom")


if __name__ == "__main__":
    main()
