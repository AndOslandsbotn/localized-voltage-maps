from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from experiments.common import style

HERE = Path(__file__).resolve().parent
PANELS = {"lvm": "(a) LVM", "lvm_chart": "(b) LVM + local chart"}


def main() -> None:
    embedding = np.load(HERE / "embeddings.npz")
    colours = style.category_colours(10)
    fig, axes = style.figure("full", aspect=0.55, ncols=len(PANELS))
    for ax, (key, title) in zip(axes, PANELS.items()):
        Z = embedding[key]
        ax.scatter(Z[:, 0], Z[:, 1], c=embedding["labels"], cmap=colours, vmin=-0.5, vmax=9.5, s=0.15, linewidths=0,
                   rasterized=True)
        ax.set_title(title, loc="left")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_axis_off()
    digits = [Line2D([], [], marker="o", linestyle="", color=colours(d), label=str(d)) for d in range(10)]
    fig.legend(handles=digits, title="digit", loc="outside lower center", ncols=10)
    style.save_figure(fig, HERE / "embedding")


if __name__ == "__main__":
    main()
