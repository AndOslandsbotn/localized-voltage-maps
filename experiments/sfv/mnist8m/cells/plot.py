import json
from pathlib import Path

import numpy as np

from experiments.common import style

HERE = Path(__file__).resolve().parent
N_CELLS = (300, 600, 1000, 2000)
PANELS = {"seconds": "time (s)", "trustworthiness": "trustworthiness", "continuity": "continuity",
          "knn_accuracy": "kNN accuracy",
          "density_preservation": "density preservation"}


def main() -> None:
    rows = json.loads((HERE / "results.json").read_text())["rows"]
    fig, axes = style.figure("full", aspect=0.3, ncols=len(PANELS))
    for letter, ax, (metric, label_y) in zip("abcdef", axes, PANELS.items()):
        values = np.array([[row[metric] for row in rows if row["setting"] == f"cells_{n_cells}"]
                           for n_cells in N_CELLS])
        ax.errorbar(N_CELLS, values.mean(axis=1), yerr=values.std(axis=1), color=style.method_colour("lvm"),
                    linewidth=0.7, marker="o", markersize=1.8, elinewidth=0.5, capsize=1.0)
        ax.set_xscale("log")
        ax.set_xticks(N_CELLS, [str(n) for n in N_CELLS])
        ax.minorticks_off()
        ax.set_xlabel("cells")
        ax.set_ylabel(label_y)
        ax.set_title(f"({letter})", loc="left")
    style.save_figure(fig, HERE / "cells")


if __name__ == "__main__":
    main()
