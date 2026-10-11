from pathlib import Path

import numpy as np

from experiments.common import style
from experiments.common.datasets import load, unrolled
from experiments.sfv.swiss_roll.even_density.run import DATASETS, N

HERE = Path(__file__).resolve().parent
SEED, SHOWN = 0, "lvm_chart"
TITLES = {"lvm": "LVM", "lvm_chart": "LVM + chart"}
ROWS = {"swiss_roll": "uneven density\n(angle uniform)", "swiss_roll_even": "even density\n(arc length uniform)"}
COLUMNS = {"truth": "truth, unrolled", f"{SHOWN}_3": f"{TITLES[SHOWN]},\n3 landmarks",
           f"{SHOWN}_12": f"{TITLES[SHOWN]},\n12 landmarks", "laplacian_eigenmaps": "Laplacian eigenmaps\n(best pair)"}


def main() -> None:
    fig, axes = style.figure("full", aspect=0.45, nrows=len(DATASETS), ncols=len(COLUMNS))
    for row, dataset in zip(axes, DATASETS):
        sheet = unrolled(load("eval", n=N, seed=0, dataset=dataset)[0])
        for ax, (name, title) in zip(row, COLUMNS.items()):
            Z = sheet if name == "truth" else np.load(HERE / "embeddings" / f"{dataset}_{name}_seed{SEED}.npz")["Z"]
            ax.scatter(Z[:, 0], Z[:, 1], c=sheet[:, 0], cmap=style.continuous_colours(), s=0.2, linewidths=0,
                       rasterized=True)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_axis_off()
            if dataset == DATASETS[0]:
                ax.set_title(title)
        row[0].text(-0.08, 0.5, ROWS[dataset], transform=row[0].transAxes, rotation=90, ha="right", va="center")
    style.save_figure(fig, HERE / "even_density")


if __name__ == "__main__":
    main()
