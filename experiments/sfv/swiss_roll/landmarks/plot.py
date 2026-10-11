import json
from pathlib import Path

import numpy as np

from experiments.common import style
from experiments.common.datasets import load, unrolled
from experiments.sfv.swiss_roll.landmarks.run import COUNTS, N

HERE = Path(__file__).resolve().parent
SEED = 0


def main() -> None:
    rows = json.loads((HERE / "results.json").read_text())["rows"]
    sheet = unrolled(load("eval", n=N, seed=SEED, dataset="swiss_roll")[0])
    fig, axes = style.figure("full", aspect=0.32, ncols=len(COUNTS) + 2)
    for ax, (title, Z) in zip(axes, [("(a) Truth, unrolled", sheet)] + [
            (f"({'bcde'[i]}) {count} landmarks", np.load(HERE / "embeddings" / f"landmarks_{count}_seed{SEED}.npz")["Z"])
            for i, count in enumerate(COUNTS)]):
        ax.scatter(Z[:, 0], Z[:, 1], c=sheet[:, 0], cmap=style.continuous_colours(), s=0.2, linewidths=0,
                   rasterized=True)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_axis_off()
        ax.set_title(title, loc="left")
    ax = axes[-1]
    values = np.array([[r["exact"] for r in rows if r["landmarks"] == count] for count in COUNTS])
    ax.errorbar(COUNTS, values.mean(axis=1), yerr=values.std(axis=1), color=style.method_colour("lvm"),
                linewidth=0.7, marker="o", markersize=2.0, elinewidth=0.5, capsize=1.0)
    ax.set_xticks(COUNTS)
    ax.set_xlabel("landmarks")
    ax.set_ylabel("geodesic rank corr.")
    ax.set_title("(f)", loc="left")
    style.save_figure(fig, HERE / "landmarks")


if __name__ == "__main__":
    main()
