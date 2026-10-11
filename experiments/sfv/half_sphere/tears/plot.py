import json
from pathlib import Path

import numpy as np

from experiments.common import style
from experiments.common.datasets import load
from experiments.sfv.half_sphere.tears.run import COUNTS, N, VARIANTS
from experiments.sfv.half_sphere.tears.score import torn

HERE = Path(__file__).resolve().parent
SEED = 0
TORN = "#882255"


def main() -> None:
    rows = json.loads((HERE / "results.json").read_text())["rows"]
    X = load("eval", n=N, seed=0, dataset="half_sphere")[0]
    angle = np.arctan2(X[:, 1], X[:, 0])
    for variant in VARIANTS:
        fig, axes = style.figure("full", aspect=0.45, nrows=2, ncols=len(COUNTS))
        for column, count in enumerate(COUNTS):
            saved = np.load(HERE / "embeddings" / f"{variant}_{count}_seed{SEED}.npz")
            Z, bad = saved["Z"], torn(X, saved["Z"])
            mean = {key: np.mean([r[key] for r in rows if r["variant"] == variant and r["landmarks"] == count])
                    for key in ("torn_share", "trustworthiness", "continuity", "geodesic")}
            ax = axes[0, column]
            ax.scatter(Z[:, 0], Z[:, 1], c=angle, cmap=style.cyclic_colours(), s=0.3, linewidths=0, rasterized=True)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_axis_off()
            ax.set_title(f"{count} landmarks\ntrust {mean['trustworthiness']:.3f}  cont {mean['continuity']:.3f}\n"
                         f"geodesic {mean['geodesic']:.3f}  torn {100 * mean['torn_share']:.1f}%", fontsize="small")
            axes[1, column].remove()
            ax3 = fig.add_subplot(2, len(COUNTS), len(COUNTS) + column + 1, projection="3d")
            ax3.scatter(*X[~bad].T, c=angle[~bad], cmap=style.cyclic_colours(), s=0.3, alpha=0.35, linewidths=0,
                        rasterized=True)
            ax3.scatter(*X[bad].T, color=TORN, s=1.2, linewidths=0, rasterized=True)
            ax3.scatter(*saved["landmarks"].T, color="k", marker="x", s=12, linewidths=0.6, depthshade=False)
            ax3.view_init(elev=30, azim=-60)
            ax3.set_box_aspect((1, 1, 0.5), zoom=1.3)
            ax3.set_axis_off()
        fig.text(0.5, 0.01, "Bottom: torn points dark red, landmarks $\\times$ (seed 0); numbers: mean of 3 seeds.",
                 ha="center", fontsize="small")
        style.save_figure(fig, HERE / f"tears_{variant}")


if __name__ == "__main__":
    main()
