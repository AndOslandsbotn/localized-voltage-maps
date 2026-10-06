"""A preview of the hierarchy (paper 2): zoom into the regions of the landmarks at digits {7, 9}, {3, 5} and {2, 8}
and fit LVM again there (one.py). Each fit runs in its own memory-guarded process; then figure.png:

one row per zoom: (a) level 0, the region highlighted; (b) level 0, only the region's points (zooming the camera);
(c) level 1, LVM fitted on the region's points alone. Crosses: landmarks (black: the region's; grey: the others).

    python experiments_old/sfv/mnist/zoom/run.py
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments_old")
sys.path.insert(0, str(BENCH))

import numpy as np  # noqa: E402

from common.runner import thread_env  # noqa: E402

ZOOMS = ([7, 9], [3, 5], [2, 8])


def guarded(*args: str) -> None:
    cmd = [sys.executable, str(BENCH / "common" / "guard.py"), "--limit-mb", "5000", "--", sys.executable,
           str(HERE / "one.py"), *args]
    subprocess.run(cmd, check=True, env=thread_env(16))


def plot() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from common.datasets import load

    _, y = load("eval", n=50_000, seed=0)
    y = np.asarray(y)
    e0 = np.load(HERE / "embeddings" / "level0.npz")
    Z0 = e0["Z"]
    fig, axes = plt.subplots(len(ZOOMS), 3, figsize=(21, 6.6 * len(ZOOMS)))
    scatter = lambda ax, Z, c, s: ax.scatter(Z[:, 0], Z[:, 1], c=c, cmap="tab10", vmin=-0.5, vmax=9.5, s=s,
                                             rasterized=True)

    def crosses(ax, P, colour):
        ax.scatter(P[:, 0], P[:, 1], marker="X", s=90, c=colour, edgecolors="white", linewidths=1.0, zorder=3)
    for i, digits in enumerate(ZOOMS):
        name = "zoom_" + "_".join(map(str, digits))
        r = json.loads((HERE / f"{name}.json").read_text())
        if "error" in r:
            for ax in axes[i]:
                ax.axis("off")
            axes[i, 1].text(0.5, 0.5, f"{digits}: {r['error']}", ha="center", transform=axes[i, 1].transAxes)
            continue
        e = np.load(HERE / "embeddings" / f"{name}.npz")
        rows, Z1 = e["rows"], e["Z1"]
        inside = np.zeros(len(y), dtype=bool)
        inside[rows] = True
        order = np.random.default_rng(0).permutation(len(rows))
        ax = axes[i, 0]
        ax.scatter(Z0[~inside, 0], Z0[~inside, 1], c="lightgrey", s=0.2, rasterized=True)
        scatter(ax, Z0[rows][order], y[rows][order], 0.3)
        own = np.isin(np.arange(len(e0["landmarks"])), r["landmarks"])
        crosses(ax, e0["landmarks"][~own], "grey")
        crosses(ax, e0["landmarks"][own], "black")
        lm = ", ".join(f"{n} at {d}" for d, n in r["landmarks_per_digit"].items())
        ax.set_title(f"level 0 (all 50k): the region of the landmarks at digits {digits} ({lm})\n"
                     f"{r['n']:,} points; " + ", ".join(f"{d}: {100 * s:.0f}%" for d, s in r["composition"].items()
                                                        if s >= 0.03), fontsize=10)
        for ax, Z, key, title in ((axes[i, 1], Z0[rows], "level0", "level 0, only the region's points (camera zoom)"),
                                  (axes[i, 2], Z1, "level1", "level 1: LVM fitted on the region alone")):
            scatter(ax, Z[order], y[rows][order], 0.6)
            crosses(ax, e0["landmarks"][own] if key == "level0" else e["landmarks1"], "black")
            q = r[key]
            ax.set_title(f"{title}\ntrust {q['trustworthiness']:.3f}  cont {q['continuity']:.3f}  "
                         f"5-NN {q['knn_accuracy']:.3f}  global {q['distance_correlation']:.3f}  "
                         f"visible {100 * q['visible_share']:.0f}%", fontsize=10)
        for ax in axes[i]:
            ax.set_xticks([]), ax.set_yticks([])
    cmap = plt.get_cmap("tab10")
    fig.legend([Line2D([], [], ls="", marker="o", ms=8, color=cmap((d + 0.5) / 10)) for d in range(10)],
               [str(d) for d in range(10)], title="digit", loc="center right", frameon=False)
    fig.suptitle("Preview of the hierarchy: zooming into landmark regions of MNIST (evaluation split, seed 0; "
                 "LVM + local PCA chart; scores on the region's points)\nX: landmarks (black: the region's at "
                 "level 0, and level 1's own; grey: the other level-0 landmarks)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 0.96, 0.98))
    fig.savefig(HERE / "figure.png", dpi=90)
    plt.close(fig)
    print(HERE / "figure.png")


def main() -> None:
    guarded("--level0")
    for digits in ZOOMS:
        guarded("--zoom", ",".join(map(str, digits)))
    plot()


if __name__ == "__main__":
    main()
