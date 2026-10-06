"""Uneven connectivity: both pinned variants on MNIST and the half sphere (each in its own memory-guarded process),
then figure.png.

    python experiments_old/sfv/demonstrations/uneven_connectivity/run.py
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments_old")
sys.path.insert(0, str(BENCH))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from common.runner import thread_env  # noqa: E402

DATASETS = {"mnist": "MNIST evaluation split (50k): digits of very different spread",
            "half_sphere": "half sphere (20k): uniform density (control)"}
VARIANTS = {"before": "before: one global radius (graph.radius knn)",
            "after": "after: a radius per cell (graph.radius adaptive_per_cell)"}


def main() -> None:
    for dataset in DATASETS:
        for variant in VARIANTS:
            if (HERE / "embeddings" / f"{dataset}_{variant}.npz").exists():
                continue
            cmd = [sys.executable, str(BENCH / "common" / "guard.py"), "--limit-mb", "5000", "--", sys.executable,
                   str(HERE / "one.py"), "--dataset", dataset, "--variant", variant]
            subprocess.run(cmd, check=True, env=thread_env(16))
    fig, axes = plt.subplots(len(DATASETS), len(VARIANTS), figsize=(9 * len(VARIANTS), 8.4 * len(DATASETS)))
    for i, dataset in enumerate(DATASETS):
        for j, variant in enumerate(VARIANTS):
            ax = axes[i, j]
            e = np.load(HERE / "embeddings" / f"{dataset}_{variant}.npz")
            r = json.loads((HERE / f"{dataset}_{variant}.json").read_text())
            order = np.random.default_rng(0).permutation(len(e["Z"]))
            if dataset == "mnist":
                ax.scatter(e["Z"][order, 0], e["Z"][order, 1], c=e["colour"][order], cmap="tab10", vmin=-0.5, vmax=9.5,
                           s=0.3, rasterized=True)
                extra = (f"\nroom of the 0s {r['room_per_digit']['0']:.2f}, of the 1s {r['room_per_digit']['1']:.2f}; "
                         f"edges per cell, 0s {r['edges_per_digit']['0']:.0f}, 1s {r['edges_per_digit']['1']:.0f}")
                knn = f"5-NN {r['knn_accuracy']:.3f}  "
            else:
                ax.scatter(e["Z"][order, 0], e["Z"][order, 1], c=e["colour"][order], cmap="twilight", s=0.5,
                           rasterized=True)
                ax.set_aspect("equal", adjustable="datalim")
                extra, knn = "", ""
            p10, p50, p90 = r["edges_p10_p50_p90"]
            ax.set_title(f"{DATASETS[dataset].split(':')[0]}, {VARIANTS[variant]}\n"
                         f"cell spacing spread (p90/p10) {r['spacing_spread_p90_over_p10']:.2f}; edges per cell "
                         f"{p10:.0f} / {p50:.0f} / {p90:.0f} (p10/p50/p90)\ntrust {r['trustworthiness']:.3f}  "
                         f"cont {r['continuity']:.3f}  {knn}global {r['distance_correlation']:.3f}  "
                         f"visible {100 * r['visible_share']:.0f}%" + extra, fontsize=10)
            ax.set_xticks([]), ax.set_yticks([])
    cmap = plt.get_cmap("tab10")
    fig.legend([Line2D([], [], ls="", marker="o", ms=8, color=cmap((d + 0.5) / 10)) for d in range(10)],
               [str(d) for d in range(10)], title="digit (MNIST)", loc="upper right", frameon=False)
    fig.suptitle("Uneven connectivity: one global kernel radius vs a radius per cell (LVM + local PCA chart, seed 0; "
                 "half sphere coloured by angle)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 0.95, 0.97))
    fig.savefig(HERE / "figure.png", dpi=90)
    plt.close(fig)
    print(HERE / "figure.png")


if __name__ == "__main__":
    main()
