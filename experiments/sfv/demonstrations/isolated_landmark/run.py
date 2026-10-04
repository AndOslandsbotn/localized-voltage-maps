"""Isolated landmark: run both pinned variants (each in its own memory-guarded process), draw figure.png.

    python experiments/sfv/demonstrations/isolated_landmark/run.py
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments")
sys.path.insert(0, str(BENCH))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.runner import thread_env  # noqa: E402

TITLES = {"before": "before: cell graph as built (graph.connect: false)",
          "after": "after: every piece joined to the main graph (graph.connect: true)"}


def main() -> None:
    for variant in TITLES:
        if not (HERE / "embeddings" / f"{variant}.npz").exists():
            cmd = [sys.executable, str(BENCH / "common" / "guard.py"), "--limit-mb", "5000", "--",
                   sys.executable, str(HERE / "one.py"), "--variant", variant]
            out = subprocess.run(cmd, capture_output=True, text=True, env=thread_env(16)).stdout
            print([l for l in out.splitlines() if l.startswith(variant)], flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, (variant, title) in zip(axes, TITLES.items()):
        data, r = np.load(HERE / "embeddings" / f"{variant}.npz"), json.loads((HERE / f"{variant}.json").read_text())
        Z, y = data["Z"], data["y"]
        order = np.random.default_rng(0).permutation(len(y))
        ax.scatter(Z[order, 0], Z[order, 1], c=y[order], cmap="tab10", vmin=-0.5, vmax=9.5, s=0.5, rasterized=True)
        for d in range(10):
            cx, cy = np.median(Z[y == d], axis=0)
            ax.text(cx, cy, str(d), fontsize=12, weight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.15", fc="white", alpha=0.7, lw=0))
        ax.set_title(f"{title}\n{r['graph_pieces']} graph pieces, {r['landmarks_in_size1_pieces']} landmark(s) in a size-1 piece; "
                     f"rho_g {r['rho_g']:.1e} (bound {r['rho_g_lower_bound']:.0e}), reach {r['reach']:.0f}/{r['n_landmarks']}\n"
                     f"trust {r['trustworthiness']:.3f}  cont {r['continuity']:.3f}  5-NN {r['knn_accuracy']:.3f}  "
                     f"global {r['distance_correlation']:.3f}", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("Isolated landmark (pinned settings_*.yaml): LVM on MNIST8M's first 50k points, seed 0", fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / "figure.png", dpi=110)
    print(HERE / "figure.png")


if __name__ == "__main__":
    main()
