"""Point stacking: run the three pinned variants (each in its own memory-guarded process) and draw figure.png.

    python benchmarks/demonstrations/point_stacking/run.py
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "benchmarks")
sys.path.insert(0, str(BENCH))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.metrics import stacking  # noqa: E402
from common.runner import thread_env  # noqa: E402

TITLES = {"before": "before: hard kernel from points to cells",
          "after": "after: each point weights its 3 nearest cells (sharpness 16)",
          "after_local_pca": "after + local PCA chart per cell"}


def main() -> None:
    for variant in TITLES:
        if not (HERE / "embeddings" / f"{variant}.npz").exists():
            cmd = [sys.executable, str(BENCH / "common" / "guard.py"), "--limit-mb", "4500", "--",
                   sys.executable, str(HERE / "one.py"), "--variant", variant]
            out = subprocess.run(cmd, capture_output=True, text=True, env=thread_env(16)).stdout
            print([l for l in out.splitlines() if l.startswith(variant)], flush=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4))
    for ax, (variant, title) in zip(axes, TITLES.items()):
        data, info = np.load(HERE / "embeddings" / f"{variant}.npz"), json.loads((HERE / f"{variant}.json").read_text())
        Z, y = data["Z"], data["y"]
        info.update(stacking(Z))
        order = np.random.default_rng(0).permutation(len(y))
        ax.scatter(Z[order, 0], Z[order, 1], c=y[order], cmap="tab10", vmin=-0.5, vmax=9.5, s=0.5, rasterized=True)
        for digit in range(10):
            cx, cy = np.median(Z[y == digit], axis=0)
            ax.text(cx, cy, str(digit), fontsize=12, weight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.15", fc="white", alpha=0.7, lw=0))
        ax.set_title(f"{title}\ntrust {info['trustworthiness']:.3f}  cont {info['continuity']:.3f}  "
                     f"5-NN {info['knn_accuracy']:.3f}  global {info['distance_correlation']:.3f}\n"
                     f"visible share {info['visible_share']:.0%} ({info['occupied_squares']} occupied squares of a "
                     f"1000x1000 grid for {len(Z)} points)", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("Point stacking (pinned settings_*.yaml): LVM on the MNIST tuning split (20k, seed 0), 150 cells",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / "figure.png", dpi=120)
    print(HERE / "figure.png")


if __name__ == "__main__":
    main()
