"""Draw figure.png: LVM on the half sphere by landmark count, in 2-D and on the sphere in 3-D.

Row 1: the 2-D embedding coloured by angle around the pole.
Row 2: the half sphere itself (3-D), torn points in red, landmarks as black stars.

    python benchmarks/demonstrations/sphere_tears/plot_sweep.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
COUNTS = (3, 6, 9, 12, 20)


def main() -> None:
    variant = sys.argv[1] if len(sys.argv) > 1 else "before"
    fig = plt.figure(figsize=(4.4 * len(COUNTS), 9))
    for col, L in enumerate(COUNTS):
        name = f"sweep_L{L}" if variant == "before" else f"sweep_{variant}_L{L}"
        data, info = np.load(HERE / "embeddings" / f"{name}.npz"), json.loads((HERE / f"{name}.json").read_text())
        X, Z, torn, lm = data["X"], data["Z"], data["torn"], data["landmarks"]
        angle = np.arctan2(X[:, 1], X[:, 0])
        order = np.random.default_rng(0).permutation(len(X))
        ax = fig.add_subplot(2, len(COUNTS), col + 1)
        ax.scatter(Z[order, 0], Z[order, 1], c=angle[order], cmap="hsv", s=0.6, rasterized=True)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{L} landmarks\ntrust {info['trustworthiness']:.3f}  cont {info['continuity']:.3f}  "
                     f"global corr {info['distance_correlation']:.3f}\ntorn points {info['torn_share']:.1%}",
                     fontsize=9)
        ax3 = fig.add_subplot(2, len(COUNTS), len(COUNTS) + col + 1, projection="3d")
        sub = order[:8000]
        ok, bad = sub[~torn[sub]], sub[torn[sub]]
        ax3.scatter(*X[ok].T, c=angle[ok], cmap="hsv", s=0.4, alpha=0.35, rasterized=True)
        ax3.scatter(*X[bad].T, c="red", s=2.0, rasterized=True)
        ax3.scatter(*lm.T, marker="*", s=180, c="black", edgecolors="white", linewidths=0.6, depthshade=False)
        ax3.view_init(elev=35, azim=-60)
        ax3.set_box_aspect((1, 1, 0.5))
        ax3.set_xticks([]), ax3.set_yticks([]), ax3.set_zticks([])
        ax3.set_title("on the half sphere: torn points red, landmarks ★", fontsize=9)
    title = {"before": "before (settings.yaml)", "fix_A": "fix A, distances read down to 1e-10 (settings_fix_A.yaml)"}
    fig.suptitle(f"Sphere tears, {title[variant]}: LVM on the half sphere (n = 20000) by number of landmarks; "
                 "torn = a sphere neighbour placed > 10% of the embedding's width away", fontsize=11)
    fig.tight_layout()
    out = HERE / ("figure.png" if variant == "before" else f"figure_{variant}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
