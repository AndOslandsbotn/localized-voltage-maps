"""Draw figure_landmarks.png: LVM on the half sphere with different landmark choices.

    python benchmarks/half_sphere/diagnostics/landmarks/plot_variants.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
VARIANTS = [("lvm_gpu", "mutual information, 3 landmarks (default)"), ("lvm_mi_L6", "mutual information, 6"),
            ("lvm_mi_L9", "mutual information, 9"), ("lvm_maxmin_L3", "max-min, 3"), ("lvm_maxmin_L6", "max-min, 6")]


def main() -> None:
    fig, axes = plt.subplots(2, len(VARIANTS), figsize=(4.4 * len(VARIANTS), 9))
    for col, (name, title) in enumerate(VARIANTS):
        data, info = np.load(HERE / "embeddings" / f"{name}.npz"), json.loads((HERE / f"{name}.json").read_text())
        X, Z = data["X"], data["Z"]
        order = np.random.default_rng(0).permutation(len(X))
        for row, (c, cmap) in enumerate([(X[:, 2], "viridis"), (np.arctan2(X[:, 1], X[:, 0]), "hsv")]):
            ax = axes[row, col]
            ax.scatter(Z[order, 0], Z[order, 1], c=c[order], cmap=cmap, s=0.6, rasterized=True)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_aspect("equal", adjustable="datalim")
        axes[0, col].set_title(f"{title}\ntrust {info['trustworthiness']:.3f}  cont {info['continuity']:.3f}  "
                               f"geodesic corr {info['geodesic_correlation']:.3f}", fontsize=9)
    axes[0, 0].set_ylabel("coloured by height"), axes[1, 0].set_ylabel("coloured by angle")
    fig.suptitle("LVM on the half sphere (n = 20000): landmark selection and count", fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / "figure_landmarks.png", dpi=110)
    plt.close(fig)
    print(HERE / "figure_landmarks.png")


if __name__ == "__main__":
    main()
