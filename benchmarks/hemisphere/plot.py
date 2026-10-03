"""Draw hemisphere/figure.png: the true layout and each method's embedding, coloured two ways.

Top row: colour by height (pole -> rim); bottom row: by angle around the pole.
A good embedding of the half sphere shows smooth rings (top) and a smooth colour wheel (bottom).

    python benchmarks/hemisphere/plot.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
METHODS = ["lvm_gpu", "lvm_pca_gpu", "tsne_gpu"]


def main() -> None:
    methods = [m for m in METHODS if (HERE / f"{m}.npz").exists()]
    X = np.load(HERE / f"{methods[0]}.npz")["X"]
    height, angle = X[:, 2], np.arctan2(X[:, 1], X[:, 0])
    panels = [("truth: seen from above", X[:, :2], None)] + [
        (m, np.load(HERE / f"{m}.npz")["Z"], json.loads((HERE / f"{m}.json").read_text())) for m in methods]
    fig, axes = plt.subplots(2, len(panels), figsize=(4.6 * len(panels), 9.2))
    order = np.random.default_rng(0).permutation(len(X))
    for col, (name, Z, info) in enumerate(panels):
        for row, (c, cmap, what) in enumerate([(height, "viridis", "height"), (angle, "hsv", "angle")]):
            ax = axes[row, col]
            ax.scatter(Z[order, 0], Z[order, 1], c=c[order], cmap=cmap, s=0.6, rasterized=True)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_aspect("equal", adjustable="datalim")
            if row == 0:
                title = name if info is None else (
                    f"{info['label']}\ntrust {info['trustworthiness']:.3f}  cont {info['continuity']:.3f}  "
                    f"geodesic corr {info['geodesic_correlation']:.3f}  ({info['fit_s']:.1f} s)")
                ax.set_title(title, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"coloured by {what}")
    fig.suptitle(f"Half sphere (upper half of the unit sphere in 3-D), n = {len(X)}; settings tuned on MNIST, "
                 "unchanged", fontsize=11)
    fig.tight_layout()
    out = HERE / "figure.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
