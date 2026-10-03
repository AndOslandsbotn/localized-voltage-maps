"""Draw half_sphere/gallery_3d/figure.png: the truth and each 3-D embedding, coloured by height and by angle.

    python benchmarks/half_sphere/gallery_3d/plot.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
PANELS = [("lvm", "LVM, default count"), ("lvm_L12", "LVM, fixed count"), ("umap", "UMAP")]


def main() -> None:
    panels = [(n, t) for n, t in PANELS if (HERE / f"{n}.npz").exists()]
    X = np.load(HERE / f"{panels[0][0]}.npz")["X"]
    columns = [("truth", X, None)] + [(t, np.load(HERE / f"{n}.npz")["Z"], json.loads((HERE / f"{n}.json").read_text()))
                                      for n, t in panels]
    colours = [("height", X[:, 2], "viridis"), ("angle", np.arctan2(X[:, 1], X[:, 0]), "hsv")]
    sub = np.random.default_rng(0).permutation(len(X))[:8000]
    fig = plt.figure(figsize=(5 * len(columns), 10))
    for col, (title, Z, info) in enumerate(columns):
        Z = (Z - Z.mean(0)) / np.abs(Z - Z.mean(0)).max()            # same scale for every panel
        for row, (what, c, cmap) in enumerate(colours):
            ax = fig.add_subplot(2, len(columns), row * len(columns) + col + 1, projection="3d")
            ax.scatter(*Z[sub].T, c=c[sub], cmap=cmap, s=1.0, rasterized=True)
            ax.view_init(elev=25, azim=-60)
            ax.set_box_aspect((1, 1, 1))
            ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
            if row == 0:
                if info is None:
                    ax.set_title(title, fontsize=9)
                else:
                    landmarks = (info.get("info") or {}).get("n_landmarks")
                    ax.set_title(f"{title}" + (f" ({landmarks} landmarks)" if landmarks else "")
                                 + f"\ntrust {info['trustworthiness']:.3f}  cont {info['continuity']:.3f}  "
                                   f"global {info['distance_correlation']:.3f}", fontsize=9)
            if col == 0:
                ax.text2D(-0.05, 0.5, f"coloured by {what}", transform=ax.transAxes, rotation=90, va="center")
    fig.suptitle("Half sphere embedded in 3-D (20k points; each method at its MNIST-tuned settings, 3 components)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / "figure.png", dpi=110)
    print(HERE / "figure.png")


if __name__ == "__main__":
    main()
