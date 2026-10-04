"""Draw half_sphere/gallery_3d/figure.png: the truth and each 3-D embedding, coloured by height and by angle.

Each embedding's axes come out in an arbitrary rotation, so before drawing it is turned and scaled onto the truth
(orthogonal Procrustes; this changes no score). The titles add the remaining shape error: the median distance from
a point to its true position after that alignment (the sphere has radius 1).

    python benchmarks/half_sphere/gallery_3d/plot.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.linalg import orthogonal_procrustes  # noqa: E402

HERE = Path(__file__).resolve().parent
PANELS = [("lvm", "LVM, default count"), ("lvm_pca", "LVM + local PCA, default count"), ("lvm_L12", "LVM, fixed count"),
          ("umap", "UMAP"), ("tsne", "t-SNE (CPU)")]


def aligned(Z: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Z rotated (or reflected) and scaled onto X by orthogonal Procrustes."""
    Zc, Xc = Z - Z.mean(0), X - X.mean(0)
    R, s = orthogonal_procrustes(Zc, Xc)
    return s / (Zc ** 2).sum() * Zc @ R + X.mean(0)


def main() -> None:
    panels = [(n, t) for n, t in PANELS if (HERE / "embeddings" / f"{n}.npz").exists()]
    X = np.load(HERE / "embeddings" / f"{panels[0][0]}.npz")["X"]
    columns = [("truth", X, None)] + [(t, aligned(np.load(HERE / "embeddings" / f"{n}.npz")["Z"], X),
                                       json.loads((HERE / f"{n}.json").read_text())) for n, t in panels]
    colours = [("height", X[:, 2], "viridis"), ("angle", np.arctan2(X[:, 1], X[:, 0]), "hsv")]
    sub = np.random.default_rng(0).permutation(len(X))[:8000]
    fig = plt.figure(figsize=(5 * len(columns), 10))
    for col, (title, Z, info) in enumerate(columns):
        error = float(np.median(np.linalg.norm(Z - X, axis=1)))
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
                                   f"global {info['distance_correlation']:.3f}\nshape error after alignment "
                                   f"{error:.3f}", fontsize=9)
            if col == 0:
                ax.text2D(-0.05, 0.5, f"coloured by {what}", transform=ax.transAxes, rotation=90, va="center")
    fig.suptitle("Half sphere embedded in 3-D (20k points; each method at its MNIST-tuned settings, 3 components; "
                 "each embedding turned and scaled onto the truth)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / "figure.png", dpi=110)
    print(HERE / "figure.png")


if __name__ == "__main__":
    main()
