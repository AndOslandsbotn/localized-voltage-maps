"""Draw gallery/figure.png: the saved embeddings side by side, coloured by digit.

    python benchmarks/gallery/plot.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PANELS = ["lvm", "lvm_pca", "umap", "tsne", "tsne_global", "le", "lisomap"]


def main() -> None:
    panels = [p for p in PANELS if (HERE / f"{p}.npz").exists()]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10.5))
    for ax, name in zip(axes.flat, panels):
        data, info = np.load(HERE / f"{name}.npz"), json.loads((HERE / f"{name}.json").read_text())
        Z, y = data["Z"], data["y"]
        order = np.random.default_rng(0).permutation(len(y))          # no digit drawn on top of all others
        ax.scatter(Z[order, 0], Z[order, 1], c=y[order], cmap="tab10", vmin=-0.5, vmax=9.5, s=0.3,
                   rasterized=True)
        for digit in range(10):
            cx, cy = np.median(Z[y == digit], axis=0)
            ax.text(cx, cy, str(digit), fontsize=13, weight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.15", fc="white", alpha=0.7, lw=0))
        params = ", ".join(f"{k}={v}" for k, v in info["params"].items()) if not info["method"].startswith("lvm") \
            else f"{info['params']['cells']['n_cells']} cells, reach, Landmark MDS" + \
            (", local PCA per cell" if info["method"].startswith("lvm_pca") else "")
        ax.set_title(f"{info['label']}\n{params}\ntrust {info['trustworthiness']:.3f}  cont {info['continuity']:.3f}  "
                     f"5-NN {info['knn_accuracy']:.3f}  global {info['distance_correlation']:.3f}  "
                     f"({info['fit_s']:.1f} s)", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.suptitle("Embeddings of the MNIST evaluation split (50k), seed 0, coloured by digit "
                 "(labels at each digit's median)", fontsize=11)
    fig.tight_layout()
    out = HERE / "figure.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
