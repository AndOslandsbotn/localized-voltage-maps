"""Draw diagnostics/kernels/figure.png: the four kernel variants' embeddings (tuning split, seed 0).

    python benchmarks/mnist/diagnostics/kernels/plot.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
TITLES = {"radial": "current: hard kernel for graph and points", "A": "A: hard graph, tapered points",
          "B": "B: tapered graph and points", "C": "C: Gaussian graph and points (σ = r/3, cut at r)",
          "D": "D: hard graph, points to their 10 nearest cells (adaptive)",
          "E": "E: hard kernel, r so a typical point has 10 cells within r"}
SWEEP = [f"D_k{k}_s{s}" for k in (3, 5, 10) for s in (1, 4, 16)]
TITLES.update({v: f"D: {v.split('_')[1][1:]} nearest cells, sharpness {v.split('_')[2][1:]}" for v in SWEEP})
LOCAL = [f"L_s{s}_f{f}" for s in (4, 16) for f in (0.25, 0.5)]
TITLES.update({v: f"3 nearest cells, sharpness {v.split('_')[1][1:]}, local scale fill {v.split('_')[2][1:]}"
               for v in LOCAL})
FIGURES = {"figure.png": ["radial", "A", "B", "C"], "figure_DE.png": ["radial", "D", "E"], "figure_sweep.png": SWEEP,
           "figure_local.png": ["radial", "D_k3_s16", "D_k3_s4", *LOCAL[2:], *LOCAL[:2]],
           "figure_pca.png": ["radial", "P_f0.35", "P_f0.5"]}
TITLES.update({f"P_f{f}": f"3 nearest cells, sharpness 16, local PCA per cell, fill {f}" for f in (0.35, 0.5)})


def main() -> None:
    for name, variants in FIGURES.items():
        draw(name, variants)


def draw(name: str, variants: list[str]) -> None:
    results = {(r["variant"], r["seed"]): r for r in map(json.loads, (HERE / "results.jsonl").read_text().splitlines())}
    cols = 3 if len(variants) > 4 else len(variants)
    rows = -(-len(variants) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 6 * rows), squeeze=False)
    for ax, variant in zip(axes.flat, variants):
        data, r = np.load(HERE / f"Z_{variant}.npz"), results[(variant, 0)]
        Z, y = data["Z"], data["y"]
        order = np.random.default_rng(0).permutation(len(y))
        ax.scatter(Z[order, 0], Z[order, 1], c=y[order], cmap="tab10", vmin=-0.5, vmax=9.5, s=0.5, rasterized=True)
        for digit in range(10):
            cx, cy = np.median(Z[y == digit], axis=0)
            ax.text(cx, cy, str(digit), fontsize=12, weight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.15", fc="white", alpha=0.7, lw=0))
        ax.set_title(f"{TITLES[variant]}\ntrust {r['trustworthiness']:.3f}  cont {r['continuity']:.3f}  "
                     f"5-NN {r['knn_accuracy']:.3f}  global {r['distance_correlation']:.3f}\n"
                     f"{r['occupied_squares']} occupied squares (1/1000 grid), "
                     f"{r['share_in_150_fullest']:.0%} of points in the 150 fullest", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
    for ax in list(axes.flat)[len(variants):]:
        ax.axis("off")
    fig.suptitle("LVM kernel variants, MNIST tuning split (20k), seed 0, coloured by digit", fontsize=11)
    fig.tight_layout()
    out = HERE / name
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
