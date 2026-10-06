"""Single-level LVM on all 70,000 MNIST images.

Fits one level (cells -> masses -> graph -> rho_g -> landmarks -> embedding),
embeds every image, and reports stage timings plus two quality measures:
trustworthiness (are embedding neighbours real neighbours?) and 5-NN label
accuracy in the embedding. Saves a scatter plot to examples/output/.

    python examples/mnist_single_level.py [--device cpu]

MNIST is fetched from OpenML via scikit-learn and cached in data/.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.manifold import trustworthiness
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from lvm_old.config import load_config
from lvm_old.pipeline import fit_level
from lvm_old.stream import array_source

ROOT = Path(__file__).resolve().parents[1]


def load_mnist(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """All 70,000 images as float64 in [0, 1], shuffled (the pipeline expects shuffled input)."""
    X, y = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff", data_home=ROOT / "data", return_X_y=True)
    order = np.random.default_rng(seed).permutation(X.shape[0])
    return X[order].astype(np.float64) / 255.0, y[order].astype(int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help="cuda or cpu (default: config)")
    parser.add_argument("--config", default=None, help="run config YAML merged over the defaults")
    parser.add_argument("--eval-size", type=int, default=5000, help="points used for the quality measures")
    args = parser.parse_args()

    config = load_config(args.config)
    print("Loading MNIST...")
    X, y = load_mnist(config.compute.seed)
    print(f"  {X.shape[0]} images, {X.shape[1]} dims")

    source = array_source(X, config.data.chunk_size)
    t0 = time.perf_counter()
    model = fit_level(source, config, device=args.device)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    Z = np.concatenate(list(model.transform_source(source)))
    t_transform = time.perf_counter() - t0

    print(f"\nFit: {t_fit:.1f}s   embed all {X.shape[0]} points: {t_transform:.1f}s")
    for stage, seconds in model.timings.items():
        print(f"  {stage:10s} {seconds:7.2f}s")
    print(
        f"cells={model.centroids.shape[0]}  r={model.r:.3g}  avg degree={model.K.sum(1).mean():.1f}  "
        f"rho_g={model.rho.rho_g:.3g}  support={model.rho.support_fraction:.3f}  "
        f"MI={model.landmarks.scores[-1]:.2f} nats"
    )

    idx = np.random.default_rng(1).choice(X.shape[0], args.eval_size, replace=False)
    trust = trustworthiness(X[idx], Z[idx], n_neighbors=10)
    knn = cross_val_score(KNeighborsClassifier(5), Z[idx], y[idx], cv=5).mean()
    print(f"trustworthiness(k=10)={trust:.3f}   5-NN label accuracy={knn:.3f}   (on {args.eval_size} points)")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = ROOT / "examples" / "output" / "mnist_single_level.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.get_cmap("tab10")(np.arange(10))
    ax.scatter(Z[:, 0], Z[:, 1], c=colors[y], s=0.3, alpha=0.5, rasterized=True)
    # Digit labels at each class's median instead of a legend: matplotlib 3.10's
    # legend deep-copies marker paths, which recurses forever on Python 3.14.
    for d in range(10):
        cx, cy = np.median(Z[y == d], axis=0)
        ax.text(cx, cy, str(d), fontsize=16, weight="bold", ha="center", va="center",
                color="black", bbox={"facecolor": colors[d], "alpha": 0.8, "boxstyle": "round"})
    ax.set_xlabel("LVM component 1")
    ax.set_ylabel("LVM component 2")
    ax.set_title(f"Single-level LVM on MNIST (70k), {len(model.landmark_cells)} landmarks")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"plot: {out}")


if __name__ == "__main__":
    main()
