import json
from pathlib import Path

import numpy as np
from scipy.spatial import procrustes
from scipy.stats import spearmanr

from experiments.common.datasets import load
from experiments.common.guard import watch_memory

HERE = Path(__file__).resolve().parent
EMBEDDINGS = HERE.parent / "embedding_3d" / "embeddings"
N, SEEDS = 20_000, (0, 1, 2)
METHODS = ("lvm", "lvm_chart", "umap", "tsne", "laplacian_eigenmaps", "landmark_isomap")


def shape(X: np.ndarray, Z: np.ndarray) -> dict:
    """The 3-D map turned, reflected and scaled onto the half sphere itself (orthogonal Procrustes): what is left over
    (the disparity), and the rank correlation between each point's true height and its height in the turned map."""
    sphere, turned, disparity = procrustes(X, Z)
    return {"procrustes": float(disparity), "height": float(spearmanr(sphere[:, 2], turned[:, 2])[0])}


def main() -> None:
    watch_memory(limit_mb=3000)
    rows = []
    for seed in SEEDS:
        X = load("eval", n=N, seed=seed, dataset="half_sphere")[0]
        for name in METHODS:
            Z = np.load(EMBEDDINGS / f"{name}_seed{seed}.npz")["Z"]
            rows.append({"method": name, "seed": seed, **shape(X, Z)})
    (HERE / "results.json").write_text(json.dumps({"dataset": "half_sphere", "split": "eval", "n": N, "seeds": SEEDS,
                                                   "dimensions": 3, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
