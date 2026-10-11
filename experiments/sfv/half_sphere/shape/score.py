import json
from pathlib import Path

import numpy as np
from scipy.spatial import procrustes
from scipy.stats import spearmanr

from experiments.common.datasets import load
from experiments.common.guard import watch_memory

HERE = Path(__file__).resolve().parent
EMBEDDINGS = HERE.parent / "embedding" / "embeddings"
N, SEEDS = 20_000, (0, 1, 2)
METHODS = ("lvm", "lvm_chart", "umap", "tsne", "laplacian_eigenmaps", "landmark_isomap")


def unfolding(X: np.ndarray) -> np.ndarray:
    """The half sphere's exact flat unrolling (azimuthal equidistant): each point keeps its angle around the pole,
    and its distance from the centre is its distance from the pole along the sphere, arccos(x_3)."""
    radius, angle = np.arccos(np.clip(X[:, 2], -1, 1)), np.arctan2(X[:, 1], X[:, 0])
    return np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])


def rings(X: np.ndarray, Z: np.ndarray) -> float:
    """Rank correlation between a point's height and its closeness to the map's centre (the mean of the map): 1 when
    higher points lie nearer the middle and the rim lies outside."""
    return float(spearmanr(X[:, 2], -np.linalg.norm(Z - Z.mean(axis=0), axis=1))[0])


def main() -> None:
    watch_memory(limit_mb=3000)
    rows = []
    for seed in SEEDS:
        X = load("eval", n=N, seed=seed, dataset="half_sphere")[0]
        flat = unfolding(X)
        for name in METHODS:
            Z = np.load(EMBEDDINGS / f"{name}_seed{seed}.npz")["Z"]
            rows.append({"method": name, "seed": seed, "procrustes": float(procrustes(flat, Z)[2]),
                         "rings": rings(X, Z)})
    (HERE / "results.json").write_text(json.dumps({"dataset": "half_sphere", "split": "eval", "n": N, "seeds": SEEDS,
                                                   "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
