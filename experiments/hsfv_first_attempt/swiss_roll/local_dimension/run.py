import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from experiments.common.datasets import load
from experiments.common.guard import watch_memory

HERE = Path(__file__).resolve().parent
N, SEEDS = 100_000, (0, 1, 2)
NOISE = (0.0, 0.3, 0.6, 1.0)
N_CELLS = (5, 10, 20, 50, 100, 200, 500, 1000, 2000)


def participation_ratio(points: np.ndarray) -> float:
    """(sum of the local PCA variances)^2 / sum of their squares: 1 for a line, 2 for a flat sheet, 3 for an isotropic
    blob in three dimensions."""
    variances = np.linalg.eigvalsh(np.cov(points.T))
    return float(variances.sum() ** 2 / np.square(variances).sum())


def main() -> None:
    """For each noise level and cell count: the median cell size (RMS distance of a cell's points to its centre) and
    the median over cells of the local PCA participation ratio."""
    watch_memory(limit_mb=3000)
    rows = []
    for sigma in NOISE:
        for seed in SEEDS:
            X = load("eval", n=N, seed=seed, dataset="swiss_roll")[0]
            X = X + np.random.default_rng(1000 + seed).normal(scale=sigma, size=X.shape)
            for n_cells in N_CELLS:
                km = KMeans(n_clusters=n_cells, n_init=1, random_state=seed).fit(X)
                sizes, dims = [], []
                for c in range(n_cells):
                    members = X[km.labels_ == c]
                    if len(members) < 5:
                        continue
                    sizes.append(float(np.sqrt(np.square(members - members.mean(axis=0)).sum(axis=1).mean())))
                    dims.append(participation_ratio(members))
                rows.append({"noise": sigma, "seed": seed, "n_cells": n_cells, "cell_size": float(np.median(sizes)),
                             "dimension": float(np.median(dims))})
            print(sigma, seed, flush=True)
    (HERE / "results.json").write_text(json.dumps({"n": N, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
