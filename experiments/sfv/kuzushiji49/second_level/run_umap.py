import json
import time
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.methods import umap_gpu
from experiments.common.record import environment
from experiments.sfv.kuzushiji49.second_level.run import DATASET, MIN_POINTS, N_FIT, SEEDS, rows_of

HERE = Path(__file__).resolve().parent


def main() -> None:
    watch_memory(limit_mb=3000)
    for seed in SEEDS:
        path = HERE / "embeddings" / f"umap_seed{seed}.npz"
        if not path.exists():
            run(seed, path)


def run(seed: int, path: Path) -> None:
    X = load("eval", n=N_FIT, seed=0, dataset=DATASET)[0]
    regions = np.load(HERE / "embeddings" / f"seed{seed}.npz")
    region_of_fit, region_of_row = regions["region_of_fit"], regions["region_of_row"]
    umap_gpu(X[:2000], seed=seed)
    embeddings = {"same": np.zeros((N_FIT, 2)), "new": np.zeros((N_FIT, 2))}
    seconds = {"same": 0.0, "new": 0.0}
    for r in np.unique(region_of_fit):
        fit_rows = np.flatnonzero(region_of_fit == r)
        if len(fit_rows) < MIN_POINTS:
            continue
        start = time.perf_counter()
        embeddings["same"][fit_rows] = umap_gpu(X[fit_rows], seed=seed)
        seconds["same"] += time.perf_counter() - start
        all_rows = np.flatnonzero(region_of_row == r)
        new_points = rows_of(all_rows)
        start = time.perf_counter()
        Z = umap_gpu(new_points, seed=seed)
        seconds["new"] += time.perf_counter() - start
        del new_points
        embeddings["new"][fit_rows] = Z[np.searchsorted(all_rows, fit_rows)]
    np.savez(path, **embeddings, seconds=json.dumps(seconds), environment=json.dumps(environment("cuda")))


if __name__ == "__main__":
    main()
