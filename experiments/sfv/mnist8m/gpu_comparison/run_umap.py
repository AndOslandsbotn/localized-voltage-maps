import json
import time
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.methods import umap_gpu
from experiments.common.record import environment

HERE = Path(__file__).resolve().parent
SIZES = (5_000, 10_000, 20_000, 50_000, 70_000, 100_000)
SEEDS = (0, 1, 2)


def main() -> None:
    watch_memory(limit_mb=3000)
    (HERE / "embeddings").mkdir(exist_ok=True)
    umap_gpu(load("eval", n=2000, seed=0, dataset="mnist8m")[0], seed=0)
    for n in SIZES:
        for seed in SEEDS:
            path = HERE / "embeddings" / f"umap_n{n}_seed{seed}.npz"
            if path.exists():
                continue
            X = load("eval", n=n, seed=seed, dataset="mnist8m")[0]
            start = time.perf_counter()
            Z = umap_gpu(X, seed=seed)
            seconds = time.perf_counter() - start
            np.savez(path, Z=Z, seconds=seconds, environment=json.dumps(environment("cuda")))


if __name__ == "__main__":
    main()
