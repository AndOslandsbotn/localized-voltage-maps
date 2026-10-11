import json
import time
from pathlib import Path

import numpy as np

from experiments.common.datasets import stream
from experiments.common.guard import watch_memory
from experiments.common.methods import laplacian_eigenmaps_gpu
from experiments.common.record import environment

HERE = Path(__file__).resolve().parent
N, SEEDS = 50_000, tuple(range(5))


def main() -> None:
    watch_memory(limit_mb=3000)
    (HERE / "embeddings").mkdir(exist_ok=True)
    X = np.concatenate(list(stream("mnist8m", n=N)))
    laplacian_eigenmaps_gpu(X[:2000], seed=0)
    for seed in SEEDS:
        path = HERE / "embeddings" / f"laplacian_eigenmaps_seed{seed}.npz"
        if path.exists():
            continue
        start = time.perf_counter()
        Z = laplacian_eigenmaps_gpu(X, seed=seed)
        seconds = time.perf_counter() - start
        np.savez(path, Z=Z, seconds=seconds, environment=json.dumps(environment("cuda")))


if __name__ == "__main__":
    main()
