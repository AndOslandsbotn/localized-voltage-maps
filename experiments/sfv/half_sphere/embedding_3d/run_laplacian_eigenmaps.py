import json
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.methods import laplacian_eigenmaps_gpu
from experiments.common.record import environment

HERE = Path(__file__).resolve().parent
N, SEED = 20_000, 0


def main() -> None:
    watch_memory(limit_mb=3000)
    (HERE / "embeddings").mkdir(exist_ok=True)
    X = load("eval", n=N, seed=SEED, dataset="half_sphere")[0].astype(np.float32)
    Z = laplacian_eigenmaps_gpu(X, seed=SEED, n_components=3)
    np.savez(HERE / "embeddings" / "laplacian_eigenmaps.npz", Z=Z, environment=json.dumps(environment("cuda")))


if __name__ == "__main__":
    main()
