import json
from pathlib import Path

import numpy as np

from experiments.common.datasets import stream
from experiments.common.guard import watch_memory
from experiments.common.methods import laplacian_eigenmaps_gpu
from experiments.common.record import environment

HERE = Path(__file__).resolve().parent
N_FIT, SEED = 50_000, 0


def main() -> None:
    watch_memory(limit_mb=3000)
    (HERE / "embeddings").mkdir(exist_ok=True)
    X = np.concatenate(list(stream("mnist8m", n=N_FIT)))
    Z = laplacian_eigenmaps_gpu(X, seed=SEED).astype(np.float32)
    np.savez(HERE / "embeddings" / "laplacian_eigenmaps.npz", Z=Z, n_fit=N_FIT, environment=json.dumps(environment("cuda")))


if __name__ == "__main__":
    main()
