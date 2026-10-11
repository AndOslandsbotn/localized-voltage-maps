import json
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.methods import umap_gpu
from experiments.common.record import environment

HERE = Path(__file__).resolve().parent
N, SEEDS = 20_000, (0, 1, 2)


def main() -> None:
    watch_memory(limit_mb=3000)
    (HERE / "embeddings").mkdir(exist_ok=True)
    for seed in SEEDS:
        path = HERE / "embeddings" / f"umap_seed{seed}.npz"
        if path.exists():
            continue
        X = load("eval", n=N, seed=seed, dataset="swiss_roll")[0].astype(np.float32)
        Z = umap_gpu(X, seed=seed)
        np.savez(path, Z=Z, environment=json.dumps(environment("cuda")))


if __name__ == "__main__":
    main()
