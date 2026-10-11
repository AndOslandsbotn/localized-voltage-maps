import json
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.methods import landmark_isomap_gpu
from experiments.common.record import environment

HERE = Path(__file__).resolve().parent
N, SEEDS = 20_000, (0, 1, 2)


def main() -> None:
    watch_memory(limit_mb=3000)
    (HERE / "embeddings").mkdir(exist_ok=True)
    for seed in SEEDS:
        path = HERE / "embeddings" / f"landmark_isomap_seed{seed}.npz"
        if path.exists():
            continue
        X = load("eval", n=N, seed=seed, dataset="half_sphere")[0].astype(np.float32)
        Z = landmark_isomap_gpu(X, seed=seed, n_components=3)
        np.savez(path, Z=Z, environment=json.dumps(environment("cuda")))


if __name__ == "__main__":
    main()
