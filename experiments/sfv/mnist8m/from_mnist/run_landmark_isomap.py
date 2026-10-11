import json
import time
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.methods import fit_landmark_isomap_gpu
from experiments.common.record import environment
from experiments.sfv.mnist8m.from_mnist.run import SEEDS, place

HERE = Path(__file__).resolve().parent


def main() -> None:
    watch_memory(limit_mb=3000)
    (HERE / "embeddings").mkdir(exist_ok=True)
    mnist = load("eval")[0].astype(np.float32)
    fit_landmark_isomap_gpu(mnist[:2000], seed=0).transform(mnist[:2000])
    for seed in SEEDS:
        path = HERE / "embeddings" / f"landmark_isomap_seed{seed}.npz"
        if path.exists():
            continue
        start = time.perf_counter()
        model = fit_landmark_isomap_gpu(mnist, seed=seed)
        fit_seconds = time.perf_counter() - start
        Z, sample, transform_seconds, n_placed = place(model, seed)
        np.savez(path, Z=Z, rows=sample, fit_seconds=fit_seconds, transform_seconds=transform_seconds,
                 n_placed=n_placed, environment=json.dumps(environment("cuda")))


if __name__ == "__main__":
    main()
