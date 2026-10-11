import json
import time
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.methods import landmark_isomap_gpu
from experiments.common.record import environment
from experiments.sfv.mnist.distribution_shift.run import SEEDS, shifted

HERE = Path(__file__).resolve().parent


def main() -> None:
    watch_memory(limit_mb=3000)
    (HERE / "embeddings").mkdir(exist_ok=True)
    datasets = {seed: shifted(seed)[0] for seed in SEEDS}
    landmark_isomap_gpu(load("eval", n=2000, seed=0)[0].astype(np.float32), seed=0)
    for seed, data in datasets.items():
        path = HERE / "embeddings" / f"landmark_isomap_seed{seed}.npz"
        if path.exists():
            continue
        start = time.perf_counter()
        Z = landmark_isomap_gpu(data, seed=seed)
        seconds = time.perf_counter() - start
        np.savez(path, Z=Z, seconds=seconds, environment=json.dumps(environment("cuda")))


if __name__ == "__main__":
    main()
