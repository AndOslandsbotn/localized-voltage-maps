import json
import time
from pathlib import Path

import numpy as np

from experiments.common.datasets import stream
from experiments.common.guard import watch_memory
from experiments.common.methods import tsne_gpu
from experiments.common.record import environment
from experiments.sfv.kuzushiji49.hierarchy_scaling.run_lvm import DATASET, SEEDS, SIZES, read, window

HERE = Path(__file__).resolve().parent


def main() -> None:
    watch_memory(limit_mb=3000)
    (HERE / "embeddings").mkdir(exist_ok=True)
    warm_up = np.concatenate(list(stream(DATASET, n=2000)))
    tsne_gpu(warm_up, seed=0)
    for n in SIZES:
        for seed in SEEDS:
            path = HERE / "embeddings" / f"tsne_n{n}_seed{seed}.npz"
            if not path.exists():
                run(n, seed, path)


def run(n: int, seed: int, path: Path) -> None:
    start = time.perf_counter()
    X = read(n, seed)
    read_seconds = time.perf_counter() - start
    Z = tsne_gpu(X, seed=seed)
    seconds = time.perf_counter() - start
    np.savez(path, start=window(n, seed), Z=Z, seconds=seconds, read_seconds=read_seconds, environment=json.dumps(environment("cuda")))


if __name__ == "__main__":
    main()
