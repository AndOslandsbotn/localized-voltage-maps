import json
import time
from pathlib import Path

import numpy as np

from experiments.common.guard import watch_memory
from experiments.common.methods import umap_gpu
from experiments.common.record import environment
from experiments.sfv.kuzushiji49.streaming.run_lvm import IN_MEMORY, SEEDS, read

HERE = Path(__file__).resolve().parent


def main() -> None:
    watch_memory(limit_mb=3000)
    (HERE / "embeddings").mkdir(exist_ok=True)
    umap_gpu(read(2000), seed=0)
    for n in IN_MEMORY:
        for seed in SEEDS:
            path = HERE / "embeddings" / f"umap_n{n}_seed{seed}.npz"
            if not path.exists():
                embed(n, seed, path)


def embed(n: int, seed: int, path: Path) -> None:
    start = time.perf_counter()
    X = read(n)
    Z = umap_gpu(X, seed=seed)
    seconds = time.perf_counter() - start
    np.savez(path, Z=Z, seconds=seconds, environment=json.dumps(environment("cuda")))


if __name__ == "__main__":
    main()
