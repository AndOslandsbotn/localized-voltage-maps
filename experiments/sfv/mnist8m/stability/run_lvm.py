import json
import time
from pathlib import Path

import numpy as np

from experiments.common.datasets import stream
from experiments.common.guard import watch_memory
from experiments.common.methods import fit_lvm
from experiments.common.record import environment

HERE = Path(__file__).resolve().parent
N, SEEDS = 50_000, tuple(range(5))


def main() -> None:
    watch_memory(limit_mb=3000)
    (HERE / "embeddings").mkdir(exist_ok=True)
    X = np.concatenate(list(stream("mnist8m", n=N)))
    fit_lvm(X[:2000], seed=0).transform(X[:2000])
    for seed in SEEDS:
        path = HERE / "embeddings" / f"lvm_seed{seed}.npz"
        if path.exists():
            continue
        start = time.perf_counter()
        model = fit_lvm(X, seed=seed)
        Z = model.transform(X)
        seconds = time.perf_counter() - start
        np.savez(path, Z=Z, seconds=seconds, environment=json.dumps(environment("cuda")))
        model.save(HERE / "models" / path.stem)


if __name__ == "__main__":
    main()
