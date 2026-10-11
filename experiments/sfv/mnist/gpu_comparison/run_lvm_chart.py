import json
import time
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.methods import fit_lvm_chart
from experiments.common.record import environment

HERE = Path(__file__).resolve().parent
SIZES = (5_000, 10_000, 20_000, 40_000, 50_000)
SEEDS = tuple(range(10))


def main() -> None:
    watch_memory(limit_mb=3000)
    (HERE / "embeddings").mkdir(exist_ok=True)
    warm_up = load("eval", n=2000, seed=0)[0].astype(np.float32)
    fit_lvm_chart(warm_up, seed=0).transform(warm_up)
    for n in SIZES:
        for seed in SEEDS:
            path = HERE / "embeddings" / f"lvm_chart_n{n}_seed{seed}.npz"
            if not path.exists():
                embed(n, seed, path)


def embed(n: int, seed: int, path: Path) -> None:
    X = load("eval", n=n, seed=0)[0].astype(np.float32)
    start = time.perf_counter()
    model = fit_lvm_chart(X, seed=seed)
    Z = model.transform(X)
    seconds = time.perf_counter() - start
    np.savez(path, Z=Z, seconds=seconds, environment=json.dumps(environment("cuda")))
    model.save(HERE / "models" / path.stem)


if __name__ == "__main__":
    main()
