import json
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.methods import lvm
from experiments.common.record import environment

HERE = Path(__file__).resolve().parent
N, SEEDS, COUNTS = 20_000, (0, 1, 2), (3, 6, 9, 12)
GRAPH = {"graph": {"radius": {"adaptive_per_cell": {"k": 5}}}}


def main() -> None:
    watch_memory(limit_mb=3000)
    (HERE / "embeddings").mkdir(exist_ok=True)
    for seed in SEEDS:
        X = load("eval", n=N, seed=seed, dataset="swiss_roll")[0].astype(np.float32)
        for count in COUNTS:
            path = HERE / "embeddings" / f"landmarks_{count}_seed{seed}.npz"
            if path.exists():
                continue
            config = {**GRAPH, "landmarks": {"count": {"strategy": "fixed", "fixed": {"n": count}}}}
            np.savez(path, Z=lvm(X, seed=seed, config=config), environment=json.dumps(environment("cuda")))


if __name__ == "__main__":
    main()
