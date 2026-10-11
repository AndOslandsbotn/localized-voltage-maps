import json
import time
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.methods import fit_lvm
from experiments.common.record import environment, model_record

HERE = Path(__file__).resolve().parent
N, SEEDS = 100_000, (0, 1, 2)
N_CELLS = (300, 600, 1000, 2000)


def main() -> None:
    watch_memory(limit_mb=3000)
    (HERE / "embeddings").mkdir(exist_ok=True)
    X = load("eval", n=2000, seed=0, dataset="mnist8m")[0]
    fit_lvm(X, seed=0).transform(X)
    for seed in SEEDS:
        for n_cells in N_CELLS:
            embed(n_cells, seed)


def embed(n_cells: int, seed: int) -> None:
    path = HERE / "embeddings" / f"cells_{n_cells}_seed{seed}.npz"
    if path.exists():
        return
    X = load("eval", n=N, seed=seed, dataset="mnist8m")[0]
    start = time.perf_counter()
    model = fit_lvm(X, seed=seed, config={"cells": {"n_cells": n_cells, "masses": {"rel_error": None}}})
    Z = model.transform(X)
    seconds = time.perf_counter() - start
    np.savez(path, Z=Z, seconds=seconds, environment=json.dumps(environment("cuda")),
             model=json.dumps(model_record(model)))
    model.save(HERE / "models" / path.stem)


if __name__ == "__main__":
    main()
