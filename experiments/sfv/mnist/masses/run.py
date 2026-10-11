import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.methods import fit_lvm, uniform_masses
from experiments.common.record import environment, model_record

HERE = Path(__file__).resolve().parent
N, SEEDS = 50_000, (0, 1, 2)
CELLS_FROM = (1_000, 5_000, 50_000)
MASSES_FROM = (1_000, 5_000, 10_000, 50_000)


def main() -> None:
    watch_memory(limit_mb=3000)
    (HERE / "embeddings").mkdir(exist_ok=True)
    X = load("eval", n=2000, seed=0)[0].astype(np.float32)
    fit_lvm(X, seed=0).transform(X)
    for seed in SEEDS:
        for cells_from in CELLS_FROM:
            for masses_from in MASSES_FROM:
                if masses_from >= cells_from:
                    embed(cells_from, masses_from, seed)
            with patch("lvm.region.cell_masses", uniform_masses):
                embed(cells_from, "uniform", seed)


def embed(cells_from: int, masses_from: int | str, seed: int) -> None:
    path = HERE / "embeddings" / f"cells_{cells_from}_masses_{masses_from}_seed{seed}.npz"
    if path.exists():
        return
    masses = {"rel_error": None} | ({} if masses_from == "uniform" else {"max_points": masses_from})
    config = {"data": {"chunk_size": 1_000}, "sample": {"size": cells_from}, "cells": {"masses": masses}}
    X = load("eval", n=N, seed=seed)[0].astype(np.float32)
    start = time.perf_counter()
    model = fit_lvm(X, seed=seed, config=config)
    Z = model.transform(X)
    seconds = time.perf_counter() - start
    np.savez(path, Z=Z, seconds=seconds, environment=json.dumps(environment("cuda")),
             model=json.dumps(model_record(model)))


if __name__ == "__main__":
    main()
