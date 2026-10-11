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
N, SEEDS = 100_000, (0, 1, 2)
COUNTED = (10_000, 20_000, 50_000, 100_000)


def main() -> None:
    watch_memory(limit_mb=3000)
    (HERE / "embeddings").mkdir(exist_ok=True)
    X = load("eval", n=2000, seed=0, dataset="mnist8m")[0]
    fit_lvm(X, seed=0).transform(X)
    for seed in SEEDS:
        for counted in COUNTED:
            embed(f"counted_{counted}", seed, {"cells": {"masses": {"rel_error": None, "max_points": counted}}})
        with patch("lvm.region.cell_masses", uniform_masses):
            embed("uniform", seed, {})


def embed(name: str, seed: int, config: dict) -> None:
    path = HERE / "embeddings" / f"{name}_seed{seed}.npz"
    if path.exists():
        return
    X = load("eval", n=N, seed=seed, dataset="mnist8m")[0]
    start = time.perf_counter()
    model = fit_lvm(X, seed=seed, config=config)
    Z = model.transform(X)
    seconds = time.perf_counter() - start
    np.savez(path, Z=Z, seconds=seconds, environment=json.dumps(environment("cuda")),
             model=json.dumps(model_record(model)))


if __name__ == "__main__":
    main()
