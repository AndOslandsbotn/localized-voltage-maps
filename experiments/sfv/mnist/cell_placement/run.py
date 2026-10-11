import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.methods import fit_lvm, fit_lvm_chart, uniform_masses
from experiments.common.record import environment, model_record

HERE = Path(__file__).resolve().parent
N, SEEDS = 50_000, (0, 1, 2)
PLACEMENTS = ("kmeans", "farthest")
METHODS = {"lvm": fit_lvm, "lvm_chart": fit_lvm_chart}


def main() -> None:
    watch_memory(limit_mb=3000)
    (HERE / "embeddings").mkdir(exist_ok=True)
    X = load("eval", n=2000, seed=0)[0].astype(np.float32)
    for fit in METHODS.values():
        fit(X, seed=0).transform(X)
    for seed in SEEDS:
        for placement in PLACEMENTS:
            for method in METHODS:
                embed(method, placement, "counted", seed)
                with patch("lvm.region.cell_masses", uniform_masses):
                    embed(method, placement, "uniform", seed)


def embed(method: str, placement: str, masses: str, seed: int) -> None:
    path = HERE / "embeddings" / f"{method}_{placement}_{masses}_seed{seed}.npz"
    if path.exists():
        return
    config = {"cells": {"placement": placement, "masses": {"rel_error": None}}}
    X = load("eval", n=N, seed=seed)[0].astype(np.float32)
    start = time.perf_counter()
    model = METHODS[method](X, seed=seed, config=config)
    Z = model.transform(X)
    seconds = time.perf_counter() - start
    np.savez(path, Z=Z, seconds=seconds, environment=json.dumps(environment("cuda")),
             model=json.dumps(model_record(model)))


if __name__ == "__main__":
    main()
