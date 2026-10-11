import json
import time
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.methods import fit_lvm_chart
from experiments.common.record import environment, model_record

HERE = Path(__file__).resolve().parent
SIZES = (5_000, 10_000, 20_000, 50_000, 70_000, 100_000)
SEEDS = (0, 1, 2)


def all_points(n: int) -> dict:
    return {"sample": {"size": n}, "cells": {"masses": {"rel_error": None}}}


def main() -> None:
    watch_memory(limit_mb=3000)
    (HERE / "embeddings").mkdir(exist_ok=True)
    X = load("eval", n=2000, seed=0, dataset="mnist8m")[0]
    fit_lvm_chart(X, seed=0, config=all_points(len(X))).transform(X)
    for n in SIZES:
        for seed in SEEDS:
            path = HERE / "embeddings" / f"lvm_chart_n{n}_seed{seed}.npz"
            if not path.exists():
                embed(n, seed, path)


def embed(n: int, seed: int, path: Path) -> None:
    X = load("eval", n=n, seed=seed, dataset="mnist8m")[0]
    start = time.perf_counter()
    model = fit_lvm_chart(X, seed=seed, config=all_points(n))
    Z = model.transform(X)
    seconds = time.perf_counter() - start
    np.savez(path, Z=Z, seconds=seconds, environment=json.dumps(environment("cuda")),
             model=json.dumps(model_record(model)))
    model.save(HERE / "models" / path.stem)


if __name__ == "__main__":
    main()
