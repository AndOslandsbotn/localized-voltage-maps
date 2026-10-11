import json
import time
from pathlib import Path

import numpy as np

from experiments.common.datasets import DATA, stream
from experiments.common.guard import watch_memory
from experiments.common.methods import fit_lvm
from experiments.common.record import environment, model_record

HERE = Path(__file__).resolve().parent
DATASET, N_FEATURES = "kuzushiji49", 784
SIZES = (5_000, 20_000, 50_000, 100_000, 150_000, 200_000, 270_912)
IN_MEMORY = (5_000, 20_000, 50_000, 100_000, 150_000)
SEEDS = (0, 1, 2)


def read(n: int) -> np.ndarray:
    """The first n points of the shuffled file, read into one array: the data sets are nested and the same for every
    seed and method, so the seeds vary only the methods."""
    X, offset = np.empty((n, N_FEATURES), dtype=np.float32), 0
    for chunk in stream(DATASET, n=n):
        X[offset:offset + len(chunk)] = chunk
        offset += len(chunk)
    return X


def labels(n: int) -> np.ndarray:
    return np.fromfile(DATA / DATASET / "y.i8", dtype=np.int8)[:n].astype(np.int64)


def config(n: int) -> dict:
    return {"sample": {"size": n}, "cells": {"masses": {"rel_error": None}}}


def main() -> None:
    watch_memory(limit_mb=3000)
    (HERE / "embeddings").mkdir(exist_ok=True)
    warm_up = read(2000)
    fit_lvm(warm_up, seed=0, config=config(2000)).transform(warm_up)
    for n in IN_MEMORY:
        for seed in SEEDS:
            path = HERE / "embeddings" / f"lvm_n{n}_seed{seed}.npz"
            if not path.exists():
                embed(n, seed, path)


def embed(n: int, seed: int, path: Path) -> None:
    start = time.perf_counter()
    X = read(n)
    model = fit_lvm(X, seed=seed, config=config(n))
    Z = model.transform(X)
    seconds = time.perf_counter() - start
    np.savez(path, Z=Z, seconds=seconds, environment=json.dumps(environment("cuda")),
             model=json.dumps(model_record(model)))
    model.save(HERE / "models" / path.stem)


if __name__ == "__main__":
    main()
