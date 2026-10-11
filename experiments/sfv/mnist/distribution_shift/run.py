import json
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.methods import fit_lvm, fit_lvm_chart, uniform_masses
from experiments.common.record import environment, model_record
from lvm.cells import cell_masses
from lvm.data import as_chunks

HERE = Path(__file__).resolve().parent
SEEDS, COMMON, RARE = (0, 1, 2), (1, 7), 500
CONFIG = {"cells": {"masses": {"rel_error": None}}}
COUNTED = (1_000, 2_000, 5_000, 10_000)
CONDITIONS = ("uniform", "balanced", *(f"shifted_{m}" for m in COUNTED), "shifted", "refit_uniform", "refit")
METHODS = {"lvm": fit_lvm, "lvm_chart": fit_lvm_chart}


def shifted(seed: int) -> tuple[np.ndarray, np.ndarray]:
    X, labels = load("eval", seed=seed)
    rows = np.concatenate([np.flatnonzero(labels == d)[:5_000 if d in COMMON else RARE] for d in range(10)])
    rows = np.random.default_rng(seed).permutation(rows)
    return X[rows].astype(np.float32), labels[rows]


def main() -> None:
    watch_memory(limit_mb=3000)
    (HERE / "embeddings").mkdir(exist_ok=True)
    datasets = {seed: (load("tune", seed=seed)[0].astype(np.float32), shifted(seed)[0]) for seed in SEEDS}
    X = load("eval", n=2000, seed=0)[0].astype(np.float32)
    for fit in METHODS.values():
        fit(X, seed=0).transform(X)
    for seed, (balanced, data) in datasets.items():
        for method in METHODS:
            for condition in CONDITIONS:
                embed(method, condition, seed, balanced, data)


def embed(method: str, condition: str, seed: int, balanced: np.ndarray, data: np.ndarray) -> None:
    path = HERE / "embeddings" / f"{method}_{condition}_seed{seed}.npz"
    if path.exists():
        return
    fitted_on = data if condition.startswith("refit") else balanced
    masses = nullcontext()
    if condition in ("uniform", "refit_uniform"):
        masses = patch("lvm.region.cell_masses", uniform_masses)
    if condition.startswith("shifted"):
        counted = data[:int(condition.removeprefix("shifted_"))] if "_" in condition else data
        masses = patch("lvm.region.cell_masses",
                       lambda chunks, centroids, *, config, device: cell_masses(
                           as_chunks(counted, config.data.chunk_size), centroids, config=config, device=device))
    start = time.perf_counter()
    with masses:
        model = METHODS[method](fitted_on, seed=seed, config=CONFIG)
    Z = model.transform(data)
    seconds = time.perf_counter() - start
    region = model.root_
    np.savez(path, Z=Z, seconds=seconds, environment=json.dumps(environment("cuda")),
             model=json.dumps(model_record(model)), centroids=region.centroids.cpu().numpy(),
             radius=region.radius.cpu().numpy(), maps=region.maps.cpu().numpy(),
             landmarks=region.landmarks.cpu().numpy())


if __name__ == "__main__":
    main()
