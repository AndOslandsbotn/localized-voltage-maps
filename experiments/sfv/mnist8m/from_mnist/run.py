import json
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.common.datasets import load, stream
from experiments.common.guard import watch_memory
from experiments.common.methods import fit_lvm, fit_lvm_chart, uniform_masses
from experiments.common.record import environment, model_record
from lvm.cells import cell_masses

HERE = Path(__file__).resolve().parent
SEEDS, SAMPLE, N_REFIT = (0, 1, 2), 5_000, 50_000
COUNTED = (10_000, 100_000, 1_000_000)
CONDITIONS = ("uniform", "mnist", *(f"mnist8m_{m}" for m in COUNTED), "mnist8m", "refit")
METHODS = {"lvm": fit_lvm, "lvm_chart": fit_lvm_chart}
CONFIG = {"cells": {"masses": {"rel_error": None}}}


def main() -> None:
    watch_memory(limit_mb=3000)
    (HERE / "embeddings").mkdir(exist_ok=True)
    mnist = load("eval")[0].astype(np.float32)
    mnist8m = np.concatenate(list(stream("mnist8m", n=N_REFIT)))
    warm_up = mnist[:2000]
    for fit in METHODS.values():
        fit(warm_up, seed=0).transform(warm_up)
    for seed in SEEDS:
        for method in METHODS:
            for condition in CONDITIONS:
                embed(method, condition, seed, mnist, mnist8m)


def embed(method: str, condition: str, seed: int, mnist: np.ndarray, mnist8m: np.ndarray) -> None:
    path = HERE / "embeddings" / f"{method}_{condition}_seed{seed}.npz"
    if path.exists():
        return
    masses = nullcontext()
    if condition == "uniform":
        masses = patch("lvm.region.cell_masses", uniform_masses)
    if condition.startswith("mnist8m"):
        counted = stream("mnist8m", n=int(condition.removeprefix("mnist8m_"))) if "_" in condition else stream("mnist8m")
        masses = patch("lvm.region.cell_masses", lambda chunks, centroids, *, config, device: cell_masses(
            counted, centroids, config=config, device=device))
    start = time.perf_counter()
    with masses:
        model = METHODS[method](mnist8m if condition == "refit" else mnist, seed=seed, config=CONFIG)
    fit_seconds = time.perf_counter() - start
    Z, sample, transform_seconds, n_placed = place(model, seed)
    np.savez(path, Z=Z, rows=sample, fit_seconds=fit_seconds, transform_seconds=transform_seconds, n_placed=n_placed,
             environment=json.dumps(environment("cuda")), model=json.dumps(model_record(model)))


def place(model, seed: int):
    data = stream("mnist8m")
    sample = np.sort(np.random.default_rng(seed).choice(data.n, SAMPLE, replace=False))
    start, placed, offset = time.perf_counter(), [], 0
    for chunk in data:
        Z = model.transform(chunk)
        inside = sample[(sample >= offset) & (sample < offset + len(chunk))]
        placed.append(Z[inside - offset])
        offset += len(chunk)
    return np.concatenate(placed), sample, time.perf_counter() - start, offset


if __name__ == "__main__":
    main()
