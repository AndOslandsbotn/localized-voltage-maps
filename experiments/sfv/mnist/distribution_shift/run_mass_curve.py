import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.methods import fit_lvm, fit_lvm_chart
from experiments.common.record import environment, model_record
from experiments.sfv.mnist.distribution_shift.run import COMMON, RARE
from lvm.cells import cell_masses
from lvm.data import as_chunks

HERE = Path(__file__).resolve().parent
SEEDS = tuple(range(10))
COUNTED = (0, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 14_000)
METHODS = {"lvm": fit_lvm, "lvm_chart": fit_lvm_chart}
CONFIG = {"cells": {"masses": {"rel_error": None}}}


def shifted_rows(labels: np.ndarray, seed: int) -> np.ndarray:
    order = np.random.default_rng(seed).permutation(len(labels))
    rows = np.concatenate([order[labels[order] == d][:5_000 if d in COMMON else RARE] for d in range(10)])
    return np.random.default_rng(seed).permutation(rows)


def smoothed_masses(points: np.ndarray):
    """Add-one smoothing: mass_i = (count_i + 1) / (n + n_cells); uniform with no points."""
    def masses(chunks, centroids: torch.Tensor, *, config, device: str) -> torch.Tensor:
        counts = (cell_masses(as_chunks(points, config.data.chunk_size), centroids, config=config, device=device)
                  * len(points) if len(points) else torch.zeros(len(centroids), dtype=torch.float64, device=device))
        return (counts + 1.0) / (len(points) + len(centroids))
    return masses


def main() -> None:
    watch_memory(limit_mb=3000)
    (HERE / "embeddings").mkdir(exist_ok=True)
    mnist, labels = load("eval")
    mnist = mnist.astype(np.float32)
    balanced = load("tune")[0].astype(np.float32)
    for fit in METHODS.values():
        fit(mnist[:2000], seed=0).transform(mnist[:2000])
    for seed in SEEDS:
        data = mnist[shifted_rows(labels, seed)]
        for method in METHODS:
            for counted in COUNTED:
                embed(method, counted, seed, balanced, data)


def embed(method: str, counted: int, seed: int, balanced: np.ndarray, data: np.ndarray) -> None:
    path = HERE / "embeddings" / f"curve_{method}_{counted}_seed{seed}.npz"
    if path.exists():
        return
    with patch("lvm.region.cell_masses", smoothed_masses(data[:counted])):
        model = METHODS[method](balanced, seed=seed, config=CONFIG)
    np.savez(path, Z=model.transform(data), environment=json.dumps(environment("cuda")),
             model=json.dumps(model_record(model)))


if __name__ == "__main__":
    main()
