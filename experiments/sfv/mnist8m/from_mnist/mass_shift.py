import json
from pathlib import Path

import numpy as np

from experiments.common.datasets import load, stream
from experiments.common.guard import watch_memory
from experiments.common.methods import fit_lvm
from lvm.cells import cell_masses
from lvm.config import load_config
from lvm.data import as_chunks

HERE = Path(__file__).resolve().parent
SEEDS = (0, 1, 2)
CONFIG = {"cells": {"masses": {"rel_error": None}}}


def main() -> None:
    watch_memory(limit_mb=3000)
    mnist = load("eval")[0].astype(np.float32)
    other_split = load("tune")[0].astype(np.float32)
    rows = []
    for seed in SEEDS:
        centroids = fit_lvm(mnist, seed=seed, config=CONFIG).root_.centroids
        on_mnist = masses(as_chunks(mnist, 10_000), centroids)
        rows.append({"seed": seed, "other_mnist_split": shift(on_mnist, masses(as_chunks(other_split, 10_000), centroids)),
                     "mnist8m": shift(on_mnist, masses(stream("mnist8m"), centroids))})
    (HERE / "mass_shift.json").write_text(json.dumps({"seeds": SEEDS, "rows": rows}, indent=2))


def masses(chunks, centroids) -> np.ndarray:
    config = load_config(overrides=CONFIG)
    return cell_masses(chunks, centroids, config=config, device="cuda").cpu().numpy()


def shift(a: np.ndarray, b: np.ndarray) -> dict:
    ratio = np.maximum(a, 1e-12) / np.maximum(b, 1e-12)
    return {"total_variation": float(0.5 * np.abs(a - b).sum()),
            "cells_off_by_2x": float(np.mean((ratio > 2) | (ratio < 0.5))),
            "log_mass_correlation": float(np.corrcoef(np.log(np.maximum(a, 1e-9)), np.log(np.maximum(b, 1e-9)))[0, 1])}


if __name__ == "__main__":
    main()
