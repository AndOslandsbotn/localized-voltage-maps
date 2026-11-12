from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.hsfv_first_attempt.common.levels import fit_two_levels, load_two_levels
from experiments.hsfv_first_attempt.common.overlap import fit_overlapping
from experiments.hsfv_first_attempt.common.repair import repaired_root
from experiments.hsfv_first_attempt.swiss_roll.chained_distances.run import CONFIG, DATASET, N, SEEDS
from lvm import LocalizedVoltageMaps
from lvm.region import save_region

HERE = Path(__file__).resolve().parent
NOISE, ROOT_CELLS, FINE_CELLS = (0.0, 0.3, 0.6, 1.0), 30, 300


def noisy(seed: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """The clean points (for the true distances) and the points with Gaussian noise of standard deviation sigma."""
    X = load("eval", n=N, seed=seed, dataset=DATASET)[0].astype(np.float32)
    return X, X + np.random.default_rng(1000 + seed).normal(scale=sigma, size=X.shape).astype(np.float32)


def main() -> None:
    """Per noise level and seed, all fitted on the noisy points: the coarse root with its level 2, the repaired root,
    level 2 with two regions per point, and a fine root."""
    watch_memory(limit_mb=3000)
    for sigma in NOISE:
        for seed in SEEDS:
            folder = HERE / "models" / f"noise{sigma}" / f"seed{seed}"
            if (folder / "fine" / "model.yaml").exists():
                continue
            _, X = noisy(seed, sigma)
            fit_two_levels(X, config=CONFIG, level1_cells=ROOT_CELLS, seed=seed, folder=folder / "base")
            level1, level2, _ = load_two_levels(folder / "base")
            root, _ = repaired_root(level1.root_, level2, level1.config_)
            save_region(root, folder / "root.npz")
            fit_overlapping(X, root, config=CONFIG, m=2, seed=seed, folder=folder / "m2")
            LocalizedVoltageMaps(local_chart="none", random_state=seed,
                                 config={**CONFIG, "cells": {"n_cells": FINE_CELLS}}).fit(X).save(folder / "fine")


if __name__ == "__main__":
    main()
