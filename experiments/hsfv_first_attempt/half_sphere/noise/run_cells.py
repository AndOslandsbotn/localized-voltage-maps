from pathlib import Path

from experiments.common.guard import watch_memory
from experiments.hsfv_first_attempt.common.overlap import fit_overlapping
from experiments.hsfv_first_attempt.half_sphere.chained_distances.run import CONFIG, SEEDS
from experiments.hsfv_first_attempt.half_sphere.noise.run import NOISE, noisy
from lvm.region import load_region

HERE = Path(__file__).resolve().parent
LEVEL2_CELLS = (30, 100)


def main() -> None:
    """Level 2 with fewer, larger cells per region (two regions per point), on the saved repaired noisy roots."""
    watch_memory(limit_mb=3000)
    for sigma in NOISE:
        for seed in SEEDS:
            folder = HERE / "models" / f"noise{sigma}" / f"seed{seed}"
            _, X = noisy(seed, sigma)
            root = load_region(folder / "root.npz", "cuda")
            for cells in LEVEL2_CELLS:
                target = folder / f"m2_cells{cells}"
                if not (target / "regions.npy").exists():
                    fit_overlapping(X, root, config={**CONFIG, "cells": {"n_cells": cells}}, m=2, seed=seed,
                                    folder=target)


if __name__ == "__main__":
    main()
