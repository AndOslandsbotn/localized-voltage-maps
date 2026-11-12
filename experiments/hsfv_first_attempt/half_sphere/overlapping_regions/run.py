from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.hsfv_first_attempt.common.levels import fit_two_levels, load_two_levels
from experiments.hsfv_first_attempt.common.overlap import fit_overlapping
from experiments.hsfv_first_attempt.common.repair import repaired_root
from experiments.hsfv_first_attempt.half_sphere.chained_distances.run import CONFIG, DATASET, N, SEEDS
from lvm.region import save_region

HERE = Path(__file__).resolve().parent
ROOT_CELLS, MS = 30, (1, 2)


def main() -> None:
    """Per seed: a coarse root with its single-region level 2, the root repaired by it, and level 2 refitted with
    each point in its m strongest regions (m = 1 and 2)."""
    watch_memory(limit_mb=3000)
    for seed in SEEDS:
        folder = HERE / "models" / f"seed{seed}"
        if (folder / f"m{MS[-1]}" / "regions.npy").exists():
            continue
        X = load("eval", n=N, seed=seed, dataset=DATASET)[0].astype(np.float32)
        fit_two_levels(X, config=CONFIG, level1_cells=ROOT_CELLS, seed=seed, folder=folder / "base")
        level1, level2, _ = load_two_levels(folder / "base")
        root, _ = repaired_root(level1.root_, level2, level1.config_)
        save_region(root, folder / "root.npz")
        for m in MS:
            fit_overlapping(X, root, config=CONFIG, m=m, seed=seed, folder=folder / f"m{m}")


if __name__ == "__main__":
    main()
