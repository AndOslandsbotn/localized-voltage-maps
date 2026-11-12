import json
from pathlib import Path

from experiments.common.guard import watch_memory
from experiments.hsfv_first_attempt.common.overlap import fit_overlapping, load_overlapping
from experiments.hsfv_first_attempt.common.real_data import read
from experiments.hsfv_first_attempt.common.repair import repaired_root
from lvm import LocalizedVoltageMaps
from lvm.region import save_region

HERE = Path(__file__).resolve().parent
DATASET, N, SEEDS, ROOT_CELLS = "mnist8m", 100_000, (0, 1, 2), (30, 50, 300)
CONFIG = {}


def main() -> None:
    """Per root size and seed: level 1, level 2 in its regions, the root repaired by that level 2, and level 2
    refitted in the repaired root's regions."""
    watch_memory(limit_mb=3000)
    X, _ = read(DATASET, N)
    for cells in ROOT_CELLS:
        for seed in SEEDS:
            folder = HERE / "models" / f"root{cells}" / f"seed{seed}"
            if (folder / "repaired" / "regions.npy").exists():
                continue
            level1 = LocalizedVoltageMaps(local_chart="none", random_state=seed,
                                          config={**CONFIG, "cells": {"n_cells": cells}}).fit(X)
            level1.save(folder / "level1")
            fit_overlapping(X, level1.root_, config=CONFIG, m=1, seed=seed, folder=folder / "plain")
            level2, _, _ = load_overlapping(folder / "plain", level1.device_)
            root, changes = repaired_root(level1.root_, level2, level1.config_)
            save_region(root, folder / "root_repaired.npz")
            (folder / "repair.json").write_text(json.dumps(changes, indent=2))
            fit_overlapping(X, root, config=CONFIG, m=1, seed=seed, folder=folder / "repaired")


if __name__ == "__main__":
    main()
