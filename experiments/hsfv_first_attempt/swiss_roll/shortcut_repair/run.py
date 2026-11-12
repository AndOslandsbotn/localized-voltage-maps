import shutil
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.hsfv_first_attempt.common.levels import fit_level2, load_two_levels
from experiments.hsfv_first_attempt.common.repair import repaired_root
from experiments.hsfv_first_attempt.swiss_roll.chained_distances.run import CONFIG, DATASET, LEVEL1_CELLS, N, SEEDS
from lvm.region import save_region

HERE = Path(__file__).resolve().parent
MODELS = HERE.parent / "chained_distances" / "models"


def main() -> None:
    """Per run: the root repaired by level 2, its new regions, and level 2 refitted inside them."""
    watch_memory(limit_mb=3000)
    for setting in LEVEL1_CELLS:
        for seed in SEEDS:
            folder = HERE / "models_repaired" / setting / f"seed{seed}"
            if (folder / "region_of_point.npy").exists():
                continue
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True)
            level1, level2, _ = load_two_levels(MODELS / setting / f"seed{seed}")
            root, _ = repaired_root(level1.root_, level2, level1.config_)
            save_region(root, folder / "root.npz")
            X = load("eval", n=N, seed=seed, dataset=DATASET)[0].astype(np.float32)
            fit_level2(X, root, config=CONFIG, seed=seed, folder=folder)


if __name__ == "__main__":
    main()
