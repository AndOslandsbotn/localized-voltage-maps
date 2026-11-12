from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.hsfv_first_attempt.common.levels import fit_two_levels

HERE = Path(__file__).resolve().parent
DATASET, N, SEEDS, LANDMARKS = "half_sphere", 20_000, (0, 1, 2), 8
LEVEL1_CELLS = {"fine": 300, "coarse": 30}
CONFIG = {"landmarks": {"count": {"strategy": "fixed", "fixed": {"n": LANDMARKS}}}}


def main() -> None:
    watch_memory(limit_mb=3000)
    for setting, cells in LEVEL1_CELLS.items():
        for seed in SEEDS:
            folder = HERE / "models" / setting / f"seed{seed}"
            if not (folder / "region_of_point.npy").exists():
                X = load("eval", n=N, seed=seed, dataset=DATASET)[0].astype(np.float32)
                fit_two_levels(X, config=CONFIG, level1_cells=cells, seed=seed, folder=folder)


if __name__ == "__main__":
    main()
