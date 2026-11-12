import json
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.record import environment
from experiments.hsfv.common.levels import fit_two_levels
from experiments.hsfv.swiss_roll.level2_regions.run import CONFIG, DATASET

HERE = Path(__file__).resolve().parent
N, SEEDS, LANDMARKS = 60_000, (0, 1, 2), (3, 9)
POINTS_PER_CELL = N / 900


def main() -> None:
    watch_memory(limit_mb=3000)
    for seed in SEEDS:
        X = load("eval", n=N, seed=seed, dataset=DATASET)[0].astype(np.float32)
        for count in LANDMARKS:
            name = f"L{count}_seed{seed}"
            if (HERE / f"maps_{name}.npz").exists():
                continue
            config = {**CONFIG, "landmarks": {"count": {"strategy": "fixed", "fixed": {"n": count}}}}
            maps = fit_two_levels(X, config=config, seed=seed, models=HERE / "models" / name,
                                  points_per_cell=POINTS_PER_CELL)
            fits = {"level1": maps.pop("level1_fitted"), "level2": maps.pop("level2_fitted")}
            np.savez(HERE / f"maps_{name}.npz", **maps, fitted=json.dumps(fits), config=json.dumps(config),
                     environment=json.dumps(environment("cuda")))


if __name__ == "__main__":
    main()
