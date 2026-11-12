import json
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.record import environment
from experiments.hsfv.common.levels import fit_two_levels

HERE = Path(__file__).resolve().parent
DATASET, N, SEED = "swiss_roll", 20_000, 0
CONFIG = {"graph": {"radius": {"adaptive_per_cell": {"k": 5}}}}


def main() -> None:
    watch_memory(limit_mb=3000)
    X = load("eval", n=N, seed=SEED, dataset=DATASET)[0].astype(np.float32)
    maps = fit_two_levels(X, config=CONFIG, seed=SEED, models=HERE / "models")
    fits = {"level1": maps.pop("level1_fitted"), "level2": maps.pop("level2_fitted")}
    np.savez(HERE / "maps.npz", **maps, fitted=json.dumps(fits), config=json.dumps(CONFIG),
             environment=json.dumps(environment("cuda")))


if __name__ == "__main__":
    main()
