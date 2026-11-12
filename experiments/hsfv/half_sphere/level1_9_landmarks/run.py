import json
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.record import environment, model_record
from experiments.hsfv.common.levels import fit_level1
from experiments.hsfv.half_sphere.level2_regions.run import CONFIG, DATASET, SEED

HERE = Path(__file__).resolve().parent
RUNS = {20_000: "level2_regions", 60_000: "level2_regions_60k"}
LANDMARKS = 9


def main() -> None:
    watch_memory(limit_mb=3000)
    config = {**CONFIG, "landmarks": {"count": {"strategy": "fixed", "fixed": {"n": LANDMARKS}}}}
    for n in RUNS:
        X = load("eval", n=n, seed=SEED, dataset=DATASET)[0].astype(np.float32)
        level1, Z, Z_chart = fit_level1(X, config=config, seed=SEED, models=HERE / "models" / f"n{n}")
        np.savez(HERE / f"maps_n{n}.npz", level1_9=Z, level1_9_chart=Z_chart,
                 fitted=json.dumps(model_record(level1)["fitted"]), config=json.dumps(config),
                 environment=json.dumps(environment("cuda")))


if __name__ == "__main__":
    main()
