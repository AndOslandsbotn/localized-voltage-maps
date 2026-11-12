import json
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.record import environment, fitted
from experiments.hsfv.common.frames import choose_pins, pins_by_region, second_regions
from experiments.hsfv.common.levels import fit_level1, fit_level2
from experiments.hsfv.swiss_roll.level2_regions.run import CONFIG as PAPER1_CONFIG, DATASET, SEED

HERE = Path(__file__).resolve().parent
SIZES, PER_PAIR, LANDMARKS = (60_000,), 50, 12
CONFIG = {**PAPER1_CONFIG, "landmarks": {"count": {"strategy": "fixed", "fixed": {"n": LANDMARKS}}}}


def main() -> None:
    watch_memory(limit_mb=3000)
    for n in SIZES:
        path = HERE / f"maps_n{n}.npz"
        if path.exists():
            continue
        X = load("eval", n=n, seed=SEED, dataset=DATASET)[0].astype(np.float32)
        models = HERE / "models" / f"n{n}"
        level1, Z1, Z1_chart = fit_level1(X, config=CONFIG, seed=SEED, models=models)
        first, second = second_regions(level1.root_, X, level1.device_)
        pins = choose_pins(first, second, per_pair=PER_PAIR, seed=SEED)
        level2 = fit_level2(X, level1, config=CONFIG, seed=SEED, models=models, pins=pins_by_region(pins))
        placed = level2.pop("pins")
        fits = {"level1": fitted(level1.root_), "level2": level2.pop("level2_fitted")}
        arrays = {f"pair_{c}_{d}": ids for (c, d), ids in pins.items()}
        arrays |= {f"region_{r}_{key}": value for r, item in placed.items() for key, value in item.items()}
        np.savez(path, level1=Z1, level1_chart=Z1_chart, **level2, second=second, **arrays, fitted=json.dumps(fits),
                 config=json.dumps(CONFIG), environment=json.dumps(environment("cuda")))


if __name__ == "__main__":
    main()
