import json
import time
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.record import environment
from experiments.hsfv.common.tree import fit_tree
from experiments.hsfv.swiss_roll.level2_regions.run import CONFIG as PAPER1_CONFIG, DATASET, SEED

HERE = Path(__file__).resolve().parent
N, DEPTH, LANDMARKS = 864_000, 3, 12
CONFIG = {**PAPER1_CONFIG, "landmarks": {"count": {"strategy": "fixed", "fixed": {"n": LANDMARKS}}}}


def main() -> None:
    watch_memory(limit_mb=3000)
    if (HERE / "paths.npz").exists():
        return
    X = load("eval", n=N, seed=SEED, dataset=DATASET)[0].astype(np.float32)
    start = time.perf_counter()
    levels, _, device = fit_tree(X, config=CONFIG, seed=SEED, depth=DEPTH, models=HERE / "models")
    seconds = time.perf_counter() - start
    paths = np.empty((N, DEPTH - 1), dtype=np.int64)
    for region_id, (_, rows) in levels[-1].items():
        paths[rows] = region_id
    np.savez(HERE / "paths.npz", paths=paths, seconds=seconds, config=json.dumps(CONFIG),
             environment=json.dumps(environment(device)))


if __name__ == "__main__":
    main()
