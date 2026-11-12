import json
import sys
import time
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.record import environment
from experiments.hsfv.common.tree import fit_tree
from experiments.hsfv.half_sphere.level2_regions.run import CONFIG as PAPER1_CONFIG, DATASET, SEED

HERE = Path(__file__).resolve().parent
N, DEPTH, LANDMARKS = 12_000_000, 4, 12
SIGMAS = (0.0, 0.001, 0.003, 0.01, 0.05)
CONFIG = {**PAPER1_CONFIG, "landmarks": {"count": {"strategy": "fixed", "fixed": {"n": LANDMARKS}}}}


def name(sigma: float) -> str:
    return f"seed{SEED}_sigma{sigma:g}"


def points(sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """The clean points and the same points with Gaussian noise of standard deviation sigma added in the data
    space."""
    clean = load("eval", n=N, seed=SEED, dataset=DATASET)[0].astype(np.float32)
    if sigma == 0:
        return clean, clean
    noise = np.random.default_rng([SEED, round(sigma * 1e6)]).normal(scale=sigma, size=clean.shape)
    return clean, clean + noise.astype(np.float32)


def main(sigma: float) -> None:
    watch_memory(limit_mb=4000)
    path = HERE / f"paths_{name(sigma)}.npz"
    if path.exists():
        return
    X = points(sigma)[1]
    start = time.perf_counter()
    levels, _, device = fit_tree(X, config=CONFIG, seed=SEED, depth=DEPTH, models=HERE / "models" / name(sigma))
    seconds = time.perf_counter() - start
    paths = np.empty((len(X), DEPTH - 1), dtype=np.uint8)
    for region_id, (_, rows) in levels[-1].items():
        paths[rows] = region_id
    np.savez(path, paths=paths, sigma=sigma, n=len(X), seconds=seconds, config=json.dumps(CONFIG),
             environment=json.dumps(environment(device)))


if __name__ == "__main__":
    main(float(sys.argv[1]))
