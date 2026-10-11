import json
import time
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.methods import fit_lvm_chart
from experiments.common.record import environment
from experiments.sfv.kuzushiji49.second_level.run import DATASET, MASSES, MIN_POINTS, N_FIT, SEEDS, rows_of

HERE = Path(__file__).resolve().parent


def main() -> None:
    watch_memory(limit_mb=3000)
    warm_up = load("eval", n=2000, seed=0, dataset=DATASET)[0]
    fit_lvm_chart(warm_up, seed=0).transform(warm_up)
    for seed in SEEDS:
        path = HERE / "embeddings" / f"chart_seed{seed}.npz"
        if not path.exists():
            run(seed, path)


def run(seed: int, path: Path) -> None:
    regions = np.load(HERE / "embeddings" / f"seed{seed}.npz")
    region_of_fit, region_of_row = regions["region_of_fit"], regions["region_of_row"]
    embeddings = {"same": np.zeros((N_FIT, 2)), "new": np.zeros((N_FIT, 2))}
    seconds = {"same": 0.0, "new": 0.0}
    for r in np.unique(region_of_fit):
        fit_rows = np.flatnonzero(region_of_fit == r)
        if len(fit_rows) < MIN_POINTS:
            continue
        all_rows = np.flatnonzero(region_of_row == r)
        new_points = rows_of(all_rows)
        points = new_points[np.searchsorted(all_rows, fit_rows)]
        for data, fitted_on, config in (("same", points, MASSES),
                                        ("new", new_points, {"sample": {"size": len(all_rows)}, **MASSES})):
            start = time.perf_counter()
            model = fit_lvm_chart(fitted_on, seed=seed, config=config)
            embeddings[data][fit_rows] = model.transform(points)
            seconds[data] += time.perf_counter() - start
        del new_points, points
    np.savez(path, **embeddings, seconds=json.dumps(seconds), environment=json.dumps(environment("cuda")))


if __name__ == "__main__":
    main()
