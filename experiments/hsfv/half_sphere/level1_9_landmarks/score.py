import json
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.hsfv.common.levels import MAPS, METRICS, score_regions, weighted
from experiments.hsfv.half_sphere.level1_9_landmarks.run import RUNS
from experiments.hsfv.half_sphere.level2_regions.run import DATASET, SEED
from experiments.hsfv.half_sphere.level2_regions.score import SAMPLE, geodesic

HERE = Path(__file__).resolve().parent
CONTROL = ("level1_9", "level1_9_chart")


def main() -> None:
    watch_memory(limit_mb=3000)
    for n, run in RUNS.items():
        X = load("eval", n=n, seed=SEED, dataset=DATASET)[0].astype(np.float32)
        levels, control = np.load(HERE.parent / run / "maps.npz"), np.load(HERE / f"maps_n{n}.npz")
        maps = {**{name: levels[name] for name in MAPS}, **{name: control[name] for name in CONTROL}}
        rows = score_regions(X, maps, levels["region"], geodesic, sample=SAMPLE, seed=SEED)
        summary = {name: {metric: weighted(rows, name, metric) for metric in METRICS} for name in maps}
        results = {"dataset": DATASET, "n": n, "seed": SEED, "sample": SAMPLE, "regions_from": run,
                   "summary": summary, "rows": rows, "fitted": json.loads(str(levels["fitted"])),
                   "control_fitted": json.loads(str(control["fitted"])), "config": json.loads(str(control["config"]))}
        (HERE / f"results_n{n}.json").write_text(json.dumps(results, indent=2))
        print(f"== n = {n}, level 1 landmarks: {results['control_fitted']['n_landmarks']}")
        print(f"{'':16s}" + "".join(f"{metric:>12s}" for metric in METRICS[:4]) + "  recall@k " + " ".join(f"{k:>5d}" for k in [5, 20, 78, 305, 1203]))
        for name, scores in summary.items():
            print(f"{name:16s}" + "".join(f"{scores[metric]:12.3f}" for metric in METRICS[:4]) + "           "
                  + " ".join(f"{scores[f'recall@{k}']:.3f}" for k in [5, 20, 78, 305, 1203]))


if __name__ == "__main__":
    main()
