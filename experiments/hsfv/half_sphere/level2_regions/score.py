import json
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.hsfv.common.levels import MAPS, METRICS, score_regions, weighted
from experiments.hsfv.half_sphere.level2_regions.run import DATASET, N, SEED

HERE = Path(__file__).resolve().parent
SAMPLE = 1_500


def geodesic(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.arccos(np.clip(A.astype(np.float64) @ B.astype(np.float64).T, -1.0, 1.0))


def main() -> None:
    watch_memory(limit_mb=3000)
    X = load("eval", n=N, seed=SEED, dataset=DATASET)[0].astype(np.float32)
    saved = np.load(HERE / "maps.npz")
    rows = score_regions(X, {name: saved[name] for name in MAPS}, saved["region"], geodesic, sample=SAMPLE, seed=SEED)
    summary = {name: {metric: weighted(rows, name, metric) for metric in METRICS} for name in MAPS}
    results = {"dataset": DATASET, "n": N, "seed": SEED, "sample": SAMPLE, "summary": summary, "rows": rows,
               "fitted": json.loads(str(saved["fitted"])), "config": json.loads(str(saved["config"]))}
    (HERE / "results.json").write_text(json.dumps(results, indent=2))
    print(f"{'':14s}" + "".join(f"{metric:>13s}" for metric in METRICS))
    for name, scores in summary.items():
        print(f"{name:14s}" + "".join(f"{scores[metric]:13.3f}" for metric in METRICS))


if __name__ == "__main__":
    main()
