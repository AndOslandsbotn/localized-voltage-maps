import json
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.hsfv.common.levels import MAPS, METRICS, score_regions, weighted
from experiments.hsfv.swiss_roll.landmark_ablation.run import LANDMARKS, N, SEEDS
from experiments.hsfv.swiss_roll.level2_regions.run import DATASET
from experiments.hsfv.swiss_roll.level2_regions.score import geodesic

HERE = Path(__file__).resolve().parent
SAMPLE, WHOLE_SAMPLE, MIN_PIECE = 1_500, 500, 500
SHOWN = ("rank", "rank_close", "rank_middle", "rank_far", "recall@5", "recall@20", "recall@78", "recall@305",
         "recall@1203")


def main() -> None:
    watch_memory(limit_mb=3000)
    rows, covered = [], {}
    for seed in SEEDS:
        X = load("eval", n=N, seed=seed, dataset=DATASET)[0].astype(np.float32)
        saved = {count: np.load(HERE / f"maps_L{count}_seed{seed}.npz") for count in LANDMARKS}
        maps = {f"{name}_L{count}": saved[count][name] for count in LANDMARKS for name in MAPS}
        regions = np.column_stack([saved[count]["region"] for count in LANDMARKS])
        piece = np.unique(regions, axis=0, return_inverse=True)[1].ravel()
        sizes = np.bincount(piece)
        covered[seed] = float(sizes[sizes >= MIN_PIECE].sum() / N)
        for row in score_regions(X, maps, piece, geodesic, sample=SAMPLE, seed=seed, min_points=MIN_PIECE):
            rows.append({"seed": seed, "scope": "pieces", **row})
        level1 = {name: Z for name, Z in maps.items() if name.startswith("level1")}
        for row in score_regions(X, level1, np.zeros(N, dtype=int), geodesic, sample=WHOLE_SAMPLE, seed=seed):
            rows.append({"seed": seed, "scope": "whole", **row})
    summary = {}
    for scope in ("pieces", "whole"):
        names = dict.fromkeys(row["map"] for row in rows if row["scope"] == scope)
        for name in names:
            per_seed = {metric: [weighted([row for row in rows if row["scope"] == scope and row["seed"] == seed],
                                          name, metric) for seed in SEEDS] for metric in METRICS}
            summary[f"{scope}/{name}"] = {metric: {"mean": float(np.mean(values)), "std": float(np.std(values))}
                                          for metric, values in per_seed.items()}
    fitted = {f"L{count}_seed{seed}": json.loads(str(np.load(HERE / f"maps_L{count}_seed{seed}.npz")["fitted"]))
              for count in LANDMARKS for seed in SEEDS}
    results = {"dataset": DATASET, "n": N, "seeds": SEEDS, "landmarks": LANDMARKS, "sample": SAMPLE,
               "whole_sample": WHOLE_SAMPLE, "min_piece": MIN_PIECE, "covered": covered, "summary": summary,
               "rows": rows, "fitted": fitted}
    (HERE / "results.json").write_text(json.dumps(results, indent=2))
    print("share of points in scored pieces:", {seed: round(share, 3) for seed, share in covered.items()})
    print(f"{'':28s}" + "".join(f"{metric:>13s}" for metric in SHOWN))
    for name, scores in summary.items():
        print(f"{name:28s}" + "".join(f"{scores[m]['mean']:7.3f}±{scores[m]['std']:.3f}" for m in SHOWN))


if __name__ == "__main__":
    main()
