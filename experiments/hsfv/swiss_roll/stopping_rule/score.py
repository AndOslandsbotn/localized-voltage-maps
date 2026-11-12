import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

from experiments.common.datasets import unrolled
from experiments.common.guard import watch_memory
from experiments.hsfv.common.tree import (banded_rank, cell_statistics, grouped_rank, load_tree, placed_at,
                                          region_dimension, rule_depths, tree_transforms)
from experiments.hsfv.swiss_roll.stopping_rule.run import SEED, name, points

HERE = Path(__file__).resolve().parent
GROUP_SAMPLE, WHOLE_SAMPLE, TOLERANCES = 1_500, 3_000, (0.05, 0.1, 0.2)


def intrinsic(clean: np.ndarray) -> np.ndarray:
    return unrolled(clean)


def truth(positions: np.ndarray) -> np.ndarray:
    return pdist(positions)


def main(sigma: float) -> None:
    watch_memory(limit_mb=4000)
    saved = np.load(HERE / f"paths_{name(sigma)}.npz")
    paths = saved["paths"].astype(np.int64)
    levels, config = load_tree(HERE / "models" / name(sigma), paths, "cuda")
    clean, X = points(sigma)
    true_positions = intrinsic(clean)
    del clean
    depth = len(levels)
    stats = {level: cell_statistics(levels, level, X, true_positions, "cuda") for level in range(1, depth + 1)}
    del X, true_positions
    transforms = tree_transforms(levels, config)
    results = {"sigma": sigma, "n": int(saved["n"]), "fit_seconds": float(saved["seconds"]), "levels": {},
               "rule": {}, "scores": {}}
    for level in range(1, depth + 1):
        cells = np.concatenate([s["dimension"] for s in stats[level].values()])
        counts = np.concatenate([s["counts"] for s in stats[level].values()])
        region = [region_dimension(s) for s in stats[level].values()]
        results["levels"][level] = {
            "regions": len(levels[level - 1]), "cells": int(len(cells)),
            "points_per_cell_median": float(np.median(counts)),
            "cell_dimension_median": region_dimension({"dimension": cells, "counts": counts}),
            "region_dimension_quartiles": np.quantile(region, [0.25, 0.5, 0.75]).tolist(),
            "library_dimension_mean": float(np.mean([r.dimension for r, _ in levels[level - 1].values()]))}
    for tolerance in TOLERANCES:
        chosen = rule_depths(stats, depth, tolerance)
        weight = {leaf: len(levels[depth - 1][leaf][1]) for leaf in chosen}
        total = sum(weight.values())
        results["rule"][tolerance] = {d: sum(w for leaf, w in weight.items() if chosen[leaf] == d) / total
                                      for d in range(1, depth + 1)}
    rng = np.random.default_rng(SEED)
    for level in range(2, depth + 1):
        ids = list(levels[level - 1])
        filled = np.concatenate([stats[level][i]["counts"] > 0 for i in ids])
        positions = np.concatenate([stats[level][i]["truth"] for i in ids])[filled]
        owner = np.concatenate([np.full(len(stats[level][i]["counts"]), k) for k, i in enumerate(ids)])[filled]
        whole = np.sort(rng.choice(len(positions), min(WHOLE_SAMPLE, len(positions)), replace=False))
        exact = truth(positions[whole])
        groups = {f"level-{g} region": np.array([hash(ids[k][:g - 1]) for k in owner]) for g in range(2, level + 1)}
        for by in range(1, level + 1):
            Z = np.concatenate([placed_at(levels, transforms, i, levels[level - 1][i][0].centroids, by, config)
                                for i in ids])[filled]
            scores = {f"within {g}": grouped_rank(Z, truth, positions, group, GROUP_SAMPLE, SEED)
                      for g, group in groups.items()}
            scores |= {f"whole {band}": value for band, value in banded_rank(Z, exact, whole).items()}
            results["scores"][f"level-{level} centres by level {by}"] = scores
    (HERE / f"results_{name(sigma)}.json").write_text(json.dumps(results, indent=2))
    print(f"== sigma {sigma}: fit {results['fit_seconds']:.0f} s")
    for level, info in results["levels"].items():
        print(f"   level {level}: {info}")
    print(f"   rule (share of points stopping at each level): {results['rule']}")
    for key, scores in results["scores"].items():
        print(f"   {key:30s} " + " | ".join(f"{k} {v:.3f}" for k, v in scores.items()))


if __name__ == "__main__":
    main(float(sys.argv[1]))
