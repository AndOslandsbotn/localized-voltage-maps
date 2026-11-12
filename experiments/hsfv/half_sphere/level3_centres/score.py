import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

from experiments.common.guard import watch_memory
from experiments.hsfv.common.tree import banded_rank, grouped_rank, load_tree, placed_at, tree_transforms
from experiments.hsfv.half_sphere.level3_centres.run import SEED

HERE = Path(__file__).resolve().parent
GROUP_SAMPLE, WHOLE_SAMPLE = 1_500, 3_000


def truth(centres: np.ndarray) -> np.ndarray:
    unit = centres / np.linalg.norm(centres, axis=1, keepdims=True)
    return np.arccos(np.clip(1 - pdist(unit, "sqeuclidean") / 2, -1, 1))


def main() -> None:
    watch_memory(limit_mb=3000)
    saved = np.load(HERE / "paths.npz")
    paths = saved["paths"]
    levels, config = load_tree(HERE / "models", paths, "cuda")
    transforms = tree_transforms(levels, config)
    depth, results = len(levels), {"regions": {}, "scores": {}}
    for level, found in enumerate(levels, start=1):
        cells = [len(region.centroids) for region, _ in found.values()]
        points = [len(rows) / len(region.centroids) for region, rows in found.values()]
        results["regions"][level] = {"regions": len(found), "cells_min_median_max": [min(cells), int(np.median(cells)), max(cells)],
                                     "points_per_cell_median": float(np.median(points))}
    rng = np.random.default_rng(SEED)
    for level in range(2, depth + 1):
        ids = list(levels[level - 1])
        centres = np.concatenate([levels[level - 1][i][0].centroids.cpu().numpy().astype(np.float64) for i in ids])
        owner = np.concatenate([np.full(len(levels[level - 1][i][0].centroids), k) for k, i in enumerate(ids)])
        whole = np.sort(rng.choice(len(centres), min(WHOLE_SAMPLE, len(centres)), replace=False))
        exact = truth(centres[whole])
        groups = {f"level-{g} region": np.array([hash(ids[k][:g - 1]) for k in owner]) for g in range(2, level + 1)}
        for by in range(1, level + 1):
            Z = np.concatenate([placed_at(levels, transforms, i, levels[level - 1][i][0].centroids, by, config)
                                for i in ids])
            scores = {f"within {name}": grouped_rank(Z, truth, centres, group, GROUP_SAMPLE, SEED)
                      for name, group in groups.items()}
            scores |= {f"whole {band}": value for band, value in banded_rank(Z, exact, whole).items()}
            results["scores"][f"level-{level} centres placed by level {by}"] = scores
    results |= {"seed": SEED, "group_sample": GROUP_SAMPLE, "whole_sample": WHOLE_SAMPLE,
                "fit_seconds": float(saved["seconds"]), "config": json.loads(str(saved["config"]))}
    (HERE / "results.json").write_text(json.dumps(results, indent=2))
    print("regions:", results["regions"], f"| fit {results['fit_seconds']:.0f} s")
    for name, scores in results["scores"].items():
        print(f"  {name:36s} " + " | ".join(f"{key} {value:.3f}" for key, value in scores.items()))


if __name__ == "__main__":
    main()
