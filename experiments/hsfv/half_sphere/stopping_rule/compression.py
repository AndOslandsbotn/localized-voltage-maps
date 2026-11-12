import json
import sys
from pathlib import Path

import numpy as np

from experiments.common.guard import watch_memory
from experiments.hsfv.common.tree import (cell_statistics, load_tree, own_scales, placed_at, sampled_pairs,
                                          split_ratios, tree_transforms)
from experiments.hsfv.half_sphere.stopping_rule.errors import UNITS, intrinsic, true_distances
from experiments.hsfv.half_sphere.stopping_rule.run import SEED, name, points

HERE = Path(__file__).resolve().parent
BINS = 30


def centres(levels, stats, level):
    """A level's cell centres that hold points: (region id, rows of its cells), their true positions, and each
    one's region path."""
    ids = list(levels[level - 1])
    filled = np.concatenate([stats[level][i]["counts"] > 0 for i in ids])
    positions = np.concatenate([stats[level][i]["truth"] for i in ids])[filled]
    owner = np.concatenate([np.tile(np.array(i, dtype=np.int64), (len(stats[level][i]["counts"]), 1))
                            for i in ids])[filled].reshape(int(filled.sum()), level - 1)
    return ids, filled, positions, owner


def main(sigma: float) -> None:
    watch_memory(limit_mb=4000)
    paths = np.load(HERE / f"paths_{name(sigma)}.npz")["paths"].astype(np.int64)
    levels, config = load_tree(HERE / "models" / name(sigma), paths, "cuda")
    transforms = tree_transforms(levels, config, own_scales(levels, config))
    clean, X = points(sigma)
    true_positions = intrinsic(clean)
    del clean
    depth = len(levels)
    stats = {level: cell_statistics(levels, level, X, true_positions, "cuda") for level in range(1, depth + 1)}
    del X, true_positions
    result = {"sigma": sigma, "seed": SEED, "units": UNITS, "deepest": {}, "own": {}}
    for scored in (depth, *range(1, depth + 1)):
        ids, filled, positions, owner = centres(levels, stats, scored)
        pairs, whole = sampled_pairs(owner, SEED)
        truth = true_distances(positions)(pairs[:, 0], pairs[:, 1])
        keep = truth > 0
        pairs, whole, truth = pairs[keep], whole[keep], truth[keep]
        edges = np.geomspace(np.quantile(truth, 0.001), truth.max() * 1.000001, BINS + 1)
        placers = range(1, depth + 1) if scored == depth and not result["deepest"] else (scored,)
        for by in placers:
            Z = np.concatenate([placed_at(levels, transforms, i, levels[scored - 1][i][0].centroids, by, config)
                                for i in ids])[filled]
            same = (owner[pairs[:, 0], :by - 1] == owner[pairs[:, 1], :by - 1]).all(axis=1)
            curves = split_ratios(Z, truth, pairs, whole, same, edges)
            curves["distance"] = np.sqrt(edges[:-1] * edges[1:]).tolist()
            target = result["deepest"] if len(placers) > 1 else result["own"]
            target[by] = curves
    (HERE / f"compression_{name(sigma)}.json").write_text(json.dumps(result, indent=2))
    for kind, label in (("deepest", f"level-{depth} centres placed by level"), ("own", "own centres of level")):
        print(f"== {label}: median ratio (same region | different regions) at every 5th distance bin")
        for by, curves in result[kind].items():
            row = [f"{d:.3g}: {'-' if s is None else f'{s:.2f}'}|{'-' if t is None else f'{t:.2f}'}"
                   for d, s, t in list(zip(curves["distance"], curves["same"], curves["different"]))[::5]]
            print(f"   {by}: " + "  ".join(row))


if __name__ == "__main__":
    main(float(sys.argv[1]))
