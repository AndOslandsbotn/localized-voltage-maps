import json
import sys
from pathlib import Path

import numpy as np

from experiments.common.guard import watch_memory
from experiments.hsfv.common.tree import (cell_statistics, distance_errors, load_tree, own_scales, placed_at,
                                          sampled_pairs, tree_transforms)
from experiments.hsfv.half_sphere.stopping_rule.run import SEED, name, points

HERE = Path(__file__).resolve().parent
UNITS = "radians"


def intrinsic(clean: np.ndarray) -> np.ndarray:
    return clean


def true_distances(positions: np.ndarray):
    unit = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    return lambda i, j: np.arccos(np.clip(np.sum(unit[i] * unit[j], axis=1), -1.0, 1.0))


def main(sigma: float, mode: str) -> None:
    watch_memory(limit_mb=4000)
    paths = np.load(HERE / f"paths_{name(sigma)}.npz")["paths"].astype(np.int64)
    levels, config = load_tree(HERE / "models" / name(sigma), paths, "cuda")
    transforms = tree_transforms(levels, config, own_scales(levels, config) if mode == "calibrated" else None)
    clean, X = points(sigma)
    true_positions = intrinsic(clean)
    del clean
    depth = len(levels)
    stats = cell_statistics(levels, depth, X, true_positions, "cuda")
    del X, true_positions
    ids = list(levels[depth - 1])
    filled = np.concatenate([stats[i]["counts"] > 0 for i in ids])
    positions = np.concatenate([stats[i]["truth"] for i in ids])[filled]
    owner = np.concatenate([np.tile(i, (len(stats[i]["counts"]), 1)) for i in ids])[filled]
    placed = {by: np.concatenate([placed_at(levels, transforms, i, levels[depth - 1][i][0].centroids, by, config)
                                  for i in ids])[filled] for by in range(1, depth + 1)}
    pairs, whole = sampled_pairs(owner, SEED)
    result = distance_errors(placed, true_distances(positions), pairs, whole)
    result |= {"mode": mode, "sigma": sigma, "seed": SEED, "units": UNITS, "centres": int(len(positions)),
               "pair_count": int(len(pairs)), "distance_centres": result.pop("centres")}
    (HERE / f"errors_{name(sigma)}_{mode}.json").write_text(json.dumps(result, indent=2))
    print(f"== {mode}, sigma {sigma}: {len(positions)} centres, {len(pairs)} pairs")
    for level, values in result["levels"].items():
        shown = [f"{d:.3g}: {a:.3g}" for d, a in zip(result["distance_centres"][::6], values["absolute"][::6]) if a]
        print(f"   level {level}: median absolute error at distance  " + " | ".join(shown))


if __name__ == "__main__":
    main(float(sys.argv[1]), sys.argv[2])
