import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform

from experiments.common.datasets import load, unrolled
from experiments.common.geodesics import graph_geodesics, rank_correlation
from experiments.common.guard import watch_memory
from experiments.hsfv_first_attempt.common.chaining import (calibration, chained_simple, chained_triangulated, chained_voltage,
                                              coordinates, extend, frame, to_landmarks)
from experiments.hsfv_first_attempt.common.levels import load_two_levels
from experiments.hsfv_first_attempt.swiss_roll.chained_distances.run import DATASET, LANDMARKS, LEVEL1_CELLS, N, SEEDS

HERE = Path(__file__).resolve().parent
SAMPLE, GRAPH_NEIGHBOURS = 1_500, 10
GROUPS = ("all", "within", "across", "short", "middle", "long")


def truth(X: np.ndarray) -> np.ndarray:
    """Distances along the unrolled sheet (the roll's exact geodesics)."""
    return pdist(unrolled(X))


def pair_distances(X: np.ndarray, sample: np.ndarray, level1, level2: dict, region_of_point: np.ndarray) -> dict:
    """Each method's distances between the sampled points (condensed)."""
    root, config = level1.root_, level1.config_
    points = torch.as_tensor(X[sample], device=level1.device_)
    root_frame = frame(root)
    direct = to_landmarks(root, points, config=config)
    simple, triangulated, voltage = direct.clone(), direct.clone(), direct.clone()
    within = np.full((len(sample), len(sample)), np.nan)
    regions = region_of_point[sample]
    for r, child in level2.items():
        idx = np.flatnonzero(regions == r)
        if len(idx) == 0:
            continue
        inside = points[idx]
        simple[idx] = extend(child, inside, chained_simple(child, root, config=config), config=config).to(simple.dtype)
        triangulated[idx] = extend(child, inside, chained_triangulated(child, root, config=config),
                                   config=config).to(triangulated.dtype)
        voltage[idx] = extend(child, inside, chained_voltage(child, root, config=config), config=config).to(voltage.dtype)
        local = coordinates(to_landmarks(child, inside, config=config), frame(child))
        within[np.ix_(idx, idx)] = calibration(child, root, config=config) * torch.cdist(local, local).cpu().numpy()
    np.fill_diagonal(within, 0.0)
    on_root = lambda d: pdist(coordinates(d, root_frame).cpu().numpy())
    return {"level1": on_root(direct), "chained_simple": on_root(simple), "chained_triangulated": on_root(triangulated),
            "chained_voltage": on_root(voltage),
            "level2_within": squareform(within, checks=False), "graph": graph_geodesics(X, sample, GRAPH_NEIGHBOURS)}


def scores(distances: np.ndarray, true: np.ndarray, mask: np.ndarray) -> dict:
    """Rank correlation with the truth, and the median relative error after one scale factor fitted on all pairs."""
    known = np.isfinite(distances) & (distances > 0)
    scale = np.median(true[known] / distances[known])
    use = mask & known
    if use.sum() < 10:
        return {"rank": float("nan"), "relative_error": float("nan")}
    return {"rank": rank_correlation(distances[use], true[use]),
            "relative_error": float(np.median(np.abs(scale * distances[use] - true[use]) / true[use]))}


def main() -> None:
    watch_memory(limit_mb=3000)
    rows = []
    for setting, seed in [(setting, seed) for setting in LEVEL1_CELLS for seed in SEEDS]:
        X = load("eval", n=N, seed=seed, dataset=DATASET)[0].astype(np.float32)
        level1, level2, region_of_point = load_two_levels(HERE / "models" / setting / f"seed{seed}")
        sample = np.random.default_rng(seed).choice(N, SAMPLE, replace=False)
        true = truth(X[sample])
        regions = region_of_point[sample]
        same = squareform((regions[:, None] == regions[None, :]).astype(float), checks=False) > 0
        thirds = np.quantile(true, [1 / 3, 2 / 3])
        masks = {"all": np.ones_like(same), "within": same, "across": ~same, "short": true <= thirds[0],
                 "middle": (true > thirds[0]) & (true <= thirds[1]), "long": true > thirds[1]}
        for method, distances in pair_distances(X, sample, level1, level2, region_of_point).items():
            for group in GROUPS:
                rows.append({"setting": setting, "level1_cells": LEVEL1_CELLS[setting], "seed": seed,
                             "method": method, "group": group,
                             **scores(distances, true, masks[group])})
    (HERE / "results.json").write_text(json.dumps({"dataset": DATASET, "n": N, "seeds": SEEDS, "sample": SAMPLE,
                                                   "landmarks": LANDMARKS, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
