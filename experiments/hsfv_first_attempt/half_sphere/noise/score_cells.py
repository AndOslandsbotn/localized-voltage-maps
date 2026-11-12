import json
from pathlib import Path

import numpy as np
import torch

from experiments.common.guard import watch_memory
from experiments.hsfv_first_attempt.common.landmark_graph import geodesic_curves, global_coordinates, landmark_graph
from experiments.hsfv_first_attempt.common.overlap import load_overlapping
from experiments.hsfv_first_attempt.half_sphere.chained_distances.run import CONFIG, N, SEEDS
from experiments.hsfv_first_attempt.half_sphere.chained_distances.score import SAMPLE, scores, truth
from experiments.hsfv_first_attempt.half_sphere.noise.run import NOISE, noisy
from experiments.hsfv_first_attempt.half_sphere.noise.run_cells import LEVEL2_CELLS
from experiments.hsfv_first_attempt.half_sphere.overlapping_regions.score import condensed
from lvm.config import load_config
from lvm.region import load_region

HERE = Path(__file__).resolve().parent
DEVICE = "cuda"
GROUPS = ("all", "short", "middle", "long")


def main() -> None:
    """The hierarchy (voltages to lengths, voltage-weighted placement) with 300, 100 and 30 level-2 cells per region,
    against the clean true distances; and each level 2's typical cell spacing (median distance to the nearest
    other cell) compared with the noise."""
    watch_memory(limit_mb=3000)
    config = load_config(overrides=CONFIG)
    rows = []
    for sigma in NOISE:
        for seed in SEEDS:
            folder = HERE / "models" / f"noise{sigma}" / f"seed{seed}"
            clean, X = noisy(seed, sigma)
            points = torch.as_tensor(X, device=DEVICE)
            sample = np.random.default_rng(seed).choice(N, SAMPLE, replace=False)
            S = torch.as_tensor(sample, device=DEVICE)
            true = truth(clean[sample])
            thirds = np.quantile(true, [1 / 3, 2 / 3])
            masks = {"all": np.ones(len(true), dtype=bool), "short": true <= thirds[0],
                     "middle": (true > thirds[0]) & (true <= thirds[1]), "long": true > thirds[1]}
            root = load_region(folder / "root.npz", DEVICE)
            for cells, sub in ((300, "m2"), *[(c, f"m2_cells{c}") for c in LEVEL2_CELLS]):
                level2, regions, strengths = load_overlapping(folder / sub, DEVICE)
                spacing = []
                for child in level2.values():
                    g = torch.cdist(child.centroids.double(), child.centroids.double())
                    g.fill_diagonal_(torch.inf)
                    spacing.append(float(g.min(dim=1).values.median()))
                curves = {r: geodesic_curves(level2[r], config) for r in level2}
                graph = landmark_graph(root, level2, points, regions, config, curves=curves)
                coords = global_coordinates(graph, level2, points, regions, strengths, S, config)
                d = condensed(torch.cdist(coords, coords))
                for group in GROUPS:
                    rows.append({"noise": sigma, "seed": seed, "level2_cells": cells, "group": group,
                                 "cell_spacing": float(np.median(spacing)), **scores(d, true, masks[group])})
            print(sigma, seed, flush=True)
    (HERE / "results_cells.json").write_text(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
