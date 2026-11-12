import json
from pathlib import Path

import numpy as np
import torch

from experiments.common.guard import watch_memory
from experiments.hsfv_first_attempt.common.overlap import build_state, distances, level1_distances, load_overlapping
from experiments.hsfv_first_attempt.common.real_data import read
from experiments.hsfv_first_attempt.mnist8m.coarse_root.run import DATASET, N, ROOT_CELLS, SEEDS
from experiments.hsfv_first_attempt.mnist8m.overlapping_regions.score import QUERIES, neighbour_scores, true_neighbours
from lvm import LocalizedVoltageMaps

HERE = Path(__file__).resolve().parent
DIRECTIONS = ("fixed", "dimension", "per_cell")


def main() -> None:
    """Within-cell positions with each way of choosing how many directions a cell uses, on the saved coarse-root
    models (level 2 in the original root's regions)."""
    watch_memory(limit_mb=3000)
    X, y = read(DATASET, N)
    rows = []
    for seed in SEEDS:
        truth = None
        for cells in ROOT_CELLS:
            folder = HERE / "models" / f"root{cells}" / f"seed{seed}"
            level1 = LocalizedVoltageMaps.load(folder / "level1")
            config, device = level1.config_, level1.device_
            points, labels = torch.as_tensor(X, device=device), torch.as_tensor(y, device=device)
            queries = torch.as_tensor(np.random.default_rng(seed).choice(N, QUERIES, replace=False), device=device)
            if truth is None:
                truth = true_neighbours(points, queries)
            everyone = torch.arange(N, device=device)
            level2, regions, strengths = load_overlapping(folder / "plain", device)
            for directions in DIRECTIONS:
                state = build_state(X, level1.root_, level2, regions, strengths, config, within=True,
                                    directions=directions)
                for method, distance in {"level1": lambda q: level1_distances(state, q, everyone),
                                          "level2": lambda q: distances(state, q, everyone, "weighted")}.items():
                    rows.append({"root_cells": cells, "seed": seed, "directions": directions, "method": method,
                                 **neighbour_scores(distance, queries, truth, labels)})
                    print(rows[-1], flush=True)
                del state
                torch.cuda.empty_cache()
    (HERE / "directions.json").write_text(json.dumps({"dataset": DATASET, "n": N, "queries": QUERIES, "rows": rows},
                                                     indent=2))


if __name__ == "__main__":
    main()
