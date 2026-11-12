import json
from pathlib import Path

import numpy as np
import torch

from experiments.common.guard import watch_memory
from experiments.hsfv_first_attempt.common.landmark_graph import geodesic_curves, global_coordinates, landmark_graph
from experiments.hsfv_first_attempt.common.overlap import build_state, level1_distances, load_overlapping
from experiments.hsfv_first_attempt.common.real_data import read
from experiments.hsfv_first_attempt.kuzushiji49.overlapping_regions.score import QUERIES, neighbour_scores, true_neighbours
from experiments.hsfv_first_attempt.kuzushiji49.root_choice.run import DATASET, N, SAVED, SEEDS
from lvm import LocalizedVoltageMaps
from lvm.region import load_region

HERE = Path(__file__).resolve().parent
FLAGGED = 0.02


def global_frame(root, folder: Path, X: np.ndarray, config) -> torch.Tensor:
    """Every point's coordinates in the global frame of the landmark graph in lengths."""
    points = torch.as_tensor(X, device=root.centroids.device)
    level2, regions, strengths = load_overlapping(folder, str(root.centroids.device))
    curves = {r: geodesic_curves(level2[r], config) for r in level2}
    graph = landmark_graph(root, level2, points, regions, config, curves=curves)
    rows = torch.arange(len(X), device=points.device)
    return torch.cat([global_coordinates(graph, level2, points, regions, strengths, rows[s:s + 20_000], config)
                      for s in range(0, len(X), 20_000)])


def main() -> None:
    """Option (a) on real data: keep the default 300-cell root unless its repair flagged more than FLAGGED of its
    edges, else use the dimension-sized root; against both hierarchies and the best single level so far (300 cells
    with within-cell positions in all d̂ directions). Scored by recall@10 and kNN accuracy."""
    watch_memory(limit_mb=3000)
    X, y = read(DATASET, N)
    rows = []
    for seed in SEEDS:
        folder = HERE / "models" / f"seed{seed}"
        choice = json.loads((folder / "choice.json").read_text())
        level1 = LocalizedVoltageMaps.load(SAVED / "root300" / f"seed{seed}" / "level1")
        config, device = level1.config_, level1.device_
        points, labels = torch.as_tensor(X, device=device), torch.as_tensor(y, device=device)
        queries = torch.as_tensor(np.random.default_rng(seed).choice(N, QUERIES, replace=False), device=device)
        truth = true_neighbours(points, queries)
        everyone = torch.arange(N, device=device)
        plain, regions, strengths = load_overlapping(SAVED / "root300" / f"seed{seed}" / "plain", device)
        state = build_state(X, level1.root_, plain, regions, strengths, config, within=True, directions="dimension")
        methods = {"level1_300_within": lambda q: level1_distances(state, q, everyone)}
        default = global_frame(load_region(SAVED / "root300" / f"seed{seed}" / "root_repaired.npz", device),
                               folder / "default_m2", X, config)
        chosen = global_frame(load_region(folder / "chosen_root.npz", device), folder / "chosen_m2", X, config)
        option = chosen if choice["default_flagged_share"] > FLAGGED else default
        methods |= {"hierarchy_default": lambda q: torch.cdist(default[q], default),
                    "hierarchy_chosen": lambda q: torch.cdist(chosen[q], chosen),
                    "option_a": lambda q: torch.cdist(option[q], option)}
        for method, distance in methods.items():
            rows.append({"seed": seed, "method": method, "chosen_cells": choice["root_cells"],
                         "default_flagged_share": choice["default_flagged_share"],
                         **neighbour_scores(distance, queries, truth, labels)})
            print(rows[-1], flush=True)
        del state, default, chosen, option
        torch.cuda.empty_cache()
    (HERE / "results.json").write_text(json.dumps({"dataset": DATASET, "n": N, "queries": QUERIES,
                                                   "flagged_threshold": FLAGGED, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
