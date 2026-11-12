import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import squareform

from experiments.common.geodesics import graph_geodesics
from experiments.common.guard import watch_memory
from experiments.hsfv_first_attempt.common.landmark_graph import (geodesic_curves, global_coordinates, landmark_graph,
                                                    region_paths, route_coordinates)
from experiments.hsfv_first_attempt.common.overlap import build_state, level1_distances, load_overlapping
from experiments.hsfv_first_attempt.swiss_roll.chained_distances.run import CONFIG, N, SEEDS
from experiments.hsfv_first_attempt.swiss_roll.chained_distances.score import SAMPLE, scores, truth
from experiments.hsfv_first_attempt.swiss_roll.noise.run import NOISE, noisy
from experiments.hsfv_first_attempt.swiss_roll.overlapping_regions.score import condensed
from lvm import LocalizedVoltageMaps
from lvm.config import load_config
from lvm.region import load_region

HERE = Path(__file__).resolve().parent
DEVICE = "cuda"
GROUPS = ("all", "short", "middle", "long")


def main() -> None:
    """Against the clean true distances, with the models fitted on noisy points: the global frame from voltages
    turned into lengths (V), the same from graph paths only (P, the ablation), a fine LVM root, and plain graph
    shortest paths over all noisy points (Isomap's distances)."""
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
            root = load_region(folder / "root.npz", DEVICE)
            level2, regions, strengths = load_overlapping(folder / "m2", DEVICE)
            curves = {r: geodesic_curves(level2[r], config) for r in level2}
            graph_v = landmark_graph(root, level2, points, regions, config, curves=curves)
            coords_v = global_coordinates(graph_v, level2, points, regions, strengths, S, config)
            paths = {r: region_paths(level2[r], config) for r in level2}
            graph_p = landmark_graph(root, level2, points, regions, config, paths=paths)
            coords_p = route_coordinates(graph_p, S)
            fine = LocalizedVoltageMaps.load(folder / "fine")
            fine_state = build_state(X, fine.root_, {}, np.zeros((N, 1), dtype=np.int64), np.ones((N, 1)),
                                     fine.config_, within=True)
            dists = {"voltage_lengths": condensed(torch.cdist(coords_v, coords_v)),
                     "graph_paths": condensed(torch.cdist(coords_p, coords_p)),
                     "fine_lvm": condensed(level1_distances(fine_state, S, S)),
                     "isomap_paths": graph_geodesics(X, sample, 10)}
            thirds = np.quantile(true, [1 / 3, 2 / 3])
            masks = {"all": np.ones(len(true), dtype=bool), "short": true <= thirds[0],
                     "middle": (true > thirds[0]) & (true <= thirds[1]), "long": true > thirds[1]}
            for method, dd in dists.items():
                for group in GROUPS:
                    rows.append({"noise": sigma, "seed": seed, "method": method, "group": group,
                                 **scores(dd, true, masks[group])})
            print(sigma, seed, flush=True)
    (HERE / "results.json").write_text(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
