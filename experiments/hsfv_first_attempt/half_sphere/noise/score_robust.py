import json
from pathlib import Path

import numpy as np
import torch

from experiments.common.guard import watch_memory
from experiments.hsfv_first_attempt.common.landmark_graph import (geodesic_curves, global_coordinates, landmark_graph,
                                                    region_paths, route_coordinates)
from experiments.hsfv_first_attempt.common.overlap import load_overlapping
from experiments.hsfv_first_attempt.half_sphere.chained_distances.run import CONFIG, N, SEEDS
from experiments.hsfv_first_attempt.half_sphere.chained_distances.score import SAMPLE, scores, truth
from experiments.hsfv_first_attempt.half_sphere.noise.run import NOISE, noisy
from experiments.hsfv_first_attempt.half_sphere.overlapping_regions.score import condensed
from lvm.config import load_config
from lvm.region import load_region

HERE = Path(__file__).resolve().parent
DEVICE = "cuda"
GROUPS = ("all", "short", "middle", "long")
BRIDGES = {"min": None, "q05": 0.05, "q10": 0.10, "q25": 0.25}


def main() -> None:
    """(1) voltages (V) against graph paths (P) with the same placement rule, both rules; (2) cross-region edges from
    a low quantile of the relays instead of their minimum."""
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
            paths = {r: region_paths(level2[r], config) for r in level2}
            dists = {}
            for bname, b in BRIDGES.items():
                for source, kw in (("V", {"curves": curves}), ("P", {"paths": paths})):
                    if source == "P" and bname not in ("min", "q10"):
                        continue
                    graph = landmark_graph(root, level2, points, regions, config, bridge=b, **kw)
                    weighted = global_coordinates(graph, level2, points, regions, strengths, S, config)
                    route = route_coordinates(graph, S)
                    dists[f"{source}_{bname}_weighted"] = condensed(torch.cdist(weighted, weighted))
                    dists[f"{source}_{bname}_route"] = condensed(torch.cdist(route, route))
            thirds = np.quantile(true, [1 / 3, 2 / 3])
            masks = {"all": np.ones(len(true), dtype=bool), "short": true <= thirds[0],
                     "middle": (true > thirds[0]) & (true <= thirds[1]), "long": true > thirds[1]}
            for method, dd in dists.items():
                for group in GROUPS:
                    rows.append({"noise": sigma, "seed": seed, "method": method, "group": group,
                                 **scores(dd, true, masks[group])})
            print(sigma, seed, flush=True)
    (HERE / "results_robust.json").write_text(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
