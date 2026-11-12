import json
from pathlib import Path

import numpy as np
import torch

from experiments.common.guard import watch_memory
from experiments.hsfv_first_attempt.common.landmark_graph import geodesic_curves, global_coordinates, landmark_graph
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


def hierarchy(folder: Path, X: np.ndarray, points: torch.Tensor, S: torch.Tensor, config) -> np.ndarray:
    root = load_region(folder / "root.npz", DEVICE)
    level2, regions, strengths = load_overlapping(folder / "m2", DEVICE)
    curves = {r: geodesic_curves(level2[r], config) for r in level2}
    graph = landmark_graph(root, level2, points, regions, config, curves=curves)
    coords = global_coordinates(graph, level2, points, regions, strengths, S, config)
    return condensed(torch.cdist(coords, coords))


def main() -> None:
    """The hierarchy on a root sized by the local-dimension curve, against the fixed 30-cell root and the single
    300-cell level, on the clean true distances."""
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
            fine = LocalizedVoltageMaps.load(folder / "fine")
            fine_state = build_state(X, fine.root_, {}, np.zeros((N, 1), dtype=np.int64), np.ones((N, 1)),
                                     fine.config_, within=True)
            chosen = json.loads((folder / "chosen" / "choice.json").read_text())
            dists = {"root30": hierarchy(folder, X, points, S, config),
                     "chosen_root": hierarchy(folder / "chosen", X, points, S, config),
                     "fine_lvm": condensed(level1_distances(fine_state, S, S))}
            thirds = np.quantile(true, [1 / 3, 2 / 3])
            masks = {"all": np.ones(len(true), dtype=bool), "short": true <= thirds[0],
                     "middle": (true > thirds[0]) & (true <= thirds[1]), "long": true > thirds[1]}
            for method, dd in dists.items():
                for group in GROUPS:
                    rows.append({"noise": sigma, "seed": seed, "method": method, "group": group,
                                 "chosen_cells": chosen["root_cells"], **scores(dd, true, masks[group])})
            print(sigma, seed, chosen["root_cells"], flush=True)
    (HERE / "results_chosen.json").write_text(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
