import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import squareform

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.hsfv_first_attempt.common.overlap import build_state, distances, level1_distances, load_overlapping
from experiments.hsfv_first_attempt.half_sphere.chained_distances.run import CONFIG, DATASET, N, SEEDS
from experiments.hsfv_first_attempt.half_sphere.chained_distances.score import GROUPS, SAMPLE, scores, truth
from experiments.hsfv_first_attempt.half_sphere.overlapping_regions.score import condensed
from lvm import LocalizedVoltageMaps
from lvm.config import load_config
from lvm.region import load_region

HERE = Path(__file__).resolve().parent
FINE = HERE.parent / "chained_distances" / "models" / "fine"
DEVICE = "cuda"


def main() -> None:
    """Within-cell positions on and off, against the true distances: the repaired 30-cell root alone, level 2 on it
    (one region per point), and a 300-cell root alone."""
    watch_memory(limit_mb=3000)
    config = load_config(overrides=CONFIG)
    rows = []
    for seed in SEEDS:
        X = load("eval", n=N, seed=seed, dataset=DATASET)[0].astype(np.float32)
        sample = np.random.default_rng(seed).choice(N, SAMPLE, replace=False)
        S = torch.as_tensor(sample, device=DEVICE)
        true = truth(X[sample])
        root = load_region(HERE / "models" / f"seed{seed}" / "root.npz", DEVICE)
        level2, regions, strengths = load_overlapping(HERE / "models" / f"seed{seed}" / "m1", DEVICE)
        fine = LocalizedVoltageMaps.load(FINE / f"seed{seed}" / "level1")
        none = np.zeros((N, 1), dtype=np.int64)
        dists = {}
        for within in (False, True):
            tag = "_cells" if within else ""
            state = build_state(X, root, level2, regions, strengths, config, within=within)
            dists[f"coarse_level1{tag}"] = condensed(level1_distances(state, S, S))
            dists[f"coarse_level2{tag}"] = condensed(distances(state, S, S, "weighted"))
            state = build_state(X, fine.root_, {}, none, np.ones((N, 1)), fine.config_, within=within)
            dists[f"fine_level1{tag}"] = condensed(level1_distances(state, S, S))
        top = regions[sample, 0]
        same = squareform((top[:, None] == top[None, :]).astype(float), checks=False) > 0
        thirds = np.quantile(true, [1 / 3, 2 / 3])
        masks = {"all": np.ones_like(same), "within": same, "across": ~same, "short": true <= thirds[0],
                 "middle": (true > thirds[0]) & (true <= thirds[1]), "long": true > thirds[1]}
        for method, d in dists.items():
            for group in GROUPS:
                rows.append({"seed": seed, "method": method, "group": group, **scores(d, true, masks[group])})
    (HERE / "within.json").write_text(json.dumps({"dataset": DATASET, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
