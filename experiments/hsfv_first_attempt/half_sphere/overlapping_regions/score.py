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
from experiments.hsfv_first_attempt.half_sphere.overlapping_regions.run import MS
from lvm.config import load_config
from lvm.region import load_region

HERE = Path(__file__).resolve().parent
DEVICE = "cuda"


def condensed(D: torch.Tensor) -> np.ndarray:
    D = ((D + D.T) / 2).cpu().numpy()
    np.fill_diagonal(D, 0.0)
    return squareform(D, checks=False)


def main() -> None:
    watch_memory(limit_mb=3000)
    config = load_config(overrides=CONFIG)
    rows = []
    for seed in SEEDS:
        folder = HERE / "models" / f"seed{seed}"
        X = load("eval", n=N, seed=seed, dataset=DATASET)[0].astype(np.float32)
        root = load_region(folder / "root.npz", DEVICE)
        sample = np.random.default_rng(seed).choice(N, SAMPLE, replace=False)
        S = torch.as_tensor(sample, device=DEVICE)
        true = truth(X[sample])
        dists = {}
        for m in MS:
            level2, regions, strengths = load_overlapping(folder / f"m{m}", DEVICE)
            state = build_state(X, root, level2, regions, strengths, config)
            if m == 1:
                dists["level1"] = condensed(level1_distances(state, S, S))
                dists["m1"] = condensed(distances(state, S, S, "weighted"))
                top = regions[sample, 0]
            else:
                for rule in ("strongest", "weighted"):
                    dists[f"m{m}_{rule}"] = condensed(distances(state, S, S, rule))
        same = squareform((top[:, None] == top[None, :]).astype(float), checks=False) > 0
        thirds = np.quantile(true, [1 / 3, 2 / 3])
        masks = {"all": np.ones_like(same), "within": same, "across": ~same, "short": true <= thirds[0],
                 "middle": (true > thirds[0]) & (true <= thirds[1]), "long": true > thirds[1]}
        for method, d in dists.items():
            for group in GROUPS:
                rows.append({"seed": seed, "method": method, "group": group, **scores(d, true, masks[group])})
    (HERE / "results.json").write_text(json.dumps({"dataset": DATASET, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
