import json
from pathlib import Path

import numpy as np
import torch

from experiments.common.guard import watch_memory
from experiments.hsfv_first_attempt.half_sphere.chained_distances.run import CONFIG, N, SEEDS
from experiments.hsfv_first_attempt.half_sphere.chained_distances.score import SAMPLE, scores, truth
from experiments.hsfv_first_attempt.half_sphere.noise.run import NOISE, noisy
from experiments.hsfv_first_attempt.half_sphere.noise.score_chosen import hierarchy
from experiments.hsfv_first_attempt.half_sphere.overlapping_regions.score import condensed
from experiments.hsfv_first_attempt.common.overlap import build_state, level1_distances
from lvm import LocalizedVoltageMaps
from lvm.config import load_config

HERE = Path(__file__).resolve().parent
DEVICE = "cuda"
GROUPS = ("all", "short", "middle", "long")
FLAGGED = 0.02


def main() -> None:
    """Option (a): keep the default 300-cell root unless its repair flags more than FLAGGED of its edges as short-cuts,
    else switch to the dimension-sized root. Against both roots' hierarchies and the single 300-cell level."""
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
            share = json.loads((folder / "default" / "choice.json").read_text())["flagged_share"]
            default = hierarchy(folder / "default", X, points, S, config)
            chosen = hierarchy(folder / "chosen", X, points, S, config)
            dists = {"default_hierarchy": default, "chosen_hierarchy": chosen,
                     "option_a": chosen if share > FLAGGED else default,
                     "fine_lvm": condensed(level1_distances(fine_state, S, S))}
            thirds = np.quantile(true, [1 / 3, 2 / 3])
            masks = {"all": np.ones(len(true), dtype=bool), "short": true <= thirds[0],
                     "middle": (true > thirds[0]) & (true <= thirds[1]), "long": true > thirds[1]}
            for method, dd in dists.items():
                for group in GROUPS:
                    rows.append({"noise": sigma, "seed": seed, "method": method, "group": group,
                                 "flagged_share": share, "switched": share > FLAGGED,
                                 **scores(dd, true, masks[group])})
            print(sigma, seed, share, flush=True)
    (HERE / "results_default.json").write_text(json.dumps({"flagged_threshold": FLAGGED, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
