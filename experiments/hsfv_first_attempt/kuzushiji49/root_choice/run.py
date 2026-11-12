import json
from pathlib import Path

import torch

from experiments.common.guard import watch_memory
from experiments.hsfv_first_attempt.common.overlap import fit_overlapping, load_overlapping
from experiments.hsfv_first_attempt.common.real_data import read
from experiments.hsfv_first_attempt.common.repair import repaired_root
from experiments.hsfv_first_attempt.common.scale import dimension_curve_gpu, root_cells
from lvm import LocalizedVoltageMaps
from lvm.region import load_region, save_region

HERE = Path(__file__).resolve().parent
SAVED = HERE.parent / "coarse_root" / "models"
DATASET, N, SEEDS = "kuzushiji49", 100_000, (0, 1, 2)
CONFIG = {}


def main() -> None:
    """Per seed: the hierarchy (level 2 with two regions per point) on the default 300-cell repaired root (saved by
    coarse_root) and on a root sized by the local-dimension curve."""
    watch_memory(limit_mb=3000)
    X, _ = read(DATASET, N)
    for seed in SEEDS:
        folder = HERE / "models" / f"seed{seed}"
        if (folder / "chosen_m2" / "regions.npy").exists():
            continue
        default_root = load_region(SAVED / "root300" / f"seed{seed}" / "root_repaired.npz", "cuda")
        repair = json.loads((SAVED / "root300" / f"seed{seed}" / "repair.json").read_text())
        fit_overlapping(X, default_root, config=CONFIG, m=2, seed=seed, folder=folder / "default_m2")
        curve = dimension_curve_gpu(torch.as_tensor(X, device="cuda"), seed=seed)
        cells = root_cells(curve)
        if (SAVED / f"root{cells}").exists():
            chosen_root = load_region(SAVED / f"root{cells}" / f"seed{seed}" / "root_repaired.npz", "cuda")
        else:
            level1 = LocalizedVoltageMaps(local_chart="none", random_state=seed,
                                          config={**CONFIG, "cells": {"n_cells": cells}}).fit(X)
            level1.save(folder / "chosen_level1")
            fit_overlapping(X, level1.root_, config=CONFIG, m=1, seed=seed, folder=folder / "chosen_plain")
            level2, _, _ = load_overlapping(folder / "chosen_plain", level1.device_)
            chosen_root, _ = repaired_root(level1.root_, level2, level1.config_)
        save_region(chosen_root, folder / "chosen_root.npz")
        fit_overlapping(X, chosen_root, config=CONFIG, m=2, seed=seed, folder=folder / "chosen_m2")
        (folder / "choice.json").write_text(json.dumps({"curve": curve, "root_cells": cells,
                                                        "default_flagged_share": repair["removed_edges"] / repair["edges"]},
                                                       indent=2))
        print(seed, cells, curve, flush=True)


if __name__ == "__main__":
    main()
