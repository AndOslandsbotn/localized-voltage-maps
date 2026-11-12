import json
from pathlib import Path

from experiments.common.guard import watch_memory
from experiments.hsfv_first_attempt.common.levels import fit_two_levels, load_two_levels
from experiments.hsfv_first_attempt.common.overlap import fit_overlapping
from experiments.hsfv_first_attempt.common.repair import repaired_root
from experiments.hsfv_first_attempt.common.scale import dimension_curve, root_cells
from experiments.hsfv_first_attempt.half_sphere.chained_distances.run import CONFIG, SEEDS
from experiments.hsfv_first_attempt.half_sphere.noise.run import NOISE, noisy
from lvm.region import save_region

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Per noise level and seed: the root's cell count chosen from the local-dimension curve, then the same hierarchy
    as before on that root (repair, level 2 with two regions per point)."""
    watch_memory(limit_mb=3000)
    for sigma in NOISE:
        for seed in SEEDS:
            folder = HERE / "models" / f"noise{sigma}" / f"seed{seed}" / "chosen"
            if (folder / "m2" / "regions.npy").exists():
                continue
            _, X = noisy(seed, sigma)
            curve = dimension_curve(X, seed=seed)
            cells = root_cells(curve)
            fit_two_levels(X, config=CONFIG, level1_cells=cells, seed=seed, folder=folder / "base")
            level1, level2, _ = load_two_levels(folder / "base")
            root, changes = repaired_root(level1.root_, level2, level1.config_)
            save_region(root, folder / "root.npz")
            fit_overlapping(X, root, config=CONFIG, m=2, seed=seed, folder=folder / "m2")
            (folder / "choice.json").write_text(json.dumps({"curve": curve, "root_cells": cells, "repair": changes},
                                                           indent=2))
            print(sigma, seed, cells, flush=True)


if __name__ == "__main__":
    main()
