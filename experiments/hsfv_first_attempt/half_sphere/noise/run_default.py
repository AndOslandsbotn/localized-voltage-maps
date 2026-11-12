import json
from pathlib import Path

from experiments.common.guard import watch_memory
from experiments.hsfv_first_attempt.common.levels import fit_two_levels, load_two_levels
from experiments.hsfv_first_attempt.common.overlap import fit_overlapping
from experiments.hsfv_first_attempt.common.repair import repaired_root
from experiments.hsfv_first_attempt.half_sphere.chained_distances.run import CONFIG, SEEDS
from experiments.hsfv_first_attempt.half_sphere.noise.run import NOISE, noisy
from lvm.region import save_region

HERE = Path(__file__).resolve().parent
DEFAULT_CELLS = 300


def main() -> None:
    """Per noise level and seed: the hierarchy on the default 300-cell root (its level 2, the repair and how many
    edges it flags, level 2 with two regions per point)."""
    watch_memory(limit_mb=3000)
    for sigma in NOISE:
        for seed in SEEDS:
            folder = HERE / "models" / f"noise{sigma}" / f"seed{seed}" / "default"
            if (folder / "m2" / "regions.npy").exists():
                continue
            _, X = noisy(seed, sigma)
            fit_two_levels(X, config=CONFIG, level1_cells=DEFAULT_CELLS, seed=seed, folder=folder / "base")
            level1, level2, _ = load_two_levels(folder / "base")
            root, changes = repaired_root(level1.root_, level2, level1.config_)
            save_region(root, folder / "root.npz")
            fit_overlapping(X, root, config=CONFIG, m=2, seed=seed, folder=folder / "m2")
            (folder / "choice.json").write_text(json.dumps({"repair": changes,
                                                            "flagged_share": changes["removed_edges"] / changes["edges"]},
                                                           indent=2))
            print(sigma, seed, changes, flush=True)


if __name__ == "__main__":
    main()
