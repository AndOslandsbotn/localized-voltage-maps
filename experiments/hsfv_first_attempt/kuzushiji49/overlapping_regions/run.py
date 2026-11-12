from pathlib import Path

from experiments.common.guard import watch_memory
from experiments.hsfv_first_attempt.common.overlap import fit_overlapping
from experiments.hsfv_first_attempt.common.real_data import read
from lvm import LocalizedVoltageMaps

HERE = Path(__file__).resolve().parent
DATASET, N, SEEDS, MS = "kuzushiji49", 100_000, (0, 1, 2), (1, 2)
CONFIG = {}


def main() -> None:
    """Per seed: level 1 on the first N images, then level 2 with each point in its m strongest regions."""
    watch_memory(limit_mb=3000)
    X, _ = read(DATASET, N)
    for seed in SEEDS:
        folder = HERE / "models" / f"seed{seed}"
        if (folder / f"m{MS[-1]}" / "regions.npy").exists():
            continue
        level1 = LocalizedVoltageMaps(local_chart="none", random_state=seed, config=CONFIG).fit(X)
        level1.save(folder / "level1")
        for m in MS:
            fit_overlapping(X, level1.root_, config=CONFIG, m=m, seed=seed, folder=folder / f"m{m}")


if __name__ == "__main__":
    main()
