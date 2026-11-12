import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.distance import squareform

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.hsfv_first_attempt.common.levels import load_two_levels
from experiments.hsfv_first_attempt.common.repair import repaired_root
from lvm.region import load_region
from experiments.hsfv_first_attempt.half_sphere.chained_distances.run import DATASET, LEVEL1_CELLS, N, SEEDS
from experiments.hsfv_first_attempt.half_sphere.chained_distances.score import GROUPS, SAMPLE, pair_distances, scores, truth

HERE = Path(__file__).resolve().parent
MODELS = HERE.parent / "chained_distances" / "models"


def methods(level1, level2: dict, region_of_point: np.ndarray, X: np.ndarray, sample: np.ndarray) -> dict:
    """Pair distances on the original and on the repaired root: level 1 alone, voltage-weighted chaining through
    level 2, and the combination (level 2 directly within a region, chaining across)."""
    repaired, changes = repaired_root(level1.root_, level2, level1.config_)
    fixed = SimpleNamespace(root_=repaired, config_=level1.config_, device_=level1.device_)
    out = {}
    for suffix, model in (("", level1), ("_repaired", fixed)):
        d = pair_distances(X, sample, model, level2, region_of_point)
        within = np.isfinite(d["level2_within"])
        out[f"level1{suffix}"] = d["level1"]
        out[f"chained_voltage{suffix}"] = d["chained_voltage"]
        out[f"combined{suffix}"] = np.where(within, d["level2_within"], d["chained_voltage"])
    return out, changes


def new_regions(level1, folder: Path, X: np.ndarray, sample: np.ndarray) -> dict:
    """Pair distances with the repaired root and level 2 refitted in its own regions (saved by run.py)."""
    root = load_region(folder / "root.npz", level1.device_)
    level2 = {int(p.stem.split("_")[1]): load_region(p, level1.device_) for p in (folder / "level2").glob("region_*.npz")}
    model = SimpleNamespace(root_=root, config_=level1.config_, device_=level1.device_)
    d = pair_distances(X, sample, model, level2, np.load(folder / "region_of_point.npy"))
    within = np.isfinite(d["level2_within"])
    return {"chained_voltage_new_regions": d["chained_voltage"],
            "combined_new_regions": np.where(within, d["level2_within"], d["chained_voltage"])}


def main() -> None:
    watch_memory(limit_mb=3000)
    rows = []
    for setting in LEVEL1_CELLS:
        for seed in SEEDS:
            X = load("eval", n=N, seed=seed, dataset=DATASET)[0].astype(np.float32)
            level1, level2, region_of_point = load_two_levels(MODELS / setting / f"seed{seed}")
            sample = np.random.default_rng(seed).choice(N, SAMPLE, replace=False)
            true = truth(X[sample])
            regions = region_of_point[sample]
            same = squareform((regions[:, None] == regions[None, :]).astype(float), checks=False) > 0
            thirds = np.quantile(true, [1 / 3, 2 / 3])
            masks = {"all": np.ones_like(same), "within": same, "across": ~same, "short": true <= thirds[0],
                     "middle": (true > thirds[0]) & (true <= thirds[1]), "long": true > thirds[1]}
            distances, changes = methods(level1, level2, region_of_point, X, sample)
            distances |= new_regions(level1, HERE / "models_repaired" / setting / f"seed{seed}", X, sample)
            for method, d in distances.items():
                for group in GROUPS:
                    rows.append({"setting": setting, "seed": seed, "method": method, "group": group, **changes,
                                 **scores(d, true, masks[group])})
            print(setting, seed, changes, flush=True)
    (HERE / "results.json").write_text(json.dumps({"dataset": DATASET, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
