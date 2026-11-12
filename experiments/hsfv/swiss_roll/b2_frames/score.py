import json
from pathlib import Path

import numpy as np

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.hsfv.common.directions import nested
from experiments.hsfv.common.frames import global_coordinates, pair_transforms, routes
from experiments.hsfv.common.levels import score_regions
from experiments.hsfv.swiss_roll.b2_frames.run import SIZES
from experiments.hsfv.swiss_roll.level2_regions.run import DATASET, SEED
from experiments.hsfv.swiss_roll.level2_regions.score import geodesic

HERE = Path(__file__).resolve().parent
SAMPLE = 500
SHOWN = ("rank", "rank_close", "rank_middle", "rank_far", "recall@5", "recall@20", "recall@78", "recall@305",
         "recall@1203")


def saved_pins(saved) -> tuple[dict, dict]:
    pins = {tuple(int(r) for r in key.split("_")[1:]): saved[key] for key in saved.files if key.startswith("pair_")}
    placed = {}
    for key in saved.files:
        if key.startswith("region_"):
            _, r, name = key.split("_", 2)
            placed.setdefault(int(r), {})[name] = saved[key]
    return pins, placed


def main() -> None:
    watch_memory(limit_mb=3000)
    for n in SIZES:
        X = load("eval", n=n, seed=SEED, dataset=DATASET)[0].astype(np.float32)
        saved = np.load(HERE / f"maps_n{n}.npz")
        region, Z2 = saved["region"], saved["level2_chart"]
        pins, placed = saved_pins(saved)
        transforms, misfits = pair_transforms(pins, placed)
        regions = [int(r) for r in np.unique(region)]
        reference = int(np.bincount(region).argmax())
        costs = {"search": {pair: (misfits[pair] + misfits[pair[::-1]]) / 2 for pair in pins},
                 "parent": {pair: 1.0 for pair in pins}}
        found = {option: routes(cost, regions, reference) for option, cost in costs.items()}
        maps = {"level1_chart": saved["level1_chart"], "nested": nested(saved["level1_chart"], Z2, region)}
        maps |= {f"b2_{option}": global_coordinates(Z2, region, transforms, found[option]) for option in found}
        np.savez(HERE / f"global_n{n}.npz", **{name: Z for name, Z in maps.items() if name != "level1_chart"})
        rows = score_regions(X, maps, np.zeros(n, dtype=int), geodesic, sample=SAMPLE, seed=SEED)
        pairs = {f"{c}-{d}": {"pins": len(ids), "misfit": costs["search"][(c, d)]} for (c, d), ids in pins.items()}
        results = {"dataset": DATASET, "n": n, "seed": SEED, "sample": SAMPLE, "reference": reference,
                   "pairs": pairs, "routes": found, "rows": rows}
        (HERE / f"results_n{n}.json").write_text(json.dumps(results, indent=2))
        print(f"== n = {n}, reference region {reference}, pairs {pairs}")
        print(f"   routes: {found}")
        print(f"{'':14s}" + "".join(f"{metric:>12s}" for metric in SHOWN))
        for row in rows:
            print(f"{row['map']:14s}" + "".join(f"{row[metric]:12.3f}" for metric in SHOWN))


if __name__ == "__main__":
    main()
