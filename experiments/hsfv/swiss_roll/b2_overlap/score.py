import json
from pathlib import Path

import numpy as np
import torch

from experiments.common.datasets import load, unrolled
from experiments.common.guard import watch_memory
from experiments.hsfv.common.directions import nested
from experiments.hsfv.common.frames import align, global_coordinates, overlap_pins, pair_transforms, routes
from experiments.hsfv.common.levels import score_regions
from experiments.hsfv.swiss_roll.b2_overlap.run import SIZES
from experiments.hsfv.swiss_roll.level2_regions.run import DATASET, SEED
from experiments.hsfv.swiss_roll.level2_regions.score import geodesic
from lvm import LocalizedVoltageMaps
from lvm.cells import assign_cells

HERE = Path(__file__).resolve().parent
SAMPLE, PIN_RATIO, EDGE_SAMPLE = 500, 0.8, 500
SHEET = unrolled
SHOWN = ("rank", "rank_close", "rank_middle", "rank_far", "recall@5", "recall@20", "recall@78", "recall@305",
         "recall@1203")


def saved_covered(saved) -> dict:
    covered = {}
    for key in saved.files:
        if key.startswith("region_"):
            _, r, name = key.split("_", 2)
            covered.setdefault(int(r), {})[name] = saved[key]
    return covered


def edge_misfits(X: np.ndarray, region: np.ndarray, covered: dict, pins: dict, seed: int) -> dict:
    """Each region's map against the true flat sheet (similarity fit): at its pins, and at as many of its own
    points that are not pins."""
    rng, misfits = np.random.default_rng(seed), {}
    for r, item in covered.items():
        at_pins = np.unique(np.concatenate([ids for pair, ids in pins.items() if r in pair] or [np.empty(0, int)]))
        if len(at_pins) < 3:
            continue
        own = np.setdiff1d(np.flatnonzero(region == r), at_pins)
        inside = rng.choice(own, min(EDGE_SAMPLE, len(own)), replace=False)
        where = lambda rows: np.searchsorted(item["ids"], rows)
        misfits[r] = {"pins": align(item["level2_chart"][where(at_pins)], SHEET(X[at_pins].astype(np.float64)))[1],
                      "inside": align(item["level2_chart"][where(inside)], SHEET(X[inside].astype(np.float64)))[1]}
    return misfits


def main() -> None:
    watch_memory(limit_mb=3000)
    for n in SIZES:
        X = load("eval", n=n, seed=SEED, dataset=DATASET)[0].astype(np.float32)
        saved = np.load(HERE / f"maps_n{n}.npz")
        level1 = LocalizedVoltageMaps.load(HERE / "models" / f"n{n}" / "level1")
        region, Z2, membership = saved["region"], saved["level2_chart"], saved["membership"]
        covered = saved_covered(saved)
        cell = assign_cells(torch.as_tensor(X, device=level1.device_), level1.root_.centroids).cpu().numpy()
        pins = overlap_pins(level1.root_, cell, membership, PIN_RATIO)
        transforms, misfits = pair_transforms(pins, covered)
        regions = [int(r) for r in np.unique(region)]
        reference = int(np.bincount(region).argmax())
        costs = {"search": {pair: (misfits[pair] + misfits[pair[::-1]]) / 2 for pair in pins},
                 "parent": {pair: 1.0 for pair in pins}}
        found = {option: routes(cost, regions, reference) for option, cost in costs.items()}
        maps = {"level1_chart": saved["level1_chart"], "nested": nested(saved["level1_chart"], Z2, region)}
        maps |= {f"b2_{option}": global_coordinates(Z2, region, transforms, found[option]) for option in found}
        np.savez(HERE / f"global_n{n}.npz", **{name: Z for name, Z in maps.items() if name != "level1_chart"})
        rows = score_regions(X, maps, np.zeros(n, dtype=int), geodesic, sample=SAMPLE, seed=SEED)
        overlap = {"duplicated_share": sum(len(item["ids"]) for item in covered.values()) / n - 1,
                   "level1_cells_per_region": membership.sum(axis=1).tolist()}
        pairs = {f"{c}-{d}": {"pins": len(ids), "misfit": costs["search"][(c, d)]} for (c, d), ids in pins.items()}
        edges = None if SHEET is None else edge_misfits(X, region, covered, pins, SEED)
        results = {"dataset": DATASET, "n": n, "seed": SEED, "sample": SAMPLE, "pin_ratio": PIN_RATIO,
                   "reference": reference, "overlap": overlap, "pairs": pairs, "edge_misfits": edges,
                   "routes": found, "rows": rows}
        (HERE / f"results_n{n}.json").write_text(json.dumps(results, indent=2))
        print(f"== n = {n}, overlap {overlap}")
        print(f"   pairs (pins, misfit): " + ", ".join(f"{p} {v['pins']}/{v['misfit']:.2f}" for p, v in pairs.items()))
        if edges:
            print("   region map vs truth (pins / inside): "
                  + ", ".join(f"{r} {m['pins']:.2f}/{m['inside']:.2f}" for r, m in edges.items()))
        print(f"   routes: {found}")
        print(f"{'':14s}" + "".join(f"{metric:>12s}" for metric in SHOWN))
        for row in rows:
            print(f"{row['map']:14s}" + "".join(f"{row[metric]:12.3f}" for metric in SHOWN))


if __name__ == "__main__":
    main()
