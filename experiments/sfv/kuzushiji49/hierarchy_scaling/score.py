import json
from pathlib import Path

import numpy as np

from experiments.common.guard import watch_memory
from experiments.common.metrics import quality
from experiments.sfv.kuzushiji49.hierarchy_scaling.run_lvm import DATASET, SEEDS, SIZES, points

HERE = Path(__file__).resolve().parent
MIN_SCORED = 50
GLOBAL = ("umap", "tsne", "lvm_level1", "lvm_level1_chart")
LOCAL = ("umap", "tsne", "lvm_level1", "lvm_level1_chart", "lvm_level2", "lvm_level2_chart")


def main() -> None:
    watch_memory(limit_mb=3000)
    results = {"global": [], "local": [], "time": []}
    for n in SIZES:
        for seed in SEEDS:
            if (HERE / "embeddings" / f"lvm_n{n}_seed{seed}.npz").exists():
                for part, rows in score_run(n, seed).items():
                    results[part].extend(rows)
    (HERE / "results.json").write_text(json.dumps({"dataset": DATASET, "sizes": SIZES, "seeds": SEEDS,
                                                   "min_scored": MIN_SCORED, **results}, indent=2))


def score_run(n: int, seed: int) -> dict[str, list[dict]]:
    """Every method's scores and times for one n and seed: global, within the LVM run's level-1 regions, and time."""
    results = {"global": [], "local": [], "time": []}
    lvm = np.load(HERE / "embeddings" / f"lvm_n{n}_seed{seed}.npz")
    maps = {f"lvm_{name}": lvm[name] for name in ("level1", "level1_chart", "level2", "level2_chart")}
    for method in ("umap", "tsne"):
        path = HERE / "embeddings" / f"{method}_n{n}_seed{seed}.npz"
        if path.exists():
            saved = np.load(path)
            maps[method] = saved["Z"]
            results["time"].append({"method": method, "n": n, "seed": seed, "seconds": float(saved["seconds"])})
    results["time"].extend(lvm_times(lvm, n, seed))
    X, labels = points(n, seed)
    region_of_point = lvm["region_of_point"]
    for method in GLOBAL:
        if method in maps:
            results["global"].append({"method": method, "n": n, "seed": seed,
                                      **quality(X, maps[method], labels, seed=seed)})
    for method in LOCAL:
        if method in maps:
            results["local"].append({"method": method, "n": n, "seed": seed,
                                     "neighbours_in_region": float(lvm["neighbours_in_region"]),
                                     **within_regions(maps[method], X, labels, region_of_point, seed)})
    return results


def lvm_times(lvm, n: int, seed: int) -> list[dict]:
    """LVM's times, every one from reading the data: level 1 alone, level 1 + chart, and levels 1 + 2 + chart with
    the regions done one after another (their sum) or in parallel (the slowest region)."""
    s = json.loads(str(lvm["seconds"]))
    regions = json.loads(str(lvm["region_seconds"])).values()
    level2 = [r["fit"] + r["chart_fit"] + r["chart_place"] for r in regions]
    before = s["level1_fit"] + s["bucketing"]
    times = {"lvm_level1": s["level1_fit"] + s["level1_place"],
             "lvm_level1_chart": s["level1_fit"] + s["level1_chart_fit"] + s["level1_chart_place"],
             "lvm_level2_chart": before + sum(level2),
             "lvm_level2_chart_parallel": before + max(level2)}
    return [{"method": method, "n": n, "seed": seed, "seconds": value} for method, value in times.items()]


def within_regions(Z: np.ndarray, X: np.ndarray, labels: np.ndarray, region_of_point: np.ndarray,
                   seed: int) -> dict:
    """Every metric within each region of at least MIN_SCORED points, on that region's points only, averaged
    weighted by its points (a metric that is undefined in a region is left out of that region's average)."""
    totals, weights, scored = {}, {}, 0
    for r in np.unique(region_of_point):
        members = np.flatnonzero(region_of_point == r)
        if len(members) < MIN_SCORED:
            continue
        scored += len(members)
        for key, value in quality(X[members], Z[members], labels[members], seed=seed).items():
            if np.isfinite(value):
                totals[key] = totals.get(key, 0.0) + len(members) * value
                weights[key] = weights.get(key, 0) + len(members)
    return {**{key: totals[key] / weights[key] for key in totals}, "scored_share": scored / len(X)}


if __name__ == "__main__":
    main()
