"""Shared hyperparameter tuning: run a grid on the tuning split and keep the best setting.

Protocol (the same for every method):
* data: the ``tune`` split only (20k images), disjoint from the evaluation split;
* each configuration runs with every seed in ``SEEDS``;
* selection: the highest mean trustworthiness (no labels are used to choose);
* budget: about 15 configurations per method, over its most influential parameters.

Every run is its own process (``runner.run_isolated``) with the memory
guard: running many fits in one process leaks memory between runs and once
exhausted the machine's RAM.
Writes ``results.csv``, ``best.yaml`` and ``metadata.json`` into the method's
tuning folder.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import yaml

from common.results import read_rows, write_metadata, write_rows
from common.runner import run_isolated

SPLIT = "tune"
N = 20000
SEEDS = (0, 1, 2)
CRITERION = "trustworthiness"


def run_grid(method: str, grid: list[dict], folder: Path, *, threads: int = 16) -> dict:
    """Run every configuration in ``grid`` with every seed; write results and the best setting."""
    csv_path = folder / "results.csv"
    rows = read_rows(csv_path)
    done = {(r["params"], int(r["seed"])) for r in rows if r.get("status", "ok") == "ok"}
    write_metadata(folder, {"method": method, "split": SPLIT, "n": N, "seeds": list(SEEDS), "criterion": CRITERION,
                            "grid": grid, "threads": threads})
    for params in grid:
        for seed in SEEDS:
            key = json.dumps(params, sort_keys=True)
            if (key, seed) in done:
                continue
            row = run_isolated(method, split=SPLIT, n=N, seed=seed, params=params, threads=threads)
            rows.append(row)
            write_rows(csv_path, rows)
            if row["status"] != "ok":
                print(f"{key:60s} seed={seed}  {row['status']}", flush=True)
                continue
            print(f"{key:60s} seed={seed}  {row['total_s']:6.2f}s  trust {row['trustworthiness']:.3f}  "
                  f"knn {row['knn_accuracy']:.3f}  dcorr {row['distance_correlation']:.3f}", flush=True)
    return select_best(folder)


def summarize(rows: list[dict]) -> dict[str, dict]:
    """Mean and spread over seeds, per configuration."""
    by_params: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("status", "ok") == "ok":
            by_params.setdefault(r["params"], []).append(r)
    out = {}
    for params, rs in by_params.items():
        stats = {}
        for key in ("trustworthiness", "continuity", "knn_accuracy", "distance_correlation", "total_s"):
            values = [float(r[key]) for r in rs]
            stats[key] = {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0}
        stats["seeds"] = len(rs)
        out[params] = stats
    return out


def select_best(folder: Path) -> dict:
    """Pick the configuration with the highest mean criterion and write ``best.yaml``."""
    summary = summarize(read_rows(folder / "results.csv"))
    best_key = max(summary, key=lambda k: summary[k][CRITERION]["mean"])
    best = {
        "params": json.loads(best_key),
        "selected_by": f"highest mean {CRITERION} on the '{SPLIT}' split (n={N}, seeds={list(SEEDS)})",
        "scores": {k: round(v["mean"], 4) for k, v in summary[best_key].items() if isinstance(v, dict)},
    }
    (folder / "best.yaml").write_text(yaml.safe_dump(best, sort_keys=False))
    print(f"best: {best['params']}  ({CRITERION} {best['scores'][CRITERION]:.4f})")
    return best
