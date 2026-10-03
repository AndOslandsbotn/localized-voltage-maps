"""Shared comparison protocol: methods at their tuned settings, on the evaluation split.

* data: the ``eval`` split (50k images), never seen during tuning;
* sizes: random subsets of the split, the largest being the whole split;
* each (method, size) runs with every seed, each in its own process
  (``runner.run_isolated``) with the same thread count;
* hyperparameters: ``tuning/<family>/best.yaml``, identical for a family's
  GPU and CPU versions.

Results go to ``results.csv`` (rerunning skips finished rows) and
``metadata.json`` in the experiment folder.
"""

from __future__ import annotations

from pathlib import Path

from common.methods import method_params
from common.results import read_rows, write_metadata, write_rows
from common.runner import run_isolated

SPLIT = "eval"
SIZES = (5000, 10000, 20000, 40000, 50000)
SEEDS = (0, 1, 2)
THREADS = 16


def run_comparison(methods: list[str], folder: Path, *, max_n: dict[str, int] | None = None,
                   sizes=SIZES, seeds=SEEDS, threads: int = THREADS, timeout: float = 1800.0) -> None:
    max_n = max_n or {}
    from common.methods import METHODS

    params = {m: method_params(m) for m in methods}
    csv_path = folder / "results.csv"
    rows = read_rows(csv_path)
    done = {(r["method"], int(r["n"]), int(r["seed"])) for r in rows}
    write_metadata(folder, {"methods": methods, "params": params, "split": SPLIT, "sizes": list(sizes),
                            "seeds": list(seeds), "threads": threads, "max_n": max_n, "timeout_s": timeout})
    for n in sizes:
        for method in methods:
            if n > max_n.get(method, n):
                print(f"{method:9s} n={n:6d}  skipped (above max_n={max_n[method]})", flush=True)
                continue
            for seed in seeds:
                if (method, n, seed) in done:
                    continue
                row = run_isolated(method, split=SPLIT, n=n, seed=seed, params=params[method], threads=threads,
                                   timeout=timeout)
                rows.append(row)
                write_rows(csv_path, rows)
                if row["status"] == "ok":
                    gpu = f"  gpu {row['gpu_peak_mb']:5.0f}MB" if row.get("gpu_peak_mb") is not None else ""
                    print(f"{method:9s} n={n:6d} seed={seed}  {row['total_s']:7.2f}s  rss {row['peak_rss_mb']:5.0f}MB{gpu}  "
                          f"trust {row['trustworthiness']:.3f}  knn {row['knn_accuracy']:.3f}  "
                          f"dcorr {row['distance_correlation']:.3f}", flush=True)
                else:
                    print(f"{method:9s} n={n:6d} seed={seed}  {row['status']}", flush=True)
