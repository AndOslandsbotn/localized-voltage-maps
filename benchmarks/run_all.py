"""Scaling benchmark on MNIST: time, peak memory and quality against n, per method.

Each (method, n) runs in its own process (``run_one.py``) with a timeout;
results are appended to a CSV as they finish, so an interrupted sweep keeps
what it has, and a rerun skips rows already present. Then a figure is drawn.

    python benchmarks/run_all.py                       # full sweep + plot
    python benchmarks/run_all.py --methods lvm umap --sizes 5000 10000
    python benchmarks/run_all.py --plot-only
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIELDS = ["method", "n", "status", "fit_s", "total_s", "peak_rss_mb", "extra_rss_mb", "gpu_peak_mb",
          "trustworthiness", "continuity", "knn_accuracy", "info"]
LABELS = {
    "lvm": "LVM (GPU)",
    "umap_gpu": "UMAP (GPU, cuML)",
    "lvm_cpu": "LVM (CPU)",
    "umap": "UMAP (CPU, umap-learn)",
    "le": "Laplacian Eigenmaps (CPU)",
    "pca": "PCA",
}
# cuML lives in its own venv so its numba/numpy pins don't touch the main one.
PYTHON = {"umap_gpu": str(HERE.parent / ".venv-rapids" / "bin" / "python")}
# Largest n a method may run at. Laplacian Eigenmaps at 70k exhausts the 7 GB
# of RAM and takes the whole machine down (40k already took 8 minutes).
MAX_N = {"le": 40000}


def run(method: str, n: int, timeout: float) -> dict:
    cmd = [PYTHON.get(method, sys.executable), str(HERE / "run_one.py"), "--method", method, "--n", str(n)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"method": method, "n": n, "status": f"timeout>{timeout:.0f}s"}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            row = json.loads(line[len("RESULT "):])
            row["status"] = "ok"
            if "info" in row:
                row["info"] = json.dumps(row["info"])
            return row
    # Out of memory shows up as a kill signal (-9) with no result line.
    tail = (proc.stderr.strip().splitlines() or ["no output"])[-1]
    return {"method": method, "n": n, "status": f"failed (exit {proc.returncode}): {tail[:200]}"}


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def plot(rows: list[dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in rows if r["status"] == "ok"]
    methods = [m for m in LABELS if any(r["method"] == m for r in ok)]
    panels = [
        ("total_s", "time to embed all points [s]", True),
        ("peak_rss_mb", "peak process memory [MB]", True),
        ("trustworthiness", "trustworthiness (k=10)", False),
        ("knn_accuracy", "5-NN label accuracy", False),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.2))
    for ax, (key, label, log) in zip(axes, panels):
        for m in methods:
            pts = sorted((int(r["n"]), float(r[key])) for r in ok if r["method"] == m and r[key] not in ("", None))
            if pts:
                ax.plot(*zip(*pts), marker="o", label=LABELS[m])
        ax.set_xscale("log")
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("number of points n")
        ax.set_ylabel(label)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend()
    fig.suptitle("MNIST scaling benchmark (2-D embeddings)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"plot: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["pca", "lvm", "umap_gpu", "lvm_cpu", "umap", "le"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[5000, 10000, 20000, 40000, 70000])
    parser.add_argument("--timeout", type=float, default=1800.0, help="seconds per run")
    parser.add_argument("--name", default="mnist_scaling")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    csv_path = RESULTS / f"{args.name}.csv"
    rows = read_rows(csv_path)
    if not args.plot_only:
        done = {(r["method"], int(r["n"])) for r in rows}
        for n in args.sizes:
            for method in args.methods:
                if (method, n) in done:
                    continue
                if n > MAX_N.get(method, n):
                    print(f"{method:8s} n={n:6d}  skipped (above MAX_N={MAX_N[method]})", flush=True)
                    continue
                row = run(method, n, args.timeout)
                rows.append(row)
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(rows)
                summary = (f"{row['total_s']:8.1f}s  rss {row['peak_rss_mb']:7.0f} MB  trust {row['trustworthiness']:.3f}  "
                           f"knn {row['knn_accuracy']:.3f}") if row["status"] == "ok" else row["status"]
                print(f"{method:8s} n={n:6d}  {summary}", flush=True)
    plot(read_rows(csv_path), RESULTS / f"{args.name}.png")


if __name__ == "__main__":
    main()
