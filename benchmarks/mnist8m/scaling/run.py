"""MNIST8M scaling: LVM streamed from disk vs UMAP held in memory, from 50k to all 8.1M points.

Every run is its own memory-guarded process (5 GB, this machine has 7.8 GB); a run over the limit is recorded as
such. Results go to results.jsonl (finished runs are skipped); then plot.py draws figure.png.

    python benchmarks/mnist8m/scaling/run.py
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "benchmarks")
sys.path.insert(0, str(BENCH))

from common.runner import thread_env  # noqa: E402

FULL = 8_100_000
GRID = (50_000, 200_000, 500_000, 1_000_000, 4_000_000, FULL)
STREAMED = ("lvm_gpu", "lvm_pca_gpu")                 # every size: streamed from disk
IN_MEMORY = ("umap_gpu", "tsne_gpu", "le_gpu", "lisomap_gpu")   # fitted on all n, up the grid until memory runs out
TRANSFORM = ("umap_gpu", "lisomap_gpu")               # then: fit on the largest size that fitted, transform the rest
LIMIT_MB = 5000


def label(method: str, n: int, fit_n: int | None) -> str:
    return f"{method} n={n:,}" + (f" (fit on {fit_n:,})" if fit_n else "")


def render(rows: list, current: str | None = None) -> None:
    """progress.md: an overall bar, every planned run with its status, and the running run's stage."""
    done = {label(r["method"], r["n"], (r.get("info") or {}).get("fit_n") if (r.get("info") or {}).get("fit_n", r["n"]) < r["n"] else None): r
            for r in rows}
    failed = {r["method"] for r in rows if r["status"] != "ok" and r["method"] in IN_MEMORY}
    largest = {m: max([r["n"] for r in rows if r["method"] == m and r["status"] == "ok"
                       and (r.get("info") or {}).get("fit_n", r["n"]) == r["n"]], default=None) for m in TRANSFORM}
    plan = [(m, n, None) for m in STREAMED for n in GRID] + [(m, n, None) for m in IN_MEMORY for n in GRID] + \
           [(m, FULL, "largest") for m in TRANSFORM]
    lines, finished, total = [], 0, 0
    for m, n, fit in plan:
        fit_n = largest.get(m) if fit == "largest" else None
        name = label(m, n, fit_n) if fit != "largest" else f"{m} n={FULL:,} (fit on the largest size that fitted" + \
            (f": {fit_n:,})" if fit_n else ")")
        r = done.get(label(m, n, fit_n)) if fit != "largest" or fit_n else None
        if r is not None:
            status = ("done  " + f"{r['total_s']:.0f} s, peak {r['peak_rss_mb'] / 1024:.1f} GB, trust {r['trustworthiness']:.3f}"
                      if r["status"] == "ok" else r["status"])
            finished += 1
        elif current and name.startswith(current.split(" (fit")[0]) and (fit_n is None or current == label(m, n, fit_n)):
            status = "**running**"
        elif m in failed and fit != "largest" and any(x["method"] == m and x["status"] != "ok" and x["n"] < n for x in rows):
            status = "skipped (a smaller size already exceeded memory)"
            finished += 1
        else:
            status = "pending"
        total += 1
        lines.append(f"| {name} | {status} |")
    bar = "#" * round(30 * finished / total) + "-" * (30 - round(30 * finished / total))
    text = [f"# MNIST8M scaling progress", "", f"`[{bar}]` {finished}/{total} runs finished", ""]
    run_file = HERE / "progress_run.json"
    if current and run_file.exists():
        try:
            st = json.loads(run_file.read_text())
            pct = f" {100 * st['done'] / st['total']:.0f}% ({st['done']:,} / {st['total']:,} points)" if st.get("total") else ""
            text += [f"**Now:** {st['run']}: {st['stage']}{pct}, {st['elapsed_s']:.0f} s", ""]
        except (ValueError, KeyError):
            pass
    text += ["| run | status |", "|---|---|", *lines, ""]
    (HERE / "progress.md").write_text("\n".join(text))


def run(rows: list, path: Path, method: str, n: int, fit_n: int | None = None) -> dict:
    """One run in its own memory-guarded process (skipped if already in results.jsonl)."""
    import tempfile
    import time

    fit = fit_n or n
    for r in rows:   # finished or out of memory counts as done; a failed run (e.g. after sleep) is retried
        if r["method"] == method and r["n"] == n and (r.get("info") or {}).get("fit_n", n) == fit \
                and r["status"] in ("ok", "memory_limit"):
            return r
    cmd = [sys.executable, str(BENCH / "common" / "guard.py"), "--limit-mb", str(LIMIT_MB), "--",
           sys.executable, str(HERE / "one.py"), "--method", method, "--n", str(n)] + \
          (["--fit-n", str(fit_n)] if fit_n else [])
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        proc = subprocess.Popen(cmd, stdout=out, stderr=err, text=True, env=thread_env(16))
        while proc.poll() is None:                     # the overview refreshes every 10 s; the run is not touched
            render(rows, label(method, n, fit_n))
            time.sleep(10)
        out.seek(0), err.seek(0)
        stdout, stderr = out.read(), err.read()
    lines = [l for l in stdout.splitlines() if l.startswith("RESULT ")]
    if lines:
        row = {**json.loads(lines[-1][len("RESULT "):]), "status": "ok"}
    else:
        status = "memory_limit" if proc.returncode == 137 else f"failed ({proc.returncode})"
        row = {"method": method, "n": n, "status": status, "info": {"fit_n": fit}, "stderr": stderr[-600:]}
    rows.append(row)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
    render(rows)
    brief = {k: round(row[k], 3) for k in ("total_s", "fit_s", "peak_rss_mb", "gpu_peak_mb", "trustworthiness",
                                           "continuity", "knn_accuracy", "distance_correlation")
             if isinstance(row.get(k), (int, float))}
    print(f"{method:12s} n={n:>9,}" + (f" fit {fit_n:,}" if fit_n else "") + f"  {row['status']}  {brief}", flush=True)
    return row


def main() -> None:
    path = HERE / "results.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    render(rows)
    for method in STREAMED:
        for n in GRID:
            run(rows, path, method, n)
    largest = {}
    for method in IN_MEMORY:
        for n in GRID:
            if run(rows, path, method, n)["status"] != "ok":
                break                                     # the next size needs even more memory
            largest[method] = n
    for method in TRANSFORM:
        if largest.get(method, FULL) < FULL:
            run(rows, path, method, FULL, fit_n=largest[method])
    subprocess.run([sys.executable, str(HERE / "plot.py")])


if __name__ == "__main__":
    main()
