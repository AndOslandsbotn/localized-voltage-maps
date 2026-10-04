"""MNIST8M scaling: LVM streamed from disk vs the other methods held in memory, from 1k to all 8.1M points.

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
GRID = (1_000, 5_000, 10_000, 20_000, 30_000, 40_000, 50_000, 200_000, 500_000, 1_000_000, 4_000_000, FULL)
# Up to 50k every method is fitted on all n (A and B coincide): which method needs the fewest points for good quality
STREAMED = ("lvm_gpu", "lvm_pca_gpu")                 # every size: streamed from disk
IN_MEMORY = ("umap_gpu", "tsne_gpu", "le_gpu", "lisomap_gpu")   # fitted on all n, up the grid until memory runs out
TRANSFORM = ("umap_gpu", "lisomap_gpu")               # A: fit on the largest size that fitted, transform the rest
SAMPLE = 50_000                                        # B: LVM's k-means sample; UMAP and Landmark Isomap fit on it too
LIMIT_MB = 5000


def label(method: str, n: int, fit: int) -> str:
    return f"{method} n={n:,} (fit on {fit:,})"


def default_fit(method: str, n: int) -> int:
    return min(n, SAMPLE) if method.startswith("lvm") else n


def planned(rows: list) -> list[tuple[str, int, int, str]]:
    """Every run of the experiment as (method, n, fit_n, part); the A transform runs depend on earlier results."""
    plan = [(m, n, default_fit(m, n), "B") for m in STREAMED for n in GRID]
    plan += [(m, FULL, SAMPLE, "B") for m in TRANSFORM]
    plan += [(m, n, n, "A") for m in IN_MEMORY for n in GRID]
    plan += [("lvm_gpu", n, n, "A") for n in GRID if n > SAMPLE]
    for m in TRANSFORM:
        fitted = [r["n"] for r in rows if r["method"] == m and r["status"] == "ok" and fit_of(r) == r["n"]]
        if fitted and max(fitted) < FULL:
            plan.append((m, FULL, max(fitted), "A"))
    return plan


def find(rows: list, method: str, n: int, fit: int) -> dict | None:
    for r in rows:
        if r["method"] == method and r["n"] == n and fit_of(r) == fit and r["status"] in ("ok", "memory_limit"):
            return r
    return None


def render(rows: list, current: tuple | None = None) -> None:
    """progress.md: an overall bar, every planned run with its status, and the running run's stage."""
    lines, finished, plan = [], 0, planned(rows)
    for m, n, fit, part in plan:
        r = find(rows, m, n, fit)
        if r is not None:
            status = (f"done  {r['total_s']:.0f} s, peak {r['peak_rss_mb'] / 1024:.1f} GB, trust {r['trustworthiness']:.3f}"
                      if r["status"] == "ok" else r["status"])
            finished += 1
        elif current == (m, n, fit):
            status = "**running**"
        elif fit == n and any(x["method"] == m and x["status"] == "memory_limit" and fit_of(x) == x["n"] and x["n"] < n
                              for x in rows):
            status = "skipped (a smaller full fit already exceeded memory)"
            finished += 1
        else:
            status = "pending"
        lines.append(f"| {part} | {label(m, n, fit)} | {status} |")
    total = len(plan)
    bar = "#" * round(30 * finished / total) + "-" * (30 - round(30 * finished / total))
    text = ["# MNIST8M scaling progress", "", f"`[{bar}]` {finished}/{total} runs finished", "",
            "B = fitted on LVM's 50k sample, the rest streamed/transformed (main); A = fitted on everything", ""]
    run_file = HERE / "progress_run.json"
    if current and run_file.exists():
        try:
            st = json.loads(run_file.read_text())
            pct = f" {100 * st['done'] / st['total']:.0f}% ({st['done']:,} / {st['total']:,} points)" if st.get("total") else ""
            text += [f"**Now:** {st['run']}: {st['stage']}{pct}, {st['elapsed_s']:.0f} s", ""]
        except (ValueError, KeyError):
            pass
    text += ["| part | run | status |", "|---|---|---|", *lines, ""]
    (HERE / "progress.md").write_text("\n".join(text))


def fit_of(r: dict) -> int:
    """How many points a run was fitted on (LVM: its k-means sample, 50k unless set, or the streamed points)."""
    default = min(r["n"], SAMPLE) if r["method"].startswith("lvm") else r["n"]
    return (r.get("info") or {}).get("fit_n", default)


def run(rows: list, path: Path, method: str, n: int, fit_n: int | None = None) -> dict:
    """One run in its own memory-guarded process (skipped if already in results.jsonl)."""
    import tempfile
    import time

    fit = fit_n or default_fit(method, n)
    done = find(rows, method, n, fit)     # finished or out of memory counts as done; a failed run is retried
    if done is not None:
        return done
    cmd = [sys.executable, str(BENCH / "common" / "guard.py"), "--limit-mb", str(LIMIT_MB), "--",
           sys.executable, str(HERE / "one.py"), "--method", method, "--n", str(n)] + \
          (["--fit-n", str(fit_n)] if fit_n else [])
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        proc = subprocess.Popen(cmd, stdout=out, stderr=err, text=True, env=thread_env(16))
        while proc.poll() is None:                     # the overview refreshes every 10 s; the run is not touched
            render(rows, (method, n, fit))
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
    # B (the main comparison): fitted on the same 50k sample as LVM, the rest placed by transform; one run to 8.1M
    # per method, with time and quality recorded at every grid size on the way.
    for method in TRANSFORM:
        run(rows, path, method, FULL, fit_n=SAMPLE)
    # A for LVM: the cells fitted on all n points -- k-means on the 50k sample, refined by streaming k-means over
    # all n (memory independent of n).
    for n in GRID:
        if n > SAMPLE and run(rows, path, "lvm_gpu", n, fit_n=n)["status"] != "ok":
            break
    subprocess.run([sys.executable, str(HERE / "plot.py")])


if __name__ == "__main__":
    main()
