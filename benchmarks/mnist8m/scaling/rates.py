"""Where the streaming time goes: each method's computation rate with the data already in memory, and this disk's
read rate. Placing 8.1M points from disk costs reading + computing; LVM computes faster than the disk reads, so its
end-to-end rate in results.jsonl is the disk's, and varies with it between runs.

* compute: fitted on the 50k sample (as in B), then 200k further points held in memory placed in 10k chunks,
  repeated (the fast methods more often, for at least ~2 s in all); the rate uses the median repeat.
* disk: the 8.1M stream read and converted to floats (exactly what LVM's stream does), no computation. "cold":
  the file is first dropped from Linux's page cache (posix_fadvise DONTNEED, no root needed), as on a first read;
  "warm": read again straight after, partly from the cache. (WSL2: Windows may still cache the virtual disk.)

Every measurement is its own memory-guarded process. Results: rates.json, drawn by plot.py.

    python benchmarks/mnist8m/scaling/rates.py
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "benchmarks")
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from common.runner import thread_env  # noqa: E402
from run import FULL, LIMIT_MB, SAMPLE  # noqa: E402

METHODS = ("lvm_gpu", "lvm_pca_gpu", "umap_gpu", "lisomap_gpu")
CHUNK, PLACED, MIN_REPEATS, MIN_SECONDS = 10_000, 200_000, 3, 2.0


def _model(method: str, seed: int):
    """A function placing a float32 chunk, fitted on the first 50k points exactly as one.py does in B."""
    from one import _float32, lisomap_model

    if method.startswith("lvm"):
        from common.datasets import stream
        from common.methods import _merge, method_params
        from lvm.config import load_config
        from lvm.pipeline import fit_level

        cfg = load_config(overrides=_merge(method_params(method), {"compute": {"device": "cuda", "seed": seed},
                                                                   "data": {"chunk_size": CHUNK}}))
        return fit_level(stream("mnist8m", n=SAMPLE, chunk_size=CHUNK), cfg).transform
    if method == "umap_gpu":
        from cuml.manifold import UMAP

        from common.methods import method_params

        params = method_params("umap_gpu")
        model = UMAP(n_components=2, n_neighbors=params["n_neighbors"], min_dist=params["min_dist"])
        model.fit(_float32(SAMPLE))
        return lambda chunk: np.asarray(model.transform(chunk), dtype=np.float64)
    return lisomap_model(_float32(SAMPLE), seed)[1]


def compute_rate(method: str, seed: int) -> dict:
    import torch

    from common.datasets import mnist8m_rows

    place = _model(method, seed)
    X = mnist8m_rows(SAMPLE, SAMPLE + PLACED).astype(np.float32)
    X /= 255.0
    place(X[:CHUNK])                                                                  # warm-up
    times = []
    while len(times) < MIN_REPEATS or sum(times) < MIN_SECONDS:
        torch.cuda.synchronize()
        t = time.perf_counter()
        for s in range(0, PLACED, CHUNK):
            place(X[s:s + CHUNK])
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return {"kind": "compute", "method": method, "points": PLACED, "seconds": times,
            "rate": PLACED / float(np.median(times))}


def disk_rate() -> dict:
    import os

    from common.datasets import MNIST8M_DIR, stream

    def read(cold: bool) -> float:
        if cold:
            fd = os.open(MNIST8M_DIR / "X.u8", os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        t = time.perf_counter()
        for _chunk in stream("mnist8m", n=FULL, chunk_size=CHUNK)():
            pass
        return time.perf_counter() - t

    cold = [read(True), read(True)]
    warm = read(False)
    return {"kind": "disk", "points": FULL, "seconds": cold, "warm_seconds": warm,
            "rate": FULL / float(np.median(cold)), "rate_warm": FULL / warm}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--one", default=None, help="measure one thing in this process: a method, or 'disk'")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.one:
        row = disk_rate() if args.one == "disk" else compute_rate(args.one, args.seed)
        print("RESULT " + json.dumps(row), flush=True)
        return
    rows = []
    for what in (*METHODS, "disk"):
        cmd = [sys.executable, str(BENCH / "common" / "guard.py"), "--limit-mb", str(LIMIT_MB), "--",
               sys.executable, str(Path(__file__).resolve()), "--one", what, "--seed", str(args.seed)]
        out = subprocess.run(cmd, capture_output=True, text=True, env=thread_env(16))
        lines = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
        if not lines:
            print(f"{what}: failed ({out.returncode})\n{out.stderr[-800:]}", flush=True)
            continue
        rows.append(json.loads(lines[-1][len("RESULT "):]))
        print(f"{what:12s} {rows[-1]['rate'] / 1e3:8.0f}k points/s  {[round(t, 2) for t in rows[-1]['seconds']]} s"
              + (f"  (warm: {rows[-1]['rate_warm'] / 1e3:.0f}k)" if what == "disk" else ""), flush=True)
    (HERE / "rates.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
