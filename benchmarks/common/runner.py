"""Run one benchmark configuration and measure it.

``run_isolated`` starts a fresh Python process per run (used by the
comparisons), so peak memory belongs to that run alone and every method gets
the same thread count. A memory guard stops a run whose process tree exceeds
``MEMORY_LIMIT_MB`` and records it as ``memory_limit``, so a method that needs
more memory than the machine has shows up as a result instead of crashing it. ``measure`` does the work inside that process, and
tuning calls it directly in one long-lived process, where only quality and
time matter.

Measurements, identical for every method:
* fit_s / total_s: fitting / until every point has coordinates. Data loading
  is excluded; a warm-up on 2000 points first absorbs one-off costs (numba
  compilation, CUDA and cuML start-up).
* peak_rss_mb: the process's peak resident memory (includes libraries).
* gpu_peak_mb: peak GPU memory used above the level before the run, sampled
  from the driver (NVML) every few ms. Unlike a library's own counter this
  sees every allocator (torch, cuML/RMM, CuPy). Allocator caches are emptied
  before the baseline is taken.
* the quality measures of ``metrics.quality``.

    python benchmarks/common/runner.py --method umap_gpu [--dataset mnist] --split eval --n 20000 --seed 0 \
        --params '{"n_neighbors": 15}'
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")
# The machine has 7.8 GB of RAM, ~2 GB of which the OS, WSL and the editor need.
MEMORY_LIMIT_MB = 5000


def thread_env(threads: int) -> dict[str, str]:
    """Environment that gives every library (BLAS, OpenMP/FAISS, numba/UMAP, torch) ``threads`` threads."""
    return {**os.environ, **{var: str(threads) for var in THREAD_VARS}}


def run_isolated(method: str, *, split: str, n: int, seed: int, params: dict, threads: int, dataset: str = "mnist",
                 timeout: float = 1800.0, memory_limit_mb: float = MEMORY_LIMIT_MB) -> dict:
    """Run one configuration in a fresh process; returns the result row (with a ``status``).

    status is "ok", "memory_limit" (the run's process tree went over
    ``memory_limit_mb`` and was stopped), "timeout" or "failed: ...".
    ``guard_peak_mb`` is the highest memory the guard saw, sampled every 0.2 s.
    """
    import tempfile

    import psutil

    cmd = [sys.executable, str(HERE / "runner.py"), "--method", method, "--dataset", dataset, "--split", split,
           "--n", str(n), "--seed", str(seed), "--params", json.dumps(params), "--threads", str(threads)]
    base = {"dataset": dataset, "method": method, "split": split, "n": n, "seed": seed,
            "params": json.dumps(params, sort_keys=True), "memory_limit_mb": memory_limit_mb}
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        proc = subprocess.Popen(cmd, stdout=out, stderr=err, text=True, env=thread_env(threads))
        handle = psutil.Process(proc.pid)
        start, peak, status = time.monotonic(), 0.0, None
        while proc.poll() is None:
            try:
                rss = sum(p.memory_info().rss for p in [handle, *handle.children(recursive=True)]) / 2**20
            except psutil.NoSuchProcess:
                rss = 0.0
            peak = max(peak, rss)
            if rss > memory_limit_mb:
                status = "memory_limit"
            elif time.monotonic() - start > timeout:
                status = "timeout"
            if status:
                for p in [*handle.children(recursive=True), handle]:
                    try:
                        p.kill()
                    except psutil.NoSuchProcess:
                        pass
                proc.wait()
                return {**base, "status": status, "guard_peak_mb": peak}
            time.sleep(0.2)
        out.seek(0)
        err.seek(0)
        stdout, stderr = out.read(), err.read()
    for line in stdout.splitlines():
        if line.startswith("RESULT "):
            return {**json.loads(line[len("RESULT "):]), "status": "ok", "guard_peak_mb": peak,
                    "memory_limit_mb": memory_limit_mb}
    tail = (stderr.strip().splitlines() or ["no output"])[-1]
    return {**base, "status": f"failed (exit {proc.returncode}): {tail[:200]}", "guard_peak_mb": peak}


class _GpuSampler:
    """Peak GPU memory in use (MB) above the level when ``start`` was called, via NVML."""

    def __init__(self, interval: float = 0.005):
        import pynvml

        pynvml.nvmlInit()
        self._nvml = pynvml
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self._interval = interval
        self._stop = threading.Event()

    def _used(self) -> int:
        return self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used

    def start(self) -> None:
        self.baseline = self.peak = self._used()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self._used())
            time.sleep(self._interval)

    def stop(self) -> float:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self._used())
        return (self.peak - self.baseline) / 2**20


def _free_gpu_caches() -> None:
    """Return cached blocks to the driver so the NVML baseline reflects real use."""
    if "torch" in sys.modules:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    if "cupy" in sys.modules:
        import cupy

        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()


_warmed_up: set[str] = set()


def measure(method: str, *, split: str, n: int, seed: int, params: dict, threads: int | None = None,
            eval_size: int = 5000, dataset: str = "mnist") -> dict:
    """Load data, run one method once, and return time, memory and quality."""
    sys.path.insert(0, str(BENCH))
    from common.datasets import DATASETS, load
    from common.methods import METHODS
    from common.metrics import quality

    m = METHODS[method]
    if m.family.startswith("lvm") and threads:
        import torch

        torch.set_num_threads(threads)
    X, y = load(split, n=n, seed=seed, dataset=dataset)
    if method not in _warmed_up:
        m.embed(X[:2000].copy(), params, seed)
        _warmed_up.add(method)

    sampler = None
    if m.device == "gpu":
        _free_gpu_caches()
        sampler = _GpuSampler()
        sampler.start()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    t0 = time.perf_counter()
    Z, fit_s, info = m.embed(X, params, seed)
    total_s = time.perf_counter() - t0
    gpu_peak_mb = sampler.stop() if sampler else None
    rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    return {
        "dataset": dataset, "method": method, "family": m.family, "device": m.device, "split": split, "n": n,
        "seed": seed,
        "params": json.dumps(params, sort_keys=True), "threads": threads,
        "fit_s": fit_s, "total_s": total_s,
        "peak_rss_mb": rss_peak, "extra_rss_mb": max(0.0, rss_peak - rss_before), "gpu_peak_mb": gpu_peak_mb,
        **quality(X, Z, y, eval_size=eval_size, seed=seed, extra=DATASETS[dataset].extra_metrics),
        "info": json.dumps(info) if info else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--split", choices=["tune", "eval"], required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--params", default="{}")
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    row = measure(args.method, split=args.split, n=args.n, seed=args.seed, params=json.loads(args.params),
                  threads=args.threads, dataset=args.dataset)
    print("RESULT " + json.dumps(row))


if __name__ == "__main__":
    main()
