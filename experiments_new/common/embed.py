from __future__ import annotations

import sys
import time

import numpy as np
import torch

from experiments_new.common.datasets import load
from experiments_new.common.methods import METHODS


def main() -> None:
    """One embedding in this process (started by runner.embed, under the memory guard): fit the method to n points
    of a dataset split and save the coordinates, labels and fit time to an .npz."""
    method, dataset, split, n, seed, path = sys.argv[1:]
    n, seed = int(n), int(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, labels = load(split, n=n, seed=seed, dataset=dataset)
    X = np.ascontiguousarray(X, dtype=np.float32)
    fit = METHODS[method]
    fit(X[:2000], seed=seed, device=device)                 # warm-up: libraries and GPU kernels load once
    _synchronize(device)
    start = time.perf_counter()
    Z = fit(X, seed=seed, device=device)
    _synchronize(device)
    seconds = time.perf_counter() - start
    np.savez(path, Z=Z, labels=np.array([]) if labels is None else labels, seconds=seconds, device=device)


def _synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
