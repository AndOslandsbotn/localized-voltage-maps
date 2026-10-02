"""Benchmark data: MNIST split once into a tuning set and a disjoint evaluation set.

One fixed shuffle of all 70,000 images (``SPLIT_SEED``) gives

* ``tune``: the first 20,000 -- used only to choose hyperparameters;
* ``eval``: the remaining 50,000 -- used for every reported comparison.

No image used for tuning appears in a reported result. Within a split, a run
draws a random subset of size n with its own ``seed`` (a different subset per
seed), in shuffled order so LVM can stream it.

Images are cached as uint8 .npy files under data/; only the rows a run needs
are converted to float64.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data"
SPLIT_SEED = 20261001
SPLITS = {"tune": (0, 20000), "eval": (20000, 70000)}


def _mnist_uint8() -> tuple[np.ndarray, np.ndarray]:
    X_path, y_path = DATA / "mnist_X.npy", DATA / "mnist_y.npy"
    if not X_path.exists():
        from sklearn.datasets import fetch_openml

        X, y = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff", data_home=DATA, return_X_y=True)
        DATA.mkdir(exist_ok=True)
        np.save(X_path, X.astype(np.uint8))
        np.save(y_path, y.astype(np.int64))
    return np.load(X_path), np.load(y_path)


def split_size(split: str) -> int:
    start, stop = SPLITS[split]
    return stop - start


def load(split: str, *, n: int | None = None, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """n images (default: the whole split) from ``split``, float64 in [0, 1], shuffled by ``seed``."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {list(SPLITS)}, got {split!r}")
    X, y = _mnist_uint8()
    start, stop = SPLITS[split]
    members = np.random.default_rng(SPLIT_SEED).permutation(X.shape[0])[start:stop]
    if n is not None and n > members.size:
        raise ValueError(f"split {split!r} has {members.size} images, asked for {n}")
    rows = members[np.random.default_rng(seed).permutation(members.size)[:n]]
    return X[rows].astype(np.float64) / 255.0, y[rows]
