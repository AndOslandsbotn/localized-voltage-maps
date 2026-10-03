"""Benchmark datasets, cached as .npy files under data/ for fast repeated loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"


def load_mnist(seed: int = 0, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The first ``n`` (default all 70,000) MNIST images in a seeded shuffled order, as float64 in [0, 1].

    The first n rows of the shuffle are a uniform random subset of size n.
    Only those rows are converted to float64, so a small subset never costs
    the memory of the full float array.
    """
    X_path, y_path = DATA / "mnist_X.npy", DATA / "mnist_y.npy"
    if not X_path.exists():
        from sklearn.datasets import fetch_openml

        X, y = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff", data_home=DATA, return_X_y=True)
        DATA.mkdir(exist_ok=True)
        np.save(X_path, X.astype(np.uint8))
        np.save(y_path, y.astype(np.int64))
    X, y = np.load(X_path), np.load(y_path)
    order = np.random.default_rng(seed).permutation(X.shape[0])[:n]
    return X[order].astype(np.float64) / 255.0, y[order]
