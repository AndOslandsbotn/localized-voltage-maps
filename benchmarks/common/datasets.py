"""Benchmark datasets: one registry, so every experiment folder runs the same way on any dataset.

Each dataset gives

* ``load(split, n, seed)`` -> (X, labels or None): n points of ``split`` in a
  random order (``seed`` picks the subset and the order), float64;
* ``extra_metrics(X, Z)``: quality measures specific to it (computed on the
  global-structure subset), if any;
* ``colourings(X, labels)``: how its pictures are coloured;
* ``reference(X)``: a 2-D view of the truth for pictures, or None.

Datasets:

* ``mnist``: one fixed shuffle of all 70,000 images (``SPLIT_SEED``) gives
  ``tune`` (the first 20,000, used only to choose hyperparameters) and
  ``eval`` (the other 50,000, used for every reported comparison). Downloaded
  from OpenML on first use (``download``), cached as uint8 .npy files under
  data/ (not in git) and checked against SHA-256 fingerprints, so every
  reproduction uses exactly the same images in the same order. Only the rows a
  run needs are converted to float64.

Datasets are never committed. To fetch them up front:

    python benchmarks/common/datasets.py --download
* ``half_sphere``: points uniform on the upper half of the unit sphere in R^3
  (z >= 0), generated; ``tune`` and ``eval`` use disjoint random streams. It
  can be flattened into a disc, so a good 2-D embedding keeps every
  neighbourhood and changes colour smoothly. Its global measure needs no extra:
  straight-line distance is a monotone function of distance along the sphere,
  so ``distance_correlation`` (a rank correlation) already is the geodesic one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Colouring:
    values: np.ndarray
    cmap: str
    categorical: bool          # categorical: label each category at its median in the picture


@dataclass(frozen=True)
class Dataset:
    name: str
    title: str
    splits: dict[str, int | None]        # size of each split (None: generated, any size)
    loader: Callable[[str, int | None, int], tuple[np.ndarray, np.ndarray | None]]
    extra_metrics: Callable[[np.ndarray, np.ndarray], dict[str, float]] | None = None
    colourings: Callable[[np.ndarray, np.ndarray | None], dict[str, Colouring]] | None = None
    reference: Callable[[np.ndarray], np.ndarray] | None = None


def load(split: str, *, n: int | None = None, seed: int = 0, dataset: str = "mnist"):
    """n points (default: the whole split) of ``split`` of ``dataset``, shuffled by ``seed`` -> (X, labels or None)."""
    ds = DATASETS[dataset]
    if split not in ds.splits:
        raise ValueError(f"{dataset} splits are {list(ds.splits)}, got {split!r}")
    return ds.loader(split, n, seed)


def split_size(split: str, dataset: str = "mnist") -> int | None:
    return DATASETS[dataset].splits[split]


# --- MNIST ------------------------------------------------------------------

SPLIT_SEED = 20261001
MNIST_SPLITS = {"tune": (0, 20000), "eval": (20000, 70000)}


# SHA-256 of the cached arrays' bytes (X: (70000, 784) uint8, y: (70000,) int64), as used for every result.
MNIST_SHA256 = {"mnist_X.npy": "5b92582eb909c82a9bfefcd44acbe9233be3eb5b419a44cf211022ad2bf4bf96",
                "mnist_y.npy": "818800b46032126b329f9306cb69a6842cc53ea30318374769bb6f46cc861467"}


def download(dataset: str = "mnist") -> None:
    """Fetch a dataset into data/ if it is missing, then check it against its fingerprints."""
    if dataset != "mnist":
        return                                   # the others are generated
    X_path, y_path = DATA / "mnist_X.npy", DATA / "mnist_y.npy"
    if not (X_path.exists() and y_path.exists()):
        from sklearn.datasets import fetch_openml

        print("downloading MNIST (mnist_784, version 1) from OpenML ...", flush=True)
        X, y = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff", data_home=DATA, return_X_y=True)
        DATA.mkdir(exist_ok=True)
        np.save(X_path, X.astype(np.uint8))
        np.save(y_path, y.astype(np.int64))
        verify("mnist")


def verify(dataset: str = "mnist") -> None:
    """Raise if the cached arrays differ from the ones every reported result used."""
    import hashlib

    for name, expected in MNIST_SHA256.items():
        a = np.load(DATA / name)
        got = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
        if got != expected:
            raise RuntimeError(f"data/{name} differs from the data the results were made with "
                               f"(sha256 {got[:12]}..., expected {expected[:12]}...); delete it and download again")


_MNIST_CACHE: tuple[np.ndarray, np.ndarray] | None = None


def _mnist_uint8() -> tuple[np.ndarray, np.ndarray]:
    global _MNIST_CACHE
    if _MNIST_CACHE is None:
        download("mnist")
        _MNIST_CACHE = np.load(DATA / "mnist_X.npy"), np.load(DATA / "mnist_y.npy")
    return _MNIST_CACHE


def _mnist(split: str, n: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = _mnist_uint8()
    start, stop = MNIST_SPLITS[split]
    members = np.random.default_rng(SPLIT_SEED).permutation(X.shape[0])[start:stop]
    if n is not None and n > members.size:
        raise ValueError(f"split {split!r} has {members.size} images, asked for {n}")
    rows = members[np.random.default_rng(seed).permutation(members.size)[:n]]
    return X[rows].astype(np.float64) / 255.0, y[rows]


def _digits(X: np.ndarray, labels: np.ndarray | None) -> dict[str, Colouring]:
    return {"digit": Colouring(labels, "tab10", True)}


# --- half sphere --------------------------------------------------------------

HALF_SPHERE_STREAMS = {"tune": 1, "eval": 2}


def half_sphere(n: int, seed: int) -> np.ndarray:
    """n points uniform on {x in R^3 : |x| = 1, x_3 >= 0}, in random order."""
    X = np.random.default_rng(seed).normal(size=(n, 3))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    X[:, 2] = np.abs(X[:, 2])
    return X


def _half_sphere(split: str, n: int | None, seed: int) -> tuple[np.ndarray, None]:
    return half_sphere(20000 if n is None else n, seed=[HALF_SPHERE_STREAMS[split], seed]), None


def _height_and_angle(X: np.ndarray, labels: np.ndarray | None) -> dict[str, Colouring]:
    return {"height": Colouring(X[:, 2], "viridis", False),
            "angle": Colouring(np.arctan2(X[:, 1], X[:, 0]), "hsv", False)}


DATASETS: dict[str, Dataset] = {
    "mnist": Dataset("mnist", "MNIST", {s: b - a for s, (a, b) in MNIST_SPLITS.items()}, _mnist,
                     colourings=_digits),
    "half_sphere": Dataset("half_sphere", "half sphere", {"tune": None, "eval": None}, _half_sphere,
                           colourings=_height_and_angle,
                           reference=lambda X: X[:, :2]),
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch and verify the benchmark datasets (they are not in git).")
    parser.add_argument("--download", action="store_true", help="download any missing dataset, then verify it")
    args = parser.parse_args()
    if args.download:
        for name in DATASETS:
            download(name)
    verify("mnist")
    print("datasets ok:", ", ".join(DATASETS))
