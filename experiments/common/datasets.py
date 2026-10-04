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

    python experiments/common/datasets.py --download
* ``half_sphere``: points uniform on the upper half of the unit sphere in R^3
  (z >= 0), generated; ``tune`` and ``eval`` use disjoint random streams. It
  can be flattened into a disc, so a good 2-D embedding keeps every
  neighbourhood and changes colour smoothly. Its global measure needs no extra:
  straight-line distance is a monotone function of distance along the sphere,
  so ``distance_correlation`` (a rank correlation) already is the geodesic one.
* ``mnist8m``: 8.1 million deformed MNIST digits (Loosli, Canu & Bottou 2007,
  "infimnist"), 784 pixels, labels; downloaded from the LIBSVM site (2.35 GB
  xz), converted to a uint8 file on disk (6.4 GB, never held in memory) and
  shuffled once with a fixed seed, as the streaming method assumes. It does not
  fit in this machine's memory: LVM streams it (``stream``), other methods get
  what fits. Settings are MNIST's, unchanged (no tuning split).
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
    if dataset == "mnist8m":
        return _download_mnist8m()
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

    if dataset == "mnist8m":
        return _verify_mnist8m()
    if dataset != "mnist":
        return

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


# --- MNIST8M -------------------------------------------------------------------

MNIST8M_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist8m.xz"
MNIST8M_BYTES = 2354965120                     # size of the xz file
MNIST8M_N, MNIST8M_D = 8_100_000, 784
MNIST8M_DIR = DATA / "mnist8m"
MNIST8M_SHUFFLE_SEED = 20261003
# SHA-256 of the shuffled arrays (X.u8: (8100000, 784) uint8 row-major, y.i8: (8100000,) int8), filled after the
# first build; every result on mnist8m used exactly these.
MNIST8M_SHA256: dict[str, str] = {"X.u8": "018d51306e385e89d1d3259565c6d25576a4447ba61485240f2c1e7f4c341181",
                                  "y.i8": "fae93bd6851cde2d506597a3860d7db863adf17e8fb255fbd51e5eaeb1546102"}


_MNIST8M_ROW = MNIST8M_D                       # bytes per image row (uint8)


def _mnist8m_arrays():
    """Read-only memory maps of the shuffled images and labels; for small random reads only (scoring rows,
    labels). Pages read through a map count towards the process's resident memory, so never stream
    through it: use ``mnist8m_rows`` or ``stream``."""
    X = np.memmap(MNIST8M_DIR / "X.u8", dtype=np.uint8, mode="r", shape=(MNIST8M_N, MNIST8M_D))
    y = np.memmap(MNIST8M_DIR / "y.i8", dtype=np.int8, mode="r", shape=(MNIST8M_N,))
    return X, y


def mnist8m_rows(start: int, stop: int) -> np.ndarray:
    """Rows [start, stop) of the shuffled images as uint8, read with a plain file read (only these rows in memory)."""
    download("mnist8m")
    with open(MNIST8M_DIR / "X.u8", "rb") as f:
        f.seek(start * _MNIST8M_ROW)
        return np.fromfile(f, dtype=np.uint8, count=(stop - start) * _MNIST8M_ROW).reshape(-1, MNIST8M_D)


def _mnist8m_complete() -> bool:
    return (MNIST8M_DIR / "COMPLETE").exists()


def _download_mnist8m(chunk_rows: int = 200_000, n_buckets: int = 64) -> None:
    """Download, convert to uint8 on disk, and shuffle once; each step is skipped if it already finished.

    Files only get their final names when a step has completed (a ``COMPLETE`` marker for the whole build), so
    an interrupted build can never be mistaken for a finished one. Everything is read and written with plain,
    sequential file I/O in bounded chunks: memory use stays small whatever the size of the data.
    """
    import io
    import shutil
    import subprocess

    from sklearn.datasets import load_svmlight_file

    if _mnist8m_complete():
        return
    MNIST8M_DIR.mkdir(parents=True, exist_ok=True)
    xz = MNIST8M_DIR / "mnist8m.xz"
    if not xz.exists() or xz.stat().st_size != MNIST8M_BYTES:
        print(f"downloading {MNIST8M_URL} (2.35 GB) ...", flush=True)
        for _ in range(30):                                  # the server drops long connections; resume
            subprocess.run(["curl", "-s", "-C", "-", "--retry", "5", "-o", str(xz), MNIST8M_URL])
            if xz.exists() and xz.stat().st_size == MNIST8M_BYTES:
                break
        else:
            raise RuntimeError("could not download mnist8m.xz completely")
    # 1. Convert: stream-decompress, parse LIBSVM text in blocks, append images and labels to files.
    raw_X, raw_y = MNIST8M_DIR / "X_unshuffled.u8", MNIST8M_DIR / "y_unshuffled.i8"
    if not (MNIST8M_DIR / "CONVERTED").exists():
        print("converting to uint8 ...", flush=True)
        proc = subprocess.Popen(["xz", "-dc", "-T0", str(xz)], stdout=subprocess.PIPE)
        rows, rest = 0, b""
        with open(raw_X, "wb") as fx, open(raw_y, "wb") as fy:
            while True:
                block = proc.stdout.read(256 << 20)
                text = rest + block
                cut = text.rfind(b"\n") + 1 if block else len(text)
                text, rest = text[:cut], text[cut:]
                if text.strip():
                    Xc, yc = load_svmlight_file(io.BytesIO(text), n_features=MNIST8M_D, zero_based=False,
                                                dtype=np.float32)
                    fx.write(np.rint(Xc.toarray()).astype(np.uint8).tobytes())
                    fy.write(yc.astype(np.int8).tobytes())
                    rows += Xc.shape[0]
                if not block:
                    break
        if proc.wait() != 0 or rows != MNIST8M_N:
            raise RuntimeError(f"conversion read {rows} rows (expected {MNIST8M_N}); delete data/mnist8m and retry")
        (MNIST8M_DIR / "CONVERTED").touch()
    # 2. Shuffle once (the file comes in generation order, cycling through the 60k base digits): the two-pass
    #    external shuffle. Pass 1 sends every row to a random bucket file; pass 2 permutes each bucket in memory
    #    (~100 MB) and appends it. Random bucket sizes and random order within buckets give a uniformly random
    #    permutation, fixed by the seed.
    print("shuffling ...", flush=True)
    rng = np.random.default_rng(MNIST8M_SHUFFLE_SEED)
    bucket_of = rng.integers(0, n_buckets, MNIST8M_N)
    tmp = MNIST8M_DIR / "buckets"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    bx = [open(tmp / f"X{b}", "wb") for b in range(n_buckets)]
    by = [open(tmp / f"y{b}", "wb") for b in range(n_buckets)]
    with open(raw_X, "rb") as fx, open(raw_y, "rb") as fy:
        for start in range(0, MNIST8M_N, chunk_rows):
            k = min(chunk_rows, MNIST8M_N - start)
            X = np.fromfile(fx, dtype=np.uint8, count=k * MNIST8M_D).reshape(k, MNIST8M_D)
            y = np.fromfile(fy, dtype=np.int8, count=k)
            b = bucket_of[start:start + k]
            for j in range(n_buckets):
                sel = b == j
                bx[j].write(X[sel].tobytes())
                by[j].write(y[sel].tobytes())
    for f in bx + by:
        f.close()
    with open(MNIST8M_DIR / "X.u8.tmp", "wb") as fx, open(MNIST8M_DIR / "y.i8.tmp", "wb") as fy:
        for j in range(n_buckets):
            X = np.fromfile(tmp / f"X{j}", dtype=np.uint8).reshape(-1, MNIST8M_D)
            y = np.fromfile(tmp / f"y{j}", dtype=np.int8)
            perm = rng.permutation(len(y))
            fx.write(X[perm].tobytes())
            fy.write(y[perm].tobytes())
    shutil.rmtree(tmp)
    (MNIST8M_DIR / "X.u8.tmp").replace(MNIST8M_DIR / "X.u8")
    (MNIST8M_DIR / "y.i8.tmp").replace(MNIST8M_DIR / "y.i8")
    (MNIST8M_DIR / "COMPLETE").touch()
    raw_X.unlink(), raw_y.unlink(), (MNIST8M_DIR / "CONVERTED").unlink()
    _verify_mnist8m()


def _sha256_file(path: Path, block: int = 64 << 20) -> str:
    """SHA-256 of a file read in blocks (plain reads, not a memory map)."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def _verify_mnist8m() -> None:
    if not _mnist8m_complete():
        print("mnist8m: not built (python experiments/common/datasets.py --download --only mnist8m)", flush=True)
        return
    got = {name: _sha256_file(MNIST8M_DIR / name) for name in ("X.u8", "y.i8")}
    if not MNIST8M_SHA256:
        print("mnist8m fingerprints (record them in MNIST8M_SHA256):", got, flush=True)
        return
    for name, expected in MNIST8M_SHA256.items():
        if got[name] != expected:
            raise RuntimeError(f"data/mnist8m/{name} differs from the data the results were made with "
                               f"(sha256 {got[name][:12]}..., expected {expected[:12]}...); delete it and rebuild")


def _mnist8m(split: str, n: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """n consecutive rows of the shuffled file (a random sample), window ``seed``; in memory, so keep n small."""
    download("mnist8m")
    n = MNIST8M_N if n is None else n
    start = (seed * n) % MNIST8M_N
    if start + n <= MNIST8M_N:
        X = mnist8m_rows(start, start + n)
    else:
        X = np.concatenate([mnist8m_rows(start, MNIST8M_N), mnist8m_rows(0, start + n - MNIST8M_N)])
    _, y = _mnist8m_arrays()
    rows = np.arange(start, start + n) % MNIST8M_N
    return X.astype(np.float64) / 255.0, y[rows].astype(np.int64)


def stream(dataset: str, *, n: int | None = None, chunk_size: int = 10_000):
    """A chunk source (``lvm.stream.ChunkSource``) over the first n points, read from disk chunk by chunk with
    plain file reads, so only the current chunk is in memory. Chunks are float32 in [0, 1], the precision every
    method computes in (the other methods get float32 too), so no chunk is converted twice.

    Only datasets stored on disk stream this way (``mnist8m``); for the others use
    ``lvm.stream.array_source(load(...))``.
    """
    if dataset != "mnist8m":
        raise ValueError(f"{dataset} is held in memory; use lvm.stream.array_source(load(...))")
    download("mnist8m")
    n = MNIST8M_N if n is None else n

    def chunks():
        with open(MNIST8M_DIR / "X.u8", "rb") as f:
            for start in range(0, n, chunk_size):
                k = min(chunk_size, n - start)
                chunk = np.fromfile(f, dtype=np.uint8, count=k * MNIST8M_D).reshape(k, MNIST8M_D).astype(np.float32)
                chunk /= 255.0
                yield chunk
    return chunks


DATASETS: dict[str, Dataset] = {
    "mnist": Dataset("mnist", "MNIST", {s: b - a for s, (a, b) in MNIST_SPLITS.items()}, _mnist,
                     colourings=_digits),
    "half_sphere": Dataset("half_sphere", "half sphere", {"tune": None, "eval": None}, _half_sphere,
                           colourings=_height_and_angle,
                           reference=lambda X: X[:, :2]),
    "mnist8m": Dataset("mnist8m", "MNIST8M", {"eval": MNIST8M_N}, _mnist8m, colourings=_digits),
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch and verify the benchmark datasets (they are not in git).")
    parser.add_argument("--download", action="store_true", help="download any missing dataset, then verify it")
    parser.add_argument("--only", default=None, help="just this dataset")
    args = parser.parse_args()
    names = [args.only] if args.only else list(DATASETS)
    if args.download:
        for name in names:
            download(name)
    for name in names:
        verify(name)
    print("datasets ok:", ", ".join(DATASETS))
