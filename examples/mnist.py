"""MNIST via direct download (IDX). Demo / example data only — not part of the lvm API."""

from __future__ import annotations

import gzip
import os
import struct
import urllib.request
from pathlib import Path

import numpy as np

_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
_FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)


def _default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME", "")
    if base:
        return Path(base) / "localized-voltage-maps" / "mnist"
    return Path.home() / ".cache" / "localized-voltage-maps" / "mnist"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as f:
        f.write(resp.read())
    tmp.replace(dest)


def _ensure_file(name: str, cache_dir: Path) -> Path:
    dest = cache_dir / name
    if not dest.is_file():
        _download(_BASE + name, dest)
    return dest


def _read_images(gz_path: Path) -> np.ndarray:
    with gzip.open(gz_path, "rb") as f:
        _magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        buf = f.read()
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows * cols)


def _read_labels(gz_path: Path) -> np.ndarray:
    with gzip.open(gz_path, "rb") as f:
        _magic, n = struct.unpack(">II", f.read(8))
        buf = f.read()
    return np.frombuffer(buf, dtype=np.uint8)


def load_mnist(
    *,
    cache_dir: Path | None = None,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    root = cache_dir or _default_cache_dir()
    paths = [_ensure_file(name, root) for name in _FILES]
    X_train = _read_images(paths[0])
    y_train = _read_labels(paths[1])
    X_test = _read_images(paths[2])
    y_test = _read_labels(paths[3])
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    if normalize:
        X = X.astype(np.float64) / 255.0
    else:
        X = X.astype(np.float64)
    return X, y
