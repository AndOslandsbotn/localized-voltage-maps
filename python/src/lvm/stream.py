"""Chunked data sources.

Every level of the hierarchy rescans the dataset, so a data source is a
*factory*: calling it starts a fresh pass and returns an iterator over
(chunk_size, d) float64 arrays.
"""

from __future__ import annotations

from itertools import islice
from typing import Callable, Iterator

import numpy as np

from .config import DataConfig

ChunkSource = Callable[[], Iterator[np.ndarray]]


def iter_csv_chunks(path, chunk_size: int, *, delimiter: str = ",", skip_header: int = 0) -> Iterator[np.ndarray]:
    """Read a numeric CSV file ``chunk_size`` rows at a time."""
    with open(path) as f:
        for _ in range(skip_header):
            next(f, None)
        while lines := list(islice(f, chunk_size)):
            yield np.loadtxt(lines, delimiter=delimiter, ndmin=2, dtype=np.float64)


def iter_array_chunks(X: np.ndarray, chunk_size: int) -> Iterator[np.ndarray]:
    """Stream an in-memory array in chunks (small datasets and tests)."""
    X = np.asarray(X, dtype=np.float64)
    for start in range(0, X.shape[0], chunk_size):
        yield X[start : start + chunk_size]


def array_source(X: np.ndarray, chunk_size: int) -> ChunkSource:
    return lambda: iter_array_chunks(X, chunk_size)


def make_source(cfg: DataConfig) -> ChunkSource:
    """Build the chunk source described by the ``data`` config section."""
    if cfg.path is None:
        raise ValueError("data.path is not set")
    if cfg.format == "csv":
        return lambda: iter_csv_chunks(
            cfg.path, cfg.chunk_size, delimiter=cfg.csv.delimiter, skip_header=cfg.csv.skip_header
        )
    raise ValueError(f"unsupported data format {cfg.format!r}")
