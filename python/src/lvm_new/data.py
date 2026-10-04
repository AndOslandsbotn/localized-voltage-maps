from __future__ import annotations

import collections.abc
from typing import Iterable, Iterator, TypeAlias

import numpy as np
from numpy.typing import ArrayLike

from lvm_new.precision import DATA_DTYPE

DataLike: TypeAlias = ArrayLike | Iterable[ArrayLike]
"""Aanything with a 2-D shape that can be sliced: NumPy,
memory-mapped, HDF5, Zarr, torch, CuPy), or a re-iterable
of chunks (each chunk an array of points)."""


class ArrayChunks:
    """An array as a re-iterable source: each pass yields slices of ``chunk_size`` rows."""

    def __init__(self, X: ArrayLike, chunk_size: int):
        self.X = X
        self.chunk_size = chunk_size

    def __iter__(self) -> Iterator[np.ndarray]:
        for start in range(0, self.X.shape[0], self.chunk_size):
            yield np.asarray(self.X[start:start + self.chunk_size], dtype=DATA_DTYPE)


class IterableChunks:
    """A re-iterable of chunks, passed on as they come: the loader's batch size is the chunk size (re-cutting would
    copy every chunk). A tuple or list piece gives its first element (a DataLoader yields (features, labels))."""

    def __init__(self, source: Iterable[ArrayLike]):
        self.source = source

    def __iter__(self) -> Iterator[np.ndarray]:
        for piece in self.source:
            if isinstance(piece, (tuple, list)):
                piece = piece[0]
            if not hasattr(piece, "shape"):
                raise TypeError(f"Expected arrays or (features, ...) tuples, got {type(piece).__name__}")
            piece = np.asarray(piece, dtype=DATA_DTYPE)
            if piece.ndim != 2:
                raise ValueError(f"Expected 2-D chunks (points x features), got shape {piece.shape}")
            yield piece


def as_chunks(X: DataLike, chunk_size: int) -> ArrayChunks | IterableChunks:
    if isinstance(X, collections.abc.Iterator):
        raise TypeError(
            f"X is a one-shot {type(X).__name__} (an iterator or generator). Pass an array, or a re-iterable"
        )
    if hasattr(X, "shape"):
        if len(X.shape) != 2:
            raise ValueError(
                f"Expected a 2-D array (points x features), got shape {tuple(X.shape)}"
            )
        return ArrayChunks(X, chunk_size)
    if isinstance(X, collections.abc.Iterable):
        return IterableChunks(X)
    raise TypeError(f"Expected an array or a re-iterable, got {type(X).__name__}")

