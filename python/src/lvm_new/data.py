from __future__ import annotations

import collections.abc
from typing import Iterable, Iterator, TypeAlias

import numpy as np
import torch
from numpy.typing import ArrayLike

from lvm_new.compute import DATA_DTYPE, TORCH_DATA_DTYPE
from lvm_new.config import SampleConfig

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


def sample_region(chunks, *, config: SampleConfig, device: str, seed: int) -> torch.Tensor:
    """(size, n_features) sample of the region's points on ``device``, by the strategy in ``config`` (all points if
    fewer)."""
    match config.strategy:
        case "prefix":
            return _prefix(chunks, config.size, device=device)
        case "reservoir":
            return _reservoir(chunks, config.size, device=device, seed=seed)


def _prefix(chunks, size: int, *, device: str) -> torch.Tensor:
    """The first ``size`` points; stops reading once full. The data must be in random order."""
    buffer, filled = None, 0
    for chunk in chunks:
        buffer = _buffer(buffer, size, chunk, device)
        take = min(size - filled, len(chunk))
        buffer[filled:filled + take] = torch.as_tensor(chunk[:take], device=device)
        filled += take
        if filled == size:
            break
    return _result(buffer, filled)


def _reservoir(chunks, size: int, *, device: str, seed: int) -> torch.Tensor:
    """A uniform sample from one full pass (reservoir sampling), for data in any order."""
    rng = np.random.default_rng(seed)
    buffer, filled, seen = None, 0, 0
    for chunk in chunks:
        buffer = _buffer(buffer, size, chunk, device)
        take = min(size - filled, len(chunk))
        buffer[filled:filled + take] = torch.as_tensor(chunk[:take], device=device)
        filled += take
        rest = chunk[take:]
        if len(rest):
            # The t-th point seen (1-based) replaces a random slot with probability size / t.
            t = seen + take + 1 + np.arange(len(rest))
            slot = rng.integers(0, t)
            keep = slot < size
            slot, rest = slot[keep], rest[keep]
            # Two points of one chunk can draw the same slot; read one at a time, the later one would win.
            _, last_from_end = np.unique(slot[::-1], return_index=True)
            last = len(slot) - 1 - last_from_end
            buffer[torch.as_tensor(slot[last], device=device)] = torch.as_tensor(rest[last], device=device)
        seen += len(chunk)
    return _result(buffer, filled)


def _buffer(buffer: torch.Tensor | None, size: int, chunk: np.ndarray, device: str) -> torch.Tensor:
    if buffer is None:
        return torch.empty((size, chunk.shape[1]), dtype=TORCH_DATA_DTYPE, device=device)
    return buffer


def _result(buffer: torch.Tensor | None, filled: int) -> torch.Tensor:
    if buffer is None:
        raise ValueError("X has no points")
    return buffer if filled == len(buffer) else buffer[:filled].clone()
