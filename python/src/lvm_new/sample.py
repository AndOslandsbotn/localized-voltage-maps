from __future__ import annotations

import numpy as np

from lvm_new.config import SampleConfig
from lvm_new.precision import DATA_DTYPE


def sample_region(chunks, *, config: SampleConfig, seed: int) -> np.ndarray:
    """(size, n_features) sample of the region's points, by the strategy in ``config`` (all points if fewer)."""
    return _STRATEGIES[config.strategy](chunks, options=config.options, size=config.size, seed=seed)


def _prefix(chunks, *, size: int, seed: int, options: None) -> np.ndarray:
    """The first ``size`` points; stops reading once full. The data must be in random order."""
    buffer, filled = None, 0
    for chunk in chunks:
        buffer = _buffer(buffer, size, chunk)
        take = min(size - filled, len(chunk))
        buffer[filled:filled + take] = chunk[:take]
        filled += take
        if filled == size:
            break
    return _result(buffer, filled)


def _reservoir(chunks, *, size: int, seed: int, options: None) -> np.ndarray:
    """A uniform sample from one full pass (reservoir sampling), for data in any order."""
    rng = np.random.default_rng(seed)
    buffer, filled, seen = None, 0, 0
    for chunk in chunks:
        buffer = _buffer(buffer, size, chunk)
        take = min(size - filled, len(chunk))
        buffer[filled:filled + take] = chunk[:take]
        filled += take
        rest = chunk[take:]
        if len(rest):
            t = seen + take + 1 + np.arange(len(rest))
            slot = rng.integers(0, t)
            keep = slot < size
            slot, rest = slot[keep], rest[keep]
            _, last_from_end = np.unique(slot[::-1], return_index=True)
            last = len(slot) - 1 - last_from_end
            buffer[slot[last]] = rest[last]
        seen += len(chunk)
    return _result(buffer, filled)


def _buffer(buffer: np.ndarray | None, size: int, chunk: np.ndarray) -> np.ndarray:
    return np.empty((size, chunk.shape[1]), dtype=DATA_DTYPE) if buffer is None else buffer


def _result(buffer: np.ndarray | None, filled: int) -> np.ndarray:
    if buffer is None:
        raise ValueError("X has no points")
    return buffer if filled == len(buffer) else buffer[:filled].copy()


_STRATEGIES = {
    "prefix": _prefix,
    "reservoir": _reservoir,
}
