"""Region tree and one-scan-per-level routing of the data stream.

A region is a node in the hierarchy. The root is the whole dataset. Once a
region's Voronoi cells are built it gets ``centroids``, and each child is a
subset of those cells (``parent_cells``). A raw point belongs to a child if it
belongs to the parent and its nearest parent centroid is one of the child's
cells. Children may overlap (partition strategy ``support``), in which case a
point is routed to several of them.

The regions that are currently being built are the leaves of the tree (the
*frontier*). ``route`` sends each chunk of the stream down the tree to the
frontier, so a single scan of the data serves every region on a level.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from .cells import assign_cells
from .stream import ChunkSource


@dataclass
class Region:
    id: tuple[int, ...] = ()                   # path from the root; the root is ()
    parent_cells: np.ndarray | None = None     # parent's cell indices that form this region (None: root)
    centroids: np.ndarray | None = None        # this region's cell centroids, once built
    children: list[Region] = field(default_factory=list)

    @property
    def depth(self) -> int:
        return len(self.id)

    def add_child(self, parent_cells) -> Region:
        if self.centroids is None:
            raise ValueError(f"region {self.id} has no cells yet, so it cannot have children")
        child = Region(id=self.id + (len(self.children),), parent_cells=np.asarray(parent_cells, dtype=int))
        self.children.append(child)
        return child

    def leaves(self) -> list[Region]:
        if not self.children:
            return [self]
        return [leaf for child in self.children for leaf in child.leaves()]


def route(chunk: np.ndarray, region: Region) -> Iterator[tuple[Region, np.ndarray]]:
    """Yield ``(leaf, rows)`` for every frontier region that receives rows of ``chunk``.

    ``rows`` are indices into ``chunk``. Points whose cell belongs to no child are
    dropped (step 9's ``uncovered`` strategy makes sure that doesn't happen).
    """
    yield from _route(chunk, region, np.arange(chunk.shape[0]))


def _route(chunk: np.ndarray, region: Region, rows: np.ndarray) -> Iterator[tuple[Region, np.ndarray]]:
    if rows.size == 0:
        return
    if not region.children:
        yield region, rows
        return
    cell = assign_cells(chunk[rows], region.centroids)
    for child in region.children:
        yield from _route(chunk, child, rows[np.isin(cell, child.parent_cells)])


@dataclass
class RegionSample:
    points: np.ndarray    # (m, d) sampled points, m <= sample_size
    n_seen: int           # points of this region read during the scan


class _Reservoir:
    """Uniform sample of fixed size from a stream (Algorithm R, vectorized per chunk).

    With ``prefix=True`` it just keeps the first ``size`` points, which is a
    uniform sample when the input is pre-shuffled. Points are stored as float32
    (what k-means and the local charts use anyway): the sample is the one part of
    a level that is held in memory, so it should cost no more than needed.
    """

    def __init__(self, size: int, rng: np.random.Generator, prefix: bool):
        self.size = size
        self.rng = rng
        self.prefix = prefix
        self.buf: np.ndarray | None = None
        self.filled = 0
        self.n_seen = 0

    @property
    def full(self) -> bool:
        return self.filled == self.size

    def add(self, X: np.ndarray) -> None:
        if self.buf is None:
            self.buf = np.empty((self.size, X.shape[1]), dtype=np.float32)
        n_fill = min(self.size - self.filled, X.shape[0])
        self.buf[self.filled : self.filled + n_fill] = X[:n_fill]
        self.filled += n_fill
        rest = X[n_fill:]
        if rest.shape[0] and not self.prefix:
            # Point t (1-based position in this region's stream) replaces a random
            # slot with probability size/t.
            t = self.n_seen + n_fill + 1 + np.arange(rest.shape[0])
            slot = self.rng.integers(0, t)
            keep = slot < self.size
            slot, rest = slot[keep], rest[keep]
            # Sequentially, a later point overwrites an earlier one in the same
            # slot; numpy doesn't guarantee that order, so keep only the last.
            _, first_from_end = np.unique(slot[::-1], return_index=True)
            last = slot.size - 1 - first_from_end
            self.buf[slot[last]] = rest[last]
        self.n_seen += X.shape[0]

    def result(self) -> RegionSample:
        """The sample; the buffer itself when it is full (no second copy), a trimmed copy otherwise."""
        if self.buf is None:
            return RegionSample(points=np.empty((0, 0), dtype=np.float32), n_seen=self.n_seen)
        points = self.buf if self.filled == self.size else self.buf[: self.filled].copy()
        return RegionSample(points=points, n_seen=self.n_seen)


def sample_regions(
    source: ChunkSource,
    root: Region,
    sample_size: int,
    *,
    shuffled: bool,
    seed: int | None = None,
) -> dict[tuple[int, ...], RegionSample]:
    """One scan of the data: a uniform sample of up to ``sample_size`` points per frontier region.

    With ``shuffled=True`` each region keeps its first points and the scan stops
    as soon as every region is full, so ``n_seen`` is then only the count up to
    that point. With ``shuffled=False`` the whole dataset is read.
    """
    rng = np.random.default_rng(seed)
    leaves = root.leaves()
    reservoirs = {leaf.id: _Reservoir(sample_size, rng, prefix=shuffled) for leaf in leaves}
    for chunk in source():
        for leaf, rows in route(chunk, root):
            reservoirs[leaf.id].add(chunk[rows])
        if shuffled and all(r.full for r in reservoirs.values()):
            break
    return {rid: r.result() for rid, r in reservoirs.items()}


@dataclass
class CellMasses:
    counts: np.ndarray    # (n_cells,) points of the region that fell in each cell
    n_seen: int           # points of the region read during the scan (= counts.sum())
    converged: bool       # every cell reached min_count

    @property
    def p(self) -> np.ndarray:
        """Estimated cell masses, relative to the region (sums to 1)."""
        return self.counts / self.n_seen if self.n_seen else np.zeros_like(self.counts, dtype=np.float64)


def min_count_for(rel_error: float) -> int:
    """Points a cell needs for a typical relative mass error of ``rel_error``.

    A cell's count is Binomial(n_p, p_i), so std(p_hat_i)/p_i ~= 1/sqrt(count).
    """
    return math.ceil(1.0 / rel_error**2)


def estimate_masses(
    source: ChunkSource,
    root: Region,
    *,
    min_count: int,
    shuffled: bool,
    max_points: int | None = None,
    device: str | None = None,
) -> dict[tuple[int, ...], CellMasses]:
    """One scan of the data: count points per cell for every frontier region.

    Every frontier region must already have ``centroids``. With ``shuffled=True``
    the scan stops once every cell of every region has ``min_count`` points;
    otherwise it reads the whole dataset (a prefix of unshuffled data would give
    biased masses). ``max_points`` caps the points read in either case.
    ``device`` is where the nearest-cell search runs (``cells.assign_cells``).
    """
    leaves = root.leaves()
    for leaf in leaves:
        if leaf.centroids is None:
            raise ValueError(f"region {leaf.id} has no cells yet")
    counts = {leaf.id: np.zeros(leaf.centroids.shape[0], dtype=np.int64) for leaf in leaves}
    n_read = 0
    for chunk in source():
        for leaf, rows in route(chunk, root):
            cell = assign_cells(chunk[rows], leaf.centroids, device=device)
            counts[leaf.id] += np.bincount(cell, minlength=counts[leaf.id].size)
        n_read += chunk.shape[0]
        if shuffled and all(c.min() >= min_count for c in counts.values()):
            break
        if max_points is not None and n_read >= max_points:
            break
    return {
        rid: CellMasses(counts=c, n_seen=int(c.sum()), converged=bool(c.min() >= min_count))
        for rid, c in counts.items()
    }
