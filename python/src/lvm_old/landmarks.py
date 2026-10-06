"""Select a region's landmarks from its candidate voltage maps (config ``landmarks``).

Step (5) of Fig. 7 in the paper: greedily add the candidate landmark whose
voltage map most increases the mutual information between a cell's identity
J and the noisy voltages it sees, Y = v(J) + eps with eps ~ N(0, sigma^2 I).

Y is a Gaussian mixture, whose entropy has no closed form. We use the
pairwise-distance estimator of Kolchinsky & Tracey (2017, "Estimating mixture
entropy with pairwise distances"), with KL divergence between the
equal-covariance components:

    I_hat = -sum_i p_i log sum_j p_j exp(-||v(i) - v(j)||^2 / (2 sigma^2))

with cells weighted by their mass p. It is 0 when every cell has the same
voltages and H(p) when all cells are separated by >> sigma, i.e. when the
voltages determine the cell. It needs only squared distances between cells'
voltage vectors, and adding a landmark adds one term to each, so a greedy
step scores all candidates at once.

A second strategy, ``maxmin``, picks landmarks for spread instead: each new
landmark is the candidate farthest (in voltage distance) from all landmarks
chosen so far. That is what triangulation needs (``embedding: landmark_mds``):
landmarks surrounding the data, in general position.

``combined`` uses both: the first ``fraction`` of the landmarks by one, the
rest by the other. Both are greedy, so the second simply continues from the
landmarks the first chose (``start``): maxmin measures distance to all of
them, mutual information scores what a candidate adds to all of them.

How many landmarks is its own choice point, ``landmarks.count``: a fixed
number, or ceil(multiplier * (d + 1)) from the region's intrinsic dimension d,
since d + 1 landmarks pin down a position in d dimensions.

Configurable choice points ``landmarks`` and ``landmarks.count``; see
``lvm.strategies`` for the convention shared by every choice point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import math

import numpy as np
import torch

from lvm_old.config import DimensionMultiplierConfig, LandmarkCountConfig, LandmarksConfig, MutualInformationConfig
from lvm_old.strategies import resolve
from lvm_old.voltage import chained_distances, voltage_distances


@dataclass(frozen=True)
class LandmarkSelection:
    indices: np.ndarray   # rows of V chosen, in the order they were added
    scores: np.ndarray    # after each addition: estimated I(J; Y) in nats (mutual_information),
                          # or the new landmark's distance to the nearest earlier one (maxmin)


# --- how many landmarks (config landmarks.count) ---------------------------

def landmark_count(d: float, *, config: LandmarksConfig) -> int:
    """Number of landmarks for a region of intrinsic dimension ``d`` (strategy ``config.count.strategy``)."""
    strategy, options = resolve(_COUNT_STRATEGIES, config.count)
    return strategy(d, options=options, n_landmarks=config.n_landmarks)


def _fixed_count(d: float, *, options: None, n_landmarks: int) -> int:
    return n_landmarks


def _dimension_count(d: float, *, options: DimensionMultiplierConfig, n_landmarks: int) -> int:
    """ceil(multiplier * (d + 1)): d + 1 landmarks triangulate a position in d dimensions."""
    return max(1, int(np.ceil(options.multiplier * (d + 1))))


_COUNT_STRATEGIES: dict[str, Callable[..., int]] = {
    "fixed": _fixed_count,
    "dimension": _dimension_count,
}


# --- which landmarks (config landmarks.strategy) ---------------------------

def choose_landmarks(
    V: np.ndarray, p: np.ndarray, *, config: LandmarksConfig, n_landmarks: int | None = None,
    cells: np.ndarray | None = None, tau: float = 1e-3, device: str = "cuda",
) -> LandmarkSelection:
    """Pick ``n_landmarks`` rows of ``V`` (candidates x cells) with the strategy in ``config.strategy``.

    ``n_landmarks`` defaults to ``config.n_landmarks`` (see ``landmark_count``).
    ``cells[i]`` is the cell candidate i sits at (default: candidate i is
    cell i); ``tau`` is the voltage threshold below which a map has no reach.
    """
    n = config.n_landmarks if n_landmarks is None else n_landmarks
    cells = np.arange(np.asarray(V).shape[0]) if cells is None else np.asarray(cells)

    def pick(name: str, count: int, start: LandmarkSelection | None) -> LandmarkSelection:
        return _STRATEGIES[name](V, p, options=getattr(config, name, None), n_landmarks=count, cells=cells,
                                 tau=tau, device=device, start=start)

    if config.strategy != "combined":
        return pick(config.strategy, n, None)
    c = config.combined
    first = pick(c.first, min(n, math.ceil(c.fraction * n)), None)
    return pick(c.second, n, first)


def mutual_information(V: np.ndarray, p: np.ndarray, noise_std: float) -> float:
    """I_hat(J; Y) in nats for voltage maps ``V`` (landmarks x cells) and cell masses ``p``.

    Direct O(n^2) evaluation; the greedy selection computes the same value
    incrementally.
    """
    X = torch.as_tensor(np.asarray(V, dtype=np.float64).T)  # (cells, landmarks)
    w = torch.as_tensor(np.asarray(p, dtype=np.float64))
    keep = w > 0
    X, w = X[keep], w[keep]
    D2 = torch.cdist(X, X).square()
    log_mix = torch.logsumexp(torch.log(w) - D2 / (2.0 * noise_std**2), dim=1)
    return float(-(log_mix * w).sum())


def _mutual_information(
    V: np.ndarray, p: np.ndarray, *, options: MutualInformationConfig, n_landmarks: int, cells: np.ndarray,
    tau: float, device: str, start: LandmarkSelection | None = None,
) -> LandmarkSelection:
    """Greedy forward selection maximizing I_hat, see the module docstring.

    ``start``: landmarks already chosen; selection continues from them up to
    ``n_landmarks`` in all (they come first in the result).

    Keeps E_ij = exp(-||v(i) - v(j)||^2 / (2 sigma^2)) over the landmarks
    chosen so far and each cell's mixture sum S_i = sum_j p_j E_ij. Adding a
    candidate x multiplies E_ij by F_ij = exp(-(x_i - x_j)^2 / (2 sigma^2)),
    which is exactly 1 unless i or j is in x's support U = {x > 0}. So

        S'_i = F_i0 (S_i - sum_{j in U} p_j E_ij) + sum_{j in U} p_j E_ij F_ij,

    with F_i0 = exp(-x_i^2 / (2 sigma^2)), costs n |U| per candidate
    instead of n^2. Pass thresholded maps (``voltage.threshold_voltages``) to
    get this saving; unthresholded maps give the same result, just slower.
    """
    if options.noise_std <= 0:
        raise ValueError(f"noise_std must be > 0, got {options.noise_std}")
    V = np.asarray(V, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if V.ndim != 2 or V.shape[1] != p.shape[0]:
        raise ValueError(f"V must be (candidates, cells) with cells = len(p), got {V.shape} and {p.shape}")

    # Cells without mass have weight 0 in every sum, so leave them out.
    keep = p > 0
    dtype = getattr(torch, options.precision)
    X = torch.as_tensor(V[:, keep], dtype=dtype, device=device)   # (candidates, n)
    w = torch.as_tensor(p[keep], dtype=dtype, device=device)
    n_cand, n = X.shape
    inv_2var = 1.0 / (2.0 * options.noise_std**2)

    # Chunks of candidates with similar support size, each padded (with a
    # mask) only to its own largest support, since sizes vary a lot.
    chunks = []
    for rows, s in _chunks_by_support(X > 0, n, budget=options.scratch_mb * 2**20, itemsize=X.element_size()):
        in_support = X[rows] > 0
        U = torch.argsort(in_support.to(torch.int8), dim=1, descending=True, stable=True)[:, :s]
        mask = torch.arange(s, device=device)[None, :] < in_support.sum(dim=1)[:, None]
        chunks.append((rows, U, mask))

    E = torch.ones((n, n), dtype=X.dtype, device=device)
    chosen: list[int] = [] if start is None else [int(i) for i in start.indices]
    mis: list[float] = [] if start is None else [float(s) for s in start.scores]
    available = torch.ones(n_cand, dtype=torch.bool, device=device)
    for i in chosen:
        xb = X[i]
        E *= torch.exp(-(xb[:, None] - xb[None, :]).square() * inv_2var)
        available[i] = False
    S = E @ w
    for _ in range(min(n_landmarks, n_cand) - len(chosen)):
        scores = torch.empty(n_cand, dtype=X.dtype, device=device)
        for rows, U, mask in chunks:
            x = X[rows]                                                 # (b, n)
            wU = w[U] * mask                                            # (b, s), 0 on padding
            wEU = E[:, U].permute(1, 0, 2) * wU[:, None, :]             # (b, n, s)
            F = torch.exp(-(x[:, :, None] - torch.gather(x, 1, U)[:, None, :]).square() * inv_2var)
            F0 = torch.exp(-x.square() * inv_2var)                      # (b, n)
            S_new = F0 * (S[None] - wEU.sum(dim=2)) + (wEU * F).sum(dim=2)
            # S'_i >= p_i (a cell always matches itself); guard against rounding.
            scores[rows] = -(w * torch.log(torch.maximum(S_new, w))).sum(dim=1)
        scores[~available] = -torch.inf
        best = int(torch.argmax(scores))
        chosen.append(best)
        mis.append(float(scores[best]))
        available[best] = False
        xb = X[best]
        E *= torch.exp(-(xb[:, None] - xb[None, :]).square() * inv_2var)
        S = E @ w  # recompute exactly so rounding doesn't accumulate
    return LandmarkSelection(indices=np.array(chosen, dtype=int), scores=np.array(mis))


def _chunks_by_support(
    in_support: torch.Tensor, n: int, *, budget: int, itemsize: int
) -> list[tuple[torch.Tensor, int]]:
    """Split candidates, sorted by support size, into chunks of at most ``budget`` bytes.

    Returns (candidate rows, padded support width) per chunk. One chunk-sized
    tensor holds b * n * s values, where s is the chunk's largest support.
    """
    sizes = in_support.sum(dim=1)
    order = torch.argsort(sizes)
    sizes_sorted = sizes[order].tolist()
    chunks = []
    start = 0
    while start < len(order):
        end = start + 1
        # Sorted ascending, so the chunk's width is set by its last member.
        while end < len(order) and (end + 1 - start) * n * max(1, sizes_sorted[end]) * itemsize <= budget:
            end += 1
        chunks.append((order[start:end], max(1, sizes_sorted[end - 1])))
        start = end
    return chunks


def _maxmin(
    V: np.ndarray, p: np.ndarray, *, options: None, n_landmarks: int, cells: np.ndarray, tau: float, device: str,
    start: LandmarkSelection | None = None,
) -> LandmarkSelection:
    """Farthest-point selection on voltage distance between candidates.

    The distance from candidate a to candidate b is -log of a's voltage at
    b's cell (symmetrised). Thresholded maps leave far pairs without a
    voltage, so gaps are bridged by shortest paths through other candidates
    (``voltage.chained_distances``). The first landmark is the candidate
    farthest from the mass-weighted medoid; each next one maximises its
    distance to the nearest landmark chosen so far. Candidates in a part of
    the graph no map reaches (infinite distance) are taken first, so every
    connected part gets a landmark. ``start``: landmarks already chosen;
    selection continues from them (no medoid step), up to ``n_landmarks`` in all.
    """
    V = np.asarray(V, dtype=np.float64)
    D = chained_distances(voltage_distances(V[:, cells], tau))   # (candidates, candidates), symmetric
    taken = np.zeros(V.shape[0], dtype=bool)
    if start is None:
        w = np.asarray(p, dtype=np.float64)[cells]
        finite = np.where(np.isfinite(D), D, np.nan)
        medoid = int(np.nanargmin(np.nansum(finite * w[None, :], axis=1)))
        first = int(np.argmax(D[medoid]))
        chosen, scores = [first], [float(D[medoid, first])]
    else:
        chosen, scores = [int(i) for i in start.indices], [float(s) for s in start.scores]
    taken[chosen] = True
    nearest = D[chosen].min(axis=0)               # each candidate's distance to its nearest landmark
    for _ in range(min(n_landmarks, V.shape[0]) - len(chosen)):
        best = int(np.argmax(np.where(taken, -1.0, nearest)))
        chosen.append(best)
        scores.append(float(nearest[best]))
        taken[best] = True
        nearest = np.minimum(nearest, D[best])
    return LandmarkSelection(indices=np.array(chosen, dtype=int), scores=np.array(scores))


_STRATEGIES: dict[str, Callable[..., LandmarkSelection]] = {
    "mutual_information": _mutual_information,
    "maxmin": _maxmin,
}
