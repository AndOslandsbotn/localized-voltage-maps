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

Configurable choice point ``landmarks``; see ``lvm.strategies`` for the
convention shared by every choice point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from lvm.config import LandmarksConfig, MutualInformationConfig
from lvm.strategies import resolve



@dataclass(frozen=True)
class LandmarkSelection:
    indices: np.ndarray   # rows of V chosen, in the order they were added
    mi: np.ndarray        # estimated I(J; Y) in nats after each addition


def choose_landmarks(V: np.ndarray, p: np.ndarray, *, config: LandmarksConfig, device: str = "cuda") -> LandmarkSelection:
    """Pick ``config.n_landmarks`` rows of ``V`` with the strategy in ``config.strategy``."""
    strategy, options = resolve(_STRATEGIES, config)
    return strategy(V, p, options=options, n_landmarks=config.n_landmarks, device=device)


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
    V: np.ndarray, p: np.ndarray, *, options: MutualInformationConfig, n_landmarks: int, device: str
) -> LandmarkSelection:
    """Greedy forward selection maximizing I_hat, see the module docstring.

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
    S = E @ w
    chosen: list[int] = []
    mis: list[float] = []
    available = torch.ones(n_cand, dtype=torch.bool, device=device)
    for _ in range(min(n_landmarks, n_cand)):
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
    return LandmarkSelection(indices=np.array(chosen, dtype=int), mi=np.array(mis))


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


_STRATEGIES: dict[str, Callable[..., LandmarkSelection]] = {
    "mutual_information": _mutual_information,
}
