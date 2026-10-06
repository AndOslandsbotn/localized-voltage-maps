from __future__ import annotations

import math

import torch

from lvm_new.compute import BLOCK_BYTES, TORCH_DATA_DTYPE
from lvm_new.config import Config
from lvm_new.config.config import ReachOptions


def voltage_maps(kernel: torch.Tensor, masses: torch.Tensor, rho_g: float, sources: torch.Tensor) -> torch.Tensor:
    """The map with source s minimises (1/2) sum_ij W_ij (v_i - v_j)^2 + rho_g sum_i p_i v_i^2 with v_s = 1, where
    W = p p^T * K. So L v = c e_s with the grounded Laplacian L = diag(W 1 + rho_g p) - W, and v = G e_s / G_ss with
    G = L^-1: one Cholesky factorisation.
    """
    weights = masses[:, None] * masses[None, :] * kernel
    laplacian = torch.diag(weights.sum(dim=1) + rho_g * masses) - weights
    columns = torch.eye(len(masses), dtype=laplacian.dtype, device=laplacian.device)[:, sources]
    G = torch.cholesky_solve(columns, torch.linalg.cholesky(laplacian))         # (n_cells, len(sources))
    rows = torch.arange(len(sources), device=G.device)
    maps = (G / G[sources, rows]).T.contiguous()
    maps[rows, sources] = 1.0                                                  # exactly, not up to rounding
    return maps


def threshold(maps: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.where(maps >= tau, maps, torch.zeros_like(maps))


def choose_landmarks(kernel: torch.Tensor, masses: torch.Tensor, dimension: float, *,
                     config: Config) -> tuple[torch.Tensor, float]:
    """The landmarks, and the largest rho_g (the most local maps) at which they still reach the data: a ``share`` of
    the data (by mass) is reached (v >= tau) by at least k of them. The landmarks are chosen at a start rho_g where
    the median candidate map reaches min(k / L, share) of the data, what each of k evenly spread landmarks would
    need; rho_g is then bisected with the landmarks fixed."""
    reach, tau = config.landmarks.reach, config.voltage.threshold
    n_landmarks = _landmark_count(dimension, len(masses), config)
    k = n_landmarks if reach.k is None else min(reach.k, n_landmarks)
    cells = torch.arange(len(masses), device=kernel.device)

    def coverage(rho_g: float) -> float:
        reached = voltage_maps(kernel, masses, rho_g, cells) >= tau
        return float(torch.median(reached.to(masses.dtype) @ masses))

    start = _largest_passing(coverage, min(k / n_landmarks, reach.share), options=reach)
    maps = threshold(voltage_maps(kernel, masses, start, cells), tau)
    landmarks, _ = _mutual_information(maps, masses, n_landmarks, config.landmarks.mutual_information.noise_std)
    rho_g = _largest_passing(lambda r: _reached(kernel, masses, r, landmarks, tau, reach.share), k, options=reach,
                             guess=start)
    return landmarks, rho_g


def _reached(kernel: torch.Tensor, masses: torch.Tensor, rho_g: float, landmarks: torch.Tensor, tau: float,
             share: float) -> float:
    """The largest number of landmarks that reaches (v >= tau) a ``share`` of the data (by mass): the mass-weighted
    (1 - share)-quantile over cells of the number of landmarks reaching them. Never increases with rho_g."""
    counts = (voltage_maps(kernel, masses, rho_g, landmarks) >= tau).sum(dim=0)
    order = torch.argsort(counts)
    cumulative = torch.cumsum(masses[order], dim=0) / masses.sum()
    index = int(torch.searchsorted(cumulative, cumulative.new_tensor([1.0 - share]))[0])
    return float(counts[order][min(index, len(order) - 1)])


def _largest_passing(value, target: float, *, options: ReachOptions, guess: float | None = None) -> float:
    """The largest rho_g within the bounds with value(rho_g) >= target, for a value that never increases with rho_g:
    bisection on log(rho_g) until the bracket's ratio is within 1 + rel_tolerance. The bracket is the whole range, or
    grows from ``guess`` by factors of 4."""
    low_bound, high_bound = options.rho_g_bounds
    seen = {}

    def passes(rho_g: float) -> bool:
        if rho_g not in seen:
            seen[rho_g] = value(rho_g) >= target
        return seen[rho_g]

    if guess is None:
        low, high = low_bound, high_bound
        if passes(high):
            return high
        if not passes(low):
            return low
    else:
        low = high = min(max(guess, low_bound), high_bound)
        if passes(low):
            while passes(high):
                if high == high_bound:
                    return high
                low, high = high, min(4.0 * high, high_bound)
        else:
            while not passes(low):
                if low == low_bound:
                    return low
                high, low = low, max(low / 4.0, low_bound)
    while high / low > 1.0 + options.rel_tolerance:
        middle = math.sqrt(low * high)
        if passes(middle):
            low = middle
        else:
            high = middle
    return low


def _landmark_count(dimension: float, n_cells: int, config: Config) -> int:
    count = config.landmarks.count
    match count.strategy:
        case "dimension":
            n = math.ceil(count.dimension.multiplier * (dimension + 1))
        case "fixed":
            n = count.fixed.n
    return min(max(n, config.embedding.landmark_mds.n_components + 1), n_cells)


def _mutual_information(maps: torch.Tensor, masses: torch.Tensor, n_landmarks: int,
                        noise_std: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy maximisation of the mutual information between a cell J and its noisy voltages Y = v(J) + N(0, s^2 I),
    estimated by I = -sum_i p_i log sum_j p_j E_ij with E_ij = exp(-|v(i) - v(j)|^2 / (2 s^2)) (Kolchinsky & Tracey,
    2017, "Estimating mixture entropy with pairwise distances").

    Adding the map x multiplies E_ij by F_ij = exp(-(x_i - x_j)^2 / (2 s^2)), which equals F_i0 = exp(-x_i^2 / (2 s^2))
    unless j is in x's support U. So with S_i = sum_j p_j E_ij,
        S'_i = F_i0 (S_i - sum_{j in U} p_j E_ij) + sum_{j in U} p_j E_ij F_ij,
    which costs n |U| per candidate instead of n^2. Returns (the chosen rows of maps, I after each addition).
    """
    x_all = maps.to(TORCH_DATA_DTYPE)
    w = masses.to(TORCH_DATA_DTYPE)
    n_candidates, n = x_all.shape
    scale = 1.0 / (2.0 * noise_std ** 2)
    blocks = []
    for rows, width in _support_blocks(x_all > 0, x_all.element_size()):
        in_support = x_all[rows] > 0
        support = torch.argsort(in_support.to(torch.int8), dim=1, descending=True, stable=True)[:, :width]
        padding = torch.arange(width, device=maps.device)[None, :] >= in_support.sum(dim=1)[:, None]
        blocks.append((rows, support, w[support].masked_fill(padding, 0.0)))

    E = torch.ones((n, n), dtype=w.dtype, device=maps.device)
    S = E @ w
    available = torch.ones(n_candidates, dtype=torch.bool, device=maps.device)
    chosen, information = [], []
    for _ in range(min(n_landmarks, n_candidates)):
        scores = torch.empty(n_candidates, dtype=w.dtype, device=maps.device)
        for rows, support, w_support in blocks:
            x = x_all[rows]                                                         # (b, n)
            wE = E[:, support].permute(1, 0, 2) * w_support[:, None, :]             # (b, n, |U|)
            F = torch.exp(-(x[:, :, None] - torch.gather(x, 1, support)[:, None, :]).square() * scale)
            S_new = torch.exp(-x.square() * scale) * (S - wE.sum(dim=2)) + (wE * F).sum(dim=2)
            scores[rows] = -(w * torch.log(torch.maximum(S_new, w))).sum(dim=1)     # S'_i >= p_i: guards rounding
        scores[~available] = -torch.inf
        best = int(torch.argmax(scores))
        chosen.append(best)
        information.append(scores[best])
        available[best] = False
        x = x_all[best]
        E *= torch.exp(-(x[:, None] - x[None, :]).square() * scale)
        S = E @ w                                                                   # exact again, no drift
    return torch.tensor(chosen, device=maps.device), torch.stack(information)


def _support_blocks(in_support: torch.Tensor, itemsize: int) -> list[tuple[torch.Tensor, int]]:
    # Candidates sorted by support size, in blocks whose (rows, n, widest support) temporaries fit BLOCK_BYTES.
    n = in_support.shape[1]
    order = torch.argsort(in_support.sum(dim=1))
    sizes = in_support.sum(dim=1)[order].clamp(min=1).tolist()
    blocks, start = [], 0
    while start < len(order):
        end = start + 1
        while end < len(order) and (end + 1 - start) * n * sizes[end] * itemsize <= BLOCK_BYTES:
            end += 1
        blocks.append((order[start:end], sizes[end - 1]))
        start = end
    return blocks

