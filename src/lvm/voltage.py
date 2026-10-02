"""Solve the grounded energy-minimizing voltage (EMV) system.

Given a grounded graph Laplacian ``L_rho`` (as produced by
``graph.grounded_laplacian``) and a set of source nodes S clamped to
voltage 1, this solves for the voltage at every node -- Definition 3 /
Lemma 4 in Structure_from_Voltage.pdf.

Lemma 4's fixed point v = D~^-1 W~(s) v says, at every free node i,
(L_rho @ v)[i] = 0, i.e. no current leaves the circuit at i except through
the ground. The only nonzero entries of L_rho @ v are therefore at the
sources, where current c is injected to hold them at 1:

    L_rho @ v = e_S @ c    =>    v = G[:, S] @ c,    with G = L_rho^-1,

and v[S] = 1 fixes c through the small |S| x |S| system G[S, S] @ c = 1.
L_rho is symmetric positive-definite when every node has a positive ground
weight, so G comes from one Cholesky factorization.
``solve_grounded_voltage_maps`` ensures this by leaving zero-mass cells out of
the solve.

L_rho is the same for every landmark on a graph -- only S changes -- so G is
computed once and every voltage map is read off its columns. That costs
O(n^3) once instead of O(n^3) per landmark.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import scipy.sparse as sp
import torch
from array_api_compat import array_namespace

from lvm.config import ExtensionConfig, SourcesConfig
from lvm.graph import grounded_laplacian
from lvm.strategies import resolve


@dataclass(frozen=True)
class Sources:
    sets: list[list[int]]   # cells clamped to voltage 1, one set per candidate landmark
    cells: np.ndarray       # (len(sets),) the cell each candidate is placed at


def choose_sources(p: np.ndarray, *, config: SourcesConfig) -> Sources:
    """Candidate landmarks' source sets with the strategy named in ``config.strategy``.

    Config choice point ``sources``; see ``lvm.strategies``.
    """
    strategy, options = resolve(_SOURCE_STRATEGIES, config)
    return strategy(np.asarray(p, dtype=np.float64), options=options)


def _single_node(p: np.ndarray, *, options: None) -> Sources:
    """Every cell with mass is a candidate, clamping only itself."""
    cells = np.flatnonzero(p > 0)
    return Sources(sets=[[int(i)] for i in cells], cells=cells)


_SOURCE_STRATEGIES: dict[str, Callable[..., Sources]] = {
    "single_node": _single_node,
}


def _node_indices(n: int, indices: Sequence[int]) -> np.ndarray:
    idx = np.unique(np.asarray(indices, dtype=int))
    if idx.size and (idx.min() < 0 or idx.max() >= n):
        raise ValueError(f"source indices must be in [0, {n}), got range [{idx.min()}, {idx.max()}]")
    return idx


def solve_voltage_maps(
    L_rho,
    source_sets: Sequence[Sequence[int]],
    *,
    device: str = "cuda",
) -> np.ndarray:
    """Solve the grounded EMV independently for each source set.

    Mirrors step (4) "Voltage Maps" in Fig. 6 of the paper, where every
    candidate centroid is used in turn as a landmark. All source sets share
    one inverse of L_rho (float64); memory is about n^2 * 8 bytes.

    Parameters
    ----------
    L_rho : (n, n) dense ndarray or scipy.sparse matrix
        Symmetric positive-definite grounded Laplacian, e.g. from
        ``graph.grounded_laplacian`` restricted to cells with mass.
    source_sets : sequence of sequences of int
        Each entry holds the node indices held at voltage 1 for one landmark.
    device : {"cuda", "cpu"}
        Torch device to solve on. Both run the exact same computation.

    Returns
    -------
    V : (len(source_sets), n) ndarray
        Row i is the voltage map for source_sets[i].
    """
    L_dense = L_rho.toarray() if sp.issparse(L_rho) else np.asarray(L_rho, dtype=float)
    n = L_dense.shape[0]
    L = torch.as_tensor(L_dense, dtype=torch.float64, device=device)
    G = torch.cholesky_inverse(torch.linalg.cholesky(L))

    # Group source sets by size so each group is one batched solve instead of
    # one tiny GPU call per landmark.
    groups: dict[int, list[int]] = {}
    indices = [_node_indices(n, sources) for sources in source_sets]
    for k, idx in enumerate(indices):
        groups.setdefault(idx.size, []).append(k)

    V = torch.empty((len(source_sets), n), dtype=L.dtype, device=device)
    for ks in groups.values():
        S = torch.as_tensor(np.stack([indices[k] for k in ks]), device=device)  # (B, m)
        # Currents injected at the sources so that they sit at voltage 1.
        c = torch.linalg.solve(G[S[:, :, None], S[:, None, :]], torch.ones(S.shape, dtype=L.dtype, device=device))
        rows = torch.as_tensor(ks, device=device)
        V[rows] = torch.einsum("bm,bmn->bn", c, G[S])  # G symmetric: G[S] rows == G[:, S] columns
        V[rows[:, None], S] = 1.0  # exact, instead of 1 up to rounding
    return V.cpu().numpy()


def solve_grounded_voltage_maps(
    K: np.ndarray,
    p: np.ndarray,
    rho_g: float,
    source_sets: Sequence[Sequence[int]],
    *,
    device: str = "cuda",
) -> np.ndarray:
    """Voltage maps on a region's mass-weighted grounded graph, for every cell.

    Cells with p_i > 0 are solved exactly with ``solve_voltage_maps``. A cell
    with p_i = 0 has no edges and no ground in L_rho (singular row), so it is
    left out of the solve and then given the voltage the stationarity
    condition assigns it:

        v_i = sum_j K_ij p_j v_j / (rho_g + sum_j K_ij p_j),

    the same mass-weighted average of its neighbours that every free cell
    satisfies (the discrete form of Def. 10's extension). It is 0 for a cell
    with no neighbours. So no cell is dropped and every cell gets a voltage.

    Parameters
    ----------
    K : (n, n) ndarray
        Kernel between cells, e.g. ``graph.choose_kernel(sq_distances(c, c), ..., exclude_self=True).K``.
    p : (n,) ndarray
        Cell masses (``CellMasses.p``).
    rho_g : float
        Ground scaling; node i's ground weight is ``rho_g * p[i]``.
    source_sets : sequence of sequences of int
        Cell indices held at voltage 1, one set per landmark. Sources must
        have p > 0: a massless cell can't inject current into the graph.
    device : {"cuda", "cpu"}

    Returns
    -------
    V : (len(source_sets), n) ndarray
    """
    K = np.asarray(K, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    n = p.shape[0]
    active = np.flatnonzero(p > 0)
    if active.size == 0:
        raise ValueError("every cell has zero mass")

    local = np.full(n, -1)
    local[active] = np.arange(active.size)
    local_sets = []
    for sources in source_sets:
        idx = _node_indices(n, sources)
        if np.any(local[idx] < 0):
            raise ValueError(f"source cells must have mass > 0, got zero-mass cells {idx[local[idx] < 0].tolist()}")
        local_sets.append(local[idx])

    L_rho = grounded_laplacian(K[np.ix_(active, active)], p[active], rho_g)
    V = np.zeros((len(source_sets), n))
    V[:, active] = solve_voltage_maps(L_rho, local_sets, device=device)

    empty = np.flatnonzero(p == 0)
    if empty.size:
        Kp = K[np.ix_(empty, active)] * p[active]  # (n_empty, n_active)
        V[:, empty] = (V[:, active] @ Kp.T) / (rho_g + Kp.sum(axis=1))
    return V


def support_mask(V, tau: float):
    """Cells where each voltage map is at least ``tau`` (config ``voltage.threshold``).

    This is a map's effective support (Sec. 6.3 of the paper): outside it the
    landmark's voltage is negligible. Used by the ``support`` partition
    strategy and the ``support_fraction`` scaling strategy.
    """
    if not 0.0 <= tau < 1.0:
        raise ValueError(f"tau must be in [0, 1), got {tau}")
    return _as_array(V) >= tau


def threshold_voltages(V, tau: float):
    """Copy of ``V`` with every voltage below ``tau`` set to 0.

    Makes each map local: a cell outside a landmark's support no longer
    carries its exponentially small voltage. Sources (v = 1) are never
    affected since ``tau < 1``. Downstream, -log(v) of a zeroed entry is
    +inf, so consumers must treat 0 as "out of range" rather than take logs.
    """
    V = _as_array(V)
    xp = array_namespace(V)
    return xp.where(support_mask(V, tau), V, xp.zeros_like(V))


def extend_voltages(
    Kx, V, p, *, config: ExtensionConfig, rho_g: float, nearest
):
    """Voltages at data points from the cells' voltage maps, with the strategy in ``config.strategy``.

    Config choice point ``extension``; see ``lvm.strategies``.

    Parameters
    ----------
    Kx : (m, n) ndarray
        Kernel from m data points to the n cells, e.g.
        ``graph.choose_kernel(cells.sq_distances(X, centroids), ...).K``.
    V : (L, n) ndarray
        Voltage maps over the cells (e.g. the landmarks' thresholded maps).
    p : (n,) ndarray
        Cell masses.
    rho_g : float
        The region's ground scaling.
    nearest : (m,) ndarray of int
        Each point's nearest cell (``cells.assign_cells``), the fallback for a
        point with no cell within the kernel's reach.

    All array inputs are NumPy arrays (computed in float64) or torch tensors
    on one device (computed there, in their dtype), so the per-point steps can
    stay on the GPU.

    Returns
    -------
    (L, m) array of the same kind as the inputs
    """
    strategy, options = resolve(_EXTENSION_STRATEGIES, config)
    if not isinstance(Kx, torch.Tensor):
        Kx, V, p = (np.asarray(a, dtype=np.float64) for a in (Kx, V, p))
        nearest = np.asarray(nearest, dtype=int)
    return strategy(Kx, V, p, options=options, rho_g=rho_g, nearest=nearest)


def _harmonic(Kx, V, p, *, options: None, rho_g: float, nearest):
    """A point is a zero-mass cell: v(x) = sum_i k(x,c_i) p_i v_i / (rho_g + sum_i k(x,c_i) p_i).

    The same fill-in ``solve_grounded_voltage_maps`` gives zero-mass cells, and
    the discrete form of Theorem 9's fixed point at a single point. A point
    with no cell within reach would get v = 0 for every landmark, which says
    nothing about where it is, so it takes its nearest cell's voltages instead.
    """
    xp = array_namespace(Kx, V, p)
    Wx = Kx * p[None, :]                              # (m, n)
    total = xp.sum(Wx, axis=1)
    out = (V @ Wx.T) / (rho_g + total)[None, :]       # (L, m)
    isolated = total == 0
    return xp.where(isolated[None, :], xp.take(V, nearest, axis=1), out)


def voltage_distances(V, tau: float):
    """Distance d = -log v from voltages; +inf where v < tau (outside the map's support, unknown).

    -log v grows roughly linearly with distance from the landmark (Theorem 12
    bounds it between two linear functions), and is 0 at the landmark itself.
    NumPy or torch.
    """
    V = _as_array(V)
    xp = array_namespace(V)
    return xp.where(V >= tau, -xp.log(xp.clip(V, min=tau)), xp.full_like(V, float("inf")))


def chained_distances(D: np.ndarray) -> np.ndarray:
    """Symmetric, complete distances from a square matrix with unknown (+inf) entries.

    Symmetrises (mean of the two directions where both are known, the known
    one otherwise), then fills unknown pairs with shortest paths through
    known ones: d(a, b) = min over routes of the summed known distances. This
    is the within-level form of chaining distances through landmarks. Pairs
    in disconnected parts stay +inf.
    """
    from scipy.sparse.csgraph import shortest_path

    D = np.asarray(D, dtype=np.float64)
    Dt = D.T
    both = np.isfinite(D) & np.isfinite(Dt)
    S = np.where(both, 0.5 * (D + Dt), np.minimum(D, Dt))
    np.fill_diagonal(S, 0.0)
    # Dense csgraph input treats inf as "no edge" (and zero as well, which only the diagonal is).
    return shortest_path(S, method="D", directed=False)


def _as_array(x):
    """Leave NumPy arrays and torch tensors alone; turn anything else (lists) into NumPy."""
    return x if isinstance(x, (np.ndarray, torch.Tensor)) else np.asarray(x)


_EXTENSION_STRATEGIES: dict[str, Callable[..., np.ndarray]] = {
    "harmonic": _harmonic,
}


def benchmark_devices(
    L_rho,
    source_sets: Sequence[Sequence[int]],
    *,
    devices: Sequence[str] = ("cpu", "cuda"),
    repeats: int = 3,
) -> dict:
    """Time ``solve_voltage_maps`` on each device for the same problem.

    Each device gets one untimed warm-up run first, so one-off costs such as
    CUDA initialization aren't counted.

    Returns a dict keyed by device, each {"mean": ..., "min": ..., "times": [...]}.
    """
    results: dict = {}
    for device in devices:
        solve_voltage_maps(L_rho, source_sets, device=device)
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            solve_voltage_maps(L_rho, source_sets, device=device)
            times.append(time.perf_counter() - start)
        results[device] = {"mean": float(np.mean(times)), "min": float(np.min(times)), "times": times}
    return results
