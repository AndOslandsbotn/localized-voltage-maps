"""Solve the grounded energy-minimizing voltage (EMV) system.

Given a grounded graph Laplacian ``L_rho = L + rho*I`` (as produced by
``graph.apply_ground_resistance``) and a set of source nodes S clamped to
voltage 1, this solves for the voltage at every node -- Definition 3 /
Lemma 4 in Structure_from_Voltage.pdf.

Lemma 4's fixed point v = D~^-1 W~(s) v says, at every free node i,
(L_rho @ v)[i] = 0, i.e. no current leaves the circuit at i except through
the ground. The only nonzero entries of L_rho @ v are therefore at the
sources, where current c is injected to hold them at 1:

    L_rho @ v = e_S @ c    =>    v = G[:, S] @ c,    with G = L_rho^-1,

and v[S] = 1 fixes c through the small |S| x |S| system G[S, S] @ c = 1.
L_rho is symmetric positive-definite because rho > 0, so G comes from one
Cholesky factorization.

L_rho is the same for every landmark on a graph -- only S changes -- so G is
computed once and every voltage map is read off its columns. That costs
O(n^3) once instead of O(n^3) per landmark.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import scipy.sparse as sp
import torch


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
        Grounded Laplacian, e.g. from ``graph.apply_ground_resistance``.
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
