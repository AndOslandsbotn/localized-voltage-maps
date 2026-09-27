"""Solve the grounded energy-minimizing voltage (EMV) system.

Given a grounded graph Laplacian ``L_rho = L + rho*I`` (as produced by
``graph.apply_ground_resistance``) and a set of source nodes clamped to
voltage 1, this solves for the voltage at every other ("floating") node --
Definition 3 / Lemma 4 in Structure_from_Voltage.pdf.

Clamping the source nodes to 1 and eliminating them from the system leaves a
reduced linear system over the floating nodes F:

    L_rho[F, F] @ v[F] = -L_rho[F, S] @ v[S]

Since off-diagonal entries of L_rho are -W_ij, the right-hand side equals the
sum of edge weights from each floating node into the source set (the
numerator in Proposition 8). L_rho[F, F] is symmetric positive-definite
because rho > 0, which is what makes the Cholesky-based backends valid.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import cho_factor, cho_solve

METHODS = ("dense", "cholesky", "cg", "cholmod")


def _node_indices(n: int, indices: Sequence[int]) -> np.ndarray:
    idx = np.unique(np.asarray(indices, dtype=int))
    if idx.size and (idx.min() < 0 or idx.max() >= n):
        raise ValueError(f"source indices must be in [0, {n}), got range [{idx.min()}, {idx.max()}]")
    return idx


def _reduced_system(L_rho, source_indices: np.ndarray, free_indices: np.ndarray):
    ones = np.ones(source_indices.size)
    if sp.issparse(L_rho):
        L_rho = L_rho.tocsr()
        A = L_rho[free_indices][:, free_indices]
        b = -(L_rho[free_indices][:, source_indices] @ ones)
    else:
        L_rho = np.asarray(L_rho)
        A = L_rho[np.ix_(free_indices, free_indices)]
        b = -(L_rho[np.ix_(free_indices, source_indices)] @ ones)
    return A, b


def solve_voltage(
    L_rho,
    source_indices: Sequence[int],
    *,
    method: str = "cg",
    tol: float = 1e-8,
    maxiter: int | None = None,
) -> np.ndarray:
    """Solve the grounded EMV for a single source set.

    Parameters
    ----------
    L_rho : (n, n) dense ndarray or scipy.sparse matrix
        Grounded Laplacian, e.g. from ``graph.apply_ground_resistance``.
    source_indices : sequence of int
        Indices of the nodes held at voltage 1 (the landmark's source ball).
    method : {"dense", "cholesky", "cg", "cholmod"}
        "dense" is a plain LU solve (numpy), used mainly as a correctness
        reference. "cholesky" is a dense SPD solve (scipy). "cg" is sparse
        conjugate gradient with a Jacobi preconditioner -- the recommended
        default for large sparse graphs, since rho > 0 lower-bounds the
        system's eigenvalues and guarantees fast convergence. "cholmod" is
        an exact sparse Cholesky factorization via scikit-sparse, generally
        the fastest exact option but requires libsuitesparse to be
        installed.
    tol : float
        Relative residual tolerance, used only by the "cg" method.
    maxiter : int, optional
        Iteration cap, used only by the "cg" method.

    Returns
    -------
    v : (n,) ndarray
        Voltage at every node; v[i] == 1 for i in source_indices.
    """
    n = L_rho.shape[0]
    source_idx = _node_indices(n, source_indices)
    free_idx = np.setdiff1d(np.arange(n), source_idx, assume_unique=True)

    v = np.ones(n, dtype=float)
    if free_idx.size == 0:
        return v

    A, b = _reduced_system(L_rho, source_idx, free_idx)

    if method == "dense":
        x = np.linalg.solve(np.asarray(A.todense() if sp.issparse(A) else A), b)
    elif method == "cholesky":
        A_dense = np.asarray(A.todense() if sp.issparse(A) else A)
        factor = cho_factor(A_dense)
        x = cho_solve(factor, b)
    elif method == "cg":
        A_sp = A if sp.issparse(A) else sp.csr_matrix(A)
        preconditioner = sp.diags(1.0 / A_sp.diagonal())
        x, info = spla.cg(A_sp, b, rtol=tol, maxiter=maxiter, M=preconditioner)
        if info != 0:
            raise RuntimeError(f"conjugate gradient failed to converge (info={info})")
    elif method == "cholmod":
        try:
            from sksparse.cholmod import cho_factor as cholmod_cho_factor
        except ImportError as exc:
            raise ImportError(
                "the 'cholmod' method needs scikit-sparse, which needs the "
                "libsuitesparse-dev system library. Install with:\n"
                "  sudo apt-get install -y libsuitesparse-dev && "
                "pip install scikit-sparse"
            ) from exc
        A_sp = (A if sp.issparse(A) else sp.csc_matrix(A)).tocsc()
        x = cholmod_cho_factor(A_sp).solve(b)
    else:
        raise ValueError(f"unknown method {method!r}, expected one of {METHODS}")

    v[free_idx] = x
    return v


def solve_voltage_maps(
    L_rho,
    source_sets: Sequence[Sequence[int]],
    *,
    method: str = "cg",
    tol: float = 1e-8,
    maxiter: int | None = None,
) -> np.ndarray:
    """Solve the grounded EMV independently for each source set.

    Mirrors step (4) "Voltage Maps" in Fig. 6 of the paper, where every
    candidate centroid is used in turn as a landmark. Each solve is
    independent, so this is the natural place to later fan out across
    processes or a batched GPU backend without changing the call site.

    Returns
    -------
    V : (len(source_sets), n) ndarray
        Row i is the voltage map for source_sets[i].
    """
    return np.stack([
        solve_voltage(L_rho, sources, method=method, tol=tol, maxiter=maxiter)
        for sources in source_sets
    ])


def benchmark_solvers(
    L_rho,
    source_indices: Sequence[int],
    *,
    methods: Sequence[str] = METHODS,
    repeats: int = 3,
    tol: float = 1e-8,
) -> dict:
    """Time each backend on the same problem for comparison.

    Returns a dict keyed by method name, each either
    {"mean": ..., "min": ..., "times": [...]} or {"error": "..."} for a
    method that isn't available (e.g. "cholmod" without scikit-sparse).
    """
    results: dict = {}
    for method in methods:
        times = []
        try:
            for _ in range(repeats):
                start = time.perf_counter()
                solve_voltage(L_rho, source_indices, method=method, tol=tol)
                times.append(time.perf_counter() - start)
        except ImportError as exc:
            results[method] = {"error": str(exc)}
            continue
        results[method] = {"mean": float(np.mean(times)), "min": float(np.min(times)), "times": times}
    return results
