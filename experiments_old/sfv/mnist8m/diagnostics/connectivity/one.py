"""Why are some digits blown up and others squeezed in LVM's distances? The cell graph per digit. One LVM fit.

Hypothesis: one radius r for the whole graph over-connects compact digits (voltage barely decays: squeezed) and
under-connects varied ones (voltage drops fast: blown up). The solver's stationarity condition
v_i = sum_j K_ij p_j v_j / (rho_g + sum_j K_ij p_j) says what sets the decay at cell i: its connectivity relative to
the ground, c_i = sum_j K_ij p_j / rho_g.

Per cell (labelled by its points' majority digit): degree (edges), connectivity c_i, distance to the nearest other
cell / r, and magnification: the voltage distance -log v_i(j) from cell i (as source) to each graph neighbour j, per
unit pixel distance |c_i - c_j| (median over neighbours) -- the local stretch of LVM's metric. Reported per digit
(median over its cells), with its share of cells vs share of the data, and the rank correlation over cells of
magnification vs connectivity.

Across a whole digit (all pairs of its cells, not only neighbours): pieces (connected pieces of the graph between
its own cells) and the share of its cells in the largest; detour (shortest path along graph edges, in pixel
lengths, / straight pixel distance); global stretch (voltage distance -log v between the pair / pixel distance,
relative to the median over all digits) and the share of pairs below the distance floor (no measured distance).

    python experiments_old/sfv/mnist8m/diagnostics/connectivity/one.py --dataset mnist8m --seed 0
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments_old")
sys.path.insert(0, str(BENCH))

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from common.methods import _merge, method_params  # noqa: E402

CHUNK = 10_000


def data(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    if dataset == "mnist":
        from common.datasets import load

        X, y = load("eval", n=50_000, seed=0)
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)
    from common.datasets import _mnist8m_arrays, mnist8m_rows

    return mnist8m_rows(0, 50_000).astype(np.float32) / 255.0, np.asarray(_mnist8m_arrays()[1][:50_000], dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "mnist8m"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    from lvm_old.cells import assign_cells, sq_distances
    from lvm_old.config import load_config
    from lvm_old.pipeline import fit_level
    from lvm_old.stream import array_source
    from lvm_old.voltage import solve_grounded_voltage_maps

    X, y = data(args.dataset)
    cfg = load_config(overrides=_merge(method_params("lvm_gpu"), {"compute": {"device": "cuda", "seed": args.seed}}))
    model = fit_level(array_source(X, CHUNK), cfg)
    C, K, p, rho_g, r = model.centroids, np.asarray(model.K, dtype=np.float64), model.masses.p, model.rho.rho_g, model.r
    n = len(C)
    cell = assign_cells(X, C, device="cuda")
    counts = np.stack([np.bincount(cell[y == d], minlength=n) for d in range(10)])      # (10, n)
    digit = np.where(counts.sum(axis=0) > 0, counts.argmax(axis=0), -1)
    dist = np.sqrt(sq_distances(C, C))
    np.fill_diagonal(dist, np.inf)
    alive = np.flatnonzero(p > 0)
    V = solve_grounded_voltage_maps(K, p, rho_g, [[int(i)] for i in alive], device="cuda")   # row k: source alive[k]
    degree = (K > 0).sum(axis=1)
    connectivity = (K * p[None, :]).sum(axis=1) / rho_g
    nearest = dist.min(axis=1) / r
    magnification = np.full(n, np.nan)
    for k, i in enumerate(alive):
        nb = np.flatnonzero((K[i] > 0) & (V[k] > 0))
        if nb.size:
            magnification[i] = float(np.median(-np.log(V[k, nb]) / dist[i, nb]))
    ok = np.isfinite(magnification) & (digit >= 0)
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components, shortest_path

    edges = np.where(K > 0, dist, 0.0)
    path = shortest_path(csr_matrix(edges), directed=False)                       # along edges, pixel lengths
    row = {int(i): k for k, i in enumerate(alive)}
    floor = model.distance_floor
    pair_stretch, pair_unreached, pair_detour, pieces = {}, {}, {}, {}
    for d in range(10):
        own = np.array([i for i in np.flatnonzero(digit == d) if i in row])
        sub = csr_matrix(edges[np.ix_(own, own)])
        n_pieces, label = connected_components(sub, directed=False)
        pieces[d] = (int(n_pieces), float(np.bincount(label).max() / len(own)))
        a, b = np.triu_indices(len(own), k=1)
        i, j = own[a], own[b]
        v = V[[row[x] for x in i], j]
        pixel = dist[i, j]
        measured = v >= floor
        pair_unreached[d] = float(1 - measured.mean())
        pair_stretch[d] = float(np.median(-np.log(v[measured]) / pixel[measured])) if measured.any() else np.nan
        finite = np.isfinite(path[i, j])
        pair_detour[d] = float(np.median(path[i, j][finite] / pixel[finite])) if finite.any() else np.nan
    typical_stretch = float(np.nanmedian(list(pair_stretch.values())))
    per_digit = {}
    for d in range(10):
        m = ok & (digit == d)
        per_digit[d] = {"data_share": float(np.mean(y == d)), "cell_share": float(m.sum() / ok.sum()),
                        "degree": float(np.median(degree[m])), "connectivity": float(np.median(connectivity[m])),
                        "nearest_over_r": float(np.median(nearest[m])),
                        "magnification": float(np.median(magnification[m]) / np.median(magnification[ok])),
                        "pieces": pieces[d][0], "largest_piece": pieces[d][1], "detour": pair_detour[d],
                        "global_stretch": pair_stretch[d] / typical_stretch, "unreached": pair_unreached[d]}
    print("RESULT " + json.dumps({
        "dataset": args.dataset, "seed": args.seed, "n_cells": n, "r": r, "rho_g": rho_g, "per_digit": per_digit,
        "corr_magnification_connectivity": float(spearmanr(magnification[ok], connectivity[ok])[0]),
        "corr_magnification_degree": float(spearmanr(magnification[ok], degree[ok])[0]),
        "corr_magnification_mass": float(spearmanr(magnification[ok], p[ok])[0]),
        "corr_magnification_nearest": float(spearmanr(magnification[ok], nearest[ok])[0])}), flush=True)


if __name__ == "__main__":
    main()
