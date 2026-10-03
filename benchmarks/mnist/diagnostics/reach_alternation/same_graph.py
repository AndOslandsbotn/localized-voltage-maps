"""Old vs new `reach` search on one fitted graph (one seed, one device), printed as a JSON line.

Fits the tuned LVM once, then runs on its kernel and masses:
  * old:  the previous search -- bisection on log(rho_g) over the full bounds,
          choosing the landmarks anew at every probe (reference copy below);
  * new:  the alternation (lvm.scaling._reach), from the default start and
          from the start x0.1 and x10.
Same graph for all, so differences come from the search alone.

    python benchmarks/mnist/diagnostics/reach_alternation/same_graph.py --seed 0 --device cuda
"""

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "benchmarks")
sys.path.insert(0, str(BENCH))

import numpy as np  # noqa: E402

from common.datasets import load  # noqa: E402
from common.methods import _merge, best_params  # noqa: E402
from lvm.config import load_config  # noqa: E402
from lvm.pipeline import fit_level  # noqa: E402
from lvm.scaling import _select, choose_rho_g, landmark_reach, typical_reach  # noqa: E402
from lvm.stream import array_source  # noqa: E402
from lvm.voltage import choose_sources  # noqa: E402

N = 20000


def old_search(K, p, *, options, tau, n_landmarks, landmarks, sources, device):
    """The search `reach` used before the alternation: select the landmarks at every probe."""
    k = n_landmarks if options.k is None else min(options.k, n_landmarks)

    def reach(rho_g):
        return typical_reach(K, p, rho_g, tau=tau, n_landmarks=n_landmarks, landmarks=landmarks,
                             quantile=options.quantile, sources=sources, device=device)

    lo, hi = options.rho_g_bounds
    n_probes = 2
    if reach(hi) >= k:
        best = hi
    elif reach(lo) < k:
        best = lo
    else:
        log_lo, log_hi, best = math.log(lo), math.log(hi), lo
        for _ in range(options.max_iter):
            if log_hi - log_lo <= math.log1p(options.rel_tolerance):
                break
            mid = 0.5 * (log_lo + log_hi)
            n_probes += 1
            if reach(math.exp(mid)) >= k:
                log_lo, best = mid, math.exp(mid)
            else:
                log_hi = mid
    # ... and the pipeline then chose the landmarks once more at the result.
    selection = _select(K, p, best, sources, tau=tau, n_landmarks=n_landmarks, landmarks=landmarks, device=device)
    return best, selection, n_probes + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], required=True)
    args = parser.parse_args()

    X, _ = load("tune", n=N, seed=args.seed)
    kmeans = "cuml" if args.device == "cuda" else "faiss"
    cfg = load_config(overrides=_merge({"compute": {"device": args.device, "seed": args.seed},
                                        "cells": {"kmeans": {"strategy": kmeans}}}, best_params("lvm")))
    model = fit_level(array_source(X, cfg.data.chunk_size), cfg)        # also warms up the device
    K, p, tau, L = model.K, model.masses.p, cfg.voltage.threshold, int(model.V.shape[0])
    sources = choose_sources(p, config=cfg.sources)
    options = cfg.scaling.reach
    k = L if options.k is None else min(options.k, L)

    def cells(selection):
        return sorted(int(sources.cells[i]) for i in selection.indices)

    def reach_of(rho_g, selection):
        return landmark_reach(K, p, rho_g, [sources.sets[i] for i in selection.indices], tau=tau,
                              quantile=options.quantile, device=args.device)

    t = time.perf_counter()
    old_rho, old_sel, old_n = old_search(K, p, options=options, tau=tau, n_landmarks=L, landmarks=cfg.landmarks,
                                         sources=sources, device=args.device)
    out = {"seed": args.seed, "device": args.device, "n_landmarks": L, "k": k,
           "old": {"rho_g": old_rho, "seconds": time.perf_counter() - t, "selections": old_n,
                   "reach_of_own_landmarks": reach_of(old_rho, old_sel), "cells": cells(old_sel)}}
    for factor in (1.0, 0.1, 10.0):
        scaling = replace(cfg.scaling, reach=replace(options, start_factor=factor))
        t = time.perf_counter()
        choice = choose_rho_g(K, p, config=scaling, tau=tau, n_landmarks=L, landmarks=cfg.landmarks,
                              sources=sources, device=args.device)
        out[f"new_x{factor:g}"] = {"rho_g": choice.rho_g, "seconds": time.perf_counter() - t,
                                   "selections": choice.n_rounds, "reach_of_own_landmarks": choice.reach,
                                   "cells": cells(choice.landmarks),
                                   "overlap_with_old": len(set(cells(choice.landmarks)) & set(cells(old_sel)))}
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
