"""Tune LVM on the tuning split (see common/tuning.py for the shared protocol).

    python benchmarks/mnist/tuning/lvm/tune.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "benchmarks")
sys.path.insert(0, str(BENCH))

from common.runner import THREAD_VARS  # noqa: E402

import os  # noqa: E402

for _var in THREAD_VARS:
    os.environ.setdefault(_var, "16")

from common.tuning import run_grid  # noqa: E402

METHOD = "lvm_gpu"   # tuned on GPU; the CPU version reuses the same settings
GRID = [
    # LVM: Landmark MDS with chained distances, MI landmarks, d_hat + 1 of them, and
    # rho_g from `reach` (every typical point reached by all landmarks), which has no
    # free parameter -- so only the cell count is tuned.
    {"cells": {"n_cells": n_cells},
     "scaling": {"strategy": "reach", "reach": {"k": None, "quantile": 0.1}},
     "landmarks": {"strategy": "mutual_information", "count": {"strategy": "dimension", "dimension": {"multiplier": 1.0}}},
     "embedding": {"strategy": "landmark_mds", "landmark_mds": {"missing": "chain"}}}
    for n_cells in [100, 150, 200, 300, 400, 600]
]


if __name__ == "__main__":
    run_grid(METHOD, GRID, HERE)
