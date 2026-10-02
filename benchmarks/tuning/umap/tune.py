"""Tune UMAP on the tuning split (see common/tuning.py for the shared protocol).

    python benchmarks/tuning/umap/tune.py
"""

import itertools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from common.runner import THREAD_VARS  # noqa: E402

import os  # noqa: E402

for _var in THREAD_VARS:
    os.environ.setdefault(_var, "16")

from common.tuning import run_grid  # noqa: E402

METHOD = "umap_gpu"   # tuned on GPU; the CPU version reuses the same settings
GRID = [
    {"n_neighbors": n_neighbors, "min_dist": min_dist}
    for n_neighbors, min_dist in itertools.product([5, 10, 15, 30, 50], [0.0, 0.1, 0.5])
]


if __name__ == "__main__":
    run_grid(METHOD, GRID, HERE)
