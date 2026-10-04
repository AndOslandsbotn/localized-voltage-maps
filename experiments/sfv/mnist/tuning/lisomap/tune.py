"""Tune Landmark Isomap on the tuning split (see common/tuning.py for the shared protocol).

    python experiments/sfv/mnist/tuning/lisomap/tune.py
"""

import itertools
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments")
sys.path.insert(0, str(BENCH))

from common.runner import THREAD_VARS  # noqa: E402

for _var in THREAD_VARS:
    os.environ.setdefault(_var, "16")

from common.tuning import run_grid  # noqa: E402

METHOD = "lisomap_gpu"   # tuned on GPU; the CPU version reuses the same settings
GRID = [
    # Neighbourhood size of the kNN graph x number of (random) landmarks.
    {"n_neighbors": n_neighbors, "n_landmarks": n_landmarks}
    for n_neighbors, n_landmarks in itertools.product([5, 10, 15, 30, 50], [15, 30, 60])
]


if __name__ == "__main__":
    run_grid(METHOD, GRID, HERE)
