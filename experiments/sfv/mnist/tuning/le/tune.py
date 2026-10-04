"""Tune LE on the tuning split (see common/tuning.py for the shared protocol).

    python experiments/sfv/mnist/tuning/le/tune.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments")
sys.path.insert(0, str(BENCH))

from common.runner import THREAD_VARS  # noqa: E402

import os  # noqa: E402

for _var in THREAD_VARS:
    os.environ.setdefault(_var, "16")

from common.tuning import run_grid  # noqa: E402

METHOD = "le_gpu"   # tuned on GPU; the CPU version reuses the same settings
GRID = [
    # The neighbourhood size is LE's only real hyperparameter in sklearn and cuML.
    {"n_neighbors": n_neighbors} for n_neighbors in [5, 10, 15, 20, 30, 50, 75, 100]
]


if __name__ == "__main__":
    run_grid(METHOD, GRID, HERE)
