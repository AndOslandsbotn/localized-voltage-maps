"""Tune t-SNE on the tuning split (see common/tuning.py for the shared protocol).

    python benchmarks/mnist/tuning/tsne/tune.py
"""

import itertools
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

METHOD = "tsne_gpu"   # tuned on GPU; the CPU version reuses the same settings
GRID = [
    {"perplexity": perplexity, "late_exaggeration": late}
    for perplexity, late in itertools.product([5, 15, 30, 50, 100], [1.0, 2.0, 4.0])
]


if __name__ == "__main__":
    run_grid(METHOD, GRID, HERE)
