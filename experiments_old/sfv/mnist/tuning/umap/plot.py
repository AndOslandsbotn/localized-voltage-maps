"""Draw tuning/umap/figure.png from tuning/umap/results.csv.

    python experiments_old/sfv/mnist/tuning/umap/plot.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments_old")
sys.path.insert(0, str(BENCH))

from common.plotting import plot_tuning  # noqa: E402

if __name__ == "__main__":
    print(plot_tuning(HERE, "UMAP tuning", x="n_neighbors", line="min_dist"))
