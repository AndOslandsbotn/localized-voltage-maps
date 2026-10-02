"""Draw tuning/umap/figure.png from tuning/umap/results.csv.

    python benchmarks/tuning/umap/plot.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from common.plotting import plot_tuning  # noqa: E402

if __name__ == "__main__":
    print(plot_tuning(HERE, "UMAP tuning", x="n_neighbors", line="min_dist"))
