"""Draw tuning/lisomap/figure.png from tuning/lisomap/results.csv.

    python experiments/sfv/mnist/tuning/lisomap/plot.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments")
sys.path.insert(0, str(BENCH))

from common.plotting import plot_tuning  # noqa: E402

if __name__ == "__main__":
    print(plot_tuning(HERE, "Landmark Isomap tuning", x="n_neighbors", line="n_landmarks"))
