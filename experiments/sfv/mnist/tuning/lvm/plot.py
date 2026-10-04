"""Draw tuning/lvm/figure.png from tuning/lvm/results.csv.

    python experiments/sfv/mnist/tuning/lvm/plot.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments")
sys.path.insert(0, str(BENCH))

from common.plotting import plot_tuning  # noqa: E402

if __name__ == "__main__":
    print(plot_tuning(HERE, "LVM tuning", x="cells.n_cells"))
