"""Draw tuning/lvm/figure.png from tuning/lvm/results.csv.

    python benchmarks/tuning/lvm/plot.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from common.plotting import plot_tuning  # noqa: E402

if __name__ == "__main__":
    print(plot_tuning(HERE, "LVM tuning", x="cells.n_cells"))
