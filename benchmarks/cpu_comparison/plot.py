"""Draw cpu_comparison/figure.png from cpu_comparison/results.csv.

    python benchmarks/cpu_comparison/plot.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))

from common.plotting import plot_comparison  # noqa: E402

if __name__ == "__main__":
    print(plot_comparison(HERE, "CPU comparison"))
