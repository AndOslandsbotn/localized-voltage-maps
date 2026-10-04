"""Draw gpu_comparison/figure.png from gpu_comparison/results.csv.

    python experiments/sfv/mnist/gpu_comparison/plot.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments")
sys.path.insert(0, str(BENCH))

from common.plotting import plot_comparison  # noqa: E402

if __name__ == "__main__":
    print(plot_comparison(HERE, "GPU comparison"))
