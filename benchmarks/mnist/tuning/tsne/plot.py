"""Draw tuning/tsne/figure.png from tuning/tsne/results.csv.

    python benchmarks/mnist/tuning/tsne/plot.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "benchmarks")
sys.path.insert(0, str(BENCH))

from common.plotting import plot_tuning  # noqa: E402

if __name__ == "__main__":
    print(plot_tuning(HERE, "t-SNE tuning", x="perplexity", line="late_exaggeration"))
