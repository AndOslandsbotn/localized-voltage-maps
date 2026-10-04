"""CPU comparison: LVM vs UMAP vs Laplacian Eigenmaps vs Landmark Isomap vs t-SNE, all on the CPU, at their tuned settings.

    python experiments/sfv/mnist/cpu_comparison/run.py      # then: python experiments/sfv/mnist/cpu_comparison/plot.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments")
sys.path.insert(0, str(BENCH))

from common.comparison import run_comparison  # noqa: E402

METHODS = ["lvm_cpu", "lvm_pca_cpu", "umap_cpu", "le_cpu", "lisomap_cpu", "tsne_cpu"]
# Laplacian Eigenmaps on CPU took the machine down at 70k with the old ARPACK
# solver (7 GB RAM in WSL). Capped at 40k until AMG's memory use is shown to be safe.
MAX_N = {"le_cpu": 40000}

if __name__ == "__main__":
    run_comparison(METHODS, HERE, max_n=MAX_N)
