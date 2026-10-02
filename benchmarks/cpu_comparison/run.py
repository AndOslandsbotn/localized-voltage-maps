"""CPU comparison: LVM vs UMAP vs Laplacian Eigenmaps, all on the CPU, at their tuned settings.

    python benchmarks/cpu_comparison/run.py      # then: python benchmarks/cpu_comparison/plot.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.comparison import run_comparison  # noqa: E402

METHODS = ["lvm_cpu", "umap_cpu", "le_cpu", "lisomap_cpu"]
# Laplacian Eigenmaps on CPU took the machine down at 70k with the old ARPACK
# solver (7 GB RAM in WSL). Capped at 40k until AMG's memory use is shown to be safe.
MAX_N = {"le_cpu": 40000}

if __name__ == "__main__":
    run_comparison(METHODS, HERE, max_n=MAX_N)
