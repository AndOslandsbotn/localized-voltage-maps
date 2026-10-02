"""GPU comparison: LVM vs UMAP vs Laplacian Eigenmaps, all on the GPU, at their tuned settings.

    python benchmarks/gpu_comparison/run.py      # then: python benchmarks/gpu_comparison/plot.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.comparison import run_comparison  # noqa: E402

METHODS = ["lvm_gpu", "umap_gpu", "le_gpu", "lisomap_gpu"]

if __name__ == "__main__":
    run_comparison(METHODS, HERE)
