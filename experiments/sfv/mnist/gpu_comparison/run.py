"""GPU comparison: LVM vs UMAP vs Laplacian Eigenmaps vs Landmark Isomap vs t-SNE, all on the GPU, at their tuned settings.

    python experiments/sfv/mnist/gpu_comparison/run.py      # then: python experiments/sfv/mnist/gpu_comparison/plot.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments")
sys.path.insert(0, str(BENCH))

from common.comparison import run_comparison  # noqa: E402

METHODS = ["lvm_gpu", "lvm_pca_gpu", "umap_gpu", "le_gpu", "lisomap_gpu", "tsne_gpu"]

if __name__ == "__main__":
    run_comparison(METHODS, HERE)
