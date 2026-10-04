"""MNIST gallery: every method's embedding of the evaluation split (50k, seed 0) at its tuned settings.

    python experiments/sfv/mnist/gallery/run.py      # figure.png (+ one .json per panel, its embedding in embeddings/<panel>.npz)
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments")
sys.path.insert(0, str(BENCH))

from common.gallery import Panel, run_gallery  # noqa: E402

PANELS = [
    Panel("lvm", "lvm_gpu"),
    Panel("lvm_pca", "lvm_pca_gpu"),
    Panel("umap", "umap_gpu"),
    Panel("tsne", "tsne_gpu"),
    Panel("tsne_global", "tsne_gpu", {"perplexity": 100, "late_exaggeration": 4.0},
          title="t-SNE (GPU, cuML), most global setting of its grid"),
    Panel("le", "le_gpu"),
    Panel("lisomap", "lisomap_gpu"),
]

if __name__ == "__main__":
    print(run_gallery(HERE, "mnist", PANELS, split="eval", n=50000))
