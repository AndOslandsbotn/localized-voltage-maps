"""Half-sphere gallery: LVM, LVM + local PCA and t-SNE on 20k points, at their MNIST-tuned settings, unchanged.

    python experiments/sfv/half_sphere/gallery/run.py      # figure.png (+ one .json per panel, its embedding in embeddings/<panel>.npz)
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments")
sys.path.insert(0, str(BENCH))

from common.gallery import Panel, run_gallery  # noqa: E402

PANELS = [Panel("lvm", "lvm_gpu"), Panel("lvm_pca", "lvm_pca_gpu"), Panel("tsne", "tsne_gpu")]

if __name__ == "__main__":
    print(run_gallery(HERE, "half_sphere", PANELS, split="eval", n=20000))
