"""Where do LVM's landmarks sit on the half sphere, and how far out are the rim points placed?"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402

from common.methods import _merge, method_params  # noqa: E402
from hemisphere.one import half_sphere  # noqa: E402
from lvm.config import load_config  # noqa: E402
from lvm.pipeline import fit_level  # noqa: E402
from lvm.stream import array_source  # noqa: E402

X = half_sphere(20000, 0)
for mult in (1.0, 2.0, 3.0):
    cfg = load_config(overrides=_merge(method_params("lvm_gpu"), {
        "compute": {"device": "cuda", "seed": 0},
        "landmarks": {"count": {"dimension": {"multiplier": mult}}}}))
    m = fit_level(array_source(X, cfg.data.chunk_size), cfg)
    heights = m.centroids[m.landmark_cells, 2]
    Z = m.transform(X)
    r = np.linalg.norm(Z - np.median(Z, axis=0), axis=1)
    rim = X[:, 2] < 0.15
    print(f"multiplier {mult}: {len(heights)} landmarks at heights {np.round(np.sort(heights), 2)}; "
          f"embedding radius (median) rim {np.median(r[rim]):.2f} vs rest {np.median(r[~rim]):.2f}; "
          f"cell heights range {m.centroids[:, 2].min():.2f}-{m.centroids[:, 2].max():.2f}", flush=True)
