"""Evidence for the cause of the tears: are the torn points exactly where the set of landmarks reaching a point changes?

A point is "next to a reach boundary" when one of its 10 nearest neighbours on
the sphere is reached (v >= tau) by a different set of landmarks.

    python benchmarks/demonstrations/sphere_tears/reach_boundary_check.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "benchmarks")
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

from sweep_one import fit, torn  # noqa: E402

for L in (3, 6):
    X, model = fit(L)
    V, tau = model.voltages(X), model.config.voltage.threshold
    t = torn(X, model.transform(X))
    reach = V >= tau
    _, idx = NearestNeighbors(n_neighbors=11).fit(X).kneighbors(X)
    boundary = (reach[:, idx[:, 1:]] != reach[:, :, None]).any(axis=(0, 2))
    print(f"{L} landmarks: torn {t.mean():.1%}; next to a reach boundary {boundary.mean():.1%}; "
          f"torn points next to a boundary {np.mean(boundary[t]):.0%}", flush=True)
