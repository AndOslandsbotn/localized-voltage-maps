"""Is the information there before the 2-D picture? One method on one dataset: its richer representation vs 2-D.

* lvm: each point's distances to the landmarks (exactly what the triangulation receives), and the 2-D result;
* lisomap: shortest-path distances to its landmarks, and the 2-D result;
* le: the first 15 eigenvectors, and the first 2;
* umap: UMAP run in 15 dimensions (a different embedding, as a reference), and in 2.
Scored on 5000 random points with the usual measures (trust, cont, 5-NN, global), as everywhere.

    python experiments/sfv/mnist/diagnostics/intermediate/one.py --dataset mnist --method lvm
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments")
sys.path.insert(0, str(BENCH))

import numpy as np  # noqa: E402

from common.datasets import load  # noqa: E402
from common.methods import _merge, method_params  # noqa: E402
from common.metrics import quality  # noqa: E402

DATA = {"mnist": ("tune", 20000), "mnist8m": ("eval", 50000)}


def lvm(X, landmarks=None):
    from lvm.config import load_config
    from lvm.embedding import point_landmark_distances
    from lvm.pipeline import fit_level
    from lvm.stream import array_source

    extra = {"landmarks": {"n_landmarks": landmarks, "count": {"strategy": "fixed"}}} if landmarks else {}
    cfg = load_config(overrides=_merge(_merge(method_params("lvm_gpu"), {"compute": {"device": "cuda", "seed": 0}}), extra))
    m = fit_level(array_source(X, cfg.data.chunk_size), cfg)
    v = m._voltages_and_cells(X, for_distances=True)[0]
    v = v.cpu().numpy() if hasattr(v, "cpu") else v
    rich = np.asarray(point_landmark_distances(v, m.embedding.landmark_D, tau=m.distance_floor, missing="chain"))
    return rich, m.transform(X), f"distances to {rich.shape[1]} landmarks"


def lisomap(X):
    from common.methods import _lisomap_distances_gpu
    from lvm.embedding import place_landmarks

    p = method_params("lisomap_gpu")
    landmarks = np.random.default_rng(0).choice(len(X), p["n_landmarks"], replace=False)
    D = _lisomap_distances_gpu(X, p["n_neighbors"], landmarks)
    D = np.where(np.isfinite(D), D, D[np.isfinite(D)].max())
    Z = np.asarray(place_landmarks(D[:, landmarks], n_components=2).triangulate(D.T))
    return D.T, Z, f"shortest-path distances to {len(landmarks)} landmarks"


def le(X):
    from cuml.manifold import SpectralEmbedding

    E = np.asarray(SpectralEmbedding(n_components=15, n_neighbors=method_params("le_gpu")["n_neighbors"],
                                     random_state=0).fit_transform(X.astype(np.float32)), dtype=np.float64)
    return E, E[:, :2], "first 15 eigenvectors"


def umap(X):
    from cuml.manifold import UMAP

    p = method_params("umap_gpu")
    kw = {"n_neighbors": p["n_neighbors"], "min_dist": p["min_dist"]}
    X32 = X.astype(np.float32)
    rich = np.asarray(UMAP(n_components=15, **kw).fit_transform(X32), dtype=np.float64)
    Z = np.asarray(UMAP(n_components=2, **kw).fit_transform(X32), dtype=np.float64)
    return rich, Z, "UMAP run in 15 dimensions"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATA), required=True)
    parser.add_argument("--method", choices=["lvm", "lisomap", "le", "umap"], required=True)
    parser.add_argument("--landmarks", type=int, default=None, help="lvm: a fixed landmark count (default d + 1)")
    args = parser.parse_args()
    split, n = DATA[args.dataset]
    X, y = load(split, n=n, seed=0, dataset=args.dataset)
    if args.method == "lvm":
        rich, Z, what = lvm(X, args.landmarks)
    else:
        rich, Z, what = {"lisomap": lisomap, "le": le, "umap": umap}[args.method](X)
    name = args.method + (f"_L{args.landmarks}" if args.landmarks else "")
    out = {"dataset": args.dataset, "method": name, "representation": what,
           "rich": quality(X, rich, y, seed=0), "2d": quality(X, Z, y, seed=0)}
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
