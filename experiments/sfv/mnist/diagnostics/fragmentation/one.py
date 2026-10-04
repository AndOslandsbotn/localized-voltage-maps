"""How fragmented is each digit? One representation of MNIST's tuning split (20k, seed 0).

For each digit: link each of its points to its 10 nearest points of the same digit (in this representation),
and report the share of the digit's points in the largest connected group, and how many groups hold >= 2%.
Representations: raw (784 pixels: is the digit really split in the data?), lvm, lvm_pca (fill 0.35 / 0.6),
umap, tsne.

    python experiments/sfv/mnist/diagnostics/fragmentation/one.py --what lvm_pca --fill 0.6
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments")
sys.path.insert(0, str(BENCH))

import numpy as np  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

from common.datasets import load  # noqa: E402
from common.methods import METHODS, _merge, method_params  # noqa: E402


def fragmentation(R: np.ndarray, y: np.ndarray, k: int = 10) -> dict:
    out = {}
    for d in range(10):
        P = R[y == d]
        _, idx = NearestNeighbors(n_neighbors=k + 1).fit(P).kneighbors(P)
        rows = np.repeat(np.arange(len(P)), k)
        G = coo_matrix((np.ones(rows.size), (rows, idx[:, 1:].ravel())), shape=(len(P), len(P)))
        _, label = connected_components(G, directed=False)
        sizes = np.sort(np.bincount(label))[::-1]
        out[d] = {"largest_share": float(sizes[0] / len(P)), "groups_over_2pct": int(np.sum(sizes >= 0.02 * len(P)))}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--what", choices=["raw", "lvm", "lvm_pca", "umap", "tsne"], required=True)
    parser.add_argument("--fill", type=float, default=0.35)
    args = parser.parse_args()
    X, y = load("tune", n=20000, seed=0)
    if args.what == "raw":
        R = X
    else:
        method = {"lvm": "lvm_gpu", "lvm_pca": "lvm_pca_gpu", "umap": "umap_gpu", "tsne": "tsne_gpu"}[args.what]
        params = method_params(method)
        if args.what == "lvm_pca":
            params = _merge(params, {"embedding": {"local_scale": {"pca": {"fill": args.fill}}}})
        R, _, _ = METHODS[method].embed(X, params, 0)
    name = args.what + (f"_fill{args.fill:g}" if args.what == "lvm_pca" else "")
    frag = fragmentation(np.asarray(R, dtype=np.float64), y)
    print("RESULT " + json.dumps({"what": name, "per_digit": frag}), flush=True)


if __name__ == "__main__":
    main()
