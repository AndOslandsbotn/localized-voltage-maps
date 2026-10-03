"""Characterise the sphere tears: where they come from, at the pinned settings (one landmark count per run).

1. jump: for sphere-neighbour pairs where landmark l is measured on one side and
   chained on the other, the jump in d(., l), against the usual neighbour difference;
2. overestimate: fit -log v = a * geodesic + b where measured; compare chained values
   with that line's prediction at the same points;
3. tears by how many landmarks reach a point;
4. control: the same fit with missing = clip (continuous at the edge of a map).

    python benchmarks/demonstrations/sphere_tears/characterise.py --landmarks 3 [--missing clip]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "benchmarks")
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

from common.datasets import half_sphere  # noqa: E402
from lvm.config import load_config  # noqa: E402
from lvm.embedding import point_landmark_distances  # noqa: E402
from lvm.pipeline import fit_level  # noqa: E402
from lvm.stream import array_source  # noqa: E402
from sweep_one import N, DATA_SEED, torn  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks", type=int, required=True)
    parser.add_argument("--missing", choices=["chain", "clip"], default="chain")
    args = parser.parse_args()
    X = half_sphere(N, DATA_SEED)
    cfg = load_config(HERE / "settings.yaml", overrides={
        "landmarks": {"n_landmarks": args.landmarks},
        "embedding": {"landmark_mds": {"missing": args.missing}}})
    model = fit_level(array_source(X, cfg.data.chunk_size), cfg)
    tau = cfg.voltage.threshold
    V = model.voltages(X)                                                    # (L, n), 0 below tau
    reached = V >= tau                                                       # (L, n)
    d = point_landmark_distances(V, model.embedding.landmark_D, tau=tau, missing=args.missing)   # (n, L)
    Z = model.transform(X)
    t = torn(X, Z)
    out = {"landmarks": args.landmarks, "missing": args.missing, "torn_share": float(t.mean())}

    # 1. Jumps across the edge of a map, vs the usual difference between neighbours (both measured).
    _, idx = NearestNeighbors(n_neighbors=11).fit(X).kneighbors(X)
    i = np.repeat(np.arange(N), 10)
    j = idx[:, 1:].ravel()
    switch, both = [], []
    for l in range(args.landmarks):
        a, b = reached[l, i], reached[l, j]
        diff = np.abs(d[i, l] - d[j, l])
        switch.append(diff[a & ~b])
        both.append(diff[a & b])
    switch, both = np.concatenate(switch), np.concatenate(both)
    out["jump_at_edge_median"] = float(np.median(switch)) if switch.size else None
    out["jump_at_edge_p90"] = float(np.quantile(switch, 0.9)) if switch.size else None
    out["neighbour_diff_inside_median"] = float(np.median(both))
    out["neighbour_diff_inside_p90"] = float(np.quantile(both, 0.9))

    # 2. Over- or underestimate: -log v vs geodesic distance to the landmark, fitted where measured.
    lm = model.centroids[model.landmark_cells]
    lm = lm / np.linalg.norm(lm, axis=1, keepdims=True)
    geo = np.arccos(np.clip(X @ lm.T, -1, 1))                               # (n, L) along the sphere
    measured = reached.T
    a, b = np.polyfit(geo[measured], d[measured], 1)
    predicted = a * geo + b
    if args.missing == "chain" and (~measured).any():
        ratio = d[~measured] / predicted[~measured]
        out["chained_over_line_median"] = float(np.median(ratio))
        out["chained_over_line_p10_p90"] = [float(np.quantile(ratio, 0.1)), float(np.quantile(ratio, 0.9))]
    out["line"] = {"slope": float(a), "intercept": float(b),
                   "r2_where_measured": float(np.corrcoef(geo[measured], d[measured])[0, 1] ** 2)}

    # 3. Tears by how many landmarks reach a point, and by whether a reach edge passes next to it.
    count = reached.sum(axis=0)
    edge = (reached[:, idx[:, 1:]] != reached[:, :, None]).any(axis=(0, 2))
    out["torn_by_reach_count"] = {int(c): [int((count == c).sum()), float(t[count == c].mean())]
                                  for c in np.unique(count)}
    out["torn_next_to_edge"] = float(t[edge].mean()) if edge.any() else None
    out["torn_away_from_edge"] = float(t[~edge].mean())
    out["share_next_to_edge"] = float(edge.mean())
    (HERE / f"characterise_L{args.landmarks}_{args.missing}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
