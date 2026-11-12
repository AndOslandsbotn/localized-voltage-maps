import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import squareform

from experiments.common.datasets import load
from experiments.common.geodesics import rank_correlation
from experiments.common.guard import watch_memory
from experiments.hsfv_first_attempt.common.landmark_graph import (geodesic_curves, global_coordinates, graph_calibration,
                                                    landmark_graph, length_state, overlap_disagreement,
                                                    route_distances)
from experiments.hsfv_first_attempt.common.overlap import build_state, distances, level1_distances, load_overlapping
from experiments.hsfv_first_attempt.swiss_roll.chained_distances.run import CONFIG, DATASET, N, SEEDS
from experiments.hsfv_first_attempt.swiss_roll.chained_distances.score import GROUPS, SAMPLE, scores, truth
from experiments.hsfv_first_attempt.swiss_roll.overlapping_regions.score import condensed
from lvm import LocalizedVoltageMaps
from lvm.config import load_config
from lvm.region import load_region

HERE = Path(__file__).resolve().parent
MODELS = HERE.parent / "overlapping_regions" / "models"
FINE = HERE.parent / "chained_distances" / "models" / "fine"
DEVICE = "cuda"


def on_surface(P: np.ndarray) -> np.ndarray:
    """Locations as truth() expects them (points on the unit sphere for the half sphere; unchanged otherwise)."""
    return P / np.linalg.norm(P, axis=1, keepdims=True) if DATASET == "half_sphere" else P


def shares_region(regions: np.ndarray, sample: np.ndarray) -> np.ndarray:
    R = regions[sample]
    shared = (R[:, None, :, None] == R[None, :, None, :]).any(axis=(2, 3))
    return squareform(shared.astype(float), checks=False) > 0


def main() -> None:
    """Global distances bottom-up through a graph of all level-2 landmarks (bridged by points in two regions),
    against the coarse root, the top-down combination and a fine root, on the true distances."""
    watch_memory(limit_mb=3000)
    config = load_config(overrides=CONFIG)
    rows, checks = [], []
    for seed in SEEDS:
        X = load("eval", n=N, seed=seed, dataset=DATASET)[0].astype(np.float32)
        points = torch.as_tensor(X, device=DEVICE)
        sample = np.random.default_rng(seed).choice(N, SAMPLE, replace=False)
        S = torch.as_tensor(sample, device=DEVICE)
        true = truth(X[sample])
        root = load_region(MODELS / f"seed{seed}" / "root.npz", DEVICE)
        l2_one, reg_one, str_one = load_overlapping(MODELS / f"seed{seed}" / "m1", DEVICE)
        l2_two, reg_two, str_two = load_overlapping(MODELS / f"seed{seed}" / "m2", DEVICE)
        one = build_state(X, root, l2_one, reg_one, str_one, config, within=True)
        two = build_state(X, root, l2_two, reg_two, str_two, config, within=True)
        fine = LocalizedVoltageMaps.load(FINE / f"seed{seed}" / "level1")
        fine_state = build_state(X, fine.root_, {}, np.zeros((N, 1), dtype=np.int64), np.ones((N, 1)), fine.config_,
                                 within=True)
        graph = landmark_graph(root, l2_two, points, reg_two, config)
        route = condensed(route_distances(graph, S, S))
        coords = global_coordinates(graph, l2_two, points, reg_two, str_two, S, config)
        frame = condensed(torch.cdist(coords, coords))
        shared = shares_region(reg_two, sample)
        level2_two = condensed(distances(two, S, S, "weighted"))
        scales, info = graph_calibration(root, l2_two, points, reg_two, config, seed=seed)
        graph_cal = landmark_graph(root, l2_two, points, reg_two, config, scale=scales)
        coords_cal = global_coordinates(graph_cal, l2_two, points, reg_two, str_two, S, config)
        frame_cal = condensed(torch.cdist(coords_cal, coords_cal))
        two_cal = build_state(X, root, l2_two, reg_two, str_two, config, within=True, scales=scales)
        level2_cal = condensed(distances(two_cal, S, S, "weighted"))
        curves = {r: geodesic_curves(l2_two[r], config) for r in l2_two}
        graph_len = landmark_graph(root, l2_two, points, reg_two, config, curves=curves)
        coords_len = global_coordinates(graph_len, l2_two, points, reg_two, str_two, S, config)
        frame_len = condensed(torch.cdist(coords_len, coords_len))
        level2_len = condensed(distances(length_state(l2_two, curves, points, reg_two, str_two, config), S, S,
                                         "weighted"))
        values = overlap_disagreement(l2_two, curves, points, reg_two, config, seed=seed)
        checks.append({"seed": seed, "what": "overlap_disagreement_lengths", "median": float(np.median(values)),
                       "max": float(np.max(values))})
        for name in ("root", "graph"):
            values = [a["disagreement"] for a in info["agreement"] if a["scales"] == name]
            checks.append({"seed": seed, "what": f"overlap_disagreement_{name}_scales", "median": float(np.median(values)),
                           "max": float(np.max(values))})
        dists = {"coarse_level1": condensed(level1_distances(one, S, S)),
                 "coarse_level2": condensed(distances(one, S, S, "weighted")),
                 "graph_route": route, "graph_frame": frame,
                 "graph_route_level2": np.where(shared, level2_two, route),
                 "graph_frame_level2": np.where(shared, level2_two, frame),
                 "graph_frame_calibrated": frame_cal,
                 "graph_frame_level2_calibrated": np.where(shared, level2_cal, frame_cal),
                 "graph_frame_lengths": frame_len,
                 "graph_frame_level2_lengths": np.where(shared, level2_len, frame_len),
                 "fine_level1": condensed(level1_distances(fine_state, S, S))}
        top = reg_one[sample, 0]
        same = squareform((top[:, None] == top[None, :]).astype(float), checks=False) > 0
        thirds = np.quantile(true, [1 / 3, 2 / 3])
        masks = {"all": np.ones_like(same), "within": same, "across": ~same, "short": true <= thirds[0],
                 "middle": (true > thirds[0]) & (true <= thirds[1]), "long": true > thirds[1]}
        for method, dd in dists.items():
            for group in GROUPS:
                rows.append({"seed": seed, "method": method, "group": group, **scores(dd, true, masks[group])})
        nodes = np.concatenate([l2_two[r].centroids[l2_two[r].landmarks].double().cpu().numpy() for r in graph["keys"]])
        node_true = truth(on_surface(nodes))
        region_of_node = np.concatenate([[r] * k for r, k in zip(graph["keys"], graph["sizes"])])
        across = squareform((region_of_node[:, None] != region_of_node[None, :]).astype(float), checks=False) > 0
        for name, matrix in (("bridge_edges", graph["direct"]), ("shortest_paths", graph["paths"])):
            m = matrix.cpu().numpy().copy()
            np.fill_diagonal(m, 0.0)
            est = squareform(np.minimum(m, m.T), checks=False)
            use = across & np.isfinite(est) & (est > 0)
            scale = np.median(node_true[use] / est[use])
            checks.append({"seed": seed, "what": name, "pairs": int(use.sum()),
                           "rank": rank_correlation(est[use], node_true[use]),
                           "relative_error": float(np.median(np.abs(scale * est[use] - node_true[use]) / node_true[use])),
                           "overestimate": float(np.median(est[use] * scale / node_true[use]))})
        checks.append({"seed": seed, "what": "bridges", "region_pairs": len(graph["bridges"]),
                       "without_bridge": sum(1 for v in graph["bridges"].values() if v == 0),
                       "median_shared_points": float(np.median(list(graph["bridges"].values())))})
        print(seed, checks[-5:], flush=True)
    (HERE / "results.json").write_text(json.dumps({"dataset": DATASET, "rows": rows, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
