import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors

from experiments.common.datasets import load, unrolled
from experiments.common.geodesics import rank_correlation
from experiments.common.guard import watch_memory
from experiments.hsfv.common.frames import align
from experiments.hsfv.common.shifted import margins
from experiments.hsfv.common.tree import own_scale, place, split_ratios
from experiments.hsfv.common.windows import neighbour_pairs, window
from experiments.hsfv.swiss_roll.shifted_regions.run import DATASET, N, SEED
from lvm import LocalizedVoltageMaps
from lvm.cells import assign_cells
from lvm.region import load_region
from lvm.voltage import point_voltages

HERE = Path(__file__).resolve().parent
SAMPLE, QUERIES, NEIGHBOURS, WHOLE, BINS = 300_000, 3_000, 30, 2_000, 25
MIN_WEIGHT, SINGLE = 0.05, 0.3
VERSIONS = (1, 2, 3)


def version_positions(version, points, cell, root, root_scale, saved, config, device):
    """The points placed by this version's regions, each region carried into level 1's frame (strategy A with the
    calibrated scale); and each point's margin inside its region of this version."""
    maps = torch.as_tensor(saved[f"version{version}_maps"], device=device)
    region = maps.argmax(dim=0).cpu().numpy()[cell]
    Z = np.empty((len(points), root.embedding.projection.shape[0]))
    for r in np.unique(region):
        child = load_region(HERE / "models" / f"version{version}" / f"region_{r}.npz", device)
        (s, R, t), _ = align(place(child, child.centroids, config), place(root, child.centroids, config),
                             own_scale(child, config) / root_scale)
        inside = np.flatnonzero(region == r)
        Z[inside] = s * place(child, points[torch.as_tensor(inside, device=device)], config) @ R + t
    return Z, margins(maps)[cell], region


def window_positions(points, region, own, root, root_scale, config, device):
    """Version 1's regions with a window over every pair of neighbouring regions (their cells, graphs and landmarks
    together, voltages solved again on the window). A window is placed into level 1's frame either independently
    (onto level 1's placement of its cells, calibrated scale) or aligned (onto its two regions' own placements of
    its cells). Each point's position either blends its own region's map (weight 1 - the largest v_B / v_A over the
    other regions B, with v its interpolated level-1 voltages and A its own region) and each window over its region
    and a neighbour B (weight v_B / v_A), or, single window, takes the window over its region and the B of its
    largest v_B / v_A alone when that exceeds SINGLE, else its own region's map."""
    children = {int(r): load_region(HERE / "models" / "version1" / f"region_{r}.npz", device) for r in np.unique(region)}
    counts = np.bincount(region, minlength=max(children) + 1)
    placed = {}
    for r, child in children.items():
        (s, R, t), _ = align(place(child, child.centroids, config), place(root, child.centroids, config),
                             own_scale(child, config) / root_scale)
        placed[r] = s * place(child, child.centroids, config) @ R + t
    voltages = point_voltages(points, root.centroids, root.masses, root.radius, root.maps, root.rho_g,
                              config=config).cpu().numpy()
    ratio = voltages / voltages[np.arange(len(region)), region][:, None]
    ratio[np.arange(len(region)), region] = 0.0
    ratio = np.clip(ratio, 0.0, 1.0)
    weight_own = 1.0 - ratio.max(axis=1)
    nearest = ratio.argmax(axis=1)
    variants = {placing: {"total": weight_own[:, None] * own, "weights": weight_own.copy(), "single": own.copy()}
                for placing in ("independent", "aligned")}
    for a, b in neighbour_pairs(children):
        joint = window(children[a], children[b], counts[a] / (counts[a] + counts[b]), config)
        own_cells = place(joint, joint.centroids, config)
        transforms = {
            "independent": align(own_cells, place(root, joint.centroids, config),
                                 own_scale(joint, config) / root_scale)[0],
            "aligned": align(own_cells, np.concatenate([placed[a], placed[b]]))[0]}
        for mine, other in ((a, b), (b, a)):
            rows = np.flatnonzero((region == mine) & (ratio[:, other] >= MIN_WEIGHT))
            local = place(joint, points[torch.as_tensor(rows, device=device)], config)
            alone = (nearest[rows] == other) & (ratio[rows, other] > SINGLE)
            for placing, (s, R, t) in transforms.items():
                z = s * local @ R + t
                variants[placing]["total"][rows] += ratio[rows, other][:, None] * z
                variants[placing]["weights"][rows] += ratio[rows, other]
                variants[placing]["single"][rows[alone]] = z[alone]
    result = {}
    for placing, parts in variants.items():
        blended = own.copy()
        positive = parts["weights"] > 0
        blended[positive] = parts["total"][positive] / parts["weights"][positive, None]
        suffix = "" if placing == "independent" else " aligned"
        result |= {f"windows{suffix}": blended, f"single window{suffix}": parts["single"]}
    return result


def main() -> None:
    watch_memory(limit_mb=4000)
    X = load("eval", n=N, seed=SEED, dataset=DATASET)[0].astype(np.float32)
    saved = np.load(HERE / "versions.npz")
    level1 = LocalizedVoltageMaps.load(HERE / "models" / "level1")
    root, config, device = level1.root_, level1.config_, level1.device_
    rng = np.random.default_rng(SEED)
    rows = np.sort(rng.choice(N, SAMPLE, replace=False))
    points = torch.as_tensor(X[rows], device=device)
    truth_positions = unrolled(X[rows].astype(np.float64))
    del X
    cell = assign_cells(points, root.centroids).cpu().numpy()
    root_scale = own_scale(root, config)
    placed = {v: version_positions(v, points, cell, root, root_scale, saved, config, device) for v in VERSIONS}
    weights = np.column_stack([placed[v][1] for v in VERSIONS])
    weights[weights.sum(axis=1) == 0] = 1.0
    stacked = np.stack([placed[v][0] for v in VERSIONS], axis=1)
    maps = {"version 1": placed[1][0],
            "blend": np.einsum("nv,nvk->nk", weights, stacked) / weights.sum(axis=1, keepdims=True),
            "best": stacked[np.arange(SAMPLE), weights.argmax(axis=1)]}
    region1 = placed[1][2]
    maps |= window_positions(points, region1, placed[1][0], root, root_scale, config, device)

    whole_rows = rng.choice(SAMPLE, WHOLE, replace=False)
    i, j = np.triu_indices(WHOLE, 1)
    whole_pairs = np.column_stack([whole_rows[i], whole_rows[j]])
    queries = rng.choice(SAMPLE, QUERIES, replace=False)
    neighbours = NearestNeighbors(n_neighbors=NEIGHBOURS + 1).fit(truth_positions).kneighbors(
        truth_positions[queries], return_distance=False)[:, 1:]
    near_pairs = np.column_stack([np.repeat(queries, NEIGHBOURS), neighbours.ravel()])
    pairs = np.concatenate([whole_pairs, near_pairs])
    whole = np.concatenate([np.ones(len(whole_pairs), bool), np.zeros(len(near_pairs), bool)])
    truth = np.linalg.norm(truth_positions[pairs[:, 0]] - truth_positions[pairs[:, 1]], axis=1)
    keep = truth > 0
    pairs, whole, truth = pairs[keep], whole[keep], truth[keep]
    straddle = region1[pairs[:, 0]] != region1[pairs[:, 1]]
    edges = np.geomspace(np.quantile(truth, 0.001), truth.max() * 1.000001, BINS + 1)

    exact_whole = pdist(truth_positions[whole_rows])
    near = ~whole
    results = {"n": N, "seed": SEED, "sample": SAMPLE, "straddling_share_near": float(straddle[near].mean()),
               "distance": np.sqrt(edges[:-1] * edges[1:]).tolist(), "maps": {}}
    for name, Z in maps.items():
        curves = split_ratios(Z, truth, pairs, whole, ~straddle, edges)
        mapped_near = np.linalg.norm(Z[pairs[near, 0]] - Z[pairs[near, 1]], axis=1)
        results["maps"][name] = {
            "same_region": curves["same"], "straddling": curves["different"],
            "rank_whole": rank_correlation(pdist(Z[whole_rows]), exact_whole),
            "rank_near_straddling": rank_correlation(mapped_near[straddle[near]], truth[near][straddle[near]]),
            "rank_near_same": rank_correlation(mapped_near[~straddle[near]], truth[near][~straddle[near]])}
    (HERE / "results.json").write_text(json.dumps(results, indent=2))
    np.savez(HERE / "positions.npz", rows=rows, truth=truth_positions, region1=region1,
             **{name.replace(" ", "_"): Z for name, Z in maps.items()})
    print(f"near pairs straddling a version-1 border: {results['straddling_share_near']:.3f}")
    shown = range(0, BINS, 3)
    print(f"{'':12s} distance " + " ".join(f"{results['distance'][b]:7.3g}" for b in shown))
    for name, values in results["maps"].items():
        for kind in ("straddling", "same_region"):
            print(f"{name:12s} {kind:11s}" + " ".join("    -  " if values[kind][b] is None else f"{values[kind][b]:7.2f}"
                                                      for b in shown))
        print(f"{'':12s} rank whole {values['rank_whole']:.3f} | near straddling {values['rank_near_straddling']:.3f}"
              f" | near same region {values['rank_near_same']:.3f}")


if __name__ == "__main__":
    main()
