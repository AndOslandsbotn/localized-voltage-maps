import numpy as np
import torch

from experiments.hsfv_first_attempt.common.chaining import (calibration, cells_to_landmarks, coordinates, extend, frame,
                                              to_landmarks)
from lvm.embedding import classical_mds, triangulate_distances


def graph_calibration(root, level2: dict, points: torch.Tensor, regions: np.ndarray, config, *,
                      max_points: int = 300, seed: int = 0) -> tuple[dict, dict]:
    """Each region's unit from its neighbours, not from the root. Two regions that share points must agree on the
    distances between them: s_A d_A(x, y) = s_B d_B(x, y), so log s_A - log s_B = log r_AB with r_AB the median of
    d_B / d_A over pairs of shared points (each region's distances in its own full frame). All log s_R are fitted at
    once by least squares weighted by the shared points (a graph-Laplacian system); the one free overall factor is
    set so the scales agree with the root's calibrations on average (geometric mean). Returns the scales and, per
    region pair, the disagreement on shared pairs (median relative difference) with the root's and the new scales."""
    keys = sorted(level2)
    device = points.device
    regions_t = torch.as_tensor(regions, device=device)
    rng = np.random.default_rng(seed)
    root_scale = {r: calibration(level2[r], root, config=config) for r in keys}
    frames = {r: frame(level2[r]) for r in keys}
    pairs = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            shared = torch.nonzero((regions_t == a).any(dim=1) & (regions_t == b).any(dim=1)).squeeze(1)
            if len(shared) < 3:
                continue
            pick = shared[torch.as_tensor(rng.choice(len(shared), min(max_points, len(shared)), replace=False),
                                          device=device)]
            ya = coordinates(to_landmarks(level2[a], points[pick], config=config), frames[a]).double()
            yb = coordinates(to_landmarks(level2[b], points[pick], config=config), frames[b]).double()
            da, db = torch.pdist(ya), torch.pdist(yb)
            use = (da > 0) & (db > 0)
            pairs.append((a, b, float(torch.log((db[use] / da[use]).median())), len(shared), da[use], db[use]))
    index = {r: i for i, r in enumerate(keys)}
    L = np.zeros((len(keys), len(keys)))
    rhs = np.zeros(len(keys))
    for a, b, log_r, w, _, _ in pairs:
        i, j = index[a], index[b]
        L[i, i] += w; L[j, j] += w; L[i, j] -= w; L[j, i] -= w
        rhs[i] += w * log_r; rhs[j] -= w * log_r
    log_s = np.linalg.pinv(L) @ rhs
    shift = np.mean([np.log(root_scale[r]) for r in keys]) - np.mean(log_s)
    scale = {r: float(np.exp(log_s[index[r]] + shift)) for r in keys}
    agreement = []
    for a, b, _, w, da, db in pairs:
        for name, s in (("root", root_scale), ("graph", scale)):
            agreement.append({"pair": [a, b], "shared": w, "scales": name,
                              "disagreement": float(((s[a] * da - s[b] * db).abs() / (0.5 * (s[a] * da + s[b] * db))).median())})
    return scale, {"agreement": agreement, "root_scale": root_scale}


def region_paths(region, config) -> np.ndarray:
    """(k, n_cells) shortest-path lengths from each landmark to every cell through the region's own cell graph (edges
    as long as the distance between cell centres)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    from lvm.graph import build_graph

    kernel, _ = build_graph(region.centroids, None, config=config, device=str(region.centroids.device), seed=0)
    C = region.centroids.double()
    linked = (kernel > 0).cpu().numpy()
    np.fill_diagonal(linked, False)
    graph = csr_matrix(np.where(linked, torch.cdist(C, C).cpu().numpy(), 0.0))
    return dijkstra(graph, directed=False, indices=region.landmarks.cpu().numpy())


def geodesic_curves(region, config) -> list:
    """One monotone curve per landmark, from its voltage distance -log v to length along the region's own cell graph
    (edges as long as the distance between cell centres): isotonic regression of the shortest-path length from the
    landmark to each cell on the cell's voltage distance. Applied to a region's distances, it turns them into
    data-space lengths, the same unit in every region, with each landmark's own stretching undone."""
    from sklearn.isotonic import IsotonicRegression

    paths = region_paths(region, config)
    voltage = cells_to_landmarks(region).double().cpu().numpy()
    curves = []
    for i in range(len(region.landmarks)):
        x, y = voltage[:, i], paths[i]
        ok = np.isfinite(x) & np.isfinite(y)
        curves.append(IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x[ok], y[ok]))
    return curves


def in_lengths(curves: list, distances: torch.Tensor) -> torch.Tensor:
    """(m, k) voltage distances to a region's k landmarks turned into lengths, column by column."""
    d = distances.double().cpu().numpy()
    out = np.column_stack([curve.predict(d[:, i]) for i, curve in enumerate(curves)])
    return torch.as_tensor(out, device=distances.device)


def landmark_lengths(region, curves: list) -> torch.Tensor:
    """(k, k) the region's landmark distances in lengths: each pair through both landmarks' curves, averaged."""
    D = region.embedding.distances.double()
    a = in_lengths(curves, D)                      # column j through landmark j's curve
    G = 0.5 * (a + a.T)
    return G.fill_diagonal_(0.0)


def landmark_graph(root, level2: dict, points: torch.Tensor, regions: np.ndarray, config,
                   scale: dict | None = None, curves: dict | None = None, paths: dict | None = None,
                   bridge: float | None = None) -> dict:
    """A graph over all level-2 landmarks, in the root's unit (``bridge``: the quantile of the relays over shared
    points used for a cross-region edge instead of their minimum, for robustness to noise). Within a region: the region's own landmark distances
    times its calibration. Between two regions: relayed through the points that belong to both,
    d(l_A, l_B) = min over shared x of s_A d_A(x, l_A) + s_B d_B(x, l_B). Shortest paths give every pair of
    landmarks a distance built from these fine hops."""
    device = points.device
    keys = sorted(level2)
    sizes = [len(level2[r].landmarks) for r in keys]
    offset = dict(zip(keys, np.cumsum([0] + sizes[:-1])))
    n = sum(sizes)
    scale = ({r: 1.0 for r in keys} if curves or paths else
             scale or {r: calibration(level2[r], root, config=config) for r in keys})
    regions_t = torch.as_tensor(regions, device=device)
    member = {r: torch.nonzero((regions_t == r).any(dim=1)).squeeze(1) for r in keys}
    own = {}
    for r in keys:
        if paths:
            cells = torch.as_tensor(paths[r].T, device=device)
            d = extend(level2[r], points[member[r]], cells, config=config).double()
        else:
            d = to_landmarks(level2[r], points[member[r]], config=config).double()
            d = in_lengths(curves[r], d) if curves else d * scale[r]
        own[r] = torch.full((len(points), len(level2[r].landmarks)), torch.inf, dtype=torch.float64, device=device)
        own[r][member[r]] = d
    D = torch.full((n, n), torch.inf, dtype=torch.float64, device=device)
    for r in keys:
        block = slice(offset[r], offset[r] + len(level2[r].landmarks))
        if paths:
            P = torch.as_tensor(paths[r][:, level2[r].landmarks.cpu().numpy()], device=device)
            D[block, block] = 0.5 * (P + P.T)
        else:
            D[block, block] = (landmark_lengths(level2[r], curves[r]) if curves else
                               scale[r] * level2[r].embedding.distances.double())
    bridges = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            shared = torch.nonzero((regions_t == a).any(dim=1) & (regions_t == b).any(dim=1)).squeeze(1)
            bridges[(a, b)] = len(shared)
            if len(shared) == 0:
                continue
            relays = own[a][shared][:, :, None] + own[b][shared][:, None, :]
            edge = relays.min(dim=0).values if bridge is None else torch.quantile(relays, bridge, dim=0)
            sa = slice(offset[a], offset[a] + sizes[keys.index(a)])
            sb = slice(offset[b], offset[b] + sizes[keys.index(b)])
            D[sa, sb] = torch.minimum(D[sa, sb], edge)
            D[sb, sa] = torch.minimum(D[sb, sa], edge.T)
    direct = D.clone()
    for via in range(n):
        D = torch.minimum(D, D[:, via, None] + D[None, via, :])
    return {"keys": keys, "offset": offset, "sizes": sizes, "scale": scale, "own": own, "member": member,
            "direct": direct, "paths": D, "bridges": bridges}


def own_to_nodes(graph: dict, rows: torch.Tensor) -> torch.Tensor:
    """(m, n) each point's distance to its own regions' landmarks (inf at all other nodes)."""
    n = sum(graph["sizes"])
    out = torch.full((len(rows), n), torch.inf, dtype=torch.float64, device=rows.device)
    for r in graph["keys"]:
        o, k = graph["offset"][r], graph["sizes"][graph["keys"].index(r)]
        out[:, o:o + k] = graph["own"][r][rows]
    return out


def route_distances(graph: dict, queries: torch.Tensor, candidates: torch.Tensor, block: int = 200) -> torch.Tensor:
    """d(x, y) = min over x's landmarks l_i and y's landmarks l_j of d(x, l_i) + D(l_i, l_j) + d(l_j, y)."""
    q_own, c_own = own_to_nodes(graph, queries), own_to_nodes(graph, candidates)
    to_nodes = (q_own[:, :, None] + graph["paths"][None, :, :]).min(dim=1).values           # (Q, n) via x's landmarks
    rows = []
    for start in range(0, len(queries), block):
        rows.append((to_nodes[start:start + block, None, :] + c_own[None, :, :]).min(dim=2).values)
    return torch.cat(rows)


def global_coordinates(graph: dict, level2: dict, points: torch.Tensor, regions: np.ndarray,
                       strengths: np.ndarray, rows: torch.Tensor, config) -> torch.Tensor:
    """Points placed in one global frame: Landmark MDS over all level-2 landmarks with the graph's distances; a point's
    distance to every landmark is the voltage-weighted average over its own regions' landmarks,
    sum_i v_i(x) D(l_i, L) / sum_i v_i(x), combined over its regions by strength."""
    n = sum(graph["sizes"])
    mean_sq, projection = classical_mds(graph["paths"], n)
    total = torch.zeros((len(rows), n), dtype=torch.float64, device=rows.device)
    weight = torch.zeros((len(rows), 1), dtype=torch.float64, device=rows.device)
    regions_t = torch.as_tensor(regions[rows.cpu().numpy()], device=rows.device)
    strengths_t = torch.as_tensor(strengths[rows.cpu().numpy()], device=rows.device, dtype=torch.float64)
    for r in graph["keys"]:
        mine = torch.nonzero((regions_t == r).any(dim=1)).squeeze(1)
        if len(mine) == 0:
            continue
        child = level2[r]
        v = extend(child, points[rows[mine]], child.maps.T, config=config).double()          # (m, k) voltages
        o, k = graph["offset"][r], graph["sizes"][graph["keys"].index(r)]
        d = v @ graph["paths"][o:o + k] / v.sum(dim=1, keepdim=True)
        s = (strengths_t[mine] * (regions_t[mine] == r)).sum(dim=1, keepdim=True)
        total[mine] += s * d
        weight[mine] += s
    return triangulate_distances(total / weight.clamp(min=1e-300), mean_sq, projection)


def length_state(level2: dict, curves: dict, points: torch.Tensor, regions: np.ndarray, strengths: np.ndarray,
                 config) -> dict:
    """Like overlap.build_state's regions, in lengths: each point's coordinates in each of its regions' frames (Landmark
    MDS over the region's landmark lengths), for level 2's distances between points that share a region."""
    device = points.device
    regions_t = torch.as_tensor(regions, device=device)
    strengths_t = torch.as_tensor(strengths, device=device, dtype=torch.float64)
    per_region = {}
    for r, child in level2.items():
        members = torch.nonzero((regions_t == r).any(dim=1)).squeeze(1)
        mean_sq, projection = classical_mds(landmark_lengths(child, curves[r]), len(child.landmarks))
        d = in_lengths(curves[r], to_landmarks(child, points[members], config=config))
        position = torch.full((len(points),), -1, dtype=torch.long, device=device)
        position[members] = torch.arange(len(members), device=device)
        per_region[r] = {"members": members, "coords": triangulate_distances(d, mean_sq, projection),
                         "strength": (strengths_t * (regions_t == r)).sum(dim=1), "position": position}
    return {"regions": per_region, "chained": torch.zeros((len(points), 1), dtype=torch.float64, device=device),
            "level1": torch.zeros((len(points), 1), dtype=torch.float64, device=device)}


def overlap_disagreement(level2: dict, curves: dict, points: torch.Tensor, regions: np.ndarray, config, *,
                         max_points: int = 300, seed: int = 0) -> list:
    """Per pair of regions sharing points: the median relative difference between the two regions' distances (in
    lengths, each in its own frame) for pairs of shared points."""
    keys = sorted(level2)
    regions_t = torch.as_tensor(regions, device=points.device)
    rng = np.random.default_rng(seed)
    frames = {r: classical_mds(landmark_lengths(level2[r], curves[r]), len(level2[r].landmarks)) for r in keys}
    out = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            shared = torch.nonzero((regions_t == a).any(dim=1) & (regions_t == b).any(dim=1)).squeeze(1)
            if len(shared) < 3:
                continue
            pick = shared[torch.as_tensor(rng.choice(len(shared), min(max_points, len(shared)), replace=False),
                                          device=points.device)]
            ya = triangulate_distances(in_lengths(curves[a], to_landmarks(level2[a], points[pick], config=config)), *frames[a])
            yb = triangulate_distances(in_lengths(curves[b], to_landmarks(level2[b], points[pick], config=config)), *frames[b])
            da, db = torch.pdist(ya), torch.pdist(yb)
            use = (da > 0) & (db > 0)
            out.append(float(((da[use] - db[use]).abs() / (0.5 * (da[use] + db[use]))).median()))
    return out


def route_coordinates(graph: dict, rows: torch.Tensor) -> torch.Tensor:
    """Points placed in the global frame from their distances to every landmark by the shortest route through their
    own regions' landmarks, min_i d(x, l_i) + D(l_i, L), with no voltage weights (for the graph-path ablation)."""
    n = sum(graph["sizes"])
    mean_sq, projection = classical_mds(graph["paths"], n)
    to_nodes = (own_to_nodes(graph, rows)[:, :, None] + graph["paths"][None, :, :]).min(dim=1).values
    return triangulate_distances(to_nodes, mean_sq, projection)


def global_within(graph: dict, level2: dict, points: torch.Tensor, regions: np.ndarray, config,
                  coords: torch.Tensor, *, directions: str = "per_cell") -> torch.Tensor:
    """All points' global coordinates with their position inside their leaf cell: each level-2 cell is placed in the
    global frame like a point (its voltages to its region's landmarks, averaged through the landmark graph), and each
    point at its cell's place plus the bounded local-PCA offset (overlap.within_cell) in the global frame, using the
    cell of its strongest region. ``coords``: every point's global coordinates without it (kept for points whose
    strongest region has no level 2)."""
    from experiments.hsfv_first_attempt.common.overlap import within_cell

    n = sum(graph["sizes"])
    mean_sq, projection = classical_mds(graph["paths"], n)
    primary = torch.as_tensor(regions[:, 0], device=points.device)
    out = coords.clone()
    for r in graph["keys"]:
        members = torch.nonzero(primary == r).squeeze(1)
        if len(members) == 0:
            continue
        child = level2[r]
        o, k = graph["offset"][r], graph["sizes"][graph["keys"].index(r)]
        v = child.maps.T.double()
        cells = triangulate_distances(v @ graph["paths"][o:o + k] / v.sum(dim=1, keepdim=True), mean_sq, projection)
        out[members] = within_cell(child, points[members], config, directions=directions,
                                   cell_coords=cells).to(out.dtype)
    return out


def global_within_root(root, points: torch.Tensor, coords: torch.Tensor, config, *,
                       directions: str = "dimension") -> torch.Tensor:
    """All points re-placed inside the ROOT's (large) cells, anchored in the hierarchy's global frame: a root cell's
    place is the mean global position of its points, and each point goes to that place plus the bounded local-PCA
    offset from its root cell (overlap.within_cell). Large cells support a rich local PCA; the global frame supplies
    the structure between them."""
    from experiments.hsfv_first_attempt.common.overlap import within_cell
    from lvm.cells import assign_cells

    cells = assign_cells(points, root.centroids.to(points.dtype))
    counts = torch.bincount(cells, minlength=len(root.centroids)).clamp(min=1)
    anchors = torch.zeros((len(root.centroids), coords.shape[1]), dtype=coords.dtype, device=coords.device)
    anchors = anchors.index_add_(0, cells, coords) / counts[:, None].to(coords.dtype)
    return within_cell(root, points, config, directions=directions, cell_coords=anchors).to(coords.dtype)
