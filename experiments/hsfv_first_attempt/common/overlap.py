import shutil
from pathlib import Path

import numpy as np
import torch

from experiments.hsfv_first_attempt.common.chaining import (calibration, cells_to_landmarks, chained_voltage, coordinates, extend,
                                              frame, to_landmarks)
from lvm.cells import assign_cells
from lvm.config import load_config
from lvm.data import as_chunks
from lvm.region import fit_region, load_region, save_region

MIN_POINTS = 100


def memberships(root, points: torch.Tensor, m: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Each point's m regions: the m landmarks with the highest voltage at its nearest root cell, strongest first,
    and those voltages (the point's strength in each region)."""
    cells = assign_cells(points, root.centroids.to(points.dtype))
    strength, regions = torch.topk(root.maps.T[cells], m, dim=1)
    return regions, strength


def fit_overlapping(X: np.ndarray, root, *, config: dict, m: int, seed: int, folder: Path) -> None:
    """Level 2 with overlapping regions: region r holds every point that has r among its m regions, and its cells
    are split from every root cell that has r among its m strongest landmarks. Saves each region's model, and each
    point's regions and strengths. Regions with fewer than MIN_POINTS points get no level 2."""
    if folder.exists():
        shutil.rmtree(folder)
    (folder / "level2").mkdir(parents=True)
    points = torch.as_tensor(X, device=root.centroids.device)
    regions, strength = memberships(root, points, m)
    cell_regions = torch.topk(root.maps.T, m, dim=1).indices
    region_config = load_config(overrides={**config, "embedding": {"local_chart": "none"}})
    for r in range(len(root.landmarks)):
        members = torch.nonzero((regions == r).any(dim=1)).squeeze(1).cpu().numpy()
        if len(members) < MIN_POINTS:
            continue
        model = fit_region(as_chunks(X[members], region_config.data.chunk_size), region_config,
                           device=str(root.centroids.device), region_id=(r,), seed=seed,
                           parents=root.centroids[(cell_regions == r).any(dim=1)])
        save_region(model, folder / "level2" / f"region_{r}.npz")
    np.save(folder / "regions.npy", regions.cpu().numpy())
    np.save(folder / "strengths.npy", strength.cpu().numpy())


def load_overlapping(folder: Path, device: str) -> tuple[dict, np.ndarray, np.ndarray]:
    level2 = {int(p.stem.split("_")[1]): load_region(p, device) for p in (folder / "level2").glob("region_*.npz")}
    return level2, np.load(folder / "regions.npy"), np.load(folder / "strengths.npy")


def within_cell(region, points: torch.Tensor, config, *, local_map: str = "bounded", fill: float = 0.5,
                directions: str = "fixed", max_directions: int = 8, block: int = 2_000,
                cell_coords: torch.Tensor | None = None, pca_points: torch.Tensor | None = None) -> torch.Tensor:
    """(m, q) points' coordinates in the region's full frame, each placed inside its cell: the cell's coordinate plus
    the point's offset from the cell's mean, taken in the cell's k principal directions and mapped into the frame.
    How many directions: fixed: k = round(d̂), at most ``max_directions`` | dimension: k = round(d̂) | per_cell:
    k_c = min(round(d̂), the cell's points - 1, its graph neighbours), what each cell can support.
    bounded: a rotation R_c (orthogonal Procrustes, k x q) turns the directions toward how the cell's graph
    neighbours lie from it in the frame, and one scale s_c puts the cell's median point at ``fill`` x the distance to
    the nearest other cell: offset = s_c u R_c (like the chart's).
    linear: a least-squares matrix fitted on the neighbours (unbounded; overshoots in high dimension).
    Cells with too few points or neighbours place their points at the cell's coordinate. ``cell_coords``: the
    cells' coordinates in another frame (e.g. the global one) to place the points in, instead of the region's own.
    ``pca_points``: the points each cell's mean and directions are estimated from (e.g. more points streamed into the
    cells), instead of the points being placed."""
    from lvm.embedding import _principal_directions
    from lvm.graph import build_graph

    if cell_coords is None:
        cell_coords = coordinates(cells_to_landmarks(region), frame(region))
    cell_coords = cell_coords.double()
    C = region.centroids.double()
    kernel, _ = build_graph(region.centroids, None, config=config, device=str(C.device), seed=0)
    W = (kernel > 0).double()
    cells = assign_cells(points, region.centroids.to(points.dtype))
    counts = torch.bincount(cells, minlength=len(C))
    if pca_points is None:
        pca_points, pca_cells, pca_counts = points, cells, counts
    else:
        pca_cells = assign_cells(pca_points, region.centroids.to(pca_points.dtype))
        pca_counts = torch.bincount(pca_cells, minlength=len(C))
    neighbours = W.sum(dim=1)
    match directions:
        case "fixed":
            k = max(1, min(int(round(region.dimension)), max_directions))
            k_cell = torch.full_like(counts, k)
        case "dimension":
            k = max(1, int(round(region.dimension)))
            k_cell = torch.full_like(counts, k)
        case "per_cell":
            k = max(1, int(round(region.dimension)))
            k_cell = torch.minimum(torch.minimum(torch.full_like(counts, k), pca_counts - 1),
                                   neighbours.long()).clamp(min=0)
    means, axes = _principal_directions(pca_points.double(), pca_cells, pca_counts, k, 256)
    kept = (torch.arange(k, device=C.device)[None, :] < k_cell[:, None]).double()      # (c, k) directions in use
    axes = axes * kept[:, None, :]
    U = torch.einsum("jd,cdk->cjk", C, axes) - torch.einsum("cd,cdk->ck", C, axes)[:, None, :]
    dY = cell_coords[None, :, :] - cell_coords[:, None, :]
    products = (U * W[:, :, None]).transpose(1, 2) @ dY                     # (c, k, q)
    local = torch.cat([torch.einsum("nd,ndk->nk", points[s:s + block].double() - means[cells[s:s + block]],
                                    axes[cells[s:s + block]]) for s in range(0, len(points), block)])
    match local_map:
        case "bounded":
            A, _, Vt = torch.linalg.svd(products, full_matrices=False)
            M = A @ Vt                                                      # rotation, orthonormal rows
            radius = local.norm(dim=1)
            order = torch.argsort(radius)
            order = order[torch.argsort(cells[order], stable=True)]
            first = torch.cumsum(counts, 0) - counts
            median = torch.zeros(len(C), dtype=radius.dtype, device=C.device)
            occupied = counts > 0
            median[occupied] = radius[order[(first + counts // 2)[occupied]]]
            gaps = torch.cdist(cell_coords, cell_coords)
            gaps.fill_diagonal_(torch.inf)
            scale = torch.where(median > 0, fill * gaps.min(dim=1).values / median.clamp(min=1e-300), 0.0)
            M = scale[:, None, None] * M
        case "linear":
            eye = torch.eye(k, dtype=C.dtype, device=C.device)
            M = torch.linalg.solve((U * W[:, :, None]).transpose(1, 2) @ U + 1e-9 * eye, products)
    usable = (neighbours >= 1) & (k_cell >= 1) & (pca_counts > k_cell)
    offset = torch.einsum("nk,nkq->nq", local, M[cells])
    return cell_coords[cells] + torch.where(usable[cells, None], offset, 0.0)


def build_state(X: np.ndarray, root, level2: dict, regions: np.ndarray, strengths: np.ndarray, config, *,
                within: bool = False, directions: str = "fixed", scales: dict | None = None) -> dict:
    """What pair distances need, for every point: its calibrated coordinates in each region it belongs to (the
    region's full frame, in the root's unit), its strength there, and two fall-back positions in the root's frame:
    level 1 alone, and the voltage-weighted chained distances (combined over the point's regions by strength).
    With ``within``, points get their position inside their cell (within_cell), in the regions and at level 1."""
    device = root.centroids.device
    points = torch.as_tensor(X, device=device)
    regions_t = torch.as_tensor(regions, device=device)
    strengths_t = torch.as_tensor(strengths, device=device, dtype=torch.float64)
    root_frame = frame(root)
    direct = to_landmarks(root, points, config=config).double()
    chained, weight = torch.zeros_like(direct), torch.zeros(len(points), 1, dtype=torch.float64, device=device)
    per_region = {}
    for r, child in level2.items():
        member = (regions_t == r).any(dim=1)
        members = torch.nonzero(member).squeeze(1)
        s = scales[r] if scales else calibration(child, root, config=config)
        local = s * (within_cell(child, points[members], config, directions=directions) if within else
                     coordinates(to_landmarks(child, points[members], config=config), frame(child)).double())
        strength = (strengths_t * (regions_t == r)).sum(dim=1)
        position = torch.full((len(points),), -1, dtype=torch.long, device=device)
        position[members] = torch.arange(len(members), device=device)
        per_region[r] = {"members": members, "coords": local, "strength": strength, "position": position}
        far = extend(child, points[members], chained_voltage(child, root, config=config), config=config).double()
        chained[members] += strength[members, None] * far
        weight[members] += strength[members, None]
    chained = torch.where(weight > 0, chained / weight.clamp(min=1e-300), direct)
    level1 = within_cell(root, points, config, directions=directions) if within else coordinates(direct, root_frame)
    return {"regions": per_region, "level1": level1, "chained": coordinates(chained, root_frame)}


def distances(state: dict, queries: torch.Tensor, candidates: torch.Tensor, rule: str) -> torch.Tensor:
    """(len(queries), len(candidates)) distances. A pair sharing regions gets level 2's distance from them: from the
    region where the pair is jointly strongest ("strongest"), or averaged with those strengths as weights
    ("weighted"); the pair's strength in a region is the smaller of its two points' strengths there. A pair sharing
    none gets the distance between the points' chained positions."""
    base = torch.cdist(state["chained"][queries], state["chained"][candidates])
    numerator, denominator = torch.zeros_like(base), torch.zeros_like(base)
    best_strength, best = torch.full_like(base, -1.0), torch.zeros_like(base)
    for region in state["regions"].values():
        q_pos, c_pos = region["position"][queries], region["position"][candidates]
        q_in, c_in = torch.nonzero(q_pos >= 0).squeeze(1), torch.nonzero(c_pos >= 0).squeeze(1)
        if len(q_in) == 0 or len(c_in) == 0:
            continue
        d = torch.cdist(region["coords"][q_pos[q_in]], region["coords"][c_pos[c_in]])
        w = torch.minimum(region["strength"][queries[q_in]][:, None], region["strength"][candidates[c_in]][None, :])
        rows, cols = q_in[:, None], c_in[None, :]
        numerator[rows, cols] += w * d
        denominator[rows, cols] += w
        stronger = w > best_strength[rows, cols]
        best[rows, cols] = torch.where(stronger, d, best[rows, cols])
        best_strength[rows, cols] = torch.where(stronger, w, best_strength[rows, cols])
    shared = denominator > 0
    match rule:
        case "weighted":
            return torch.where(shared, numerator / denominator.clamp(min=1e-300), base)
        case "strongest":
            return torch.where(shared, best, base)


def level1_distances(state: dict, queries: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    return torch.cdist(state["level1"][queries], state["level1"][candidates])
