import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from lvm.cells import assign_cells
from lvm.region import RegionModel

MIN_PINS = 10

Transform = tuple[float, np.ndarray, np.ndarray]


def second_regions(root: RegionModel, X: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Each point's region and second region: the strongest and the second-strongest landmark at its level-1
    cell."""
    cell = assign_cells(torch.as_tensor(X, device=device), root.centroids)
    top = torch.topk(root.maps[:, cell], 2, dim=0).indices.cpu().numpy()
    return top[0], top[1]


def choose_pins(first: np.ndarray, second: np.ndarray, *, per_pair: int,
                seed: int) -> dict[tuple[int, int], np.ndarray]:
    """For each pair of neighbouring regions (C < D): up to ``per_pair`` points chosen at random among its border
    points, those whose two strongest regions are C and D. Pairs with fewer than MIN_PINS border points are not
    neighbours."""
    rng, pairs = np.random.default_rng(seed), np.sort(np.column_stack([first, second]), axis=1)
    pins = {}
    for c, d in np.unique(pairs, axis=0):
        border = np.flatnonzero((pairs[:, 0] == c) & (pairs[:, 1] == d))
        if len(border) >= MIN_PINS:
            pins[(int(c), int(d))] = np.sort(rng.choice(border, min(per_pair, len(border)), replace=False))
    return pins


def pins_by_region(pins: dict[tuple[int, int], np.ndarray]) -> dict[int, np.ndarray]:
    """Each region's pins: those of every pair it belongs to."""
    regions = {r for pair in pins for r in pair}
    return {r: np.unique(np.concatenate([ids for pair, ids in pins.items() if r in pair])) for r in regions}


def overlapping_membership(root: RegionModel, ratio: float) -> np.ndarray:
    """Which level-1 cells belong to which regions (regions x cells): cell c belongs to region j if j's landmark's
    voltage there is at least ``ratio`` times the strongest landmark's; with ratio 1, the strongest only."""
    return (root.maps >= ratio * root.maps.max(dim=0).values).cpu().numpy()


def overlap_pins(root: RegionModel, cell: np.ndarray, membership: np.ndarray,
                 ratio: float) -> dict[tuple[int, int], np.ndarray]:
    """For each pair of regions C < D that share cells: the points in their shared cells where their two landmarks'
    voltages are nearly equal, the weaker at least ``ratio`` times the stronger (the middle of the overlap). Pairs
    with fewer than MIN_PINS such points are left out."""
    maps, pins = root.maps.cpu().numpy(), {}
    for c in range(len(membership)):
        for d in range(c + 1, len(membership)):
            balanced = (membership[c] & membership[d]
                        & (np.minimum(maps[c], maps[d]) >= ratio * np.maximum(maps[c], maps[d])))
            rows = np.flatnonzero(balanced[cell])
            if len(rows) >= MIN_PINS:
                pins[(c, d)] = rows
    return pins


def align(source: np.ndarray, target: np.ndarray, scale: float | None = None) -> tuple[Transform, float]:
    """The similarity transform (s, R, t) that best maps ``source`` onto ``target``, y = s x R + t with R a rotation
    or reflection (orthogonal Procrustes with scaling; with ``scale`` given, s is that and only R and t are fitted);
    and its misfit, the root mean square residual relative to the target's root mean square spread."""
    source_mean, target_mean = source.mean(0), target.mean(0)
    a, b = source - source_mean, target - target_mean
    U, S, Vt = np.linalg.svd(a.T @ b)
    R = U @ Vt
    s = S.sum() / np.square(a).sum() if scale is None else scale
    misfit = np.sqrt(np.square(s * a @ R - b).sum() / np.square(b).sum())
    return (s, R, target_mean - s * source_mean @ R), float(misfit)


def pin_coordinates(placed: dict[int, dict[str, np.ndarray]], r: int, ids: np.ndarray) -> np.ndarray:
    """The coordinates in region r's frame of the pins ``ids``."""
    return placed[r]["level2_chart"][np.searchsorted(placed[r]["ids"], ids)]


def pair_transforms(pins: dict[tuple[int, int], np.ndarray],
                    placed: dict[int, dict[str, np.ndarray]]) -> tuple[dict, dict]:
    """For each pair of neighbouring regions, both ways: the transform from one region's frame into the other's,
    fitted on their shared pins, and its misfit."""
    transforms, misfits = {}, {}
    for (c, d), ids in pins.items():
        in_c, in_d = pin_coordinates(placed, c, ids), pin_coordinates(placed, d, ids)
        transforms[(d, c)], misfits[(d, c)] = align(in_d, in_c)
        transforms[(c, d)], misfits[(c, d)] = align(in_c, in_d)
    return transforms, misfits


def routes(costs: dict[tuple[int, int], float], regions: list[int], reference: int) -> dict[int, list[int]]:
    """Each region's route to the reference region: the cheapest chain of neighbouring regions, from the region to
    the reference (shortest paths; ``costs`` has one cost per pair of neighbours, either way round)."""
    index = {r: i for i, r in enumerate(regions)}
    rows, columns, weights = zip(*((index[c], index[d], cost + 1e-12) for (c, d), cost in costs.items()))
    graph = csr_matrix((weights, (rows, columns)), shape=(len(regions), len(regions)))
    _, predecessors = dijkstra(graph, directed=False, indices=index[reference], return_predecessors=True)
    found = {}
    for r in regions:
        route, i = [r], index[r]
        while regions[i] != reference:
            if predecessors[i] < 0:
                raise ValueError(f"Region {r} has no route to the reference region {reference}")
            i = predecessors[i]
            route.append(regions[i])
        found[r] = route
    return found


def compose(transforms: dict[tuple[int, int], Transform], route: list[int]) -> Transform:
    """The transform from the first region's frame into the last's, step by step along the route."""
    k = next(iter(transforms.values()))[1].shape[0]
    s, R, t = 1.0, np.eye(k), np.zeros(k)
    for here, there in zip(route, route[1:]):
        s_step, R_step, t_step = transforms[(here, there)]
        s, R, t = s * s_step, R @ R_step, s_step * t @ R_step + t_step
    return s, R, t


def global_coordinates(Z: np.ndarray, region: np.ndarray, transforms: dict[tuple[int, int], Transform],
                       found: dict[int, list[int]]) -> np.ndarray:
    """Every region's points in the reference frame: its own coordinates moved along its route."""
    placed = np.empty_like(Z)
    for r, route in found.items():
        inside = region == r
        s, R, t = compose(transforms, route)
        placed[inside] = s * Z[inside] @ R + t
    return placed
