from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy.spatial.distance import pdist

from experiments.common.geodesics import rank_correlation
from experiments.hsfv.common.directions import own_edges
from experiments.hsfv.common.frames import Transform, align
from experiments.hsfv.common.levels import MIN_FOR_DIMENSION
from lvm import LocalizedVoltageMaps
from lvm.cells import assign_cells
from lvm.compute import TORCH_DATA_DTYPE
from lvm.config import Config, load_config
from lvm.data import as_chunks
from lvm.region import RegionModel, fit_region, load_region, save_region

Level = dict[tuple[int, ...], tuple[RegionModel, np.ndarray]]


def fit_tree(X: np.ndarray, *, config: dict, seed: int, depth: int, models: Path) -> tuple[list[Level], Config, str]:
    """Paper 1's LVM applied recursively, without charts: level 1 on all points, then the same LVM again inside each
    region (landmark basin of the cells) of the level above, its cells split from the region's cells, down to
    ``depth`` levels. Each region is fitted only from its own points and its parent's cells. Every region is saved;
    returns, per level, each region (by its path of region numbers) with the rows of its points."""
    level1 = LocalizedVoltageMaps(local_chart="none", random_state=seed, config=config).fit(X)
    level1.save(models / "level1")
    device, levels = level1.device_, [{(): (level1.root_, np.arange(len(X)))}]
    for level in range(2, depth + 1):
        children = {}
        for parent_id, (parent, rows) in levels[-1].items():
            children |= fit_children(X, parent, parent_id, rows, parent.maps.argmax(dim=0), config=config, seed=seed,
                                     folder=models / f"level{level}", device=device)
        levels.append(children)
    return levels, level1.config_, device


def fit_children(X: np.ndarray, parent: RegionModel, parent_id: tuple[int, ...], rows: np.ndarray,
                 region_of_cell: torch.Tensor, *, config: dict, seed: int, folder: Path, device: str) -> Level:
    """The parent's children: for each region (``region_of_cell`` gives the region of each of the parent's cells),
    the same LVM fitted on the points of its cells, its cells split from them; each saved in ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    basin = region_of_cell[assign_cells(torch.as_tensor(X[rows], device=device), parent.centroids)].cpu().numpy()
    children = {}
    for r in np.unique(basin):
        child_rows, child_id = rows[basin == r], parent_id + (int(r),)
        dimension = ({} if len(child_rows) >= MIN_FOR_DIMENSION
                     else {"dimension": {"strategy": "fixed", "fixed": {"d": parent.dimension}}})
        settings = load_config(config, {**dimension, "embedding": {"local_chart": "none"}})
        child = fit_region(as_chunks(X[child_rows], settings.data.chunk_size), settings, device=device,
                           region_id=child_id, seed=seed, parents=parent.centroids[region_of_cell == int(r)])
        save_region(child, folder / f"region_{'-'.join(map(str, child_id))}.npz")
        children[child_id] = (child, child_rows)
    return children


def place(region: RegionModel, points: torch.Tensor, config: Config) -> np.ndarray:
    """Points placed by the region's own LVM: voltages to its landmarks, then its landmark MDS."""
    return region.transform_chunk(points.to(TORCH_DATA_DTYPE), config=config).cpu().numpy()


def tree_transforms(levels: list[Level], config: Config,
                    scales: dict[tuple[int, ...], float] | None = None) -> dict[tuple[int, ...], Transform]:
    """Strategy A at every level: each region's transform into its parent's frame, the similarity fit of its own
    cell centres placed by itself onto the same centres placed by its parent. With ``scales`` (each region's map
    units in true units), only rotation and shift are fitted and the scale is the region's over its parent's."""
    transforms = {}
    for level in range(1, len(levels)):
        for region_id, (region, _) in levels[level].items():
            parent = levels[level - 1][region_id[:-1]][0]
            scale = None if scales is None else scales[region_id] / scales[region_id[:-1]]
            transforms[region_id] = align(place(region, region.centroids, config),
                                          place(parent, region.centroids, config), scale)[0]
    return transforms


def own_scales(levels: list[Level], config: Config) -> dict[tuple[int, ...], float]:
    """Each region's map units in true units, from its own cells: over the edges of its graph, the median ratio of
    the data-space distance between the two cell centres (close to the true distance at a cell's scale) to their
    distance in the region's own map."""
    return {region_id: own_scale(region, config) for found in levels for region_id, (region, _) in found.items()}


def own_scale(region: RegionModel, config: Config) -> float:
    """A region's map units in true units: over the edges of its graph, the median ratio of the data-space distance
    between the two cell centres to their distance in the region's own map."""
    edges = own_edges(region.centroids, region.radius, config)
    centres = region.centroids.double().cpu().numpy()
    mapped = place(region, region.centroids, config).astype(np.float64)
    data = np.linalg.norm(centres[edges[:, 0]] - centres[edges[:, 1]], axis=1)
    map_distance = np.linalg.norm(mapped[edges[:, 0]] - mapped[edges[:, 1]], axis=1)
    usable = map_distance > 0
    return float(np.median(data[usable] / map_distance[usable]))


def placed_at(levels: list[Level], transforms: dict, region_id: tuple[int, ...], points: torch.Tensor, level: int,
              config: Config) -> np.ndarray:
    """Points of a region placed by its ancestor at ``level`` (1 = level 1), then carried up to level 1's frame
    through each ancestor's transform."""
    ancestor = region_id[:level - 1]
    Z = place(levels[level - 1][ancestor][0], points, config)
    while ancestor:
        s, R, t = transforms[ancestor]
        Z, ancestor = s * Z @ R + t, ancestor[:-1]
    return Z


def grouped_rank(Z: np.ndarray, truth: Callable[[np.ndarray], np.ndarray], centres: np.ndarray, group: np.ndarray,
                 sample: int, seed: int) -> float:
    """Rank correlation of the centres' distances with the true ones over pairs in the same group, per group (at most
    ``sample`` centres each), averaged with each group weighted by its number of centres; groups whose rank is
    undefined (e.g. all their distances equal) are left out."""
    rng, ranks, weights = np.random.default_rng(seed), [], []
    for g in np.unique(group):
        inside = np.flatnonzero(group == g)
        if len(inside) < 3:
            continue
        inside = rng.choice(inside, min(sample, len(inside)), replace=False)
        rank = rank_correlation(pdist(Z[inside]), truth(centres[inside]))
        if np.isfinite(rank):
            ranks.append(rank)
            weights.append(np.sum(group == g))
    return float(np.average(ranks, weights=weights))


def banded_rank(Z: np.ndarray, exact: np.ndarray, rows: np.ndarray) -> dict[str, float]:
    """Rank correlation over all pairs of the chosen centres, and over the close, middle and far third of them."""
    embedded = pdist(Z[rows])
    band = np.searchsorted(np.quantile(exact, [1 / 3, 2 / 3]), exact)
    scores = {"all": rank_correlation(embedded, exact)}
    scores |= {name: rank_correlation(embedded[band == i], exact[band == i])
               for i, name in enumerate(("close", "middle", "far"))}
    return scores


def load_tree(models: Path, paths: np.ndarray, device: str) -> tuple[list[Level], Config]:
    """The saved tree: level 1's model and every deeper region's, with the rows of its points from ``paths`` (each
    point's region numbers, one column per level below level 1)."""
    level1 = LocalizedVoltageMaps.load(models / "level1")
    levels = [{(): (level1.root_, np.arange(len(paths)))}]
    for level in range(2, paths.shape[1] + 2):
        prefix = paths[:, :level - 1]
        found = {}
        for region_id in np.unique(prefix, axis=0):
            name = "-".join(map(str, region_id))
            found[tuple(int(r) for r in region_id)] = (load_region(models / f"level{level}" / f"region_{name}.npz", device),
                                                       np.flatnonzero((prefix == region_id).all(axis=1)))
        levels.append(found)
    return levels, level1.config_


def cell_statistics(levels: list[Level], level: int, X: np.ndarray, intrinsic: np.ndarray, device: str,
                    block: int = 1_000_000) -> dict[tuple[int, ...], dict[str, np.ndarray]]:
    """For each region at ``level``, per cell: its number of points, the participation ratio of its points in the
    data space (sum of the local PCA variances squared over the sum of their squares: 2 for a flat sheet, up to the
    ambient dimension for an isotropic blob) and the mean of their ``intrinsic`` (true, noise-free) positions."""
    stats = {}
    for region_id, (region, rows) in levels[level - 1].items():
        n_cells, n_features = len(region.centroids), X.shape[1]
        counts = torch.zeros(n_cells, dtype=torch.float64, device=device)
        sums = torch.zeros((n_cells, n_features), dtype=torch.float64, device=device)
        squares = torch.zeros((n_cells, n_features * n_features), dtype=torch.float64, device=device)
        truth = torch.zeros((n_cells, intrinsic.shape[1]), dtype=torch.float64, device=device)
        for start in range(0, len(rows), block):
            chunk = rows[start:start + block]
            points = torch.as_tensor(X[chunk], device=device)
            cells = assign_cells(points, region.centroids)
            points = points.double()
            counts.index_add_(0, cells, torch.ones(len(cells), dtype=torch.float64, device=device))
            sums.index_add_(0, cells, points)
            squares.index_add_(0, cells, (points[:, :, None] * points[:, None, :]).reshape(len(points), -1))
            truth.index_add_(0, cells, torch.as_tensor(intrinsic[chunk], device=device).double())
        filled = counts.clamp(min=1)[:, None]
        mean = sums / filled
        covariance = (squares / filled).reshape(n_cells, n_features, n_features) - mean[:, :, None] * mean[:, None, :]
        variances = torch.linalg.eigvalsh(covariance).clamp(min=0)
        ratio = variances.sum(dim=1).square() / variances.square().sum(dim=1).clamp(min=1e-300)
        stats[region_id] = {"counts": counts.cpu().numpy(), "dimension": ratio.cpu().numpy(),
                            "truth": (truth / filled).cpu().numpy()}
    return stats


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, cumulative[-1] / 2)])


def region_dimension(stats: dict[str, np.ndarray]) -> float:
    """A region's dimension at its cells' scale: the median of its cells' participation ratios, each cell weighted by
    its points."""
    return weighted_median(stats["dimension"], stats["counts"])


def rule_depths(stats: dict[int, dict], depth: int, tolerance: float) -> dict[tuple[int, ...], int]:
    """The stopping rule: a region is split only if its children's cells are not of higher dimension than its own
    (their pooled median participation ratio at most ``tolerance`` above the region's). For each deepest region,
    the level at which the rule stops along its path from level 1."""
    def split(level: int, region_id: tuple[int, ...]) -> bool:
        children = [s for child, s in stats[level + 1].items() if child[:-1] == region_id]
        pooled = {key: np.concatenate([s[key] for s in children]) for key in ("dimension", "counts")}
        return region_dimension(pooled) <= region_dimension(stats[level][region_id]) + tolerance

    decisions, chosen = {}, {}
    for leaf in stats[depth]:
        level = 1
        while level < depth:
            ancestor = leaf[:level - 1]
            if (level, ancestor) not in decisions:
                decisions[(level, ancestor)] = split(level, ancestor)
            if not decisions[(level, ancestor)]:
                break
            level += 1
        chosen[leaf] = level
    return chosen


def sampled_pairs(owner: np.ndarray, seed: int, *, groups: int = 60, points: int = 300,
                  whole_points: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Pairs of centres (rows i < j) at every scale, from each centre's region path ``owner`` (one column per level
    below level 1): for each depth, all pairs among up to ``points`` centres in each of up to ``groups`` regions of
    that depth; and all pairs among ``whole_points`` centres over the whole surface. Also which pairs are the
    whole-surface ones."""
    rng, found = np.random.default_rng(seed), []
    for depth in range(owner.shape[1], 0, -1):
        group = np.unique(owner[:, :depth], axis=0, return_inverse=True)[1].ravel()
        for g in rng.choice(group.max() + 1, min(groups, group.max() + 1), replace=False):
            inside = np.flatnonzero(group == g)
            inside = rng.choice(inside, min(points, len(inside)), replace=False)
            i, j = np.triu_indices(len(inside), 1)
            found.append((np.column_stack([inside[i], inside[j]]), False))
    sample = rng.choice(len(owner), min(whole_points, len(owner)), replace=False)
    i, j = np.triu_indices(len(sample), 1)
    found.append((np.column_stack([sample[i], sample[j]]), True))
    pairs = np.concatenate([p for p, _ in found])
    whole = np.concatenate([np.full(len(p), w) for p, w in found])
    return pairs, whole


def distance_errors(placed: dict[int, np.ndarray], true_distance: Callable[[np.ndarray, np.ndarray], np.ndarray],
                    pairs: np.ndarray, whole: np.ndarray, bins: int = 30) -> dict:
    """Each placement's distance errors in true units, by true distance: its map distances are scaled once (least
    squares over the whole-surface pairs ``whole``) into true units; per log-spaced bin of true distance, the median
    absolute error |scaled map distance - true distance|, the median relative error (divided by the true
    distance) and the median ratio of scaled map distance to true distance."""
    truth = true_distance(pairs[:, 0], pairs[:, 1])
    keep = truth > 0
    pairs, truth, whole = pairs[keep], truth[keep], whole[keep]
    edges = np.geomspace(np.quantile(truth, 0.001), truth.max(), bins + 1)
    which = np.clip(np.searchsorted(edges, truth) - 1, 0, bins - 1)
    result = {"centres": np.sqrt(edges[:-1] * edges[1:]).tolist(),
              "pairs": np.bincount(which, minlength=bins).tolist(), "levels": {}}
    for level, Z in placed.items():
        mapped = np.linalg.norm(Z[pairs[:, 0]] - Z[pairs[:, 1]], axis=1)
        scale = float(np.sum(mapped[whole] * truth[whole]) / np.sum(mapped[whole] ** 2))
        absolute = np.abs(scale * mapped - truth)
        relative = absolute / truth
        ratio = scale * mapped / truth
        result["levels"][level] = {
            "scale": scale,
            "ratio": [float(np.median(ratio[which == b])) if np.any(which == b) else None for b in range(bins)],
            "absolute": [float(np.median(absolute[which == b])) if np.any(which == b) else None for b in range(bins)],
            "relative": [float(np.median(relative[which == b])) if np.any(which == b) else None for b in range(bins)]}
    return result


def binned_medians(values: np.ndarray, truth: np.ndarray, edges: np.ndarray) -> list[float | None]:
    """The median of ``values`` in each bin of true distance (``edges``); None for an empty bin."""
    which = np.searchsorted(edges, truth) - 1
    return [float(np.median(values[which == b])) if np.any(which == b) else None for b in range(len(edges) - 1)]


def split_ratios(Z: np.ndarray, truth: np.ndarray, pairs: np.ndarray, whole: np.ndarray, same: np.ndarray,
                 edges: np.ndarray) -> dict:
    """The median ratio of map to true distance per bin of true distance, after one least-squares scale over the
    whole-surface pairs: over all pairs, over pairs in the same region and over pairs in different regions; and the
    share of same-region pairs per bin."""
    mapped = np.linalg.norm(Z[pairs[:, 0]] - Z[pairs[:, 1]], axis=1)
    scale = float(np.sum(mapped[whole] * truth[whole]) / np.sum(mapped[whole] ** 2))
    ratio = scale * mapped / truth
    return {"all": binned_medians(ratio, truth, edges),
            "same": binned_medians(ratio[same], truth[same], edges),
            "different": binned_medians(ratio[~same], truth[~same], edges),
            "same_share": binned_medians(same.astype(float), truth, edges)}
