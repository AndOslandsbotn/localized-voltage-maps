import numpy as np
import torch
from scipy.sparse.csgraph import connected_components

from lvm.graph import kernel_weights


def _linked(centroids: torch.Tensor, radius: torch.Tensor, config) -> np.ndarray:
    """(m, m) which cells the graph's kernel joins, each cell with its own radius (either side suffices)."""
    weights = kernel_weights(torch.cdist(centroids, centroids).square(), radius.square()[None, :], config.graph.kernel)
    return (torch.maximum(weights, weights.T) > 0).cpu().numpy()


def root_edges(root, config) -> np.ndarray:
    """(e, 2) the root graph's edges, each pair once."""
    linked = _linked(root.centroids.double(), root.radius.double(), config)
    np.fill_diagonal(linked, False)
    i, j = np.nonzero(np.triu(linked))
    return np.column_stack([i, j])


def children(root, level2: dict) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """All level-2 cells (centroids, radii) and the root cell each lies in (its nearest root cell)."""
    centroids = torch.cat([child.centroids.double() for child in level2.values()])
    radius = torch.cat([child.radius.double() for child in level2.values()])
    parent = torch.cdist(centroids, root.centroids.double()).argmin(dim=1).cpu().numpy()
    return centroids, radius, parent


def straddling(root, level2: dict, config) -> np.ndarray:
    """For each root cell, how many separate groups its children form at level 2's resolution (1 = one piece)."""
    centroids, radius, parent = children(root, level2)
    groups = np.zeros(len(root.centroids), dtype=int)
    for a in range(len(root.centroids)):
        mine = np.flatnonzero(parent == a)
        if len(mine):
            groups[a] = connected_components(_linked(centroids[mine], radius[mine], config), directed=False)[0]
    return groups


def flagged_edges(root, level2: dict, config) -> tuple[np.ndarray, np.ndarray]:
    """The root's edges and, for each, whether level 2 contradicts it: the children of its two cells, joined at level
    2's resolution, do not connect the two sides."""
    edges = root_edges(root, config)
    centroids, radius, parent = children(root, level2)
    flags = np.zeros(len(edges), dtype=bool)
    for e, (a, b) in enumerate(edges):
        side_a, side_b = np.flatnonzero(parent == a), np.flatnonzero(parent == b)
        if len(side_a) == 0 or len(side_b) == 0:
            continue
        both = np.concatenate([side_a, side_b])
        labels = connected_components(_linked(centroids[both], radius[both], config), directed=False)[1]
        flags[e] = not set(labels[:len(side_a)]) & set(labels[len(side_a):])
    return edges, flags


def _detour_flags(edges: np.ndarray, nodes: np.ndarray, centroids: torch.Tensor, radius: torch.Tensor,
                  parent: np.ndarray, config, ratio: float) -> np.ndarray:
    """For each edge between two nodes: whether the shortest path through level 2's graph (its cells ``centroids``,
    edge length = centroid distance) from one node's children (``parent``) to the other's is more than ``ratio``
    times the edge, or does not exist. Each search starts from one node's children and stops at that length."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    linked = _linked(centroids, radius, config)
    np.fill_diagonal(linked, False)
    graph = csr_matrix(np.where(linked, torch.cdist(centroids, centroids).cpu().numpy(), 0.0))
    span = np.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1)
    flags = np.zeros(len(edges), dtype=bool)
    for a in np.unique(edges[:, 0]):
        sources = np.flatnonzero(parent == a)
        mine = np.flatnonzero(edges[:, 0] == a)
        if len(sources) == 0:
            continue
        reach = dijkstra(graph, directed=False, indices=sources, min_only=True, limit=ratio * span[mine].max())
        for e in mine:
            targets = np.flatnonzero(parent == edges[e, 1])
            flags[e] = len(targets) == 0 or reach[targets].min() > ratio * span[e]
    return flags


def flagged_by_detour(root, level2: dict, config, ratio: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """The root's edges and, for each, whether level 2 contradicts it (see _detour_flags)."""
    edges = root_edges(root, config)
    centroids, radius, parent = children(root, level2)
    flags = _detour_flags(edges, root.centroids.double().cpu().numpy(), centroids, radius, parent, config, ratio)
    return edges, flags


def child_masses(root, level2: dict) -> torch.Tensor:
    """Each level-2 cell's share of all the data (its share of its region times the region's share), in the order of
    children()."""
    cell_region = root.maps.argmax(dim=0)
    return torch.cat([child.masses.double() * root.masses.double()[cell_region == r].sum()
                      for r, child in level2.items()])


def root_kernel(centroids: torch.Tensor, config) -> tuple[torch.Tensor, torch.Tensor]:
    """The root graph's kernel and radii, made by the library's own build_graph (radius strategies that need no data
    sample)."""
    from lvm.graph import build_graph

    return build_graph(centroids, None, config=config, device=str(centroids.device), seed=0)


def repaired_root(root, level2: dict, config, ratio: float = 2.0):
    """The root after level 2's corrections: a root cell whose children form separate groups at level 2's resolution
    is split into one node per group; then every edge that level 2 contradicts (_detour_flags) is removed, and the
    ground strength rho_g is chosen again for the repaired graph as LVM chooses it (the largest at which all the
    landmarks still reach the data), and the voltage maps and landmark distances are recomputed with the same
    landmarks. Returns the repaired root and what was changed."""
    from dataclasses import replace

    from lvm.embedding import fit_embedding
    from lvm.voltage import voltage_maps

    centroids, radius, parent = children(root, level2)
    masses = child_masses(root, level2)
    device = root.centroids.device
    nodes, node_masses, child_node, origin = [], [], np.empty(len(parent), dtype=int), []
    for a in range(len(root.centroids)):
        mine = np.flatnonzero(parent == a)
        labels = (connected_components(_linked(centroids[mine], radius[mine], config), directed=False)[1]
                  if len(mine) > 1 else np.zeros(len(mine), dtype=int))
        groups = np.unique(labels) if len(mine) else np.array([0])
        for g in groups:
            members = mine[labels == g] if len(mine) else mine
            if len(groups) == 1:
                nodes.append(root.centroids[a].double())
                node_masses.append(root.masses[a].double())
            else:
                w = masses[members]
                nodes.append((w[:, None] * centroids[members]).sum(dim=0) / w.sum())
                node_masses.append(root.masses[a].double() * w.sum() / masses[mine].sum())
            child_node[members] = len(nodes) - 1
            origin.append(a)
    nodes, node_masses, origin = torch.stack(nodes), torch.stack(node_masses), np.array(origin)
    landmarks = torch.tensor([max(np.flatnonzero(origin == int(l)), key=lambda n: float(node_masses[n]))
                              for l in root.landmarks], device=device)
    kernel, node_radius = root_kernel(nodes.to(root.centroids.dtype), config)
    linked = (kernel > 0).cpu().numpy()
    i, j = np.nonzero(np.triu(linked, 1))
    edges = np.column_stack([i, j])
    flags = _detour_flags(edges, nodes.cpu().numpy(), centroids, radius, child_node, config, ratio)
    kernel[edges[flags, 0], edges[flags, 1]] = 0.0
    kernel[edges[flags, 1], edges[flags, 0]] = 0.0
    from lvm.voltage import _largest_passing, _reached

    reach, tau = config.landmarks.reach, config.voltage.threshold
    k = len(landmarks) if reach.k is None else min(reach.k, len(landmarks))
    node_masses = node_masses.to(kernel.dtype)
    rho_g = _largest_passing(lambda r: _reached(kernel, node_masses, r, landmarks, tau, reach.share), k,
                             options=reach, guess=root.rho_g)
    maps = voltage_maps(kernel, node_masses, rho_g, landmarks)
    repaired = replace(root, centroids=nodes.to(root.centroids.dtype), masses=node_masses.to(root.masses.dtype),
                       radius=node_radius.to(root.radius.dtype), landmarks=landmarks, maps=maps, rho_g=rho_g,
                       embedding=fit_embedding(maps, landmarks, config=config))
    return repaired, {"split_cells": int(len(nodes) - len(root.centroids)), "edges": int(len(edges)),
                      "removed_edges": int(flags.sum()), "rho_g_before": float(root.rho_g), "rho_g_after": float(rho_g)}


def voltage_flags(root, level2: dict, config, *, reach: float = 1.5, spread: float = 3.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The root's edges, how well level 2's grounded voltages connect each edge's two cells, and which edges level 2
    contradicts. Per edge (a, b): a small graph of the level-2 cells within ``reach`` edge lengths of the edge's
    midpoint (LVM's kernel with each cell's radius, level 2's masses as shares of all the data), grounded with the
    median of the level-2 regions' rho_g; s = -log v with v the voltage at b's cells from sources at a's cells
    (median over sources of the mean over targets, both directions averaged). A thin, spurious connection carries
    little current, so v is small even when a short path exists. An edge is flagged when log(s / its length) lies
    more than ``spread`` robust standard deviations (MAD) above the median over all the root's edges."""
    from lvm.voltage import voltage_maps

    kernel, _ = root_kernel(root.centroids, config)
    linked = (kernel > 0).cpu().numpy()
    i, j = np.nonzero(np.triu(linked, 1))
    edges = np.column_stack([i, j])
    centroids, radius, parent = children(root, level2)
    masses = child_masses(root, level2)
    rho_g = float(np.median([child.rho_g for child in level2.values()]))
    R = root.centroids.double()
    strength = np.full(len(edges), np.nan)
    for e, (a, b) in enumerate(edges):
        middle, length = (R[a] + R[b]) / 2, float((R[a] - R[b]).norm())
        local = torch.nonzero((centroids - middle).norm(dim=1) <= reach * length).squeeze(1)
        local = torch.unique(torch.cat([local, torch.as_tensor(np.flatnonzero((parent == a) | (parent == b)),
                                                               device=local.device)]))
        side = parent[local.cpu().numpy()]
        source, target = np.flatnonzero(side == a), np.flatnonzero(side == b)
        if len(source) == 0 or len(target) == 0:
            continue
        C, r = centroids[local], radius[local]
        sq = torch.cdist(C, C).square()
        K = kernel_weights(sq, torch.maximum(r.square()[:, None], r.square()[None, :]), config.graph.kernel)
        K.fill_diagonal_(0.0)
        p = masses[local] / masses[local].sum()
        reads = []
        for s_set, t_set in ((source, target), (target, source)):
            maps = voltage_maps(K, p, rho_g, torch.as_tensor(s_set, device=K.device))
            reads.append(float(maps[:, torch.as_tensor(t_set, device=K.device)].mean(dim=1).median()))
        v = 0.5 * (reads[0] + reads[1])
        strength[e] = -np.log(v) if v > 0 else np.inf
        strength[e] = strength[e] / length if np.isfinite(strength[e]) else np.inf
    score = np.log(strength)
    finite = np.isfinite(score)
    median = np.median(score[finite])
    mad = 1.4826 * np.median(np.abs(score[finite] - median))
    flags = ~finite | (score > median + spread * mad)
    return edges, score, flags
