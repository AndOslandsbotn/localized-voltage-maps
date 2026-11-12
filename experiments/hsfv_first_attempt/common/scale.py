import numpy as np
from sklearn.cluster import KMeans

ROOT_GRID = (10, 20, 30, 50, 100, 150, 200, 300)


def participation_ratio(points: np.ndarray) -> float:
    """(sum of the local PCA variances)^2 / sum of their squares: the number of directions a cell's points spread in
    (1 for a line, 2 for a flat sheet, up to the ambient dimension for an isotropic blob)."""
    variances = np.linalg.eigvalsh(np.cov(points.T))
    return float(variances.sum() ** 2 / np.square(variances).sum())


def dimension_curve(X: np.ndarray, *, grid=ROOT_GRID, seed: int = 0) -> dict:
    """For each cell count: the median over k-means cells of the local dimension (participation ratio)."""
    curve = {}
    for n_cells in grid:
        labels = KMeans(n_clusters=n_cells, n_init=1, random_state=seed).fit(X).labels_
        dims = [participation_ratio(X[labels == c]) for c in range(n_cells) if (labels == c).sum() >= 5]
        curve[n_cells] = float(np.median(dims))
    return curve


def root_cells(curve: dict, tolerance: float = 0.1) -> int:
    """The finest root whose cells are still clean local patches: the largest cell count whose local dimension stays
    within ``tolerance`` of the lowest (the data's dimension at its best scale), before noise inflates it."""
    lowest = min(curve.values())
    return max(n for n, d in curve.items() if d <= lowest + tolerance)


def dimension_curve_gpu(points, *, grid=ROOT_GRID, seed: int = 0) -> dict:
    """dimension_curve for large, high-dimensional data on the GPU: the library's k-means (cuML), and each cell's
    participation ratio from the singular values of its centred points, (sum s^2)^2 / sum s^4."""
    import torch

    from lvm.cells import assign_cells, fit_cells
    from lvm.config import load_config

    curve = {}
    for n_cells in grid:
        config = load_config(overrides={"cells": {"n_cells": n_cells}})
        centroids = fit_cells(points, config=config, device=str(points.device), seed=seed)
        labels = assign_cells(points, centroids.to(points.dtype))
        dims = []
        for c in range(n_cells):
            members = points[labels == c]
            if len(members) >= 5:
                s2 = torch.linalg.svdvals((members - members.mean(dim=0)).double()).square()
                dims.append(float(s2.sum() ** 2 / s2.square().sum()))
        curve[n_cells] = float(np.median(dims))
    return curve
