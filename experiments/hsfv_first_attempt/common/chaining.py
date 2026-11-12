import torch

from lvm.config import Config
from lvm.embedding import classical_mds, triangulate_distances
from lvm.voltage import point_voltages


def extend(region, points: torch.Tensor, cell_values: torch.Tensor, *, config: Config) -> torch.Tensor:
    """(m, k) values at points (or any locations), from the (n_cells, k) values of the region's cells around them,
    with LVM's extension (the same weights that give a point its voltages)."""
    return point_voltages(points, region.centroids, region.masses, region.radius, cell_values.T, region.rho_g,
                          config=config)


def to_landmarks(region, points: torch.Tensor, *, config: Config) -> torch.Tensor:
    """(m, L) distances -log v from points (or any locations) to the region's landmarks, in the region's own unit."""
    return region.embedding.item_distances(extend(region, points, region.maps.T, config=config))


def cells_to_landmarks(region) -> torch.Tensor:
    """(n_cells, L) each cell's distances to the region's landmarks, in the region's own unit."""
    return region.embedding.item_distances(region.maps.T)


def frame(region) -> tuple[torch.Tensor, torch.Tensor]:
    """The region's landmarks placed by classical MDS in all their dimensions (not only the map's two)."""
    return classical_mds(region.embedding.distances, len(region.landmarks))


def coordinates(distances: torch.Tensor, frame: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Items placed in a region's full frame from their distances to its landmarks (Landmark MDS triangulation)."""
    return triangulate_distances(distances, *frame)


def calibration(child, root, *, config: Config) -> float:
    """The factor that turns the child region's distances into the root's unit: the median, over pairs of the
    child's landmarks, of their distance measured at the root (in the root's full frame) over their distance measured
    in the child."""
    places = to_landmarks(root, child.centroids[child.landmarks].to(torch.float32), config=config)
    at_root = torch.cdist(coordinates(places, frame(root)), coordinates(places, frame(root)))
    in_child = child.embedding.distances.to(at_root.dtype)
    pairs = ~torch.eye(len(in_child), dtype=torch.bool, device=in_child.device) & (in_child > 0)
    return float((at_root[pairs] / in_child[pairs]).median())


def landmarks_to_root(child, root, *, config: Config) -> torch.Tensor:
    """(k, L_root) distances from the child's landmarks to the root's landmarks, in the root's unit (the child's
    landmarks are located in the root like any point)."""
    return to_landmarks(root, child.centroids[child.landmarks].to(torch.float32), config=config)


def chained_simple(child, root, *, config: Config) -> torch.Tensor:
    """(n_cells, L_root) each child cell's distance to each root landmark by the shortest route through one of the
    child's landmarks: min_i s d(c, l_i) + D(l_i, L), with s the calibration."""
    s = calibration(child, root, config=config)
    local = s * cells_to_landmarks(child)
    far = landmarks_to_root(child, root, config=config).to(local.dtype)
    return (local[:, :, None] + far[None, :, :]).min(dim=1).values


def chained_triangulated(child, root, *, config: Config) -> torch.Tensor:
    """(n_cells, L_root) each child cell's distance to each root landmark by triangulation: the cell's position among
    the child's landmarks as affine weights w (sum 1, least squares, minimum norm), then sum_i w_i D(l_i, L). The
    weights do not depend on the child's unit, so no calibration is needed."""
    f = frame(child)
    landmarks = coordinates(child.embedding.distances, f)                   # (k, q)
    cells = coordinates(cells_to_landmarks(child).to(landmarks.dtype), f)    # (n_cells, q)
    ones = torch.ones(1, len(landmarks), dtype=landmarks.dtype, device=landmarks.device)
    A = torch.cat([landmarks.T, ones])                                      # (q + 1, k)
    b = torch.cat([cells.T, torch.ones(1, len(cells), dtype=cells.dtype, device=cells.device)])
    weights = torch.linalg.pinv(A) @ b                                      # (k, n_cells)
    far = landmarks_to_root(child, root, config=config).to(weights.dtype)
    return weights.T @ far


def chained_voltage(child, root, *, config: Config) -> torch.Tensor:
    """(n_cells, L_root) each child cell's distance to each root landmark as the average of the child's landmarks'
    distances, weighted by the cell's voltages from them: sum_i v_i(c) D(l_i, L) / sum_i v_i(c). The voltages are
    harmonic, so this interpolates in LVM's own terms; no calibration and no MDS frame are needed."""
    weights = child.maps.T                                                  # (n_cells, k)
    far = landmarks_to_root(child, root, config=config).to(weights.dtype)
    return weights @ far / weights.sum(dim=1, keepdim=True)
