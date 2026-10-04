"""Low-dimensional coordinates from landmark voltage maps (config ``embedding``).

Step (6) of Fig. 7 in the paper. Theorem 12 / Corollary 13 bound a voltage
map between two exponentials in the distance to its landmark, so -log(v) grows
roughly linearly with that distance. The features -log v therefore behave like
distances to the landmarks, and a linear method recovers the geometry from
them. A zeroed voltage (below the threshold tau) means "farther than the
map's support", so features are clipped at -log(tau) instead of going to
infinity.

Configurable choice point ``embedding``; see ``lvm.strategies``. A fitted
embedding maps any voltage vectors -- the cells' or, after
``voltage.extend_voltages``, individual data points' -- to coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from array_api_compat import array_namespace, device

from lvm.config import EmbeddingConfig, LandmarkMdsConfig, LocalScaleConfig, LogMdsConfig
from lvm.strategies import resolve
from lvm.voltage import chained_distances, voltage_distances


@dataclass(frozen=True)
class LogMdsEmbedding:
    tau: float
    mean: np.ndarray          # (L,) mass-weighted mean feature vector of the cells
    components: np.ndarray    # (n_components, L) principal axes
    cell_coords: np.ndarray   # (n_cells, n_components) coordinates of the cells

    def transform(self, V):
        """Coordinates for voltage vectors ``V`` (L landmarks x m items) -> (m, n_components).

        ``V`` may be a NumPy array or a torch tensor; the result is the same
        kind, on the same device.
        """
        F = _log_features(V, self.tau)
        xp = array_namespace(F)
        mean = xp.asarray(self.mean, dtype=F.dtype, device=device(F))
        components = xp.asarray(self.components, dtype=F.dtype, device=device(F))
        return (F - mean) @ components.T


@dataclass(frozen=True)
class LandmarkMdsEmbedding:
    tau: float
    missing: str                  # clip | chain: how a distance outside a map's support is filled
    landmark_D: np.ndarray        # (L, L) symmetric landmark-to-landmark distances (gaps filled)
    delta_mean: np.ndarray        # (L,) mean squared distance from each landmark to the landmarks
    pinv: np.ndarray              # (n_components, L) pseudo-inverse of the landmarks' MDS coordinates
    cell_coords: np.ndarray       # (n_cells, n_components)
    eigenvalues: np.ndarray       # (n_components,) MDS eigenvalues of the landmark configuration
    usable: int                   # components with a non-negligible eigenvalue; the rest are set to 0

    def transform(self, V):
        """Coordinates for voltage vectors ``V`` (L landmarks x m items) by triangulation -> (m, n_components).

        NumPy or torch; the result is the same kind, on the same device.
        """
        d = point_landmark_distances(V, self.landmark_D, tau=self.tau, missing=self.missing)   # (m, L)
        placement = LandmarkPlacement(self.delta_mean, self.pinv, self.eigenvalues, self.usable)
        return placement.triangulate(d)


@dataclass(frozen=True)
class LocalScale:
    """A second, local scale per cell: z' = anchors[c] + alpha[c] * (z - anchors[c]) for a point of cell c.

    Scaling a cell's points by a positive factor about their mean keeps their
    order, and so what they say about which points are near which.
    """
    anchors: np.ndarray   # (n_cells, n_components) mean embedded position of each cell's points
    alpha: np.ndarray     # (n_cells,) scale factor; 1 for cells with fewer than 2 points

    def apply(self, Z: np.ndarray, cell: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
        Z, cell = _numpy(Z), _numpy(cell)
        a = self.anchors[cell]
        return a + self.alpha[cell, None] * (Z - a)


@dataclass(frozen=True)
class LocalPca:
    """Points placed within their cell by the cell's own k-D PCA of the raw features (torch, on one device).

    k is the embedding's dimension. z = anchors[c] + scale[c] * ((x - mean_c) @ B_c) @ rotations[c]. The
    cells keep LVM's global layout; within a cell the k main directions of
    variation of its points give local coordinates, turned to agree with the
    LVM offsets. Cells with at most k points keep their LVM positions.
    """
    anchors: "torch.Tensor"      # (n_cells, k)
    bases: "torch.Tensor"        # (d, k n_cells): columns k c ... k c + k - 1 are cell c's directions
    mean_proj: "torch.Tensor"    # (n_cells, k) each cell's mean projected on its directions
    rotations: "torch.Tensor"    # (n_cells, k, k)
    scale: "torch.Tensor"        # (n_cells,)
    valid: "torch.Tensor"        # (n_cells,) bool

    def project(self, X, cell):
        """(m, k) coordinates of points X along their own cell's k directions (one matmul for all cells)."""
        (m, n), k = (X.shape[0], self.anchors.shape[0]), self.anchors.shape[1]
        P = (X @ self.bases).view(m, n, k)
        return P[torch.arange(m, device=X.device), cell] - self.mean_proj[cell]

    def apply(self, Z, cell, X):
        import torch

        dev = self.anchors.device
        Z = torch.as_tensor(Z, device=dev, dtype=self.anchors.dtype)
        cell = torch.as_tensor(cell, device=dev, dtype=torch.long)
        X = torch.as_tensor(X, device=dev, dtype=self.bases.dtype)
        u = self.project(X, cell).to(self.anchors.dtype)
        placed = self.anchors[cell] + self.scale[cell, None] * torch.einsum("mk,mkj->mj", u, self.rotations[cell])
        return torch.where(self.valid[cell, None], placed, Z)


def fit_local_scale(
    Z: np.ndarray, cell: np.ndarray, n_cells: int, *, config: LocalScaleConfig, X: np.ndarray | None = None,
) -> LocalScale | LocalPca | None:
    """Local scale from embedded sample points ``Z``, their cells (and raw features ``X`` for pca).

    Strategy ``config.strategy``; none -> None.
    """
    if config.strategy == "none":
        return None
    if config.strategy == "pca":
        return _fit_local_pca(Z, cell, n_cells, X, fill=config.pca.fill)
    Z, cell = np.asarray(Z, dtype=np.float64), np.asarray(cell)
    counts = np.bincount(cell, minlength=n_cells)
    anchors = np.zeros((n_cells, Z.shape[1]))
    np.add.at(anchors, cell, Z)
    occupied = counts >= 2
    anchors[counts > 0] /= counts[counts > 0, None]
    sq = np.zeros(n_cells)
    np.add.at(sq, cell, ((Z - anchors[cell]) ** 2).sum(axis=1))
    spread = np.sqrt(sq / np.maximum(counts, 1))
    alpha = np.ones(n_cells)
    if occupied.sum() >= 2:
        from scipy.spatial.distance import cdist

        D = cdist(anchors[occupied], anchors[occupied])
        np.fill_diagonal(D, np.inf)
        target = config.cell.fill * D.min(axis=1)
        s = spread[occupied]
        alpha[occupied] = np.where(s > 0, target / np.where(s > 0, s, 1.0), 1.0)
    return LocalScale(anchors, alpha)


def _fit_local_pca(Z, cell, n_cells: int, X, *, fill: float, max_points: int = 256) -> LocalPca:
    """Fit ``LocalPca`` on torch tensors Z (m, k), cell (m,), X (m, d), all on one device; k = Z's dimension.

    Each cell's k directions come from at most ``max_points`` of its points
    (a few directions need few points; this bounds memory): their Gram
    matrices are built and eigendecomposed in one batch.
    """
    import torch

    dev, m, d, k = X.device, X.shape[0], X.shape[1], Z.shape[1]
    Z = Z.to(torch.float64)
    counts = torch.bincount(cell, minlength=n_cells)
    safe = counts.clamp(min=1).to(torch.float64)
    anchors = torch.zeros(n_cells, k, dtype=torch.float64, device=dev).index_add_(0, cell, Z) / safe[:, None]
    means = torch.zeros(n_cells, d, dtype=X.dtype, device=dev).index_add_(0, cell, X) / safe[:, None].to(X.dtype)

    # Each cell's first max_points points, centred, in a padded (n_cells, max_points, d) tensor.
    order = torch.argsort(cell, stable=True)
    sorted_cell = cell[order]
    first = torch.cumsum(counts, 0) - counts
    rank = torch.arange(m, device=dev) - first[sorted_cell]
    keep = rank < max_points
    P = torch.zeros(n_cells, max_points, d, dtype=X.dtype, device=dev)
    rows = order[keep]
    P[sorted_cell[keep], rank[keep]] = X[rows] - means[sorted_cell[keep]]
    evals, evecs = torch.linalg.eigh(P @ P.transpose(1, 2))          # ascending; padded rows add zeros
    V = P.transpose(1, 2) @ evecs[:, :, -k:].flip(-1)                  # (n_cells, d, k), largest first
    norms = V.norm(dim=1)
    valid = (counts >= k + 1) & (norms > 0).all(dim=1)
    V = V / norms.clamp(min=1e-12)[:, None, :]
    bases = V.permute(1, 0, 2).reshape(d, k * n_cells)
    mean_proj = torch.einsum("nd,ndk->nk", means, V)

    local = LocalPca(anchors, bases, mean_proj, torch.eye(k, dtype=torch.float64, device=dev).repeat(n_cells, 1, 1),
                     torch.ones(n_cells, dtype=torch.float64, device=dev), valid)
    u = torch.cat([local.project(X[s:s + 10000], cell[s:s + 10000]) for s in range(0, m, 10000)]).to(torch.float64)
    # Orthogonal Procrustes per cell: the rotation/reflection that best maps u onto the LVM offsets.
    M = torch.zeros(n_cells, k, k, dtype=torch.float64, device=dev).index_add_(
        0, cell, u[:, :, None] * (Z - anchors[cell])[:, None, :])
    a, _, bt = torch.linalg.svd(M)
    rotations = a @ bt
    # Median distance from the cell mean in these coordinates, per cell (sort by cell, then by radius).
    radius = u.norm(dim=1)
    by = torch.argsort(cell.to(torch.float64) * (radius.max() + 1) + radius)
    median = torch.zeros(n_cells, dtype=torch.float64, device=dev)
    occupied = counts > 0
    median[occupied] = radius[by][(first + counts // 2)[occupied]]
    D = torch.cdist(anchors[occupied], anchors[occupied])
    D.fill_diagonal_(float("inf"))
    nearest = torch.full((n_cells,), float("inf"), dtype=torch.float64, device=dev)
    nearest[occupied] = D.min(dim=1).values
    valid = valid & torch.isfinite(nearest) & (median > 0)
    scale = torch.where(valid, fill * nearest / median.clamp(min=1e-12), torch.ones_like(median))
    return LocalPca(anchors, bases, mean_proj, rotations, scale, valid)


def _numpy(x) -> np.ndarray:
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def fit_embedding(
    V: np.ndarray, p: np.ndarray, *, config: EmbeddingConfig, tau: float, landmark_cells: np.ndarray | None = None,
):
    """Fit the embedding named in ``config.strategy`` to the cells' landmark maps.

    ``V`` is (L landmarks, n cells), typically thresholded; ``p`` the cell
    masses, which weight the fit so each cell counts as much as the data in it.
    ``landmark_cells[l]`` is the cell landmark l sits at (needed by
    ``landmark_mds`` for the distances between landmarks).
    """
    strategy, options = resolve(_STRATEGIES, config)
    return strategy(np.asarray(V, dtype=np.float64), np.asarray(p, dtype=np.float64), options=options, tau=tau,
                    landmark_cells=landmark_cells)


def _log_features(V, tau: float):
    """(m, L) features -log(max(v, tau)) for voltage vectors V (L x m), NumPy or torch."""
    if not hasattr(V, "shape"):
        V = np.asarray(V, dtype=np.float64)
    xp = array_namespace(V)
    return -xp.log(xp.clip(V, min=tau)).T


def _log_mds(V: np.ndarray, p: np.ndarray, *, options: LogMdsConfig, tau: float, landmark_cells) -> LogMdsEmbedding:
    """Classical MDS on the Euclidean distances between cells' features.

    Implemented as a weighted eigendecomposition (``np.linalg.eigh``) because
    scikit-learn's PCA takes no sample weights.
    With Euclidean distances classical MDS equals PCA of the features, so the
    fit is a linear projection that applies unchanged to new points. Cells
    are weighted by mass; zero-mass cells don't influence the axes but still
    get coordinates. Axis signs are fixed (largest loading positive) so the
    result is deterministic.
    """
    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau must be in (0, 1) for log features, got {tau}")
    F = _log_features(V, tau)                        # (n, L)
    w = p / p.sum()
    mean = w @ F
    C = (F - mean).T @ ((F - mean) * w[:, None])     # (L, L) weighted covariance
    eigvals, eigvecs = np.linalg.eigh(C)
    k = min(options.n_components, F.shape[1])
    components = eigvecs[:, ::-1][:, :k].T           # largest variance first
    signs = np.sign(components[np.arange(k), np.abs(components).argmax(axis=1)])
    components = components * signs[:, None]
    return LogMdsEmbedding(tau=tau, mean=mean, components=components, cell_coords=(F - mean) @ components.T)


def point_landmark_distances(V, landmark_D: np.ndarray, *, tau: float, missing: str):
    """(m, L) distances from m items to the L landmarks, from their voltages V (L x m).

    d = -log v where v >= tau. Outside a landmark's support the distance is
    unknown; ``missing`` decides what it becomes:
    * clip: -log(tau), i.e. "at least this far" (the same for every far item);
    * chain: through the landmarks the item does reach,
      d(x, l) = min_j d(x, j) + D(j, l), using the landmark distances D.
    An item no landmark reaches gets -log(tau) to every landmark either way.
    NumPy or torch.
    """
    d = voltage_distances(V, tau).T                            # (m, L), +inf where unknown
    xp = array_namespace(d)
    far = -float(np.log(tau))
    if missing == "chain":
        D = xp.asarray(landmark_D, dtype=d.dtype, device=device(d))
        via = xp.min(d[:, :, None] + D[None, :, :], axis=1)   # (m, L): best route through a reached landmark
        d = xp.where(xp.isfinite(d), d, via)
    elif missing != "clip":
        raise ValueError(f"missing must be 'clip' or 'chain', got {missing!r}")
    return xp.where(xp.isfinite(d), d, xp.full_like(d, far))


_RELATIVE_EIGENVALUE_FLOOR = 1e-6


def _landmark_mds(
    V: np.ndarray, p: np.ndarray, *, options: LandmarkMdsConfig, tau: float, landmark_cells,
) -> LandmarkMdsEmbedding:
    """Landmark MDS (de Silva & Tenenbaum, 2004) on voltage distances d = -log v.

    1. Landmark-to-landmark distances: -log of each landmark's voltage at the
       other landmarks' cells, symmetrised. Pairs outside each other's
       support are filled as configured: chain = shortest paths through the
       landmarks (``voltage.chained_distances``); clip = -log(tau).
    2. Classical MDS places the landmarks: double-centre the squared
       distances, keep the top eigenvectors.
    3. Every item (cell or data point) is placed by distance-based
       triangulation, y = -1/2 L# (delta_x - delta_mean), a fixed linear map
       of its squared distances to the landmarks -- so it streams and runs on
       the GPU like the rest of the per-point path.
    Unlike ``log_mds``, which runs PCA on the distance vectors, this is exact
    when the distances are Euclidean.
    """
    if landmark_cells is None:
        raise ValueError("landmark_mds needs landmark_cells")
    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau must be in (0, 1), got {tau}")
    L = V.shape[0]
    D = voltage_distances(V[:, np.asarray(landmark_cells)], tau)    # (L, L): row a = landmark a's map
    if options.missing == "chain":
        D = chained_distances(D)
    else:
        Dt = D.T
        both = np.isfinite(D) & np.isfinite(Dt)
        D = np.where(both, 0.5 * (D + Dt), np.minimum(D, Dt))
        np.fill_diagonal(D, 0.0)
    far = -np.log(tau)
    D = np.where(np.isfinite(D), D, max(far, np.nanmax(np.where(np.isfinite(D), D, np.nan))))

    lmds = place_landmarks(D, options.n_components)
    embedding = LandmarkMdsEmbedding(tau=tau, missing=options.missing, landmark_D=D,
                                     delta_mean=lmds.delta_mean, pinv=lmds.pinv, cell_coords=np.empty((0, 0)),
                                     eigenvalues=lmds.eigenvalues, usable=lmds.usable)
    return LandmarkMdsEmbedding(**{**embedding.__dict__, "cell_coords": embedding.transform(V)})


@dataclass(frozen=True)
class LandmarkPlacement:
    """Landmark MDS fitted to a landmark distance matrix; ``triangulate`` places other items."""
    delta_mean: np.ndarray    # (L,) mean squared distance from each landmark to the landmarks
    pinv: np.ndarray          # (k, L)
    eigenvalues: np.ndarray   # (k,)
    usable: int

    def triangulate(self, d):
        """Coordinates (m, k) from distances d (m, L) to the landmarks; NumPy or torch."""
        xp = array_namespace(d)
        delta_mean = xp.asarray(self.delta_mean, dtype=d.dtype, device=device(d))
        pinv = xp.asarray(self.pinv, dtype=d.dtype, device=device(d))
        return -0.5 * (d * d - delta_mean) @ pinv.T


def place_landmarks(D: np.ndarray, n_components: int) -> LandmarkPlacement:
    """Landmark MDS (de Silva & Tenenbaum, 2004) on a complete, symmetric landmark distance matrix.

    Classical MDS places the landmarks (double-centred squared distances, top
    eigenvectors); any other item is then placed by triangulation from its
    distances to the landmarks. Shared by LVM (voltage distances) and the
    Landmark Isomap baseline (shortest-path distances), so the two differ
    only in their distances.
    """
    D = np.asarray(D, dtype=np.float64)
    L = D.shape[0]
    Delta = D * D
    H = np.eye(L) - 1.0 / L
    B = -0.5 * H @ Delta @ H
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1][:n_components]
    lam, vec = eigvals[order], eigvecs[:, order]
    # Triangulation divides by sqrt(lambda). If the landmarks are not in general
    # position (e.g. nearly on a line), a later eigenvalue is ~0 and that
    # coordinate would amplify tiny distance errors without bound; such
    # components are set to 0 instead.
    usable = lam > _RELATIVE_EIGENVALUE_FLOOR * max(lam[0], 0.0)
    signs = np.sign(vec[np.abs(vec).argmax(axis=0), np.arange(vec.shape[1])])
    vec = vec * signs[None, :]
    pinv = np.where(usable[None, :], vec / np.sqrt(np.where(usable, lam, 1.0))[None, :], 0.0).T   # (k, L)
    return LandmarkPlacement(delta_mean=Delta.mean(axis=0), pinv=pinv, eigenvalues=lam, usable=int(usable.sum()))


_STRATEGIES: dict[str, Callable[..., LogMdsEmbedding | LandmarkMdsEmbedding]] = {
    "log_mds": _log_mds,
    "landmark_mds": _landmark_mds,
}
