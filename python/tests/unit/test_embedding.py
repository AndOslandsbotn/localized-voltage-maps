import numpy as np
import torch
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist

from lvm.config import load_config
from lvm.embedding import fit_embedding


def _plane(n_points=40, n_landmarks=6, seed=0):
    # Points in the unit square; "maps" exp(-scale |x - l|) give exactly Euclidean distances d = -log v / scale.
    points = np.random.default_rng(seed).random((n_points, 2))
    landmarks = np.arange(n_landmarks)
    return points, landmarks, cdist(points[landmarks], points)                       # (L, n_points)


def test_landmark_mds_places_items_exactly_when_the_distances_are_euclidean(device):
    points, landmarks, distances = _plane()
    maps = torch.as_tensor(np.exp(-distances), device=device)
    embedding = fit_embedding(maps, torch.as_tensor(landmarks, device=device), config=load_config())
    placed = embedding.triangulate(maps.T).cpu().numpy()
    np.testing.assert_allclose(cdist(placed, placed), cdist(points, points), atol=1e-9)


def test_a_distance_below_the_floor_goes_through_the_landmarks_the_item_reaches(device):
    points, landmarks, distances = _plane()
    d = distances * 10.0 / 0.7                         # pairs farther than 0.7: v < exp(-10), some between landmarks
    config = load_config(overrides={"embedding": {"distance_floor": float(np.exp(-10.0))}})
    maps = torch.as_tensor(np.exp(-d), device=device)
    embedding = fit_embedding(maps, torch.as_tensor(landmarks, device=device), config=config)

    known = np.where(d <= 10.0, d, np.inf)
    between = shortest_path(np.ascontiguousarray(np.where(np.isfinite(known[:, landmarks]), known[:, landmarks], 0.0)),
                            directed=False)
    expected = np.where(np.isfinite(known), known, (known[:, None, :] + between[:, :, None]).min(axis=0)).T
    assert np.isinf(known[:, landmarks]).any() and np.isfinite(between).all()
    np.testing.assert_allclose(embedding.item_distances(maps.T).cpu().numpy(), expected, rtol=1e-12)
