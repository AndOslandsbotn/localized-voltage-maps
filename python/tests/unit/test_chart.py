import numpy as np
import torch
from scipy.spatial.distance import cdist

from lvm.cells import assign_cells
from lvm.config import load_config
from lvm.embedding import fit_chart


def test_a_chart_keeps_each_cells_shape_and_puts_its_median_point_at_fill_of_the_gap(device):
    # Three cells far apart in 10 dimensions, each cell's points on its own flat 2-D patch: its principal directions.
    rng = np.random.default_rng(0)
    centroids = 10.0 * rng.standard_normal((3, 10))
    points = np.vstack([c + rng.uniform(-1, 1, (100, 2)) @ np.linalg.qr(rng.standard_normal((10, 2)))[0].T
                        for c in centroids]).astype(np.float32)
    coordinates = (rng.standard_normal((300, 2)) + np.repeat([[0, 0], [5, 0], [0, 8]], 100, axis=0)).astype(np.float32)
    x, C, z = (torch.as_tensor(a, device=device) for a in (points, centroids, coordinates))
    config = load_config()
    chart = fit_chart(x, C, z, config=config)
    cells = assign_cells(x, C)
    y = chart.place(x, cells, z).cpu().numpy()

    anchors = chart.anchors.cpu().numpy()
    gaps = cdist(anchors, anchors) + np.diag([np.inf] * 3)
    for c in range(3):
        rows = (cells == c).cpu().numpy()
        scale = chart.scales[c].item()
        np.testing.assert_allclose(cdist(y[rows], y[rows]), scale * cdist(points[rows], points[rows]), atol=1e-4)
        median = np.median(np.linalg.norm(y[rows] - anchors[c], axis=1))
        assert abs(median - config.embedding.chart.fill * gaps[c].min()) < 0.02 * median

    point_origin = load_config(overrides={"embedding": {"chart": {"origin": "point"}}})
    y_point = fit_chart(x, C, z, config=point_origin).place(x, cells, z).cpu().numpy()
    np.testing.assert_allclose(y_point - coordinates, y - anchors[cells.cpu().numpy()], atol=1e-4)
