from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from lvm.viz import plot_flat_image_grid


def test_plot_flat_image_grid_writes_file(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x = rng.random((4, 9)).astype(np.float64)
    dest = tmp_path / "grid.png"
    plot_flat_image_grid(x, (3, 3), path=dest, show=False, ncols=2)
    assert dest.is_file()
