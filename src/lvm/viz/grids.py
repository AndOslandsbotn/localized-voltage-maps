"""Plot grids of images from flat feature vectors (e.g. cluster centroids)."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_flat_image_grid(
    vectors: np.ndarray,
    image_shape: tuple[int, int],
    *,
    ncols: int = 8,
    cmap: str = "gray",
    figsize: tuple[float, float] | None = None,
    dpi: float | None = None,
    path: str | Path | None = None,
    show: bool = True,
    title: str | None = None,
) -> None:
    """
    Plot each row of ``vectors`` as a 2D image in a row-major grid.

    Parameters
    ----------
    vectors
        Array of shape ``(n, h * w)`` where ``image_shape == (h, w)``.
    image_shape
        Height and width of each image after reshaping from a flat row.
    ncols
        Number of columns in the grid.
    cmap
        Colormap passed to ``imshow``.
    figsize
        Figure size in inches. If ``None``, scales with grid size.
    dpi
        Resolution for figure and ``savefig``. If ``None``, Matplotlib default.
    path
        If set, save the figure to this path. Missing parent directories are created.
    show
        If ``True``, call ``plt.show()`` (set ``False`` for headless runs).
    title
        Optional figure title.
    """
    vectors = np.asarray(vectors)
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2D with shape (n, h*w); got shape {vectors.shape}")
    h, w = image_shape
    expected = h * w
    if vectors.shape[1] != expected:
        raise ValueError(
            f"vectors has feature dim {vectors.shape[1]}, expected h*w={expected} "
            f"for image_shape={image_shape}"
        )
    n = vectors.shape[0]
    if ncols < 1:
        raise ValueError("ncols must be >= 1")

    nrows = math.ceil(n / ncols)
    images = vectors.reshape(n, h, w)

    if figsize is None:
        figsize = (ncols * 1.2, nrows * 1.2)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, squeeze=False)
    if title is not None:
        fig.suptitle(title)

    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        ax.axis("off")
        if i < n:
            ax.imshow(images[i], cmap=cmap, interpolation="nearest")

    plt.tight_layout()
    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi if dpi is not None else fig.dpi)
    if show:
        plt.show()
    plt.close(fig)
