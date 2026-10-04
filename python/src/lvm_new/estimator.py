from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from lvm.config import Config
from lvm_new.data import DataLike


class LocalizedVoltageMaps(TransformerMixin, BaseEstimator):
    """Localized Voltage Maps: a streaming embedding of data into ``n_components`` dimensions.

    Parameters
    ----------
    n_components : int, default 2
        Dimension of the embedding.
    n_cells : int, default 1000
        Number of cells (graph nodes) per region.
    device : {"auto", "cuda", "cpu"}, default "auto"
        Where to compute. "auto": CUDA if a GPU is available, else the CPU. Each step picks its implementation by the
        device unless the config names one explicitly (e.g. k-means: cuML on the GPU, FAISS on the CPU).
    random_state : int, default 0
        Seed. Each region's seed is derived from it and the region's id.
    local_chart : {"last", "all", "none"}, default "last"
        The second scale: points placed within their cell by the cell's local PCA. "last": charts at the deepest level
        during ``fit``, other levels on demand; "all": charts at every level during ``fit``; "none": plain coordinates.
    chunk_size : int, default 10_000
        Points per chunk when an array is read in pieces. A re-iterable's own pieces are used as they come (re-cutting
        would copy every chunk), so for a loader its batch size is the chunk size. Streaming k-means updates once per
        chunk, so results depend on it; that's also why it is fixed rather than estimated from free memory.
    levels : int, default 1
        Depth of the hierarchy. Only 1 for now; more levels are stage 2.
    config : str, Path or None, default None
        YAML file with any other settings, merged over the package's ``config.yaml``. The direct arguments above
        override the same settings in it.

    Attributes (after ``fit``)
    --------------------------
    root_ : RegionModel
        The fitted model of level 0 (the whole data).
    config_ : Config
        The complete settings used (YAML + direct arguments).
    device_ : str
        The device actually used.
    n_features_in_ : int
        Number of input features.

    A region's own results (centroids, masses, landmarks, rho_g, dimension, ...) are attributes of its RegionModel:
    ``model.root_`` for level 0, ``model.regions(level)`` for any level.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_cells: int = 1000,
        device: str = "auto",
        random_state: int = 0,
        local_chart: str = "last",
        chunk_size: int = 10_000,
        levels: int = 1,
        config: str | Path | None = None,
    ):
        # scikit-learn: store the arguments unchanged, no checking or work here.
        self.n_components = n_components
        self.n_cells = n_cells
        self.device = device
        self.random_state = random_state
        self.local_chart = local_chart
        self.chunk_size = chunk_size
        self.levels = levels
        self.config = config

    # --- public: fitting and placing ------------------------------------------------------------------------------

    def fit(self, X: DataLike, y: None = None) -> "LocalizedVoltageMaps":
        """Fit the model to X (an array or a re-iterable of chunks); returns the model itself.

        ``y`` is ignored (scikit-learn's signature for unsupervised estimators).
        """
        # resolve settings and device -> as_chunks(X, self.chunk_size) (refuses one-shot iterators, as every method
        # taking X does) -> fit_region on the root -> set the fitted attributes
        raise NotImplementedError

    def transform(self, X: DataLike, level: int = 0) -> np.ndarray:
        """Coordinates of the points of X, as one (n, n_components) array.

        ``level``: which level's embedding (only 0 for now). A level without local charts gets them first, from X
        (see ``fit_charts``).
        """
        # np.concatenate(list(self.transform_chunks(X, level)))
        raise NotImplementedError

    def transform_chunks(self, X: DataLike, level: int = 0) -> Iterator[np.ndarray]:
        """Coordinates of X one chunk at a time, for data too big to hold: yields (chunk, n_components) arrays."""
        # for each chunk: route to its region (stage 1: the root) -> region.transform_chunk
        raise NotImplementedError

    def fit_charts(self, X: DataLike, level: int = 0) -> "LocalizedVoltageMaps":
        """Fit the local charts of one level from X (one pass, at most 256 points per cell); returns the model itself.

        Needed only for a level without charts (with local_chart="last", every level but the deepest); transform does
        it on demand. A chart is only as good as its data: a cell needs at least n_components + 1 points.
        """
        raise NotImplementedError

    def regions(self, level: int = 0) -> list:
        """The fitted regions at one depth, as a list of RegionModel (level 0: [root_])."""
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        """Store the fitted model as a folder: model.yaml (versions, settings; readable) + regions/<id>.npz (arrays)."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> "LocalizedVoltageMaps":
        """Load a model stored with ``save`` (arrays loaded without pickle; checks the format version)."""
        raise NotImplementedError

    def _resolve_config(self) -> Config:
        """The package's config.yaml, merged with the ``config`` YAML, then the direct arguments on top."""
        raise NotImplementedError

    def _resolve_device(self) -> str:
        """"cuda" or "cpu": ``device``, with "auto" resolved by whether a GPU is available."""
        raise NotImplementedError

    def _check_arguments(self) -> None:
        """Validate the direct arguments (called by fit, not __init__): ranges, allowed values, levels == 1 for now."""
        raise NotImplementedError

    def _check_fitted(self) -> None:
        """Raise scikit-learn's NotFittedError if ``fit`` hasn't been called."""
        raise NotImplementedError
