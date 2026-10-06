from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from lvm_new.config import Config, load_config
from lvm_new.data import DataLike, as_chunks
from lvm_new.region import fit_region


# Each direct argument of LocalizedVoltageMaps and the setting it stands for.
SETTINGS = {
    "n_components": ("embedding", "landmark_mds", "n_components"),
    "n_cells": ("cells", "n_cells"),
    "device": ("compute", "device"),
    "random_state": ("compute", "seed"),
    "local_chart": ("embedding", "local_chart"),
    "chunk_size": ("data", "chunk_size"),
    "levels": ("hierarchy", "levels"),
}


class LocalizedVoltageMaps(TransformerMixin, BaseEstimator):
    """Localized Voltage Maps: a streaming embedding of data into ``n_components`` dimensions.

    Every direct argument is a setting (lvm_new/config/config.yaml). None means "not given": the value then comes from
    the ``config`` YAML if it sets it, else from the package defaults. A given argument wins over both.

    Parameters
    ----------
    n_components : int or None (setting embedding.landmark_mds.n_components, default 2)
        Dimension of the embedding.
    n_cells : int or None (setting cells.n_cells, default 1000)
        Number of cells (graph nodes) per region.
    device : {"auto", "cuda", "cpu"} or None (setting compute.device, default "auto")
        Where to compute. "auto": CUDA if a GPU is available, else the CPU. Each step picks its implementation by the
        device unless the config names one explicitly (e.g. k-means: cuML on the GPU, FAISS on the CPU).
    random_state : int or None (setting compute.seed, default 0)
        Seed. Each region's seed is derived from it and the region's id.
    local_chart : {"last", "all", "none"} or None (setting embedding.local_chart, default "last")
        The second scale: points placed within their cell by the cell's local PCA. "last": charts at the deepest level
        during ``fit``, other levels on demand; "all": charts at every level during ``fit``; "none": plain coordinates.
    chunk_size : int or None (setting data.chunk_size, default 10_000)
        Points per chunk when an array is read in pieces. A re-iterable's own pieces are used as they come (re-cutting
        would copy every chunk), so for a loader its batch size is the chunk size. Streaming k-means updates once per
        chunk, so results depend on it; that's also why it is fixed rather than estimated from free memory.
    levels : int or None (setting hierarchy.levels, default 1)
        Depth of the hierarchy. Only 1 for now; more levels are stage 2.
    config : str, Path, dict or None, default None
        Any settings, merged over the package's defaults: a YAML file, or a dict of the same structure
        (e.g. ``{"graph": {"radius": {"strategy": "knn"}}}``).

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
        n_components: int | None = None,
        n_cells: int | None = None,
        device: str | None = None,
        random_state: int | None = None,
        local_chart: str | None = None,
        chunk_size: int | None = None,
        levels: int | None = None,
        config: str | Path | dict | None = None,
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


    def fit(self, X: DataLike, y: None = None) -> "LocalizedVoltageMaps":
        config = self._resolve_config()
        device = self._resolve_device(config)
        chunks = as_chunks(X, config.data.chunk_size)
        root = fit_region(chunks, config, device=device, region_id=(), seed=config.compute.seed)
        self.config_, self.device_, self.root_ = config, device, root
        self.n_features_in_ = root.n_features
        return self


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
        """The package defaults, merged with the ``config`` YAML, then the direct arguments that were given."""
        overrides = {}
        for name, path in SETTINGS.items():
            value = getattr(self, name)
            if value is not None:
                section = overrides
                for key in path[:-1]:
                    section = section.setdefault(key, {})
                section[path[-1]] = value
        config = load_config(self.config, overrides)
        if config.hierarchy.levels != 1:
            raise NotImplementedError("levels > 1 is not implemented yet")
        return config

    def _resolve_device(self, config: Config) -> str:
        """"cuda" or "cpu": the setting compute.device, with "auto" resolved by whether a GPU is available."""
        device = config.compute.device
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device is 'cuda' but no CUDA GPU is available")
        return device
