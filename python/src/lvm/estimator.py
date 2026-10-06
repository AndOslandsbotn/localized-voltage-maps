from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from lvm.config import Config, load_config
from lvm.data import DataLike, as_chunks
from lvm.region import RegionModel, fit_region, fit_region_chart, load_region, save_region


# Each direct argument of LocalizedVoltageMaps and the setting it stands for.
SETTINGS = {
    "n_components": ("embedding", "n_components"),
    "n_cells": ("cells", "n_cells"),
    "device": ("compute", "device"),
    "random_state": ("compute", "seed"),
    "local_chart": ("embedding", "local_chart"),
    "chunk_size": ("data", "chunk_size"),
    "levels": ("hierarchy", "levels"),
}

FORMAT = 1      # version of the saved model's layout (model.yaml + regions/<id>.npz)


class LocalizedVoltageMaps(TransformerMixin, BaseEstimator):
    """Localized Voltage Maps: an embedding of data into ``n_components`` dimensions by grounded voltage maps.

    Each argument left as None takes its value from ``config``, else from the package defaults.

    Parameters
    ----------
    n_components : int, default 2
        Dimension of the embedding.
    n_cells : int, default 300
        Number of cells (graph nodes) per region.
    device : {"auto", "cuda", "cpu"}, default "auto"
        Where to compute; "auto" uses a GPU if one is available.
    random_state : int, default 0
        Random seed.
    local_chart : {"last", "all", "none"}, default "last"
        Levels at which points are placed within their cell by a local PCA chart.
    chunk_size : int, default 10000
        Points per chunk when an array is read in pieces.
    levels : int, default 1
        Depth of the hierarchy.
    config : str, Path or dict, optional
        Settings merged over the package defaults: a YAML file, or a dict of the same structure.

    Attributes
    ----------
    root_ : RegionModel
        The fitted root region.
    config_ : Config
        The settings used.
    device_ : str
        The device used.
    n_features_in_ : int
        Number of input features.
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
        return np.concatenate(list(self.transform_chunks(X, level)))

    def transform_chunks(self, X: DataLike, level: int = 0) -> Iterator[np.ndarray]:
        check_is_fitted(self)
        if level != 0:
            raise NotImplementedError("levels > 0 are not implemented yet")
        for chunk in as_chunks(X, self.config_.data.chunk_size):
            points = torch.as_tensor(chunk, device=self.device_)
            yield self.root_.transform_chunk(points, config=self.config_).cpu().numpy()

    def fit_charts(self, X: DataLike, level: int = 0) -> "LocalizedVoltageMaps":
        check_is_fitted(self)
        if level != 0:
            raise NotImplementedError("levels > 0 are not implemented yet")
        chunks = as_chunks(X, self.config_.data.chunk_size)
        self.root_ = fit_region_chart(self.root_, chunks, config=self.config_, device=self.device_)
        return self

    def regions(self, level: int = 0) -> list[RegionModel]:
        check_is_fitted(self)
        found = [self.root_]
        for _ in range(level):
            found = [child for region in found for child in region.children]
        if not found:
            raise ValueError(f"The model has no regions at level {level}")
        return found

    def save(self, path: str | Path) -> None:
        check_is_fitted(self)
        folder = Path(path)
        (folder / "regions").mkdir(parents=True, exist_ok=True)
        regions = _all_regions(self.root_)
        model = {
            "format": FORMAT,
            "library": version("localized-voltage-maps"),
            "n_features": self.n_features_in_,
            "params": {name: value for name, value in self.get_params().items() if name != "config"},
            "regions": [_file_name(region.id) for region in regions],
            "settings": self.config_.model_dump(mode="json"),       # complete: defaults, YAML and arguments merged
        }
        (folder / "model.yaml").write_text(yaml.safe_dump(model, sort_keys=False))
        for region in regions:
            save_region(region, folder / "regions" / f"{_file_name(region.id)}.npz")

    @classmethod
    def load(cls, path: str | Path) -> "LocalizedVoltageMaps":
        folder = Path(path)
        model = yaml.safe_load((folder / "model.yaml").read_text())
        if model["format"] != FORMAT:
            raise ValueError(f"Saved model format {model['format']} is not supported; this version reads {FORMAT}")
        if model["regions"] != [_file_name(())]:
            raise NotImplementedError("loading more than one level is not implemented yet")
        estimator = cls(**model["params"])
        estimator.config_ = Config.model_validate(model["settings"])
        estimator.device_ = estimator._resolve_device(estimator.config_)        # this machine's device
        estimator.root_ = load_region(folder / "regions" / f"{_file_name(())}.npz", estimator.device_)
        estimator.n_features_in_ = model["n_features"]
        return estimator

    def _resolve_config(self) -> Config:
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
        device = config.compute.device
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device is 'cuda' but no CUDA GPU is available")
        return device


def _all_regions(region: RegionModel) -> list[RegionModel]:
    return [region] + [r for child in region.children for r in _all_regions(child)]


def _file_name(region_id: tuple[int, ...]) -> str:
    return "-".join(str(i) for i in region_id) if region_id else "root"

