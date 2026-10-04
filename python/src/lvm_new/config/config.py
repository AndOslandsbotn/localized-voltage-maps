from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULTS = Path(__file__).with_name("config.yaml")


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _Choice(_Section):
    """A section that chooses a strategy: ``strategy`` names it, the field of that name holds its options."""

    strategy: str

    @property
    def options(self) -> Any:
        return getattr(self, self.strategy, None)


class ComputeConfig(_Section):
    device: Literal["auto", "cuda", "cpu"]
    seed: int = Field(ge=0)


class DataConfig(_Section):
    chunk_size: int = Field(ge=1)


class SampleConfig(_Choice):
    strategy: Literal["prefix", "reservoir"]
    size: int = Field(ge=1)


class MleOptions(_Section):
    k: int = Field(ge=2)


class FixedDimensionOptions(_Section):
    d: float = Field(gt=0)


class DimensionConfig(_Choice):
    strategy: Literal["mle", "twonn", "fixed"]
    sample_size: int = Field(ge=3)
    mle: MleOptions
    fixed: FixedDimensionOptions


class CumlKMeansOptions(_Section):
    init: Literal["random", "k-means||"]
    max_iter: int = Field(ge=1)
    tol: float = Field(ge=0)


class FaissKMeansOptions(_Section):
    niter: int = Field(ge=1)


class SklearnKMeansOptions(_Section):
    max_iter: int = Field(ge=1)
    tol: float = Field(ge=0)


class KMeansConfig(_Choice):
    strategy: Literal["auto", "cuml", "faiss", "sklearn"]
    cuml: CumlKMeansOptions
    faiss: FaissKMeansOptions
    sklearn: SklearnKMeansOptions


class StreamRefineOptions(_Section):
    passes: int = Field(ge=1)
    max_points: int | None = Field(ge=1)


class RefineConfig(_Choice):
    strategy: Literal["none", "stream"]
    stream: StreamRefineOptions


class CellsConfig(_Section):
    n_cells: int = Field(ge=2)
    kmeans: KMeansConfig
    refine: RefineConfig


class LandmarkMdsConfig(_Section):
    n_components: int = Field(ge=1)


class EmbeddingConfig(_Section):
    landmark_mds: LandmarkMdsConfig
    local_chart: Literal["last", "all", "none"]


class HierarchyConfig(_Section):
    levels: int = Field(ge=1)


class Config(_Section):
    compute: ComputeConfig
    data: DataConfig
    sample: SampleConfig
    dimension: DimensionConfig
    cells: CellsConfig
    embedding: EmbeddingConfig
    hierarchy: HierarchyConfig


def load_config(path: str | Path | Mapping[str, Any] | None = None, overrides: Mapping[str, Any] | None = None) -> Config:
    """The defaults, with the run's settings (a YAML file, or a dict of the same structure) and then ``overrides``
    merged over them."""
    settings = yaml.safe_load(DEFAULTS.read_text())
    run = path if isinstance(path, Mapping) else (yaml.safe_load(Path(path).read_text()) if path else None)
    for update in (run, overrides):
        if update:
            settings = _merge(settings, update)
    return Config.model_validate(settings)


def _merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            merged[key] = _merge(base[key], value)
        else:
            merged[key] = value
    return merged
