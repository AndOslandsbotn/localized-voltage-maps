from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULTS = Path(__file__).with_name("config.yaml")


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ComputeConfig(_Section):
    device: Literal["auto", "cuda", "cpu"]
    seed: int = Field(ge=0)


class DataConfig(_Section):
    chunk_size: int = Field(ge=1)


class SampleConfig(_Section):
    strategy: Literal["prefix", "reservoir"]
    size: int = Field(ge=1)


class MleOptions(_Section):
    k: int = Field(ge=2)


class FixedDimensionOptions(_Section):
    d: float = Field(gt=0)


class DimensionConfig(_Section):
    strategy: Literal["mle", "fixed"]
    sample_size: int = Field(ge=3)
    mle: MleOptions
    fixed: FixedDimensionOptions


class CumlKMeansOptions(_Section):
    init: Literal["random", "k-means||"]
    max_iter: int = Field(ge=1)
    tol: float = Field(ge=0)


class LloydKMeansOptions(_Section):
    iterations: int = Field(ge=1)


class KMeansConfig(_Section):
    strategy: Literal["auto", "cuml", "lloyd"]
    cuml: CumlKMeansOptions
    lloyd: LloydKMeansOptions


class RefineConfig(_Section):
    passes: int = Field(ge=0)
    max_points: int | None = Field(ge=1)


class MassesConfig(_Section):
    rel_error: float = Field(gt=0, lt=1)
    max_points: int | None = Field(ge=1)


class CellsConfig(_Section):
    n_cells: int = Field(ge=2)
    kmeans: KMeansConfig
    refine: RefineConfig
    masses: MassesConfig


class KOptions(_Section):
    k: int = Field(ge=1)


class PointKnnOptions(_Section):
    k: int = Field(ge=1)
    sample_size: int = Field(ge=1)


class CentroidSpacingOptions(_Section):
    multiplier: float = Field(gt=0)


class RadiusConfig(_Section):
    strategy: Literal["adaptive_per_cell", "knn", "point_knn", "centroid_spacing"]
    adaptive_per_cell: KOptions
    knn: KOptions
    point_knn: PointKnnOptions
    centroid_spacing: CentroidSpacingOptions


class GaussianKernelOptions(_Section):
    sigma: float = Field(gt=0)
    cutoff: float = Field(gt=0)


class KernelConfig(_Section):
    strategy: Literal["radial", "tapered", "gaussian"]
    gaussian: GaussianKernelOptions


class GraphConfig(_Section):
    radius: RadiusConfig
    kernel: KernelConfig
    connect: bool


class VoltageConfig(_Section):
    threshold: float = Field(gt=0, lt=1)


class MultiplierOptions(_Section):
    multiplier: float = Field(gt=0)


class FixedCountOptions(_Section):
    n: int = Field(ge=1)


class CountConfig(_Section):
    strategy: Literal["dimension", "fixed"]
    dimension: MultiplierOptions
    fixed: FixedCountOptions


class MutualInformationOptions(_Section):
    noise_std: float = Field(gt=0)


class ReachOptions(_Section):
    k: int | None = Field(ge=1)
    share: float = Field(gt=0, le=1)
    rel_tolerance: float = Field(gt=0)
    rho_g_bounds: tuple[float, float]

    @model_validator(mode="after")
    def _ordered_bounds(self) -> ReachOptions:
        low, high = self.rho_g_bounds
        if not 0 < low < high:
            raise ValueError(f"rho_g_bounds must satisfy 0 < low < high, got {self.rho_g_bounds}")
        return self


class LandmarksConfig(_Section):
    count: CountConfig
    mutual_information: MutualInformationOptions
    reach: ReachOptions


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
    graph: GraphConfig
    voltage: VoltageConfig
    landmarks: LandmarksConfig
    embedding: EmbeddingConfig
    hierarchy: HierarchyConfig


def load_config(path: str | Path | Mapping[str, Any] | None = None, overrides: Mapping[str, Any] | None = None) -> Config:
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
