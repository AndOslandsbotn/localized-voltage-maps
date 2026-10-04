from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULTS = Path(__file__).with_name("config.yaml")


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ComputeConfig(_Section):
    device: Literal["cuda", "cpu"]
    seed: int = Field(ge=0)


class DataConfig(_Section):
    chunk_size: int = Field(ge=1)


class CellsConfig(_Section):
    n_cells: int = Field(ge=2)


class LandmarkMdsConfig(_Section):
    n_components: int = Field(ge=1)


class LocalScaleConfig(_Section):
    strategy: Literal["pca", "none"]


class EmbeddingConfig(_Section):
    landmark_mds: LandmarkMdsConfig
    local_scale: LocalScaleConfig


class Config(_Section):
    compute: ComputeConfig
    data: DataConfig
    cells: CellsConfig
    embedding: EmbeddingConfig


def load_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> Config:
    settings = yaml.safe_load(DEFAULTS.read_text())
    for update in (yaml.safe_load(Path(path).read_text()) if path else None, overrides):
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
