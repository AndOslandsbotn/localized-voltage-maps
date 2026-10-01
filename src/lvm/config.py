"""Load and validate the run configuration.

The defaults live in ``config.yaml`` next to this module and are the single
source of default values -- the dataclasses below only describe the structure
and types. A run passes its own YAML (and/or a dict of overrides) containing
just the keys it changes; those are merged over the defaults.

Every choice point uses the pattern ``strategy: <name>`` plus a ``<name>:``
subsection holding that method's options. Strategy fields are typed as
``Literal[...]`` so an unknown strategy name is rejected at load time.
"""

from __future__ import annotations

import copy
import dataclasses
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Union, get_args, get_origin, get_type_hints

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


class ConfigError(ValueError):
    """Raised when a config file has unknown keys, missing keys or bad values."""


@dataclass(frozen=True)
class ComputeConfig:
    device: Literal["cuda", "cpu"]
    dtype: Literal["float64", "float32"]
    seed: int


@dataclass(frozen=True)
class CsvConfig:
    delimiter: str
    skip_header: int


@dataclass(frozen=True)
class DataConfig:
    path: str | None
    format: Literal["csv"]
    csv: CsvConfig
    chunk_size: int
    shuffled: bool


@dataclass(frozen=True)
class KMeansConfig:
    max_iter: int
    tol: float
    n_local_trials: int
    init_sample_size: int | None


@dataclass(frozen=True)
class MassesConfig:
    rel_error: float
    max_points: int | None


@dataclass(frozen=True)
class CellsConfig:
    n_cells: int
    sample_size: int
    kmeans: KMeansConfig
    masses: MassesConfig


@dataclass(frozen=True)
class KernelConfig:
    strategy: Literal["radial"]


@dataclass(frozen=True)
class CentroidSpacingConfig:
    multiplier: float


@dataclass(frozen=True)
class KnnRadiusConfig:
    k: int


@dataclass(frozen=True)
class RadiusConfig:
    strategy: Literal["knn", "centroid_spacing"]
    knn: KnnRadiusConfig
    centroid_spacing: CentroidSpacingConfig


@dataclass(frozen=True)
class GraphConfig:
    kernel: KernelConfig
    radius: RadiusConfig
    edge_weighting: Literal["mass"]


@dataclass(frozen=True)
class GroundConfig:
    weighting: Literal["mass"]


@dataclass(frozen=True)
class SourcesConfig:
    strategy: Literal["single_node"]


@dataclass(frozen=True)
class VoltageConfig:
    solver: Literal["dense_inverse"]
    threshold: float


@dataclass(frozen=True)
class SupportFractionConfig:
    target: float
    statistic: Literal["median", "mean"]
    tolerance: float
    max_iter: int
    rho_g_bounds: tuple[float, float]


@dataclass(frozen=True)
class CoverageConfig:
    overlap: float
    statistic: Literal["median", "mean"]
    tolerance: float
    max_iter: int
    rho_g_bounds: tuple[float, float]


@dataclass(frozen=True)
class ScalingConfig:
    strategy: Literal["coverage", "support_fraction"]
    coverage: CoverageConfig
    support_fraction: SupportFractionConfig


@dataclass(frozen=True)
class MutualInformationConfig:
    noise_std: float
    precision: Literal["float32", "float64"]
    scratch_mb: int


@dataclass(frozen=True)
class LandmarksConfig:
    strategy: Literal["mutual_information"]
    n_landmarks: int
    mutual_information: MutualInformationConfig


@dataclass(frozen=True)
class ExtensionConfig:
    strategy: Literal["harmonic"]


@dataclass(frozen=True)
class LogMdsConfig:
    n_components: int


@dataclass(frozen=True)
class EmbeddingConfig:
    strategy: Literal["log_mds"]
    log_mds: LogMdsConfig


@dataclass(frozen=True)
class UncoveredConfig:
    strategy: Literal["highest_voltage"]


@dataclass(frozen=True)
class PartitionConfig:
    strategy: Literal["argmax", "support"]
    uncovered: UncoveredConfig


@dataclass(frozen=True)
class MaxDepthConfig:
    depth: int


@dataclass(frozen=True)
class StoppingConfig:
    strategy: Literal["max_depth"]
    max_depth: MaxDepthConfig


@dataclass(frozen=True)
class OutputConfig:
    dir: str


@dataclass(frozen=True)
class Config:
    compute: ComputeConfig
    data: DataConfig
    cells: CellsConfig
    graph: GraphConfig
    ground: GroundConfig
    sources: SourcesConfig
    voltage: VoltageConfig
    scaling: ScalingConfig
    landmarks: LandmarksConfig
    extension: ExtensionConfig
    embedding: EmbeddingConfig
    partition: PartitionConfig
    stopping: StoppingConfig
    output: OutputConfig

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def load_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> Config:
    """Load the default config, merge ``path`` and then ``overrides`` over it.

    Parameters
    ----------
    path : path to a YAML file, optional
        Run-specific settings. Only the keys that differ from the defaults
        are needed.
    overrides : nested dict, optional
        Applied last, e.g. ``{"compute": {"device": "cpu"}}``.
    """
    merged = _read_yaml(DEFAULT_CONFIG_PATH)
    if path is not None:
        merged = _merge(merged, _read_yaml(Path(path)), where="")
    if overrides:
        merged = _merge(merged, overrides, where="")
    return _build(Config, merged, where="")


def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        content = yaml.safe_load(f)
    if content is None:
        return {}
    if not isinstance(content, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(content).__name__}")
    return content


def _merge(base: dict, update: Mapping[str, Any], where: str) -> dict:
    """Recursively merge ``update`` into a copy of ``base``; unknown keys are an error."""
    out = copy.deepcopy(base)
    for key, value in update.items():
        name = f"{where}.{key}" if where else key
        if key not in out:
            raise ConfigError(f"unknown config key '{name}'")
        if isinstance(out[key], dict) and isinstance(value, Mapping):
            out[key] = _merge(out[key], value, where=name)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _build(cls: type, raw: Any, where: str) -> Any:
    """Turn a nested dict into the dataclass ``cls``, checking keys and types."""
    if not isinstance(raw, Mapping):
        raise ConfigError(f"'{where or '<root>'}' must be a mapping, got {raw!r}")
    fields = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - fields
    missing = fields - set(raw)
    if unknown:
        raise ConfigError(f"unknown key(s) in '{where or '<root>'}': {sorted(unknown)}")
    if missing:
        raise ConfigError(f"missing key(s) in '{where or '<root>'}': {sorted(missing)}")
    hints = get_type_hints(cls)
    return cls(**{name: _convert(hints[name], raw[name], f"{where}.{name}" if where else name) for name in fields})


def _convert(tp: Any, value: Any, where: str) -> Any:
    origin = get_origin(tp)

    if dataclasses.is_dataclass(tp):
        return _build(tp, value, where)

    if origin is Literal:
        allowed = get_args(tp)
        if value not in allowed:
            raise ConfigError(f"'{where}' must be one of {list(allowed)}, got {value!r}")
        return value

    if origin in (Union, types.UnionType):
        args = get_args(tp)
        if value is None:
            if type(None) in args:
                return None
            raise ConfigError(f"'{where}' must not be null")
        (inner,) = [a for a in args if a is not type(None)]
        return _convert(inner, value, where)

    if origin is tuple:
        args = get_args(tp)
        if not isinstance(value, (list, tuple)) or len(value) != len(args):
            raise ConfigError(f"'{where}' must be a list of {len(args)} values, got {value!r}")
        return tuple(_convert(a, v, f"{where}[{i}]") for i, (a, v) in enumerate(zip(args, value)))

    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"'{where}' must be true or false, got {value!r}")
        return value

    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"'{where}' must be an integer, got {value!r}")
        return value

    if tp is float:
        # PyYAML reads e.g. "1e-3" (no dot) as a string, so accept numeric strings.
        if isinstance(value, bool):
            raise ConfigError(f"'{where}' must be a number, got {value!r}")
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ConfigError(f"'{where}' must be a number, got {value!r}") from None

    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"'{where}' must be a string, got {value!r}")
        return value

    raise TypeError(f"unsupported config field type {tp!r} at '{where}'")
