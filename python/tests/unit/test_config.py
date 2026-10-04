import pytest

from lvm.config import Config, ConfigError, load_config


def test_default_config_loads():
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.compute.device == "cuda"
    assert cfg.cells.masses.max_points is None
    assert cfg.scaling.support_fraction.rho_g_bounds == (1.0e-6, 1.0e6)
    assert isinstance(cfg.voltage.threshold, float)


def test_yaml_file_overrides_only_given_keys(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("compute:\n  device: cpu\ncells:\n  n_cells: 500\n")
    cfg = load_config(path)
    assert cfg.compute.device == "cpu"
    assert cfg.cells.n_cells == 500
    # Untouched keys keep their defaults.
    assert cfg.compute.dtype == load_config().compute.dtype
    assert cfg.cells.kmeans == load_config().cells.kmeans


def test_dict_overrides_apply_after_file(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("partition:\n  strategy: argmax\n")
    cfg = load_config(path, overrides={"partition": {"strategy": "support"}, "stopping": {"max_depth": {"depth": 5}}})
    assert cfg.partition.strategy == "support"
    assert cfg.stopping.max_depth.depth == 5


def test_float_written_without_dot_is_accepted():
    # PyYAML parses "1e-4" as a string; the loader should still give a float.
    cfg = load_config(overrides={"voltage": {"threshold": "1e-4"}})
    assert cfg.voltage.threshold == pytest.approx(1e-4)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"compute": {"devise": "cpu"}}, "unknown config key 'compute.devise'"),
        ({"partition": {"strategy": "nearest"}}, "must be one of"),
        ({"cells": {"n_cells": 10.5}}, "must be an integer"),
        ({"data": {"shuffled": "yes"}}, "must be true or false"),
        ({"scaling": {"support_fraction": {"rho_g_bounds": [1.0]}}}, "list of 2 values"),
        ({"cells": {"n_cells": None}}, "must be an integer, got None"),
    ],
)
def test_invalid_config_is_rejected(overrides, message):
    with pytest.raises(ConfigError, match=message):
        load_config(overrides=overrides)


def test_empty_yaml_file_gives_defaults(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    assert load_config(path) == load_config()
