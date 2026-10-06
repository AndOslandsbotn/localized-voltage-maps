import numpy as np
import pytest

from lvm import LocalizedVoltageMaps


@pytest.mark.parametrize("argument", [{"n_cells": 1}, {"device": "gpu"}, {"local_chart": True}])
def test_a_bad_argument_is_refused_by_name(argument):
    (name, _), = argument.items()
    with pytest.raises(ValueError, match=name):
        LocalizedVoltageMaps(**argument)._resolve_config()


def test_more_than_one_level_is_not_implemented_yet():
    with pytest.raises(NotImplementedError):
        LocalizedVoltageMaps(levels=2)._resolve_config()


def test_a_given_argument_beats_the_yaml_which_beats_the_defaults(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("cells:\n  n_cells: 200\n")
    assert LocalizedVoltageMaps()._resolve_config().cells.n_cells == 300
    assert LocalizedVoltageMaps(config=path)._resolve_config().cells.n_cells == 200
    assert LocalizedVoltageMaps(config={"cells": {"n_cells": 200}})._resolve_config().cells.n_cells == 200
    assert LocalizedVoltageMaps(config=path, n_cells=500)._resolve_config().cells.n_cells == 500


def test_a_saved_model_loads_and_places_points_exactly_as_before(tmp_path, device):
    rng = np.random.default_rng(0)
    X = np.vstack([c + 0.3 * rng.standard_normal((500, 5)) for c in 5 * rng.standard_normal((4, 5))]).astype(np.float32)
    model = LocalizedVoltageMaps(n_cells=40, local_chart="none", device=device).fit(X)
    model.fit_charts(X)
    before = model.transform(X)
    model.save(tmp_path / "model")
    loaded = LocalizedVoltageMaps.load(tmp_path / "model")
    assert loaded.regions(0)[0].chart is not None
    np.testing.assert_array_equal(loaded.transform(X), before)
