import pytest

from lvm_new import LocalizedVoltageMaps


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
    path.write_text("cells:\n  n_cells: 300\n")
    assert LocalizedVoltageMaps()._resolve_config().cells.n_cells == 1000
    assert LocalizedVoltageMaps(config=path)._resolve_config().cells.n_cells == 300
    assert LocalizedVoltageMaps(config={"cells": {"n_cells": 300}})._resolve_config().cells.n_cells == 300
    assert LocalizedVoltageMaps(config=path, n_cells=500)._resolve_config().cells.n_cells == 500
