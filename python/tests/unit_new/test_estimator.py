import pytest

from lvm_new import LocalizedVoltageMaps


@pytest.mark.parametrize("argument", [{"n_cells": 1}, {"device": "gpu"}, {"local_chart": True}])
def test_a_bad_argument_is_refused_by_name(argument):
    (name, _), = argument.items()
    with pytest.raises(ValueError, match=name):
        LocalizedVoltageMaps(**argument)._check_arguments()


def test_more_than_one_level_is_not_implemented_yet():
    with pytest.raises(NotImplementedError):
        LocalizedVoltageMaps(levels=2)._check_arguments()
