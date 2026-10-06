"""Shared test setup.

Layout: unit/ has one file per library module (test_graph.py tests lvm/graph.py, ...). The old library's tests
(lvm_old) are in ../tests_old and run with ``pytest tests_old``.

GPU: tests marked ``gpu`` need a CUDA GPU and are skipped without one; ``pytest -m "not gpu"`` deselects them.
A test that takes a ``device`` argument runs on the CPU and, marked ``gpu``, on CUDA.
"""

import pytest
import torch


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def device(request) -> str:
    """The device to run on: "cpu", and "cuda" (marked gpu)."""
    return request.param


def pytest_collection_modifyitems(config, items):
    """Skip the tests marked gpu when there is no CUDA GPU."""
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="needs a CUDA GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
