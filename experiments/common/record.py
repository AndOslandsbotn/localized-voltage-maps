from __future__ import annotations

import inspect
from importlib.metadata import version

import torch

from lvm import LocalizedVoltageMaps
from lvm.region import RegionModel


def environment(device: str) -> dict:
    """Where a result was computed: device, GPU, library versions."""
    return {"device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "lvm": version("localized-voltage-maps"), "torch": torch.__version__}


def fitted(region: RegionModel) -> dict:
    """What a fit chose for itself, beyond the settings."""
    return {"n_cells": len(region.centroids), "n_landmarks": len(region.landmarks), "dimension": region.dimension,
            "rho_g": region.rho_g, "largest_mass": float(region.masses.max()), "refine_passes": region.refine_passes}


def model_record(model: LocalizedVoltageMaps) -> dict:
    """A fitted model's complete settings and what its fit chose."""
    return {"settings": model.config_.model_dump(mode="json"), "fitted": fitted(model.root_)}


def defaults(function) -> dict:
    """A method function's keyword settings with their default values: what it ran with when called without them."""
    return {name: parameter.default for name, parameter in inspect.signature(function).parameters.items()
            if parameter.default is not inspect.Parameter.empty}

