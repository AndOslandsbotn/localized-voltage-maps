from __future__ import annotations

import numpy as np

from lvm_new.config import Config


def estimate_dimension(sample: np.ndarray, config: Config, *, device: str, seed: int) -> float:
    """The region's intrinsic dimension d̂, estimated from its sample."""
    raise NotImplementedError
