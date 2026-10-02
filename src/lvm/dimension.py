"""Intrinsic dimension of a region (config section ``dimension``).

Estimated once per fit from the region's sample (before the cells are
built), so it is measured rather than assumed; in the hierarchy each region
gets its own estimate. It sets how many landmarks are needed: a position in
d dimensions is pinned down by its distances to d + 1 landmarks in general
position (``landmarks.count: dimension``).

The estimators come from scikit-dimension. Configurable choice point
``dimension``; see ``lvm.strategies``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from lvm.config import DimensionConfig, FixedDimensionConfig, MleDimensionConfig, TwoNnDimensionConfig
from lvm.strategies import resolve


@dataclass(frozen=True)
class Dimension:
    d: float   # estimated intrinsic dimension (not rounded)


def choose_dimension(sample: np.ndarray, *, config: DimensionConfig, seed: int = 0) -> Dimension:
    """Intrinsic dimension of the data in ``sample`` with the strategy in ``config.strategy``."""
    strategy, options = resolve(_STRATEGIES, config)
    return Dimension(float(strategy(np.asarray(sample), options=options, seed=seed)))


def _subsample(sample: np.ndarray, size: int, seed: int) -> np.ndarray:
    if sample.shape[0] <= size:
        return sample
    return sample[np.random.default_rng(seed).choice(sample.shape[0], size, replace=False)]


def _mle(sample: np.ndarray, *, options: MleDimensionConfig, seed: int) -> float:
    """Levina & Bickel (2005) maximum-likelihood estimate from k-nearest-neighbour distances."""
    import skdim

    return skdim.id.MLE(K=options.k).fit(_subsample(sample, options.sample_size, seed)).dimension_


def _twonn(sample: np.ndarray, *, options: TwoNnDimensionConfig, seed: int) -> float:
    """Facco et al. (2017): from the ratio of each point's second to first neighbour distance."""
    import skdim

    return skdim.id.TwoNN().fit(_subsample(sample, options.sample_size, seed)).dimension_


def _fixed(sample: np.ndarray, *, options: FixedDimensionConfig, seed: int) -> float:
    return options.d


_STRATEGIES: dict[str, Callable[..., float]] = {
    "mle": _mle,
    "twonn": _twonn,
    "fixed": _fixed,
}
