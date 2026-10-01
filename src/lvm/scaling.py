"""Choose the ground scaling rho_g for one region (config section ``scaling``).

rho_g sets how far voltage maps reach: node i's ground weight is rho_g * p_i,
and a stronger ground drains current sooner, so maps shrink as rho_g grows
(Corollary 14 of the paper). Each strategy is a function of one region's
kernel K and masses p, so regions can be scaled independently and in parallel.

Configurable choice point ``scaling``; see ``lvm.strategies`` for the
convention shared by every choice point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from lvm.config import CoverageConfig, ScalingConfig, SupportFractionConfig
from lvm.strategies import resolve
from lvm.voltage import solve_grounded_voltage_maps, support_mask


@dataclass(frozen=True)
class RhoChoice:
    rho_g: float
    support_fraction: float   # typical fraction of cells a map covers at rho_g
    n_iter: int               # voltage solves spent on the search


def choose_rho_g(
    K: np.ndarray, p: np.ndarray, *, config: ScalingConfig, tau: float, n_landmarks: int, device: str = "cuda"
) -> RhoChoice:
    """Pick rho_g for a region with the strategy named in ``config.strategy``.

    ``n_landmarks`` is the number of landmarks the region will get (config
    ``landmarks.n_landmarks``); strategies that tie map size to it use it.
    """
    strategy, options = resolve(_STRATEGIES, config)
    return strategy(K, p, options=options, tau=tau, n_landmarks=n_landmarks, device=device)


def typical_support_fraction(
    K: np.ndarray, p: np.ndarray, rho_g: float, *, tau: float, statistic: str, device: str = "cuda"
) -> float:
    """Median or mean over every cell with mass of the fraction of cells its map covers."""
    sources = [[i] for i in np.flatnonzero(p > 0)]
    V = solve_grounded_voltage_maps(K, p, rho_g, sources, device=device)
    fractions = support_mask(V, tau).mean(axis=1)
    return float(np.median(fractions) if statistic == "median" else np.mean(fractions))


def _coverage(
    K: np.ndarray, p: np.ndarray, *, options: CoverageConfig, tau: float, n_landmarks: int, device: str
) -> RhoChoice:
    """Size maps so the region's landmarks cover every cell about ``overlap`` times.

    L landmarks whose maps each cover a fraction f of the cells cover at most
    L f of the region, so the target is f = overlap / L (capped at 1). Too
    small a target leaves cells outside every landmark's support; those cells
    get no information from the level and collapse together in the embedding.
    """
    if n_landmarks < 1 or options.overlap <= 0:
        raise ValueError(f"need n_landmarks >= 1 and overlap > 0, got {n_landmarks} and {options.overlap}")
    target = min(1.0, options.overlap / n_landmarks)
    return _bisect_support(K, p, target=target, options=options, tau=tau, device=device)


def _support_fraction(
    K: np.ndarray, p: np.ndarray, *, options: SupportFractionConfig, tau: float, n_landmarks: int, device: str
) -> RhoChoice:
    """Fixed target: a typical map covers ``options.target`` of the cells, whatever ``n_landmarks``."""
    return _bisect_support(K, p, target=options.target, options=options, tau=tau, device=device)


def _bisect_support(
    K: np.ndarray,
    p: np.ndarray,
    *,
    target: float,
    options: CoverageConfig | SupportFractionConfig,
    tau: float,
    device: str,
) -> RhoChoice:
    """Bisect log(rho_g) until a typical map covers ``target`` of the cells.

    The typical fraction never increases with rho_g (more ground, lower
    voltages everywhere), so bisection brackets the target. Using every cell
    with mass as a candidate source is cheap: all maps come from one
    factorization per probe. If the target lies outside what the bounds can
    reach, the nearest bound is returned.
    """
    lo, hi = options.rho_g_bounds
    if not 0 < lo < hi:
        raise ValueError(f"rho_g_bounds must satisfy 0 < lo < hi, got {options.rho_g_bounds}")

    def fraction(rho_g: float) -> float:
        return typical_support_fraction(K, p, rho_g, tau=tau, statistic=options.statistic, device=device)

    f_lo, f_hi = fraction(lo), fraction(hi)
    if f_lo <= target:   # even the weakest ground gives maps that are too small
        return RhoChoice(lo, f_lo, 2)
    if f_hi >= target:   # even the strongest ground gives maps that are too large
        return RhoChoice(hi, f_hi, 2)

    log_lo, log_hi = math.log(lo), math.log(hi)
    rho_g, f = lo, f_lo
    n_iter = 2
    for _ in range(options.max_iter):
        rho_g = math.exp(0.5 * (log_lo + log_hi))
        f = fraction(rho_g)
        n_iter += 1
        if abs(f - target) <= options.tolerance:
            break
        if f > target:
            log_lo = math.log(rho_g)
        else:
            log_hi = math.log(rho_g)
    return RhoChoice(rho_g, f, n_iter)


_STRATEGIES: dict[str, Callable[..., RhoChoice]] = {
    "coverage": _coverage,
    "support_fraction": _support_fraction,
}
