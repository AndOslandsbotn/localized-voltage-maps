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
from typing import Callable, Sequence

import numpy as np

from lvm.config import CoverageConfig, LandmarksConfig, ReachConfig, ScalingConfig, SupportFractionConfig
from lvm.landmarks import LandmarkSelection, choose_landmarks
from lvm.strategies import resolve
from lvm.voltage import Sources, solve_grounded_voltage_maps, support_mask, threshold_voltages


@dataclass(frozen=True)
class RhoChoice:
    rho_g: float
    support_fraction: float   # typical fraction of cells a map covers at rho_g (nan for reach)
    n_iter: int               # voltage solves spent on the search (a landmark selection counts as one)
    reach: float | None = None  # typical number of chosen landmarks reaching a point (reach only)
    landmarks: LandmarkSelection | None = None  # reach only: the landmarks chosen with rho_g, rows of `sources`
    n_rounds: int | None = None                 # reach only: landmark selections in the alternation


def choose_rho_g(
    K: np.ndarray, p: np.ndarray, *, config: ScalingConfig, tau: float, n_landmarks: int,
    landmarks: LandmarksConfig | None = None, sources: Sources | None = None, device: str = "cuda",
) -> RhoChoice:
    """Pick rho_g for a region with the strategy named in ``config.strategy``.

    ``n_landmarks`` is the number of landmarks the region will get;
    ``landmarks`` is how they are chosen and ``sources`` the candidates they
    are chosen from (default: every cell with mass). ``reach`` needs them, as
    it measures the actually chosen landmarks; it returns its choice in
    ``RhoChoice.landmarks`` so the caller doesn't have to choose again.
    """
    strategy, options = resolve(_STRATEGIES, config)
    return strategy(K, p, options=options, tau=tau, n_landmarks=n_landmarks, landmarks=landmarks, sources=sources,
                    device=device)


def typical_support_fraction(
    K: np.ndarray, p: np.ndarray, rho_g: float, *, tau: float, statistic: str, device: str = "cuda"
) -> float:
    """Median or mean over every cell with mass of the fraction of cells its map covers."""
    sources = [[i] for i in np.flatnonzero(p > 0)]
    V = solve_grounded_voltage_maps(K, p, rho_g, sources, device=device)
    fractions = support_mask(V, tau).mean(axis=1)
    return float(np.median(fractions) if statistic == "median" else np.mean(fractions))


def _coverage(
    K: np.ndarray, p: np.ndarray, *, options: CoverageConfig, tau: float, n_landmarks: int,
    landmarks: LandmarksConfig | None, sources: Sources | None, device: str,
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
    K: np.ndarray, p: np.ndarray, *, options: SupportFractionConfig, tau: float, n_landmarks: int,
    landmarks: LandmarksConfig | None, sources: Sources | None, device: str,
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


def landmark_reach(
    K: np.ndarray, p: np.ndarray, rho_g: float, source_sets: Sequence[Sequence[int]], *, tau: float,
    quantile: float, device: str = "cuda",
) -> float:
    """At ``rho_g``: the mass-weighted ``quantile`` of how many of the given landmarks reach each cell.

    Cells are weighted by mass, so the quantile is over data points: with
    quantile 0.5 half of the data is reached by at least this many landmarks.
    For a fixed set of landmarks this never increases with rho_g (a stronger
    ground lowers every voltage), which is what makes bisection on it exact.
    """
    V = solve_grounded_voltage_maps(K, p, rho_g, source_sets, device=device)
    return _reach_quantile(support_mask(V, tau), p, quantile)


def typical_reach(
    K: np.ndarray, p: np.ndarray, rho_g: float, *, tau: float, n_landmarks: int, landmarks: LandmarksConfig,
    quantile: float, sources: Sources | None = None, device: str = "cuda",
) -> float:
    """At ``rho_g``: choose the landmarks, then ``landmark_reach`` of that choice."""
    sources = _every_cell(p) if sources is None else sources
    selection = _select(K, p, rho_g, sources, tau=tau, n_landmarks=n_landmarks, landmarks=landmarks, device=device)
    return landmark_reach(K, p, rho_g, [sources.sets[i] for i in selection.indices], tau=tau, quantile=quantile,
                          device=device)


def _reach(
    K: np.ndarray, p: np.ndarray, *, options: ReachConfig, tau: float, n_landmarks: int,
    landmarks: LandmarksConfig | None, sources: Sources | None, device: str,
) -> RhoChoice:
    """The most local maps (largest rho_g) for which a typical point is reached by k chosen landmarks.

    k defaults to all landmarks: with landmarks.count = dimension that is
    d + 1, the number needed to triangulate a position in d dimensions.

    The landmarks are chosen from the maps at rho_g, and reach is counted for
    the chosen landmarks, so the two are found by alternation:

    0. Start where a typical candidate map covers min(k / L, 1 - quantile) of
       the data -- what k of L evenly spread landmarks would each need to
       cover. This needs no landmarks, only solves.
    1. Choose the landmarks at the current rho_g (the only expensive step).
    2. With that set fixed, bisect log(rho_g) for the largest rho_g where it
       still reaches k. Exact, since a fixed set's reach only falls as rho_g
       grows.
    3. Stop once rho_g moved by at most ``round_tolerance``; else go to 1.

    Returns the last rho_g with the set its search used: that set reaches k
    at that rho_g exactly, and was chosen at a rho_g within the tolerance.
    If even the weakest ground (lower bound) can't reach k, that bound is
    returned.
    """
    if landmarks is None:
        raise ValueError("the reach strategy needs the landmarks config")
    k = n_landmarks if options.k is None else min(options.k, n_landmarks)
    bounds = options.rho_g_bounds
    sources = _every_cell(p) if sources is None else sources
    mass = np.asarray(p, dtype=np.float64) / np.sum(p)

    def coverage(rho_g: float) -> float:   # median share of the data a candidate map reaches
        V = solve_grounded_voltage_maps(K, p, rho_g, sources.sets, device=device)
        return float(np.median(support_mask(V, tau) @ mass))

    start, _, n_iter = _largest_passing(coverage, min(k / n_landmarks, 1.0 - options.quantile), bounds=bounds,
                                        rel_tolerance=options.rel_tolerance, max_iter=options.start_max_iter)
    rho_g = float(np.clip(start * options.start_factor, *bounds))
    for n_rounds in range(1, options.max_rounds + 1):
        selection = _select(K, p, rho_g, sources, tau=tau, n_landmarks=n_landmarks, landmarks=landmarks,
                            device=device)
        n_iter += 1
        chosen = [sources.sets[i] for i in selection.indices]
        new_rho, reach, evaluations = _largest_passing(
            lambda r: landmark_reach(K, p, r, chosen, tau=tau, quantile=options.quantile, device=device), k,
            bounds=bounds, guess=rho_g, rel_tolerance=options.rel_tolerance, max_iter=options.max_iter,
        )
        n_iter += evaluations
        moved = abs(math.log(new_rho / rho_g))
        rho_g = new_rho
        if moved <= math.log1p(options.round_tolerance):
            break
    return RhoChoice(rho_g, float("nan"), n_iter, reach, landmarks=selection, n_rounds=n_rounds)


def _largest_passing(
    value: Callable[[float], float], target: float, *, bounds: tuple[float, float], rel_tolerance: float,
    max_iter: int, guess: float | None = None,
) -> tuple[float, float, int]:
    """Largest rho_g in ``bounds`` with ``value(rho_g) >= target``, for a ``value`` that never increases.

    Bisects log(rho_g) until the bracket is within ``rel_tolerance`` or
    ``max_iter`` steps. Without ``guess`` the bracket is the whole of
    ``bounds``; with it, the bracket grows from ``guess`` by factors of 4,
    which is quicker when the answer is near. Returns (rho_g, value there,
    evaluations); a bound when the target is met everywhere or nowhere.
    """
    lo_bound, hi_bound = bounds
    if not 0 < lo_bound < hi_bound:
        raise ValueError(f"rho_g_bounds must satisfy 0 < lo < hi, got {bounds}")
    seen: dict[float, float] = {}

    def at(rho_g: float) -> float:
        if rho_g not in seen:
            seen[rho_g] = value(rho_g)
        return seen[rho_g]

    if guess is None:
        lo, hi = lo_bound, hi_bound
        if at(hi) >= target:
            return hi, at(hi), len(seen)
        if at(lo) < target:
            return lo, at(lo), len(seen)
    else:
        lo = hi = float(np.clip(guess, lo_bound, hi_bound))
        if at(lo) >= target:                         # passes: grow upwards until it fails
            while at(hi) >= target:
                if hi == hi_bound:
                    return hi, at(hi), len(seen)
                lo, hi = hi, min(4.0 * hi, hi_bound)
        else:                                        # fails: grow downwards until it passes
            while at(lo) < target:
                if lo == lo_bound:
                    return lo, at(lo), len(seen)
                hi, lo = lo, max(lo / 4.0, lo_bound)
    for _ in range(max_iter):                        # invariant: value(lo) >= target > value(hi)
        if hi / lo <= 1.0 + rel_tolerance:
            break
        mid = math.sqrt(lo * hi)                     # midpoint on the log scale
        if at(mid) >= target:
            lo = mid
        else:
            hi = mid
    return lo, at(lo), len(seen)


def _select(
    K: np.ndarray, p: np.ndarray, rho_g: float, sources: Sources, *, tau: float, n_landmarks: int,
    landmarks: LandmarksConfig, device: str,
) -> LandmarkSelection:
    V = threshold_voltages(solve_grounded_voltage_maps(K, p, rho_g, sources.sets, device=device), tau)
    return choose_landmarks(V, p, config=landmarks, n_landmarks=n_landmarks, cells=sources.cells, tau=tau,
                            device=device)


def _reach_quantile(reached: np.ndarray, p: np.ndarray, quantile: float) -> float:
    """Mass-weighted lower ``quantile`` over cells with mass of how many rows of ``reached`` reach each."""
    active = np.flatnonzero(p > 0)
    counts = np.asarray(reached).sum(axis=0)[active]
    order = np.argsort(counts)
    cumulative = np.cumsum(p[active][order]) / p[active].sum()
    # Lower quantile: at least a (1 - quantile) share of the mass is reached by >= this many landmarks.
    return float(counts[order][min(np.searchsorted(cumulative, quantile), len(order) - 1)])


def _every_cell(p: np.ndarray) -> Sources:
    cells = np.flatnonzero(np.asarray(p) > 0)
    return Sources(sets=[[int(i)] for i in cells], cells=cells)


_STRATEGIES: dict[str, Callable[..., RhoChoice]] = {
    "reach": _reach,
    "coverage": _coverage,
    "support_fraction": _support_fraction,
}
