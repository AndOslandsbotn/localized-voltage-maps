import numpy as np
import torch

from lvm import LocalizedVoltageMaps
from lvm.config import Config
from lvm.data import as_chunks, sample_region
from lvm.graph import build_graph
from lvm.region import RegionModel
from lvm.voltage import _largest_passing, _mutual_information, _reached, threshold, voltage_maps

CANDIDATE_SHARE = 0.2


def level1_kernel(level1: LocalizedVoltageMaps, X: np.ndarray) -> torch.Tensor:
    """Level 1's graph, rebuilt exactly as its fit built it (same sample and seed)."""
    config, device, seed = level1.config_, level1.device_, level1.config_.compute.seed
    sample = sample_region(as_chunks(X, config.data.chunk_size), config=config, device=device, seed=seed)
    kernel, radius = build_graph(level1.root_.centroids, sample, config=config, device=device, seed=seed)
    if not torch.allclose(radius, level1.root_.radius):
        raise ValueError("The rebuilt graph differs from level 1's")
    return kernel


def shifted_candidates(maps: torch.Tensor) -> dict[str, torch.Tensor]:
    """Cells near the corners of the landmarks' basins (three voltages nearly equal) and near the middle of their
    borders (two nearly equal, the third much lower): the top CANDIDATE_SHARE of cells by v3/v1 and by
    v2/v1 - v3/v1, with v1 >= v2 >= v3 a cell's three strongest voltages."""
    top = torch.topk(maps, 3, dim=0).values
    second, third = top[1] / top[0], top[2] / top[0]
    count = max(1, round(CANDIDATE_SHARE * maps.shape[1]))
    return {"corners": torch.topk(third, count).indices, "borders": torch.topk(second - third, count).indices}


def landmarks_among(kernel: torch.Tensor, masses: torch.Tensor, candidates: torch.Tensor, n_landmarks: int,
                    config: Config) -> tuple[torch.Tensor, float]:
    """The library's landmark choice (choose_landmarks) restricted to the candidate cells: candidate maps at the
    start reach, mutual-information selection among them, then the largest reach at which the landmarks still reach
    the data."""
    reach, tau = config.landmarks.reach, config.voltage.threshold
    k = n_landmarks if reach.k is None else min(reach.k, n_landmarks)

    def coverage(rho_g: float) -> float:
        reached = voltage_maps(kernel, masses, rho_g, candidates) >= tau
        return float(torch.median(reached.to(masses.dtype) @ masses))

    start = _largest_passing(coverage, min(k / n_landmarks, reach.share), options=reach)
    maps = threshold(voltage_maps(kernel, masses, start, candidates), tau)
    chosen, _ = _mutual_information(maps, masses, n_landmarks, config.landmarks.mutual_information.noise_std)
    landmarks = candidates[chosen]
    rho_g = _largest_passing(lambda r: _reached(kernel, masses, r, landmarks, tau, reach.share), k, options=reach,
                             guess=start)
    return landmarks, rho_g


def shifted_versions(root: RegionModel, kernel: torch.Tensor, config: Config) -> dict[int, dict]:
    """Three sets of landmarks on level 1's graph, each with its maps: version 1 = level 1's own landmarks, version
    2 near the corners of their basins, version 3 near the middle of their borders."""
    versions = {1: {"landmarks": root.landmarks, "rho_g": root.rho_g, "maps": root.maps}}
    candidates = shifted_candidates(root.maps)
    for version, kind in ((2, "corners"), (3, "borders")):
        landmarks, rho_g = landmarks_among(kernel, root.masses, candidates[kind], len(root.landmarks), config)
        versions[version] = {"landmarks": landmarks, "rho_g": rho_g,
                             "maps": voltage_maps(kernel, root.masses, rho_g, landmarks)}
    return versions


def margins(maps: torch.Tensor) -> np.ndarray:
    """Each cell's margin inside its region, 1 - v2/v1 from its two strongest voltages: 0 on a border, near 1 deep
    inside."""
    top = torch.topk(maps, 2, dim=0).values
    return (1 - top[1] / top[0]).cpu().numpy()
