from __future__ import annotations

import torch

from lvm_new.config import Config


def voltage_maps(kernel: torch.Tensor, masses: torch.Tensor, rho_g: float, sources: torch.Tensor) -> torch.Tensor:
    """
    The map with source s minimises (1/2) sum_ij W_ij (v_i - v_j)^2 + rho_g sum_i p_i v_i^2 with v_s = 1, where
    W = p p^T * K. So L v = c e_s with the grounded Laplacian L = diag(W 1 + rho_g p) - W, and v = G e_s / G_ss with
    G = L^-1: one Cholesky factorisation.
    """
    weights = masses[:, None] * masses[None, :] * kernel
    laplacian = torch.diag(weights.sum(dim=1) + rho_g * masses) - weights
    columns = torch.eye(len(masses), dtype=laplacian.dtype, device=laplacian.device)[:, sources]
    G = torch.cholesky_solve(columns, torch.linalg.cholesky(laplacian))         # (n_cells, len(sources))
    rows = torch.arange(len(sources), device=G.device)
    maps = (G / G[sources, rows]).T.contiguous()
    maps[rows, sources] = 1.0                                                  # exactly, not up to rounding
    return maps


def threshold(maps: torch.Tensor, tau: float) -> torch.Tensor:
    """``maps`` with every voltage below ``tau`` set to 0: each map is local, nothing outside its support."""
    return torch.where(maps >= tau, maps, torch.zeros_like(maps))


def choose_landmarks(kernel: torch.Tensor, masses: torch.Tensor, dimension: float,
                     config: Config) -> tuple[torch.Tensor, float]:
    """(landmarks, rho_g): the landmark cells and their reach, chosen together. The most local maps (largest rho_g)
    for which the chosen landmarks still reach the data."""
    raise NotImplementedError


def landmark_maps(kernel: torch.Tensor, masses: torch.Tensor, rho_g: float, landmarks: torch.Tensor, config: Config, *,
                  device: str):
    """The landmarks' voltage maps over the cells, thresholded for support and for distances."""
    raise NotImplementedError
