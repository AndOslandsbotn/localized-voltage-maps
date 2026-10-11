import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from experiments.common.datasets import load
from experiments.common.guard import watch_memory
from experiments.common.record import environment
from lvm.cells import assign_cells, fit_cells
from lvm.config import load_config
from lvm.voltage import point_voltages, voltage_maps

HERE = Path(__file__).resolve().parent
N, SEEDS, N_CELLS, N_LANDMARKS = 5_000, (0, 1, 2), 300, 15
TAU, FLOOR, REACH, SHOWN = 1e-3, 1e-10, 0.5, 4_000
K_POINTS = 10 * N // N_CELLS
PLACEMENTS, MASSES = ("kmeans", "farthest"), ("uniform", "counted")


def main() -> None:
    watch_memory(limit_mb=3000)
    rows, examples = [], {}
    for seed in SEEDS:
        X = torch.as_tensor(load("eval", n=N, seed=seed)[0].astype(np.float32), device="cuda")
        radius = float(torch.cdist(X, X).kthvalue(K_POINTS + 1, dim=1).values.median())
        landmarks = torch.as_tensor(np.random.default_rng(seed).choice(N, N_LANDMARKS, replace=False), device="cuda")
        point_kernel = radial(X, radius)
        point_masses = torch.full((N,), 1.0 / N, dtype=torch.float64, device="cuda")
        rho_g = reach_rho_g(point_kernel, point_masses, landmarks)
        reference = voltage_maps(point_kernel, point_masses, rho_g, landmarks)
        for placement in PLACEMENTS:
            config = load_config(overrides={"cells": {"n_cells": N_CELLS, "placement": placement}})
            centroids = fit_cells(X, config=config, device="cuda", seed=seed)
            counts = torch.bincount(assign_cells(X, centroids), minlength=len(centroids))
            centroids, counts = centroids[counts > 0], counts[counts > 0]
            sources = assign_cells(X[landmarks], centroids)
            cell_kernel = radial(centroids, radius)
            for masses_name in MASSES:
                masses = (counts / counts.sum() if masses_name == "counted"
                          else torch.full((len(counts),), 1.0 / len(counts), device="cuda")).to(torch.float64)
                maps = voltage_maps(cell_kernel, masses, rho_g, sources)
                full_radius = torch.full((len(centroids),), radius, device="cuda")
                voltages = point_voltages(X, centroids, masses, full_radius, maps, rho_g, config=config).T
                reference_distance, cell_distance = compare(reference, voltages, landmarks)
                rows.append({"placement": placement, "masses": masses_name, "seed": seed, "rho_g": rho_g,
                             "radius": radius, "largest_cell": int(counts.max()),
                             "rank_correlation": float(spearmanr(reference_distance, cell_distance).statistic),
                             "median_relative_error": float(np.median(np.abs(cell_distance - reference_distance)
                                                                      / reference_distance)),
                             "reached_share": float(np.mean(cell_distance <= -np.log(TAU)))})
                if seed == SEEDS[0]:
                    shown = np.random.default_rng(seed).choice(len(reference_distance), SHOWN, replace=False)
                    examples[f"{placement}_{masses_name}"] = {"reference": reference_distance[shown].tolist(),
                                                              "cells": cell_distance[shown].tolist()}
    results = {"dataset": "mnist", "n": N, "n_cells": N_CELLS, "n_landmarks": N_LANDMARKS, "k_points": K_POINTS,
               "tau": TAU, "reach": REACH, "environment": environment("cuda"), "rows": rows, "examples": examples}
    (HERE / "results.json").write_text(json.dumps(results, indent=2))


def radial(points: torch.Tensor, radius: float) -> torch.Tensor:
    kernel = (torch.cdist(points, points) <= radius).to(torch.float64)
    kernel.fill_diagonal_(0.0)
    return kernel


def reach_rho_g(kernel: torch.Tensor, masses: torch.Tensor, landmarks: torch.Tensor) -> float:
    """The rho_g at which the median landmark map reaches REACH of the points (v >= TAU), by bisection in log rho_g."""
    low, high = -10.0, 6.0
    for _ in range(40):
        middle = (low + high) / 2
        reached = (voltage_maps(kernel, masses, 10.0 ** middle, landmarks) >= TAU).double().mean(dim=1)
        low, high = (middle, high) if float(reached.median()) >= REACH else (low, middle)
    return 10.0 ** low


def compare(reference: torch.Tensor, voltages: torch.Tensor, landmarks: torch.Tensor):
    """-log v of the point graph and of the cell model at every point the point graph reaches, landmarks excluded."""
    inside = reference >= TAU
    inside[torch.arange(len(landmarks)), landmarks] = False
    reference_distance = -torch.log(reference[inside])
    cell_distance = -torch.log(voltages[inside].clamp(min=FLOOR))
    return reference_distance.cpu().numpy(), cell_distance.cpu().numpy()


if __name__ == "__main__":
    main()
