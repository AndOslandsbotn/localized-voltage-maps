import json
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

from experiments.common import style
from experiments.common.datasets import unrolled
from experiments.hsfv.common.frames import align
from lvm import LocalizedVoltageMaps

HERE = Path(__file__).resolve().parent
VERSIONS = {1: "version 1: level 1's landmarks", 2: "version 2: near the basins' corners",
            3: "version 3: near the borders' middles"}
MAPS = {"version 1": "version 1 only", "blend": "blend of 3 versions", "best": "best version per point",
        "windows": "version 1 + windows"}
CLOSE_UP = 4.0


def region_colours(maps: np.ndarray, sheet: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """A colour per region, regions ordered along the sheet so that neighbours differ (hues repeat only ten regions
    apart)."""
    palette = style.category_colours(10).colors
    order = np.argsort(np.argsort(sheet[landmarks, 0]))
    return np.array([palette[o % len(palette)] for o in order])


def draw_versions(fig, grid, root_sheet, saved):
    for row, (version, title) in enumerate(VERSIONS.items()):
        ax = fig.add_subplot(grid[row, :])
        maps, landmarks = saved[f"version{version}_maps"], saved[f"version{version}_landmarks"]
        colours = region_colours(maps, root_sheet, landmarks)
        ax.scatter(root_sheet[:, 0], root_sheet[:, 1], c=colours[maps.argmax(axis=0)], s=9, linewidths=0)
        ax.scatter(root_sheet[landmarks, 0], root_sheet[landmarks, 1], marker="x", c="0.1", s=14, linewidths=0.8)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"({'abc'[row]}) {title}", loc="left")


def draw_curves(ax, results):
    colours = dict(zip(MAPS, style.category_colours(len(MAPS)).colors))
    for name, label in MAPS.items():
        for kind, line in (("straddling", "--"), ("same_region", "-")):
            values = np.array([np.nan if v is None else v for v in results["maps"][name][kind]])
            ax.plot(results["distance"], values, line, color=colours[name], linewidth=1.2)
    ax.axhline(1.0, color="0.6", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_ylim(0, 3)
    ax.set_xlim(0.03, 100)
    ax.set_xlabel("true distance (sheet units)")
    ax.set_ylabel("median map / true distance")
    ax.set_title("(d) pairs across a version-1 border", loc="left")
    handles = [ax.plot([], [], color=colours[name], linewidth=1.2)[0] for name in MAPS]
    handles += [ax.plot([], [], line, color="0.4", linewidth=1.2)[0] for line in ("--", "-")]
    ax.legend(handles, [*MAPS.values(), "across a border", "same region"], fontsize=5, frameon=False,
              loc="upper right", handlelength=1.8)


def close_up(positions):
    """The points within CLOSE_UP of the middle of the most crossed version-1 border, in its two regions."""
    truth, region = positions["truth"], positions["region1"]
    rng = np.random.default_rng(0)
    probe = rng.choice(len(truth), 20_000, replace=False)
    neighbours = NearestNeighbors().fit(truth).radius_neighbors(truth[probe], radius=0.3, return_distance=False)
    pairs = {}
    for i, near in zip(probe, neighbours):
        for j in near[region[near] != region[i]]:
            pairs.setdefault(tuple(sorted((int(region[i]), int(region[j])))), []).append(0.5 * (truth[i] + truth[j]))
    regions, midpoints = max(pairs.items(), key=lambda item: len(item[1]))
    centre = np.median(np.array(midpoints), axis=0)
    inside = np.flatnonzero((np.linalg.norm(truth - centre, axis=1) < CLOSE_UP) & np.isin(region, regions))
    return inside, regions


def draw_close_ups(fig, grid, positions):
    inside, regions = close_up(positions)
    truth = positions["truth"][inside]
    colours = np.where(positions["region1"][inside] == regions[0], 0, 1)
    palette = np.array(style.category_colours(2).colors)
    panels = {"truth": ("true positions", truth)}
    panels |= {name: (MAPS[name], positions[name.replace(" ", "_")][inside])
               for name in ("version 1", "blend", "windows")}
    for column, (letter, (key, (title, Z))) in enumerate(zip("efgh", panels.items())):
        ax = fig.add_subplot(grid[3, column + 1])
        if key != "truth":
            (s, R, t), _ = align(Z, truth)
            Z = s * Z @ R + t
        ax.scatter(Z[:, 0], Z[:, 1], c=palette[colours], s=0.6, linewidths=0, rasterized=True)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_axis_off()
        ax.set_title(f"({letter}) {title}", loc="left")


def main() -> None:
    saved = np.load(HERE / "versions.npz")
    positions = np.load(HERE / "positions.npz")
    results = json.loads((HERE / "results.json").read_text())
    root = LocalizedVoltageMaps.load(HERE / "models" / "level1").root_
    root_sheet = unrolled(root.centroids.cpu().numpy().astype(np.float64))
    fig, ax = style.figure("full", aspect=0.95)
    ax.remove()
    grid = fig.add_gridspec(4, 5, height_ratios=[1, 1, 1, 2.2], width_ratios=[1.6, 1, 1, 1, 1])
    draw_versions(fig, grid, root_sheet, saved)
    draw_curves(fig.add_subplot(grid[3, 0]), results)
    draw_close_ups(fig, grid, positions)
    style.save_figure(fig, HERE / "shifted_regions")


if __name__ == "__main__":
    main()
