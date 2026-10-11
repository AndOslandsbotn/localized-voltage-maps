from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from matplotlib import patheffects

from experiments.common import style
from experiments.common.datasets import DATA
from experiments.sfv.kuzushiji49.hierarchy_scaling.run_lvm import DATASET, REFINE, window
from lvm import LocalizedVoltageMaps
from lvm.config import load_config
from lvm.region import load_region

HERE = Path(__file__).resolve().parent
N, SEED, REGIONS, CLASSES = 100_000, 0, 3, 49
COLOURS = style.cyclic_colours()


def placed(region, config, chart: bool) -> np.ndarray:
    """Where a region's landmarks land in its map: each landmark cell's centroid, placed like any point."""
    model = region if chart else replace(region, chart=None)
    centres = region.centroids[region.landmarks].to(torch.float32)
    return model.transform_chunk(centres, config=config).cpu().numpy()


def crosses(ax, P, colour, size) -> None:
    marks = ax.scatter(P[:, 0], P[:, 1], marker="x", s=size, color=colour, linewidths=0.4 + size / 40, zorder=3)
    marks.set_path_effects([patheffects.withStroke(linewidth=1.0 + size / 25, foreground="white")])


def shown(ax, Z, shade, size, title=None) -> None:
    ax.scatter(Z[:, 0], Z[:, 1], c=shade, cmap=COLOURS, vmin=0, vmax=CLASSES, s=size, linewidths=0,
               rasterized=True)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_axis_off()
    if title:
        ax.set_title(title)


def in_context(ax, Z, members, shade, title=None) -> None:
    """The whole map in grey with the region's points coloured on top."""
    ax.scatter(*np.delete(Z, members, axis=0).T, color="0.88", s=0.1, linewidths=0, rasterized=True)
    shown(ax, Z[members], shade, 0.6, title)


COLUMNS = {"whole": "(a) Level 1 + chart,\nthe region", "level2": "(b) Level 2,\nfitted on the region",
           "level2_chart": "(c) Level 2 + chart"}


def main() -> None:
    start = window(N, SEED)
    labels = np.fromfile(DATA / DATASET / "y.i8", dtype=np.int8)[start:start + N].astype(np.int64)
    lvm = np.load(HERE / "embeddings" / f"lvm_n{N}_seed{SEED}.npz")
    maps = {"whole": lvm["level1_chart"], "level2": lvm["level2"], "level2_chart": lvm["level2_chart"]}
    models = HERE / "models" / f"n{N}_seed{SEED}"
    config = load_config(overrides={**REFINE, "embedding": {"local_chart": "none"}})
    level1_landmarks = placed(LocalizedVoltageMaps.load(models / "level1").root_, config, chart=True)
    region = lvm["region_of_point"]
    fig, axes = style.figure("full", aspect=0.95, nrows=REGIONS, ncols=len(COLUMNS))
    for i, (row, r) in enumerate(zip(axes, np.bincount(region).argsort()[::-1][:REGIONS])):
        members = np.flatnonzero(region == r)
        members = members[np.random.default_rng(0).permutation(len(members))]
        shade = labels[members]
        level2 = load_region(models / "level2" / f"region_{r}.npz", "cuda")
        landmarks = {"level2": placed(level2, config, chart=False),
                     "level2_chart": placed(level2, config, chart=True)}
        for ax, (name, title) in zip(row, COLUMNS.items()):
            if name == "whole":
                in_context(ax, maps[name], members, shade, title if i == 0 else None)
                crosses(ax, np.delete(level1_landmarks, r, axis=0), "0.55", 14)
                crosses(ax, level1_landmarks[[r]], "black", 22)
            else:
                shown(ax, maps[name][members], shade, 0.5, title if i == 0 else None)
                crosses(ax, landmarks[name], "black", 10)
        row[0].text(-0.08, 0.5, f"region {r}\n{len(members)} points", transform=row[0].transAxes, rotation=90,
                    ha="right", va="center")
    style.save_figure(fig, HERE / "embeddings_local")


if __name__ == "__main__":
    main()
