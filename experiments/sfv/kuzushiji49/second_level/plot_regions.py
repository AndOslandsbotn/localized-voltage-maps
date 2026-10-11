import json
from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap

from experiments.common import style
from experiments.common.datasets import load
from experiments.sfv.kuzushiji49.second_level.run import DATASET, N_FIT

HERE = Path(__file__).resolve().parent
PROBE = HERE.parents[1] / "scaling_probe" / "embeddings"
SEED, SHOWN_REGIONS, TOP = 0, 3, 9
COLUMNS = {"level1": "LVM level 1", "new": "LVM level 2", "new_chart": "LVM level 2\n+ local chart",
           "umap_new": "UMAP per region", "tsne_new": "t-SNE per region"}


def main() -> None:
    labels = load("eval", n=N_FIT, seed=0, dataset=DATASET)[1]
    saved = np.load(HERE / "embeddings" / f"seed{SEED}.npz")
    embeddings = {"level1": saved["level1"], "new": saved["new"],
                  "new_chart": np.load(HERE / "embeddings" / f"chart_seed{SEED}.npz")["new"],
                  "umap_new": np.load(HERE / "embeddings" / f"umap_seed{SEED}.npz")["new"],
                  "tsne_new": np.load(HERE / "embeddings" / f"tsne_seed{SEED}.npz")["new"]}
    sizes = json.loads(str(saved["sizes"]))["fit"]
    regions = np.argsort(sizes)[::-1][:SHOWN_REGIONS]
    colours = ListedColormap([*style.category_colours(TOP).colors, "#c8c8c8"])
    fig, axes = style.figure("full", aspect=0.65, nrows=SHOWN_REGIONS, ncols=len(COLUMNS))
    for row, r in zip(axes, regions):
        members = np.flatnonzero(saved["region_of_fit"] == r)
        top = np.bincount(labels[members]).argsort()[::-1][:TOP]
        shade = np.array([np.flatnonzero(top == c)[0] if c in top else TOP for c in labels[members]])
        for ax, (name, title) in zip(row, COLUMNS.items()):
            Z = embeddings[name][members]
            ax.scatter(Z[:, 0], Z[:, 1], c=shade, cmap=colours, vmin=-0.5, vmax=TOP + 0.5, s=1.35, linewidths=0,
                       rasterized=True)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_axis_off()
            if r == regions[0]:
                ax.set_title(title)
        row[0].text(-0.08, 0.5, f"region {r}\n{len(members)} points", transform=row[0].transAxes, rotation=90,
                    ha="right", va="center")
    style.save_figure(fig, HERE / "regions")


if __name__ == "__main__":
    main()
