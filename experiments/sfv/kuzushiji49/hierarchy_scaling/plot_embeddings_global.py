from pathlib import Path

import numpy as np

from experiments.common import style
from experiments.common.datasets import DATA
from experiments.sfv.kuzushiji49.hierarchy_scaling.run_lvm import DATASET, window

HERE = Path(__file__).resolve().parent
N, SEED, CLASSES = 100_000, 0, 49
PANELS = {"umap": "UMAP", "tsne": "t-SNE", "level1": "LVM level 1", "level1_chart": "LVM level 1 + chart"}


def main() -> None:
    start = window(N, SEED)
    labels = np.fromfile(DATA / DATASET / "y.i8", dtype=np.int8)[start:start + N].astype(np.int64)
    lvm = np.load(HERE / "embeddings" / f"lvm_n{N}_seed{SEED}.npz")
    maps = {"level1": lvm["level1"], "level1_chart": lvm["level1_chart"],
            **{m: np.load(HERE / "embeddings" / f"{m}_n{N}_seed{SEED}.npz")["Z"] for m in ("umap", "tsne")}}
    order = np.random.default_rng(0).permutation(N)
    fig, axes = style.figure("full", aspect=0.27, ncols=len(PANELS))
    for ax, (name, title) in zip(axes, PANELS.items()):
        Z = maps[name][order]
        ax.scatter(Z[:, 0], Z[:, 1], c=labels[order], cmap=style.cyclic_colours(), vmin=0, vmax=CLASSES, s=0.3,
                   linewidths=0, rasterized=True)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_axis_off()
    style.save_figure(fig, HERE / "embeddings_global")


if __name__ == "__main__":
    main()
