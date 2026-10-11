from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from experiments.common import style
from experiments.common.datasets import mnist8m_labels
from experiments.sfv.mnist8m.hierarchy_scaling.run_lvm import window

HERE = Path(__file__).resolve().parent
N, SEED = 100_000, 0
PANELS = {"umap": "UMAP", "tsne": "t-SNE", "level1": "LVM level 1", "level1_chart": "LVM level 1 + chart"}


def main() -> None:
    start = window(N, SEED)
    labels = mnist8m_labels(start + N)[start:]
    lvm = np.load(HERE / "embeddings" / f"lvm_n{N}_seed{SEED}.npz")
    maps = {"level1": lvm["level1"], "level1_chart": lvm["level1_chart"],
            **{m: np.load(HERE / "embeddings" / f"{m}_n{N}_seed{SEED}.npz")["Z"] for m in ("umap", "tsne")}}
    colours = style.category_colours(10)
    order = np.random.default_rng(0).permutation(N)
    fig, axes = style.figure("full", aspect=0.3, ncols=len(PANELS))
    for ax, (name, title) in zip(axes, PANELS.items()):
        Z = maps[name][order]
        ax.scatter(Z[:, 0], Z[:, 1], c=labels[order], cmap=colours, vmin=-0.5, vmax=9.5, s=0.3, linewidths=0,
                   rasterized=True)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_axis_off()
    digits = [Line2D([], [], marker="o", linestyle="", color=colours(d), label=str(d)) for d in range(10)]
    fig.legend(handles=digits, title="digit", loc="outside lower center", ncols=10)
    style.save_figure(fig, HERE / "embeddings_global")


if __name__ == "__main__":
    main()
