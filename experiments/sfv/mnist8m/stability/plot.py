import json
from pathlib import Path

import numpy as np

from experiments.common import style

HERE = Path(__file__).resolve().parent
LABELS = {"lvm": "LVM", "lvm_chart": "LVM + local chart", "umap": "UMAP", "tsne": "t-SNE",
          "laplacian_eigenmaps": "Laplacian eigenmaps", "landmark_isomap": "Landmark Isomap"}
PANELS = (("procrustes", "(a) between seeds", "Procrustes disparity $\\downarrow$"),
          ("neighbour_agreement", "(b) between seeds", "neighbour agreement $\\uparrow$"))


def main() -> None:
    rows = json.loads((HERE / "results.json").read_text())["rows"]
    fig, axes = style.figure("full", aspect=0.32, ncols=len(PANELS), sharey=True)
    for ax, (metric, title, label) in zip(axes, PANELS):
        for y, method in enumerate(LABELS):
            values = [r[metric] for r in rows if r["method"] == method]
            ax.barh(y, np.mean(values), xerr=np.std(values), color=style.method_colour(method),
                    error_kw={"linewidth": 0.5, "capsize": 1.0})
        ax.set_xlabel(label)
        ax.set_title(title, loc="left")
    axes[0].set_yticks(range(len(LABELS)), list(LABELS.values()))
    axes[0].invert_yaxis()
    style.save_figure(fig, HERE / "stability")


if __name__ == "__main__":
    main()
