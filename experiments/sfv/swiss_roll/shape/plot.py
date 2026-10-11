import json
from pathlib import Path

import numpy as np

from experiments.common import style

HERE = Path(__file__).resolve().parent
LABELS = {"lvm": "LVM", "lvm_chart": "LVM + local chart", "umap": "UMAP", "tsne": "t-SNE",
          "laplacian_eigenmaps": "Laplacian eigenmaps", "landmark_isomap": "Landmark Isomap"}
PANELS = {"procrustes": ("(a) Procrustes vs the unrolling", "disparity (lower is better)"),
          "along_roll": ("(b) order along the roll", "rank correlation")}


def main() -> None:
    rows = json.loads((HERE / "results.json").read_text())["rows"]
    fig, axes = style.figure("full", aspect=0.32, ncols=len(PANELS), sharey=True)
    for ax, (metric, (title, label)) in zip(axes, PANELS.items()):
        values = np.array([[row[metric] for row in rows if row["method"] == method] for method in LABELS])
        mean, spread = values.mean(axis=1), values.std(axis=1)
        ax.barh(list(LABELS.values()), mean, xerr=spread, color=[style.method_colour(method) for method in LABELS],
                error_kw={"linewidth": 0.6, "capsize": 1.5})
        ax.set_xlim(0, 1)
        ax.set_xlabel(label)
        ax.set_title(title, loc="left")
    axes[0].invert_yaxis()
    style.save_figure(fig, HERE / "shape")


if __name__ == "__main__":
    main()
