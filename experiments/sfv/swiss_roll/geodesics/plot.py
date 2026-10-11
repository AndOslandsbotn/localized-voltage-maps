import json
from pathlib import Path

import numpy as np

from experiments.common import style

HERE = Path(__file__).resolve().parent
LABELS = {"lvm": "LVM", "lvm_chart": "LVM + local chart", "umap": "UMAP", "tsne": "t-SNE",
          "laplacian_eigenmaps": "Laplacian eigenmaps", "landmark_isomap": "Landmark Isomap"}
PANELS = {"exact": "(a) vs exact geodesics", "graph": "(b) vs graph geodesics"}


def main() -> None:
    rows = json.loads((HERE / "results.json").read_text())["rows"]
    graph_vs_exact = np.mean([row["exact"] for row in rows if row["method"] == "graph"])
    fig, axes = style.figure("full", aspect=0.32, ncols=len(PANELS), sharey=True)
    for ax, (reference, title) in zip(axes, PANELS.items()):
        values = [[row[reference] for row in rows if row["method"] == method] for method in LABELS]
        ax.barh(list(LABELS.values()), np.mean(values, axis=1), xerr=np.std(values, axis=1),
                color=[style.method_colour(method) for method in LABELS], error_kw={"linewidth": 0.6, "capsize": 1.5})
        ax.set_xlim(0, 1)
        ax.set_xlabel("rank correlation of distances")
        ax.set_title(title, loc="left")
    axes[0].invert_yaxis()
    for ax, reference in zip(axes, (graph_vs_exact, 1.0)):
        ax.axvline(reference, color="0.4", linestyle="--", linewidth=0.8)
        ax.text(reference, 1.0, " graph", transform=ax.get_xaxis_transform(), color="0.4", va="bottom", ha="center")
    style.save_figure(fig, HERE / "geodesics")


if __name__ == "__main__":
    main()
