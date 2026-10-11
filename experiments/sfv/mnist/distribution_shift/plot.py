import json
from pathlib import Path

import numpy as np

from experiments.common import style

HERE = Path(__file__).resolve().parent
CONDITIONS = {"uniform": "uniform\nmasses", "balanced": "masses from\nbalanced data",
              "shifted": "masses from\nshifted data", "refit_uniform": "refit,\nuniform masses",
              "refit": "refit,\ncounted masses"}
METHODS = {"lvm": "LVM", "lvm_chart": "LVM + local chart"}
PANELS = {"trustworthiness": "trustworthiness", "continuity": "continuity", "knn_accuracy": "kNN accuracy",
          "density_preservation": "density preservation", "decay_common": "decay in 1s and 7s (rel.)"}


def main() -> None:
    rows = json.loads((HERE / "results.json").read_text())["rows"]
    positions = np.arange(len(CONDITIONS))
    fig, axes = style.figure("full", aspect=0.3, ncols=len(PANELS))
    for letter, ax, (metric, label_y) in zip("abcde", axes, PANELS.items()):
        for i, method in enumerate(METHODS):
            values = np.array([[r[metric] for r in rows if r["method"] == method and r["condition"] == condition]
                               for condition in CONDITIONS])
            ax.errorbar(positions + 0.1 * (2 * i - 1), values.mean(axis=1), yerr=values.std(axis=1),
                        color=style.method_colour(method), linestyle="", marker="o", markersize=2.5, elinewidth=0.5,
                        capsize=1.5, label=METHODS[method])
        ax.set_xticks(positions, list(CONDITIONS.values()), rotation=90)
        ax.set_ylabel(label_y)
        ax.set_title(f"({letter})", loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=len(METHODS))
    style.save_figure(fig, HERE / "distribution_shift")


if __name__ == "__main__":
    main()
