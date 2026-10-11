import json
from pathlib import Path

import numpy as np

from experiments.common import style

HERE = Path(__file__).resolve().parent
LABELS = {"uniform": "uniform", "counted_10000": "10k", "counted_20000": "20k", "counted_50000": "50k",
          "counted_100000": "100k"}
PANELS = {"trustworthiness": "trustworthiness", "continuity": "continuity", "knn_accuracy": "kNN accuracy",
          "density_preservation": "density preservation"}


def main() -> None:
    rows = json.loads((HERE / "results.json").read_text())["rows"]
    fig, axes = style.figure("full", aspect=0.3, ncols=len(PANELS))
    positions = np.arange(len(LABELS))
    for letter, ax, (metric, label_y) in zip("abcdef", axes, PANELS.items()):
        values = np.array([[row[metric] for row in rows if row["setting"] == name] for name in LABELS])
        ax.errorbar(positions, values.mean(axis=1), yerr=values.std(axis=1), color=style.method_colour("lvm"),
                    linestyle="", marker="o", markersize=2.5, elinewidth=0.5, capsize=1.5)
        ax.set_xticks(positions, list(LABELS.values()))
        ax.set_xlabel("masses counted from")
        ax.set_ylabel(label_y)
        ax.set_title(f"({letter})", loc="left")
    style.save_figure(fig, HERE / "masses")


if __name__ == "__main__":
    main()
