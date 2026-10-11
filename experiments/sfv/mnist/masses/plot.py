import json
from pathlib import Path

import numpy as np

from experiments.common import style

HERE = Path(__file__).resolve().parent
CELLS_FROM = (1_000, 5_000, 50_000)
MASSES_FROM = ("uniform", "1000", "5000", "10000", "50000")
PANELS = {"trustworthiness": "trustworthiness", "continuity": "continuity", "knn_accuracy": "kNN accuracy",
          "density_preservation": "density preservation"}


def main() -> None:
    rows = json.loads((HERE / "results.json").read_text())["rows"]
    colours = style.category_colours(len(CELLS_FROM))
    positions = np.arange(len(MASSES_FROM))
    fig, axes = style.figure("full", aspect=0.3, ncols=len(PANELS))
    for letter, ax, (metric, label_y) in zip("abcdef", axes, PANELS.items()):
        for i, cells_from in enumerate(CELLS_FROM):
            shown = [(p, name) for p, name in zip(positions, MASSES_FROM)
                     if any(r["cells_from"] == cells_from and r["masses_from"] == name for r in rows)]
            values = np.array([[r[metric] for r in rows if r["cells_from"] == cells_from and r["masses_from"] == name]
                               for _, name in shown])
            ax.errorbar([p + 0.08 * (i - 1) for p, _ in shown], values.mean(axis=1), yerr=values.std(axis=1),
                        color=colours(i), linewidth=0.7, marker="o", markersize=2.0, elinewidth=0.5, capsize=1.0,
                        label=f"cells from {cells_from // 1000}k")
        ax.set_xticks(positions, ["uniform"] + [f"{int(name) // 1000}k" for name in MASSES_FROM[1:]], rotation=45)
        ax.set_xlabel("masses counted from")
        ax.set_ylabel(label_y)
        ax.set_title(f"({letter})", loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=len(CELLS_FROM))
    style.save_figure(fig, HERE / "masses")


if __name__ == "__main__":
    main()
