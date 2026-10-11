import json
from pathlib import Path

import numpy as np

from experiments.common import style

HERE = Path(__file__).resolve().parent
CONDITIONS = {("kmeans", "uniform"): "k-means, uniform", ("kmeans", "counted"): "k-means, counted",
              ("farthest", "uniform"): "farthest, uniform", ("farthest", "counted"): "farthest, counted"}
SEED = 0


def main() -> None:
    rows = json.loads((HERE / "theory.json").read_text())["rows"]
    colours = style.category_colours(len(CONDITIONS))
    positions = np.arange(len(CONDITIONS))
    fig, axes = style.figure("full", aspect=0.32, ncols=3)
    for i, (condition, label) in enumerate(CONDITIONS.items()):
        row = next(r for r in rows if (r["placement"], r["masses"]) == condition and r["seed"] == SEED)
        x, y = np.array(row["log_density"]), np.array(row["log_rate"])
        axes[0].scatter(x, y, color=colours(i), s=0.8, linewidths=0, label=label)
        slope, intercept = np.polyfit(x, y, 1)
        axes[0].plot(np.sort(x), slope * np.sort(x) + intercept, color=colours(i), linewidth=0.8)
    axes[0].set_xlabel("log data density at the cell")
    axes[0].set_ylabel("log decay rate")
    for ax, key, label_y in ((axes[1], "decay_slope", "decay rate vs density (slope)"),
                             (axes[2], "room_vs_density", "room vs density (rank corr.)")):
        for i, condition in enumerate(CONDITIONS):
            values = [r[key] for r in rows if (r["placement"], r["masses"]) == condition]
            ax.errorbar(i, np.mean(values), yerr=np.std(values), color=colours(i), marker="o", markersize=2.5,
                        elinewidth=0.5, capsize=1.5)
        ax.axhline(0.0, color="0.6", linewidth=0.5)
        ax.set_xticks(positions, [label.replace(", ", "\n") for label in CONDITIONS.values()])
        ax.set_ylabel(label_y)
    for letter, ax in zip("abc", axes):
        ax.set_title(f"({letter})", loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=len(CONDITIONS), markerscale=4)
    style.save_figure(fig, HERE / "theory")


if __name__ == "__main__":
    main()
