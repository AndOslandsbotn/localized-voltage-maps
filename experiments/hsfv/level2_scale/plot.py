import json
from pathlib import Path

from experiments.common import style
from experiments.hsfv.common.levels import KS

HERE = Path(__file__).resolve().parent
TOYS = {"swiss_roll": "swiss roll", "half_sphere": "half sphere"}
RUNS = {"level2_regions": "--", "level2_regions_60k": "-"}
MAPS = {"level1_chart": "level 1 + chart", "level2_chart": "level 2 + chart"}


def main() -> None:
    colours = dict(zip(MAPS, style.category_colours(len(MAPS)).colors))
    fig, axes = style.figure("full", aspect=0.38, ncols=len(TOYS), sharey=True)
    for letter, ax, (toy, title) in zip("ab", axes, TOYS.items()):
        for run, line in RUNS.items():
            results = json.loads((HERE.parent / toy / run / "results.json").read_text())
            cell = results["n"] / results["fitted"]["level1"]["n_cells"]
            for name, label in MAPS.items():
                ax.plot([k / cell for k in KS], [results["summary"][name][f"recall@{k}"] for k in KS], line,
                        marker="o", markersize=3, color=colours[name], label=f"{label}, n = {results['n'] // 1000}k")
        ax.axvline(1.0, color="0.6", linewidth=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("k, in points per level-1 cell")
        ax.set_title(f"({letter}) {title}", loc="left")
    axes[0].set_ylabel("recall@k within the region")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2)
    style.save_figure(fig, HERE / "recall_by_scale")


if __name__ == "__main__":
    main()
