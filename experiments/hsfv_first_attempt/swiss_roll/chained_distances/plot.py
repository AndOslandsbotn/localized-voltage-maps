import json
from pathlib import Path

import numpy as np

from experiments.common import style
from experiments.hsfv_first_attempt.swiss_roll.chained_distances.run import LEVEL1_CELLS
from experiments.hsfv_first_attempt.swiss_roll.chained_distances.score import GROUPS

HERE = Path(__file__).resolve().parent
METHODS = {"level1": "level 1 directly", "chained_simple": "chained, shortest route",
           "chained_triangulated": "chained, triangulated", "chained_voltage": "chained, voltage-weighted",
           "level2_within": "level 2 directly (within a region)", "graph": "graph shortest paths"}
PANELS = {"rank": "rank correlation with the truth $\\uparrow$", "relative_error": "relative error $\\downarrow$"}


def main() -> None:
    rows = json.loads((HERE / "results.json").read_text())["rows"]
    colours = style.category_colours(len(METHODS))
    width = 0.8 / len(METHODS)
    fig, axes = style.figure("full", aspect=0.6, nrows=len(LEVEL1_CELLS), ncols=len(PANELS))
    for row, (setting, cells) in zip(axes, LEVEL1_CELLS.items()):
        for ax, (metric, label) in zip(row, PANELS.items()):
            for i, (method, name) in enumerate(METHODS.items()):
                values = np.array([[r[metric] for r in rows if r["setting"] == setting and r["method"] == method
                                    and r["group"] == g] for g in GROUPS])
                mean, spread = np.nanmean(values, axis=1), np.nanstd(values, axis=1)
                ax.bar(np.arange(len(GROUPS)) + (i - (len(METHODS) - 1) / 2) * width, mean, width, yerr=spread,
                       color=colours(i), label=name, error_kw={"linewidth": 0.5, "capsize": 1.0})
            ax.set_xticks(range(len(GROUPS)), GROUPS)
            ax.set_ylabel(label)
            ax.set_title(f"level 1 with {cells} cells", loc="left")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncols=3)
    style.save_figure(fig, HERE / "chained_distances")


if __name__ == "__main__":
    main()
