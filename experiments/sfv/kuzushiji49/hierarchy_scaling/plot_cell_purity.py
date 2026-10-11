import json
from pathlib import Path

import numpy as np

from experiments.common import style
from experiments.sfv.kuzushiji49.hierarchy_scaling.cell_purity import SIZES

HERE = Path(__file__).resolve().parent
LINES = {"level1": ("LVM level 1 (300 cells)", "lvm_chart"), "level2": ("LVM level 2 (cells split inside regions)", "lvm")}


def main() -> None:
    rows = json.loads((HERE / "cell_purity.json").read_text())["rows"]
    fig, ax = style.figure("half", aspect=0.75)
    for key, (label, colour) in LINES.items():
        values = np.array([[r[key] for r in rows if r["n"] == n] for n in SIZES])
        mean, spread = values.mean(axis=1), values.std(axis=1)
        ax.fill_between(SIZES, mean - spread, mean + spread, color=style.method_colour(colour), alpha=0.2, linewidth=0)
        ax.plot(SIZES, mean, color=style.method_colour(colour), linewidth=0.8, marker="o", markersize=1.6, label=label)
    ax.set_xscale("log")
    ax.set_xticks(SIZES, ["5k", "20k", "50k", "100k", "200k", "271k"], rotation=45)
    ax.minorticks_off()
    ax.set_xlabel("points the cells were fitted on")
    ax.set_ylabel("cell purity (all 271k images)")
    ax.legend(loc="lower right", fontsize="small")
    style.save_figure(fig, HERE / "cell_purity")


if __name__ == "__main__":
    main()
