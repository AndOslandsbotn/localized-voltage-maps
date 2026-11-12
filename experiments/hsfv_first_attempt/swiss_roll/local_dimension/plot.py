import json
from pathlib import Path

import numpy as np

from experiments.common import style

HERE = Path(__file__).resolve().parent
CURVES = {0.3: "noise $\\sigma = 0.3$", 0.6: "noise $\\sigma = 0.6$", 1.0: "noise $\\sigma = 1.0$"}


def main() -> None:
    rows = json.loads((HERE / "results.json").read_text())["rows"]
    colours = style.category_colours(len(CURVES))
    fig, ax = style.figure("half", aspect=0.75)
    for i, (sigma, label) in enumerate([(0.0, "no noise"), *CURVES.items()]):
        by_cells = {}
        for r in rows:
            if r["noise"] == sigma:
                by_cells.setdefault(r["n_cells"], []).append((r["cell_size"], r["dimension"]))
        cells = sorted(by_cells)
        size = np.array([np.mean([s for s, _ in by_cells[c]]) for c in cells])
        dim = np.array([np.mean([d for _, d in by_cells[c]]) for c in cells])
        spread = np.array([np.std([d for _, d in by_cells[c]]) for c in cells])
        colour = "0.4" if sigma == 0.0 else colours(i - 1)
        style_line = "--" if sigma == 0.0 else "-"
        ax.fill_between(size, dim - spread, dim + spread, color=colour, alpha=0.2, linewidth=0)
        ax.plot(size, dim, color=colour, linestyle=style_line, linewidth=0.8, marker="o", markersize=1.6, label=label)
        if sigma > 0:
            ax.axvline(sigma, color=colour, linestyle=":", linewidth=0.7)
    ax.axhline(2.0, color="0.6", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xticks([0.2, 0.5, 1, 2, 5], ["0.2", "0.5", "1", "2", "5"])
    ax.minorticks_off()
    ax.set_xlabel("cell size (median RMS radius)")
    ax.set_ylabel("estimated local dimension")
    ax.set_ylim(1.0, 3.05)
    ax.legend(loc="lower right", fontsize="small")
    style.save_figure(fig, HERE / "local_dimension")


if __name__ == "__main__":
    main()
