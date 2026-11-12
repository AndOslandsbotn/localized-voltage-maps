import json
from pathlib import Path

import numpy as np

from experiments.common import style

HERE = Path(__file__).resolve().parent
RUNS = {"swiss roll": HERE.parent / "swiss_roll" / "stopping_rule" / "errors_seed3_sigma0_{mode}.json",
        "half sphere": HERE.parent / "half_sphere" / "stopping_rule" / "errors_seed0_sigma0_{mode}.json"}
MODES = ("procrustes", "calibrated")
ROWS = {"absolute": "median absolute error", "relative": "median relative error",
        "ratio": "median map / true distance"}


def main() -> None:
    for mode in MODES:
        if all(Path(str(path).format(mode=mode)).exists() for path in RUNS.values()):
            draw(mode)


def draw(mode: str) -> None:
    fig, axes = style.figure("full", aspect=1.05, nrows=len(ROWS), ncols=len(RUNS), sharex="col")
    for column, (title, path) in enumerate(RUNS.items()):
        result = json.loads(Path(str(path).format(mode=mode)).read_text())
        levels = list(result["levels"])
        colours = style.continuous_colours()(np.linspace(0.15, 0.9, len(levels)))
        distance = np.array(result["distance_centres"])
        for row, (kind, label) in enumerate(ROWS.items()):
            ax = axes[row, column]
            for colour, level in zip(colours, levels):
                values = np.array([np.nan if v is None else v for v in result["levels"][level][kind]])
                ax.plot(distance, values, "-o", color=colour, markersize=2.5, linewidth=1.2,
                        label=f"placed by level {level}")
            ax.set_xscale("log")
            if kind == "ratio":
                ax.axhline(1.0, color="0.6", linewidth=0.5)
            else:
                ax.set_yscale("log")
            if column == 0:
                ax.set_ylabel(label)
            if row == len(ROWS) - 1:
                ax.set_xlabel(f"true distance between centres ({result['units']})")
        axes[0, column].set_title(f"({'ab'[column]}) {title}", loc="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=len(handles))
    style.save_figure(fig, HERE / f"level_errors_{mode}")


if __name__ == "__main__":
    main()
