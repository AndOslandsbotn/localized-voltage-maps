import json
from pathlib import Path

import numpy as np

from experiments.common import style

HERE = Path(__file__).resolve().parent
RUNS = {"swiss roll": HERE.parent / "swiss_roll" / "stopping_rule" / "compression_seed3_sigma0.json",
        "half sphere": HERE.parent / "half_sphere" / "stopping_rule" / "compression_seed0_sigma0.json"}
ROWS = {"deepest": "level-4 centres, placed by level", "own": "each level's own centres"}
PAIRS = {"same": ("-", "same region"), "different": ("--", "different regions")}


def main() -> None:
    fig, axes = style.figure("full", aspect=0.75, nrows=len(ROWS), ncols=len(RUNS))
    for column, (title, path) in enumerate(RUNS.items()):
        result = json.loads(path.read_text())
        levels = list(result["deepest"])
        colours = dict(zip(levels, style.continuous_colours()(np.linspace(0.15, 0.9, len(levels)))))
        for row, (kind, label) in enumerate(ROWS.items()):
            ax = axes[row, column]
            for level, curves in result[kind].items():
                for pair, (line, _) in PAIRS.items():
                    values = np.array([np.nan if v is None else v for v in curves[pair]])
                    ax.plot(curves["distance"], values, line, color=colours[level], linewidth=1.2)
            ax.axhline(1.0, color="0.6", linewidth=0.5)
            ax.set_xscale("log")
            ax.set_ylim(0, 1.3)
            ax.set_ylabel(f"median map / true\n({label})" if column == 0 else "")
            ax.set_xlabel(f"true distance ({result['units']})" if row == len(ROWS) - 1 else "")
        axes[0, column].set_title(f"({'ab'[column]}) {title}", loc="left")
    handles = [axes[0, 0].plot([], [], color=c, linewidth=1.2)[0] for c in colours.values()]
    handles += [axes[0, 0].plot([], [], line, color="0.4", linewidth=1.2)[0] for line, _ in PAIRS.values()]
    labels = [f"level {level}" for level in colours] + [text for _, text in PAIRS.values()]
    fig.legend(handles, labels, loc="outside lower center", ncols=len(labels))
    style.save_figure(fig, HERE / "compression")


if __name__ == "__main__":
    main()
