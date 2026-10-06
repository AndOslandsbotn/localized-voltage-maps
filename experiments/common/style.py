from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

STYLE = Path(__file__).resolve().parents[1] / "style"
SETTINGS = yaml.safe_load((STYLE / "figures.yaml").read_text())
plt.style.use(STYLE / "paper.mplstyle")


def figure(width: str = "full", aspect: float = 0.5, **subplots):
    """(figure, axes) at the printed size: ``width`` of the text width, height = aspect x width."""
    width_in = SETTINGS["text_width_in"] * SETTINGS["widths"][width]
    return plt.subplots(figsize=(width_in, aspect * width_in), **subplots)


def method_colour(method: str) -> str:
    return SETTINGS["palettes"]["methods"][method]


def category_colours(n: int) -> ListedColormap:
    colours = SETTINGS["palettes"]["categories"]
    if n > len(colours):
        raise ValueError(f"The category palette has {len(colours)} colours, {n} are needed")
    return ListedColormap(colours[:n])


def save_figure(fig, path: Path) -> None:
    """``path`` without suffix: one file per format in figures.yaml."""
    for suffix, dpi in SETTINGS["formats"].items():
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=dpi)
    plt.close(fig)
