"""Figures for the experiments: one figure per experiment, drawn from its results.csv.

Every point is the mean over seeds; the shaded band is +-1 standard deviation.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common.methods import METHODS  # noqa: E402
from common.results import read_rows  # noqa: E402

QUALITY = [
    ("trustworthiness", "trustworthiness (k=10)"),
    ("continuity", "continuity (k=10)"),
    ("knn_accuracy", "5-NN label accuracy"),
    ("distance_correlation", "global: distance rank correlation"),
]
COLORS = {"lvm": "tab:blue", "lvm_pca": "tab:cyan", "umap": "tab:orange", "le": "tab:green", "lisomap": "tab:purple",
          "tsne": "tab:red"}


def _mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def _band(ax, xs, means, stds, **kw):
    (line,) = ax.plot(xs, means, marker="o", **kw)
    ax.fill_between(xs, [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)],
                    alpha=0.2, color=line.get_color())


def plot_comparison(folder: Path, title: str) -> Path:
    """Time, memory and quality against n, one line per method."""
    if not (folder / "results.csv").exists():
        raise SystemExit(f"no results yet: run {folder.name}/run.py first")
    rows = [r for r in read_rows(folder / "results.csv") if r["status"] == "ok"]
    methods = [m for m in METHODS if any(r["method"] == m for r in rows)]
    has_gpu = any(r.get("gpu_peak_mb") not in ("", None) for r in rows)
    panels = [("total_s", "time to embed all points [s]", True), ("peak_rss_mb", "peak process memory [MB]", True)]
    if has_gpu:
        panels.append(("gpu_peak_mb", "peak GPU memory [MB]", True))
    panels += [(key, label, False) for key, label in QUALITY]

    ncols = 4
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.8 * nrows), squeeze=False)
    for ax, (key, label, log) in zip(axes.flat, panels):
        for m in methods:
            by_n = defaultdict(list)
            for r in rows:
                if r["method"] == m and r.get(key) not in ("", None):
                    by_n[int(r["n"])].append(float(r[key]))
            if not by_n:
                continue
            xs = sorted(by_n)
            stats = [_mean_std(by_n[x]) for x in xs]
            _band(ax, xs, [s[0] for s in stats], [s[1] for s in stats],
                  label=METHODS[m].label, color=COLORS[METHODS[m].family])
        ax.set_xscale("log")
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("number of points n")
        ax.set_ylabel(label)
        ax.grid(True, which="both", alpha=0.3)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"{title}  (MNIST evaluation split; mean ± 1 std over seeds)")
    fig.tight_layout()
    out = folder / "figure.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _get(params: dict, path: str):
    for part in path.split("."):
        params = params[part]
    return params


def plot_tuning(folder: Path, title: str, x: str, line: str | None = None) -> Path:
    """Quality and time across a tuning grid: ``x`` on the horizontal axis, one line per ``line`` value.

    ``x`` and ``line`` are dotted paths into the params, e.g. "cells.n_cells".
    """
    if not (folder / "results.csv").exists():
        raise SystemExit(f"no results yet: run {folder.name}/tune.py first")
    rows = [r for r in read_rows(folder / "results.csv") if r.get("status", "ok") == "ok"]
    groups: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        params = json.loads(r["params"])
        groups[_get(params, line) if line else None][_get(params, x)].append(r)
    panels = [(key, label) for key, label in QUALITY] + [("total_s", "time [s]")]

    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.8))
    for ax, (key, label) in zip(axes, panels):
        for g in sorted(groups, key=lambda v: (v is None, v)):
            xs = sorted(groups[g])
            stats = [_mean_std([float(r[key]) for r in groups[g][xv]]) for xv in xs]
            _band(ax, xs, [s[0] for s in stats], [s[1] for s in stats],
                  label=f"{line.split('.')[-1]}={g}" if line else None)
        ax.set_xlabel(x.split(".")[-1])
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        if key == "total_s":
            ax.set_yscale("log")
    if line:
        axes[0].legend(fontsize=8)
    best = folder / "best.yaml"
    subtitle = ""
    if best.exists():
        import yaml

        subtitle = f"\nbest: {json.dumps(yaml.safe_load(best.read_text())['params'])}"
    fig.suptitle(f"{title}  (tuning split, mean ± 1 std over seeds){subtitle}")
    fig.tight_layout()
    out = folder / "figure.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
