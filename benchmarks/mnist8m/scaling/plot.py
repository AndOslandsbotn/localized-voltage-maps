"""Draw the MNIST8M figures from results.jsonl, one per purpose.

* figure_cost.png: construction (fitting a model on m points) and streaming (placing n points with a model
  fitted on 50k): time, memory and throughput. Linear axes, so a per-point cost reads as a slope; the throughput
  panel (log axis) separates computing (rates.py, data in memory) from streaming from disk.
* figure_quality.png: trust, continuity, 5-NN and global correlation, against points streamed (B: all fitted on
  the same 50k sample) and against fit size from 1k (A: fitted on all points, as far as memory allows; log axis,
  to see which method needs the fewest points).
* figure_embeddings.png: the 8.1M embeddings (B).

    python benchmarks/mnist8m/scaling/plot.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run import FULL, SAMPLE, fit_of  # noqa: E402

COLOURS = {"lvm_gpu": "tab:blue", "lvm_pca_gpu": "tab:cyan", "umap_gpu": "tab:orange", "tsne_gpu": "tab:red",
           "le_gpu": "tab:green", "lisomap_gpu": "tab:purple"}
NAMES = {"lvm_gpu": "LVM", "lvm_pca_gpu": "LVM + local PCA", "umap_gpu": "UMAP", "tsne_gpu": "t-SNE",
         "le_gpu": "Laplacian Eigenmaps", "lisomap_gpu": "Landmark Isomap"}
QUALITY = [("trustworthiness", "trustworthiness"), ("continuity", "continuity"), ("knn_accuracy", "5-NN accuracy"),
           ("distance_correlation", "global distance rank correlation")]
STYLE = {"tsne_gpu": "s--"}                       # t-SNE's memory equals Laplacian Eigenmaps': keep both visible
STREAMING = ("lvm_gpu", "lvm_pca_gpu", "umap_gpu", "lisomap_gpu")
FULL_FIT = ("lvm_gpu", "umap_gpu", "tsne_gpu", "le_gpu", "lisomap_gpu")


def streamed(rows: list) -> dict[str, list[dict]]:
    """B: per method, points with n (points embedded) and stream_s (time after the 50k fit), plus the metrics."""
    out = {}
    for m in STREAMING:
        if m.startswith("lvm"):
            rs = [r for r in rows if r["method"] == m and r["status"] == "ok" and r["n"] >= SAMPLE and fit_of(r) == SAMPLE]
            out[m] = sorted([{**r, "stream_s": r["total_s"] - r["fit_s"]} for r in rs], key=lambda r: r["n"])
        else:
            start = [r for r in rows if r["method"] == m and r["status"] == "ok" and r["n"] == SAMPLE and fit_of(r) == SAMPLE]
            full = [r for r in rows if r["method"] == m and r["status"] == "ok" and r["n"] == FULL and fit_of(r) == SAMPLE]
            pts = [{**r, "stream_s": 0.0} for r in start[:1]]
            for r in full[:1]:
                pts += [{**c, "stream_s": c["total_s"] - r["fit_s"]} for c in r.get("checkpoints", [])]
                pts.append({**r, "stream_s": r["total_s"] - r["fit_s"]})
            out[m] = sorted(pts, key=lambda r: r["n"])
    return out


def fitted(rows: list) -> tuple[dict, dict]:
    """A: per method, the full fits (fitted on all n) and the first size that ran out of memory."""
    ok, limit = {}, {}
    for m in FULL_FIT:
        ok[m] = sorted([r for r in rows if r["method"] == m and r["status"] == "ok" and fit_of(r) == r["n"]],
                       key=lambda r: r["n"])
        hit = [r["n"] for r in rows if r["method"] == m and r["status"] == "memory_limit" and fit_of(r) == r["n"]]
        if hit:
            limit[m] = min(hit)
    return ok, limit


def _out_of_memory(ax, limit: dict, scale: float) -> None:
    """Mark each method's first out-of-memory size near the top, stacked so marks at the same size stay visible,
    and add "out of memory" to the legend."""
    lo, hi = ax.get_ylim()
    for i, (m, n) in enumerate(sorted(limit.items(), key=lambda kv: kv[1])):
        same = [k for k, v in limit.items() if v == n]
        ax.plot([n / scale], [hi - (0.06 * same.index(m)) * (hi - lo)], "x", color=COLOURS[m], ms=11, mew=2.5)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], ls="", marker="x", color="grey", ms=9, mew=2))
    ax.legend(handles, labels + ["out of memory (first size)"], fontsize=8)


def a_min(a: dict) -> int:
    """The smallest fit size of part A (the memory baseline)."""
    return min((pts[0]["n"] for pts in a.values() if pts), default=SAMPLE)


def figure_cost(b: dict, a: dict, limit: dict) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.8))
    ax = axes[0]                                                # construction time
    for m, pts in a.items():
        if pts:
            ax.plot([p["n"] / 1e6 for p in pts], [p["fit_s"] for p in pts], STYLE.get(m, "o-"), color=COLOURS[m],
                    label=NAMES[m])
    _out_of_memory(ax, limit, 1e6)
    ax.set_xlabel("fit size m [million points]")
    ax.set_ylabel("time to fit the model [s]")
    ax.set_title("Construction: fitting a model on m points\n(LVM: k-means on 50k, refined by streaming k-means over "
                 "all m)", fontsize=10)
    ax = axes[1]                                                # construction memory, above each method's smallest run
    for m, pts in a.items():
        if pts:
            base = pts[0]["peak_rss_mb"]
            ax.plot([p["n"] / 1e6 for p in pts], [(p["peak_rss_mb"] - base) / 1024 for p in pts], STYLE.get(m, "o-"),
                    color=COLOURS[m], label=f"{NAMES[m]} (starts at {base / 1024:.1f} GB)")
    _out_of_memory(ax, limit, 1e6)
    ax.set_xlabel("fit size m [million points]")
    ax.set_ylabel(f"peak memory above the {a_min(a) / 1e3:.0f}k fit [GB]")
    ax.set_title("Construction: memory growth with m\n(memory guard 5 GB per run, machine 7.8 GB)", fontsize=10)
    ax = axes[2]                                                # streaming time, linear
    for m, pts in b.items():
        if pts:
            ax.plot([p["n"] / 1e6 for p in pts], [p["stream_s"] / 60 for p in pts], "o-", color=COLOURS[m], label=NAMES[m])
    ax.set_xlabel("points placed n [million]")
    ax.set_ylabel("time to place them after the fit [min]")
    ax.set_title("Streaming: placing n points with a model fitted on 50k\n(slope = cost per point; t-SNE and Laplacian "
                 "Eigenmaps cannot place new points)", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[3]                                                # throughput: computing alone, and from the disk
    rates_file = HERE / "rates.json"
    rates = json.loads(rates_file.read_text()) if rates_file.exists() else []
    compute = {r["method"]: r["rate"] for r in rates if r["kind"] == "compute"}
    disk = next((r for r in rates if r["kind"] == "disk"), None)
    names, ys = [], []
    for i, (m, pts) in enumerate(b.items()):
        last = pts[-1] if pts else None
        end_to_end = None
        if last and last["n"] == FULL and last["stream_s"] > 0:
            end_to_end = (last["n"] - (0 if m.startswith("lvm") else SAMPLE)) / last["stream_s"]
        for j, (value, alpha, kind) in enumerate(((compute.get(m), 1.0, "computing"), (end_to_end, 0.45, "from disk"))):
            if value:
                y = 3 * i + j
                ax.barh(y, value, color=COLOURS[m], alpha=alpha)
                ax.text(value, y, f" {value / 1e3:,.0f}k", va="center", fontsize=9)
        names.append(NAMES[m]), ys.append(3 * i + 0.5)
    if disk:
        ax.axvline(disk["rate"], color="black", ls="--", lw=1.2)
        ax.text(disk["rate"] * 0.93, max(ys) + 1.4, f"disk read alone: {disk['rate'] / 1e3:,.0f}k/s", fontsize=8,
                ha="right", va="center")
    ax.set_yticks(ys, names)
    ax.set_ylim(-0.8, max(ys) + 2.0)
    ax.set_xscale("log")
    ax.set_xlim(1e4, 1e7)
    ax.invert_yaxis()
    ax.set_xlabel("points per second (log scale)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="grey"), plt.Rectangle((0, 0), 1, 1, color="grey", alpha=0.45)],
              labels=["computing only (data in memory)", f"end to end, streamed from disk ({FULL / 1e6:.1f}M)"],
              fontsize=8, loc="center right", bbox_to_anchor=(1.0, 0.3))
    ax.set_title("Placing points: rate of the computation alone, and from disk\n(rates.py; dashed: reading the "
                 "stream with no computation)", fontsize=10)
    for ax in axes[:3]:
        ax.grid(True, alpha=0.3)
    fig.suptitle("MNIST8M: cost of building a model, and of placing points with it (GPU)", fontsize=12)
    fig.tight_layout()
    fig.savefig(HERE / "figure_cost.png", dpi=110)
    plt.close(fig)


def figure_quality(b: dict, a: dict, limit: dict) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    for col, (metric, label) in enumerate(QUALITY):
        ax = axes[0, col]
        for m, pts in b.items():
            if pts:
                ax.plot([p["n"] / 1e6 for p in pts], [p[metric] for p in pts], "o-", color=COLOURS[m], label=NAMES[m])
        ax.set_xlabel("points placed n [million]")
        ax.set_ylabel(label)
        ax = axes[1, col]
        for m, pts in a.items():
            if pts:
                ax.plot([p["n"] for p in pts], [p[metric] for p in pts], STYLE.get(m, "o-"), color=COLOURS[m],
                        label=NAMES[m])
        ax.set_xscale("log")
        ax.set_xlabel("fit size m = n [points, log scale]")
        ax.set_ylabel(label)
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    axes[0, 0].set_title("B: fitted on the same 50k sample, the rest placed (streamed / transformed)", fontsize=10, loc="left")
    axes[1, 0].set_title("A: fitted on all points, as far as memory allows (from 1k: how few points does each "
                         "method need?)", fontsize=10, loc="left")
    fig.suptitle("MNIST8M: embedding quality (5000 random points per run)", fontsize=12)
    fig.tight_layout()
    fig.savefig(HERE / "figure_quality.png", dpi=110)
    plt.close(fig)


def figure_embeddings(rows: list) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.6))
    for ax, m in zip(axes, STREAMING):
        r = next((r for r in rows if r["method"] == m and r["n"] == FULL and r["status"] == "ok" and fit_of(r) == SAMPLE),
                 None)
        tag = f"{m}_n{FULL}_fit{SAMPLE}"                       # one.py's name for a run fitted on fewer points than n
        if r is None or not (HERE / "embeddings" / f"Z_{tag}.npy").exists():
            ax.axis("off")
            continue
        Z, y = np.load(HERE / "embeddings" / f"Z_{tag}.npy"), np.load(HERE / "embeddings" / f"y_{tag}.npy")
        order = np.random.default_rng(0).permutation(len(y))
        ax.scatter(Z[order, 0], Z[order, 1], c=y[order], cmap="tab10", vmin=-0.5, vmax=9.5, s=0.3, rasterized=True)
        ax.set_title(f"{NAMES[m]}: {r['total_s']:.0f} s for {FULL / 1e6:.1f}M points\ntrust {r['trustworthiness']:.3f}  "
                     f"cont {r['continuity']:.3f}  5-NN {r['knn_accuracy']:.3f}  global {r['distance_correlation']:.3f}",
                     fontsize=10)
        ax.set_xticks([]), ax.set_yticks([])
    cmap = plt.get_cmap("tab10")
    fig.legend([Line2D([], [], ls="", marker="o", ms=8, color=cmap((d + 0.5) / 10)) for d in range(10)],
               [str(d) for d in range(10)], title="digit", loc="center right", frameon=False)
    fig.suptitle(f"MNIST8M: all {FULL / 1e6:.1f}M points, each method fitted on the same 50k sample "
                 "(every 40th point shown)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.96, 1))
    fig.savefig(HERE / "figure_embeddings.png", dpi=110)
    plt.close(fig)


def main() -> None:
    rows = [json.loads(l) for l in (HERE / "results.jsonl").read_text().splitlines()]
    b, (a, limit) = streamed(rows), fitted(rows)
    figure_cost(b, a, limit)
    figure_quality(b, a, limit)
    figure_embeddings(rows)
    for name in ("figure_cost.png", "figure_quality.png", "figure_embeddings.png"):
        print(HERE / name)


if __name__ == "__main__":
    main()
