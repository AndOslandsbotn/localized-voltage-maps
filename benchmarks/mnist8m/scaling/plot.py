"""Draw mnist8m/scaling/figure.png: time, memory and quality against n, and the 8.1M embeddings.

    python benchmarks/mnist8m/scaling/plot.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
STYLE = {"lvm_gpu": ("LVM, streamed", "tab:blue", "o-"), "lvm_pca_gpu": ("LVM + local PCA, streamed", "tab:cyan", "s-"),
         "umap_gpu": ("UMAP, fit on all", "tab:orange", "o-"),
         "umap_gpu_transform": ("UMAP, fit on what fits + transform", "darkorange", "D"),
         "tsne_gpu": ("t-SNE, fit on all", "tab:red", "o-"), "le_gpu": ("Laplacian Eigenmaps, fit on all", "tab:green", "o-"),
         "lisomap_gpu": ("Landmark Isomap, fit on all", "tab:purple", "o-"),
         "lisomap_gpu_transform": ("Landmark Isomap, fit on what fits + transform", "indigo", "D")}


def key(r):
    if (r.get("info") or {}).get("fit_n", r["n"]) < r["n"]:
        return r["method"] + "_transform"
    return r["method"]


def main() -> None:
    rows = [json.loads(l) for l in (HERE / "results.jsonl").read_text().splitlines()]
    ok = [r for r in rows if r["status"] == "ok"]
    panels = [("total_s", "time to coordinates for all n points [s]", True), ("peak_rss_mb", "peak process memory [MB]", True),
              ("trustworthiness", "trustworthiness (5000 random points)", False),
              ("distance_correlation", "global distance rank correlation", False)]
    fig = plt.figure(figsize=(22, 11))
    for i, (metric, label, log) in enumerate(panels):
        ax = fig.add_subplot(2, 4, i + 1)
        for k, (name, colour, fmt) in STYLE.items():
            rs = sorted([r for r in ok if key(r) == k], key=lambda r: r["n"])
            if rs:
                ax.plot([r["n"] for r in rs], [r[metric] for r in rs], fmt, color=colour, label=name)
        for r in rows:
            if r["status"] == "memory_limit" and metric == "total_s":
                ax.axvline(r["n"], color=STYLE[key(r)][1], ls=":", lw=1)
                ax.text(r["n"], ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1, " memory limit", rotation=90,
                        color=STYLE[key(r)][1], fontsize=7, va="top")
        ax.set_xscale("log")
        if log:
            ax.set_yscale("log")
        if metric == "peak_rss_mb":
            ax.axhline(5000, color="grey", ls="--", lw=1)
            ax.text(5.2e4, 5200, "memory guard (5 GB; machine 7.8 GB)", fontsize=7, color="grey")
        ax.set_xlabel("number of points n")
        ax.set_ylabel(label)
        ax.grid(True, which="both", alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)
    full = max(r["n"] for r in ok)
    pics = [r for r in ok if r["n"] == full]
    for j, r in enumerate(sorted(pics, key=lambda r: list(STYLE).index(key(r)))[:4]):   # every method that reached all n
        tag = f"{r['method']}_n{r['n']}" + (f"_fit{r['info']['fit_n']}" if key(r).endswith("_transform") else "")
        Z, y = np.load(HERE / f"Z_{tag}.npy"), np.load(HERE / f"y_{tag}.npy")
        ax = fig.add_subplot(2, 4, 5 + j)
        order = np.random.default_rng(0).permutation(len(y))
        ax.scatter(Z[order, 0], Z[order, 1], c=y[order], cmap="tab10", vmin=-0.5, vmax=9.5, s=0.2, rasterized=True)
        for d in range(10):
            cx, cy = np.median(Z[y == d], axis=0)
            ax.text(cx, cy, str(d), fontsize=11, weight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.15", fc="white", alpha=0.7, lw=0))
        ax.set_title(f"{STYLE[key(r)][0]}, n = {r['n']:,}\n{r['total_s']:.0f} s, peak {r['peak_rss_mb'] / 1024:.1f} GB; "
                     f"trust {r['trustworthiness']:.3f}  5-NN {r['knn_accuracy']:.3f}  global {r['distance_correlation']:.3f}",
                     fontsize=8)
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("MNIST8M (8.1 million digits, does not fit in this machine's 7.8 GB): LVM streams it from disk; "
                 "the other methods must hold what they fit on (memory guard 5 GB per run)", fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / "figure.png", dpi=110)
    print(HERE / "figure.png")


if __name__ == "__main__":
    main()
