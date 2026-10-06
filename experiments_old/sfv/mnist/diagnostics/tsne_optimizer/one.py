"""One t-SNE optimiser variant at perplexity 30 on the tuning split (20k); prints a JSON line.

    python experiments_old/sfv/mnist/diagnostics/tsne_optimizer/one.py --variant cuml_none_n6 --seed 0
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = next(p for p in HERE.parents if p.name == "experiments_old")
sys.path.insert(0, str(BENCH))

import numpy as np  # noqa: E402

from common.datasets import load  # noqa: E402
from common.metrics import quality  # noqa: E402

N, PERPLEXITY = 20000, 30.0
VARIANTS = ("cuml_adaptive", "cuml_none_n12", "cuml_none_n6", "cuml_none_n3", "opentsne_auto")


def run(variant: str, X: np.ndarray, seed: int):
    n = X.shape[0]
    if variant.startswith("cuml"):
        from cuml.manifold import TSNE

        common = dict(n_components=2, perplexity=PERPLEXITY, init="pca", method="fft", max_iter=750,
                      exaggeration_iter=250, random_state=seed)
        if variant == "cuml_adaptive":
            model = TSNE(**common)
        else:
            divisor = float(variant.rsplit("_n", 1)[1])
            model = TSNE(**common, learning_rate_method="none", learning_rate=n / divisor,
                         n_neighbors=int(3 * PERPLEXITY), early_exaggeration=12.0)
        Z = np.asarray(model.fit_transform(X.astype(np.float32)), dtype=np.float64)
        return Z, float(model.kl_divergence_)
    from openTSNE import TSNE

    emb = TSNE(n_components=2, perplexity=PERPLEXITY, initialization="pca", negative_gradient_method="fft",
               early_exaggeration=12, early_exaggeration_iter=250, n_iter=500, n_jobs=-1,
               random_state=seed).fit(X)
    return np.asarray(emb), float(emb.kl_divergence)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    X, y = load("tune", n=N, seed=args.seed)
    run(args.variant, X[:2000], args.seed)                                # warm-up
    t = time.perf_counter()
    Z, kl = run(args.variant, X, args.seed)
    seconds = time.perf_counter() - t
    print("RESULT " + json.dumps({"variant": args.variant, "seed": args.seed, "seconds": seconds, "kl": kl,
                                  **quality(X, Z, y, seed=args.seed)}), flush=True)


if __name__ == "__main__":
    main()
