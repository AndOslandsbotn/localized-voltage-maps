from pathlib import Path

from experiments.common import cli, plots, style
from experiments.common.methods import TITLES
from experiments.common.runner import embed

HERE = Path(__file__).resolve().parent
DATASET, N, SEED = "mnist", 50_000, 0
METHODS = ["lvm", "lvm_chart"]


def main() -> None:
    args = cli.arguments("MNIST: each method's embedding of the 50,000 evaluation images.")
    embeddings = {TITLES[m]: embed(m, dataset=DATASET, n=N, seed=SEED, folder=HERE, recompute=args.recompute)
                  for m in METHODS}
    style.save_figure(plots.embeddings(embeddings), HERE / "embedding")


if __name__ == "__main__":
    main()
