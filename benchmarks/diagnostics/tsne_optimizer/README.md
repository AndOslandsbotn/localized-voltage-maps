# t-SNE optimiser settings

**Question:** which learning-rate setting should cuML's t-SNE use? The comparison should not depend on the libraries' differing optimiser conventions.

## Why this needed deciding

cuML's default `learning_rate_method="adaptive"` does more than set a learning rate. It also:
- overrides `n_neighbors` (78 at n = 20k, 42 at 50k), which caps the perplexity in effect;
- sets early exaggeration to 24.

With adaptive mode off, the learning rate is the given value during early exaggeration and twice that afterwards. openTSNE's `"auto"` is n / exaggeration of the current phase.

## Rule

- Model settings (perplexity, late exaggeration) are tuned by trustworthiness, like every method.
- Optimiser settings are chosen by t-SNE's own objective, its KL divergence.

## Check

- `one.py`, perplexity 30, tuning split 20k, seeds 0–1, each variant in its own guarded process; output in `run.log`.
- Variants: cuML adaptive; cuML non-adaptive with rate n/12, n/6 and n/3 (3 × perplexity neighbours, early exaggeration 12); openTSNE auto as reference.

## Results (mean of 2 seeds)

| Variant | KL (cuML's own) | Trust | Cont | 5-NN | Global corr. | Time |
|---|---|---|---|---|---|---|
| cuML adaptive | 0.413 | 0.969 | 0.962 | 0.926 | 0.431 | 1.3 s |
| cuML n/12 | 0.467 | 0.968 | 0.962 | 0.926 | 0.380 | 1.4 s |
| cuML n/6 | 0.438 | 0.968 | 0.963 | 0.924 | 0.369 | 1.4 s |
| cuML n/3 | 0.403 | 0.967 | 0.962 | 0.925 | 0.364 | 1.4 s |
| openTSNE auto | 2.17 | 0.973 | 0.959 | 0.947 | 0.307 | 38 s |

## Decision

Use cuML non-adaptive with rate n/3:
- it gives the lowest KL divergence;
- it is the same value cuML's adaptive mode picks;
- the cuML variants hardly differ on our metrics.

## Notes

- KL values cannot be compared across libraries: they are computed differently.
- openTSNE is somewhat better locally than cuML at the same settings (5-NN 0.947 vs 0.925). Each library is reported with its own result.
