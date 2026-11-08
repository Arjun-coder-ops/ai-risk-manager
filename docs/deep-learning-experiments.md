# Deep-learning experiments (Phase 5)

Phase 5 is an experimental extension, not a reason to replace the serving
gradient-boosting model. `evaluation/run_deep_learning_experiments.py` runs
CPU-only, deterministic PyTorch experiments and writes
`evaluation/deep_learning_experiments.json`.

## Protocol and safeguards

- PyTorch 2.14.1+cpu; CUDA is not used.
- Fixed seed 42 covers Python, NumPy, and PyTorch; training uses one CPU thread
  and deterministic algorithms where supported.
- Feature normalization is fit on the 13,303 training rows only, then applied
  unchanged to 4,346 validation rows and sequence data.
- Every classifier uses class-weighted binary cross-entropy, early stopping,
  best-validation checkpointing, and a SHA-256 state hash for the restored
  checkpoint.
- The declared final temporal test is not loaded. No model, threshold, or
  calibration artifact is promoted by this runner.

## Reproduced validation comparison

All values below are synthetic validation evidence from 2026-04-16 through
2026-05-15 (6.8569% positive rate), not real-world fraud performance.

| Model | Hypothesis / input | Parameters | PR-AUC | CPU mean ms/row |
|---|---|---:|---:|---:|
| MLP | 17 normalized end-of-day features | 2,065 | 0.873382 | 0.000163 |
| Normal-only autoencoder | Reconstruct normal tabular behaviour; rank by error | 2,909 | 0.686569 | 0.000166 |
| LSTM | 14 observed merchant-day feature vectors | 12,913 | 0.885684 | 0.007149 |
| TCN | Same 14-step input, causal dilated convolutions | 14,833 | 0.922128 | 0.013894 |
| HistGradientBoosting | Classical supervised tabular comparator | — | **0.931039** | Not benchmarked in this neural report |

The TCN is closer to the supervised tree baseline than the other neural
models, but it is still worse on PR-AUC and slower than the LSTM in this
CPU measurement. None beats the existing tree model, so the operational model
remains `fraud-spike-v2-eod` gradient boosting.

## Architecture-specific interpretation

- **MLP:** a compact non-linear tabular baseline. It trains to epoch 25 but
  remains below gradient boosting, so it is retained as experiment code only.
  Its Phase 6 evidence includes Integrated Gradients from a stated
  train-normalized zero baseline; attribution signs describe its logit, not
  an observed raw-feature direction.
- **Autoencoder:** trained exclusively on label-negative training rows. Its
  reconstruction error is an anomaly ranking, **not** a fraud probability;
  it is compared with Isolation Forest (PR-AUC 0.469292), a transparent
  heuristic (0.443319), and supervised HistGradientBoosting (0.931039).
  Label-free novelty detection is fundamentally different from supervised
  fraud classification.
- **LSTM:** sequence targets are merchant-days; each 14-observation input
  ends at its target day and contains no later rows or labels. It did not show
  the expected multi-day pattern benefit over gradient boosting.
- **TCN:** uses the identical sequences, splits, class weighting, and metrics
  as the LSTM. Its larger parameter count and slower measured CPU inference
  do not justify a serving change.

## Conditional architectures

A Transformer is intentionally omitted: 180-day synthetic histories and the
observed LSTM/TCN results do not justify more sequence complexity. A GNN is
also omitted: merchant-day aggregates do not contain defensible
merchant-device-account graph relationships. No graph was invented to satisfy
a technology checklist.

## Reproduce

```powershell
pip install -r requirements.txt
python evaluation/run_deep_learning_experiments.py
python -m pytest tests/test_deep_learning.py -q
```

The runner writes transient local checkpoints, restores the best checkpoint,
and records checkpoint hashes in its JSON evidence. Checkpoints are not
candidate serving artifacts.
