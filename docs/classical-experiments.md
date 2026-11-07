# Classical model experiments (Phase 4.1)

The offline candidate framework in `model/experiments.py` compares seven
models under the same temporal training/validation partition:

| Candidate | Score semantics | Eligible for later production selection? |
|---|---|---:|
| Heuristic baseline | Fixed merchant-relative risk ranking | No |
| Logistic Regression | Class-one model score | Yes |
| Random Forest | Class-one model score | Yes |
| HistGradientBoosting | Class-one model score | Yes |
| XGBoost | Class-one model score | Yes |
| LightGBM | Class-one model score | Yes |
| Isolation Forest | Normal-only anomaly ranking | No |

The heuristic and Isolation Forest deliberately remain outside threshold and
deployment selection. Their values are **not calibrated fraud probabilities**;
they establish whether supervised classifiers add useful ranking value beyond a
transparent rule and unsupervised novelty detection.

Run the comparison in the locked Python 3.10 environment:

```powershell
python evaluation/run_classical_experiments.py
```

It writes `evaluation/classical_model_comparison.json`. The runner uses train
and validation dates only, records library versions and feature schema, and
does not load or tune against the final test period. Its reported validation
leader is not a production promotion decision. Phase 4.2–4.6 must still add
merchant/OOD/cold-start/stress protocols, decision metrics, calibration,
robustness checks, and ablations before any model replacement.

## Dependency decision

XGBoost 2.1.4 and LightGBM 4.5.0 are pinned in `requirements.lock` because
the master prompt specifically requires a comparison of those mature tree
boosting families under identical conditions. Scikit-learn’s
HistGradientBoosting remains the lower-dependency baseline. The trade-off is
larger binary wheels and more experiment time; no new serving process or ML
framework is introduced.
