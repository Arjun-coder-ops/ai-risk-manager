# Milestone 0 baseline evidence

Audited reference commit: `35a433608545c1ddd5cec4d7945dbbfe9cf72802`.

## Environment

- Python `3.10.8`
- NumPy `1.26.4`
- pandas `2.3.1`
- scikit-learn `1.7.2`
- joblib `1.5.3`
- Flask `3.1.0`
- Flask-CORS `6.0.5`
- pytest `9.1.1`
- SHAP `0.49.1`

The baseline was run in an isolated clone so the source checkout's committed
model and evaluation artifacts were not overwritten. Before the run, SHA-256
values were recorded for `evaluation/test_results.json`,
`model/artifacts/model.joblib`, and `model/artifacts/model_meta.json`:

```text
test_results.json   0EA7DBA36E1766AAC5002600B9A4C83870CE16E3BB3C7BF6AAC89D62B812B27A
model.joblib        12D9E0A70D4BDCB3B7A5BEEA566CFEC515394ED391BB0A73922AC6C9F8E0626D
model_meta.json     5446B6A0B888F08286C29ACDDD23C2940C23F8B485E6DFC7346F62E2A5D7751C
```

## Commands and results

```powershell
python -m pytest tests -q
python model/generate_data.py
python model/features.py
python model/train.py
python evaluation/evaluate.py
```

The generator reproduced 27,000 merchant-day records across 150 merchants,
with a 7.00% positive rate over 2026-01-01 through 2026-06-29. Model
selection reproduced validation PR-AUC values of 0.7904 (logistic
regression), 0.9439 (random forest), and 0.9634 (HistGradientBoosting).

The held-out evaluation reproduced the committed synthetic result exactly:

| Metric | Value |
|---|---:|
| Test rows | 6,750 |
| Test positive rate | 0.0618 |
| ROC-AUC | 0.9825 |
| PR-AUC | 0.9555 |
| REVIEW precision / recall | 0.7011 / 0.9448 |
| HIGH precision / recall | 0.9497 / 0.9065 |

These are deterministic **synthetic, in-distribution** results. They do not
validate real payment-fraud performance and must not be interpreted as a
production guarantee.
