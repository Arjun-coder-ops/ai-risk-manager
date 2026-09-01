# AI Risk Manager — Merchant Fraud-Spike Detector

Built for the **Razorpay AI Buildathon 2026 — Track 02: AI Risk Manager**.

## Problem

Merchants lose money to fraud, returns, and chargebacks. One recurring loss
pattern is a **fraud spike**: a short window where a merchant's transaction
behaviour deviates sharply from its own normal pattern — a burst of
high-velocity low-value transactions (card testing), a sudden jump in
failed-payment rate (credential stuffing), or a wave of new devices/regions
transacting at once (abuse rings, account takeover). These spikes are easy
to miss in aggregate dashboards because they're relative to *that merchant's*
baseline, not a global threshold — a spike for a small merchant looks
nothing like a spike for a large one.

## Solution

A merchant-relative fraud-spike detector. For each merchant-day, the system
compares that day's behaviour to the merchant's own trailing 14-day
baseline, scores the probability that the day is part of a fraud spike, and
routes the result into one of three actions (LOW / REVIEW / HIGH) using a
cost-aware policy, with the top contributing signals surfaced as
merchant-facing evidence.

## Architecture

```
Merchant-day transaction data
        │
        ▼
Feature engineering (merchant-relative rolling baselines, leakage-safe)
        │
        ▼
ML risk model (Gradient Boosting, selected via validation PR-AUC)
        │
        ▼
Risk score (0-1) ──► Policy engine (validation-tuned thresholds) ──► LOW / REVIEW / HIGH
        │                                                                  │
        ▼                                                                  ▼
Top-3 signal contributions                                    Recommended action + audit record
        │
        ▼
Optional LLM layer (rephrases verified signals only — never invents evidence)
        │
        ▼
Merchant-facing explanation
```

Backend: Flask REST API sitting directly on the trained sklearn model
(see `backend/app.py` for why — no Node/Python IPC hop for a
prototype this size). Frontend: a single-page dashboard
(`frontend/index.html`) that calls the API directly.

## How It Works

1. `model/generate_data.py` — generates synthetic merchant-day transaction
   data (documented methodology below).
2. `model/features.py` — computes 8 merchant-relative features per day
   using `.shift(1)` + rolling windows so no feature ever sees same-day or
   future data.
3. `model/split.py` — time-based train/validation/test split.
4. `model/train.py` — trains and compares 3 candidate models on train,
   selects the winner by validation PR-AUC, then picks two decision
   thresholds (REVIEW: minimizes total validation business cost; HIGH:
   lowest threshold with validation precision ≥ 0.95) — **using validation
   data only**.
5. `evaluation/evaluate.py` — runs the selected model + thresholds **once**
   against the held-out test split and writes the real, measured results
   this README quotes.
6. `model/risk_engine.py` — the production-facing engine: validates input,
   computes features, scores, applies the policy, and produces evidence —
   or returns an explicit failure/insufficient-evidence result instead of
   guessing.
7. `backend/app.py` — REST API + audit logging around the engine.
8. `frontend/index.html` — dashboard.

## Dataset

**Synthetic, and explicitly disclosed as such.** Razorpay does not expose a
public dataset or test-mode API of historical merchant-level transaction
streams with fraud-spike labels, and generic public card-fraud datasets are
transaction-level, not merchant-day-level, so they don't fit this problem
shape. We generated 150 synthetic merchants across 3 archetypes
(small/medium/large, weighted 55/32/13%) over 180 days each (27,000
merchant-day rows), with realistic weekly seasonality, per-merchant growth
trends, and Poisson/Gamma-distributed transaction counts and amounts.

Fraud-spike windows (2-7 days, randomized severity from subtle to obvious,
one of 4 spike archetypes: volume burst, failed-payment burst, new-device
velocity, amount anomaly) are injected against **each merchant's own
baseline**, not a global rule, and severity is randomized so spikes are not
trivially separable. Overall positive rate: **7.0%** of merchant-days,
reflecting realistic class imbalance for a rare-event detector.

- Source: synthetic (`model/generate_data.py`, seed=42, fully reproducible)
- Size: 27,000 merchant-day rows, 150 merchants, 180 days (2026-01-01 to 2026-06-29)
- Features: 8 engineered signals, described in `model/features.py`
- Class distribution: 7.0% positive overall (train 7.9%, val 7.1%, test 6.2%)
- **Limitation**: this is a simulation of plausible merchant behaviour, not
  real Razorpay data. Absolute metric values should be read as evidence
  that the pipeline and methodology work correctly, not as a claim about
  real-world fraud-detection performance.

## Evaluation (actual measured results on the held-out test set)

Test set: 6,750 merchant-days, 2026-05-16 → 2026-06-29 — dates strictly
after training and validation, touched exactly once.

| Metric | Value |
|---|---|
| ROC-AUC | **0.9825** |
| PR-AUC (appropriate for 6.2% positive rate) | **0.9555** |

**HIGH tier** (score ≥ 0.48, direct merchant alert, validation-tuned for precision ≥ 0.95):

| | |
|---|---|
| Precision | 0.9497 |
| Recall | 0.9065 |
| F1 | 0.9276 |
| False positive rate | 0.32% |
| Confusion matrix | TP 378, FP 20, FN 39, TN 6313 |
| Estimated cost | ₹9,000 (FP) + ₹312,000 (FN) = **₹321,000** |

**REVIEW tier** (score ≥ 0.05, flag for analyst, validation-tuned to minimize total business cost):

| | |
|---|---|
| Precision | 0.7011 |
| Recall | 0.9448 |
| F1 | 0.8049 |
| False positive rate | 2.65% |
| Confusion matrix | TP 394, FP 168, FN 23, TN 6165 |
| Estimated cost | ₹75,600 (FP) + ₹184,000 (FN) = **₹259,600** |

Model comparison on validation (PR-AUC): Logistic Regression 0.790, Random
Forest 0.944, **Gradient Boosting 0.963 (selected)**.

## False-Positive Cost Model

Configurable, disclosed assumptions (`model/cost_model.py`) — not real
Razorpay financial data:

- Manual review cost: ₹150/flagged merchant-day
- Merchant friction/trust cost: ₹300/false alarm
- Undetected fraud loss: ₹8,000/missed spike-day (deliberately conservative)

**Why REVIEW's threshold is lower (more false positives) than HIGH's:** the
cost sweep on validation data shows that, under these assumptions, missing a
real spike is ~18x costlier than one unnecessary manual review. The
REVIEW tier is validation-tuned to minimize *total* cost, which pushes it
toward higher recall even at the expense of precision — acceptable because
a human reviews before any merchant-facing action is taken. The HIGH tier
is deliberately held to a stricter, precision-first bar (≥0.95 on
validation) because it triggers a direct merchant alert with no human in
the loop first.

## Threshold Selection

Both thresholds are chosen using **validation data only**, never the test
set: REVIEW by sweeping thresholds against the cost model above and taking
the minimum-total-cost point; HIGH by taking the lowest threshold that
clears 0.95 precision on validation. This is re-run by `model/train.py`
and logged to `model/artifacts/model_meta.json`.

## Failure Handling

Tested and demonstrated (`tests/test_risk_engine.py`, `TestFailureHandling`):

- **Unknown merchant** → structured error, no guess.
- **Missing date row** → structured error.
- **Invalid data** (e.g. `n_failed > n_transactions`) → rejected before
  scoring.
- **Insufficient trailing history** (< 7 prior days) → explicit
  `insufficient_evidence` status with `"Manual review recommended until
  more history accumulates"` — the system never emits a confident
  LOW/REVIEW/HIGH label from an unreliable baseline.
- **Empty input** → structured error.

One concrete example (demo case): a merchant on day 3 of its recorded
history returns:
```json
{"status": "insufficient_evidence", "risk_level": null,
 "reason": "Insufficient trailing history for a reliable baseline (need >= 7 prior days). Manual review recommended until more history accumulates.",
 "recommended_action": "MANUAL_REVIEW_INSUFFICIENT_DATA"}
```

## Security / Defense-Only Design

- No hardcoded secrets or API keys anywhere in the codebase (verified via
  repo-wide grep before submission).
- No Razorpay API endpoints, dataset, or business figures are invented —
  everywhere a number could be mistaken for a real Razorpay figure, it's
  explicitly labeled as a configurable/illustrative assumption.
- The system only ever *detects and flags*. It contains no code path that
  could be repurposed to bypass fraud checks, generate fraudulent
  transactions, or evade payment security controls.
- The optional LLM layer is constrained by code (not just prompt) to
  restate already-verified signals; it cannot independently accuse a
  merchant of fraud or add unverified claims, and it falls back to a
  deterministic template if unavailable — see `model/llm_explainer.py`.
- CORS is enabled broadly for local demo convenience only; a real
  deployment would scope this to the actual dashboard origin.

## Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic data, build features, train, and evaluate
cd model
python generate_data.py
python features.py
python train.py
cd ../evaluation
python evaluate.py

# 3. Run tests
cd ..
pytest tests/ -v

# 4. Start the backend API (port 8000)
cd backend
python app.py

# 5. Open the dashboard
# In a browser, open frontend/index.html directly (it calls http://localhost:8000)
```

## Limitations

- Data is synthetic; real-world merchant behaviour is messier and this
  hasn't been validated against real fraud patterns.
- Only 8 features, all daily-aggregate — a production system would likely
  add transaction-level (not just daily-aggregate) signals and richer
  device/network graph features.
- Cost assumptions are illustrative placeholders, not calibrated to real
  Razorpay unit economics.
- The dashboard is a static single-page app for demo purposes, not
  production-hardened (no auth, broad CORS).
- No SHAP/formal feature-attribution library is used; the explanation
  layer ranks features by a simple normalized-deviation heuristic, which
  is transparent but less rigorous than a proper attribution method.

## Future Improvements

- Add transaction-level (not just daily-aggregate) features and a proper
  SHAP-based attribution layer.
- Calibrate cost assumptions against real (anonymized) chargeback/review
  cost data if given access.
- Add authentication and per-merchant data scoping to the API.
- Extend the policy engine to support per-merchant-archetype thresholds
  instead of one global threshold pair.
- Online/streaming feature computation instead of batch daily-aggregate.

## Project Structure

```
ai-risk-manager/
├── backend/app.py              # Flask REST API + audit logging
├── frontend/index.html         # Dashboard (single-page, no build step)
├── model/
│   ├── generate_data.py        # synthetic data generator (documented)
│   ├── features.py             # leakage-safe feature engineering
│   ├── split.py                 # time-based train/val/test split
│   ├── train.py                 # model comparison + threshold selection
│   ├── risk_engine.py           # production scoring + policy + explainability
│   ├── cost_model.py            # configurable false-positive/negative cost model
│   ├── llm_explainer.py         # optional LLM layer (verified-signals-only)
│   └── artifacts/               # trained model + metadata (generated)
├── evaluation/
│   ├── evaluate.py               # one-time held-out test evaluation
│   ├── test_results.json         # actual measured results (generated)
│   └── audit_log.jsonl           # runtime audit trail (generated)
├── tests/test_risk_engine.py    # feature leakage, policy, failure-handling tests
├── data/                         # generated CSVs (raw + processed)
├── requirements.txt
├── .env.example
└── .gitignore
```
