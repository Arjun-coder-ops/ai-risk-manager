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
2. `model/features.py` — computes the versioned end-of-day merchant-relative
   feature contract using prior-only calendar windows, same-weekday
   normalization, rolling distributions, EWM state, volatility, trend, and
   prior anomaly frequency. See [`docs/feature-contract.md`](docs/feature-contract.md).
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
shape. The configurable `synthetic-simulator-v2` generates small, medium, and
large merchant profiles over 180 calendar days. Its label-independent behaviour
stage includes per-merchant weekly/monthly seasonality, growth and decline,
cold starts, churn, quiet periods, legitimate campaign/holiday demand, missing
observations, and delayed-record metadata. A separate injection stage then
adds six kinds of simulated fraud spike against each merchant's own baseline.

Legitimate campaign demand intentionally resembles some anomaly signals but
retains a negative label. This makes the task less trivially separable, but it
does **not** make the simulation representative of real false positives.
Read the full ground-truth contract, configurable parameters, and boundaries
in [`docs/synthetic-data-provenance.md`](docs/synthetic-data-provenance.md).

- Source: synthetic (`model/generate_data.py`, seed=42, fully reproducible)
- Size and class distribution: generated values are recorded in the committed
  offline-only `data/raw/simulation_manifest.json`
- Features: 17 versioned end-of-day signals, described in `model/features.py`
  and [`docs/feature-contract.md`](docs/feature-contract.md)
- Class distribution: 7.02% positive overall (train 7.48%, validation 6.86%,
  test 7.32%)
- **Limitation**: this is a simulation of plausible merchant behaviour, not
  real Razorpay data. Absolute metric values should be read as evidence
  that the pipeline and methodology work correctly, not as a claim about
  real-world fraud-detection performance.

## Evaluation (actual measured results on the held-out test set)

Test set: 6,157 merchant-days, 2026-05-16 → 2026-06-29 — dates strictly
after training and validation, touched exactly once.

| Metric | Value |
|---|---|
| ROC-AUC | **0.9903** |
| PR-AUC (appropriate for 7.32% positive rate) | **0.9500** |

**HIGH tier** (score ≥ 0.67, direct merchant alert, validation-tuned for precision ≥ 0.95):

| | |
|---|---|
| Precision | 0.9599 |
| Recall | 0.8492 |
| F1 | 0.9012 |
| False positive rate | 0.28% |
| Confusion matrix | TP 383, FP 16, FN 68, TN 5690 |
| Estimated cost | ₹7,200 (FP) + ₹544,000 (FN) = **₹551,200** |

**REVIEW tier** (score ≥ 0.10, flag for analyst, validation-tuned to minimize total business cost):

| | |
|---|---|
| Precision | 0.8154 |
| Recall | 0.9401 |
| F1 | 0.8733 |
| False positive rate | 1.68% |
| Confusion matrix | TP 424, FP 96, FN 27, TN 5610 |
| Estimated cost | ₹43,200 (FP) + ₹216,000 (FN) = **₹259,200** |

Model comparison on validation (PR-AUC): Logistic Regression 0.753, Random
Forest 0.882, **Gradient Boosting 0.931 (selected)**.

The required Phase 4.1 classical candidate suite (including a transparent
heuristic, Isolation Forest, XGBoost, and LightGBM) is kept separate from the
currently deployed model while the remaining evaluation and calibration gates
are implemented. Its reproducible protocol and selection boundary are in
[`docs/classical-experiments.md`](docs/classical-experiments.md).

Phase 4.2 adds independent temporal, merchant-disjoint, OOD, cold-start, and
stress-test protocols without promoting a model. The report shows material
synthetic OOD and mild-anomaly degradation; details and limitations are in
[`docs/evaluation-protocols.md`](docs/evaluation-protocols.md).

Phase 4.3 adds decision-quality, fixed-review-capacity, event-detection, raw
calibration, cost-sensitivity, and cluster-bootstrap uncertainty evidence for
the frozen artifact. It does not select, calibrate, or promote a model; see
[`docs/decision-metrics.md`](docs/decision-metrics.md).

Phase 4.4 compares sigmoid and isotonic calibration on a dedicated,
chronological calibration-selection period. Neither improved the primary Brier
criterion, so no calibrated candidate is retained or deployed; see
[`docs/calibration-experiment.md`](docs/calibration-experiment.md).

Phase 4.5 records robustness evidence across anomaly severity, merchant-size
proxies, campaigns, record quality, distribution shift, label noise,
prevalence, and temporal drift. It exposes severe synthetic campaign, OOD, and
late-drift false-positive failures; see
[`docs/robustness-tests.md`](docs/robustness-tests.md).

Phase 4.6 provides a common-window feature, calibration, and model-family
ablation report. It confirms meaningful merchant-relative value but does not
promote a new model from one synthetic slice; see
[`docs/ablation-experiments.md`](docs/ablation-experiments.md).

Phase 5 adds reproducible CPU-only PyTorch MLP, autoencoder, LSTM, and TCN
experiments. None surpasses the operational tree model on synthetic validation,
so gradient boosting remains the serving choice; see
[`docs/deep-learning-experiments.md`](docs/deep-learning-experiments.md).

## Independent synthetic OOD stress test

The held-out result above is synthetic **in-distribution** evidence only. A
separately designed, 2027 synthetic OOD benchmark scores the frozen model
without retraining or tuning it. It uses a different population and generation
process (heavy-tailed values/counts, structural business shifts, promotions,
outages, credential-stuffing, mule-network, and account-takeover scenarios).

| Metric | Independent synthetic OOD result |
|---|---:|
| Raw / scored merchant-days | 16,500 / 15,730 |
| Positive rate | 6.3064% |
| ROC-AUC | 0.9390 |
| PR-AUC | 0.6681 |
| REVIEW precision / recall | 0.1410 / 0.9698 |
| HIGH precision / recall | 0.2164 / 0.9022 |

This is intentionally a much harsher synthetic stress test, not external
validation or real-world performance. Its full result is
`evaluation/ood_benchmark_results.json`; the candidate-dataset review and
benchmark boundaries are documented in
[`docs/external-data-provenance.md`](docs/external-data-provenance.md).

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
- CORS is allowlisted through `ALLOWED_ORIGINS`; it is not a substitute for
  authentication or authorization. Tenant routes use bearer authentication,
  RBAC, and organization-scoped database queries; the legacy demo routes do
  not and must never receive tenant data.

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

# 5. Serve the dashboard from an allowed local origin
# In a second terminal at the repository root:
python -m http.server 3000 -d frontend
# Open http://localhost:3000
```

The API accepts browser requests only from the comma-separated origins in
`ALLOWED_ORIGINS` (default: `http://localhost:3000,http://127.0.0.1:3000`).
Set it explicitly when using another local frontend origin, for example:

```bash
ALLOWED_ORIGINS=http://localhost:3000 python backend/app.py
```

On PowerShell:

```powershell
$env:ALLOWED_ORIGINS = "http://localhost:3000"
python backend/app.py
```

**CORS is not authentication or authorization.** The authenticated `/api/v1`
surface enforces bearer authentication, RBAC, and organization-scoped queries.
The legacy `/api/risk` and static dashboard routes remain synthetic-data demo
surfaces and must never receive tenant or personal data. The dashboard API
deliberately omits simulator ground-truth labels; they remain available only to
offline evaluation code.

## Reproducibility

`requirements.txt` installs the reviewed versions in `requirements.lock`.
Use Python 3.10 and run the pipeline in a clean virtual environment. The
baseline reproduction record, commands, artifact hashes, and measured output
are documented in [`docs/baseline-evidence.md`](docs/baseline-evidence.md).

## Durable tenant API (Milestones 2A–2B)

The authenticated, PostgreSQL-ready API and investigation workflow are documented in
[`docs/tenant-data-foundation.md`](docs/tenant-data-foundation.md). It is
separate from the legacy synthetic-demo endpoints, which must never receive
tenant data. Copy `.env.example` to `.env`, replace its placeholders, and run
`docker compose up --build` for local PostgreSQL-backed development.
`ALLOW_BOOTSTRAP_REGISTRATION` defaults to `false`; enable it temporarily only
to create an initial local administrator.

The tenant API supports atomic merchant-day JSON/CSV uploads and bounded,
asynchronous transaction batches. Transaction records use opaque provider,
user, and device identifiers only; Compose starts a separate database-backed
worker that rebuilds daily aggregates and retries safely. This is batch
processing—not a real-time streaming claim. See
[`docs/tenant-data-foundation.md`](docs/tenant-data-foundation.md) for the
strict request contract, state model, and operational limits.

## Limitations

- Data is synthetic; real-world merchant behaviour is messier and this
  hasn't been validated against real fraud patterns.
- The model uses 17 end-of-day daily-aggregate features. A production system would likely
  add transaction-level (not just daily-aggregate) signals and richer
  device/network graph features.
- Cost assumptions are illustrative placeholders, not calibrated to real
  Razorpay unit economics.
- The dashboard is a static single-page app for demo purposes, not
  production-hardened (no authentication, RBAC, or tenant isolation).
- SHAP is used when the installed model/explainer combination supports it;
  the engine uses a transparent heuristic fallback if it does not. SHAP
  impact direction is distinct from the observed feature-value direction.

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
│   ├── deep_learning.py          # offline PyTorch experiment primitives
│   ├── llm_explainer.py         # optional LLM layer (verified-signals-only)
│   └── artifacts/               # trained model + metadata (generated)
├── evaluation/
│   ├── evaluate.py               # one-time held-out test evaluation
│   ├── test_results.json         # actual measured results (generated)
│   ├── run_decision_metrics.py   # reproducible Phase 4.3 metric report
│   ├── decision_metrics.json     # frozen-artifact decision evidence
│   ├── run_calibration_experiment.py # chronological calibration comparison
│   ├── calibration_report.json   # sigmoid/isotonic selection evidence
│   ├── calibration_reliability.svg # reliability diagram
│   ├── run_robustness_tests.py   # Phase 4.5 frozen-model stress suite
│   ├── robustness_tests.json     # robustness evidence and failure modes
│   ├── run_ablation_experiments.py # Phase 4.6 common-window comparisons
│   ├── ablation_experiments.json # feature, calibration, and family evidence
│   ├── run_deep_learning_experiments.py # Phase 5 CPU experiment suite
│   ├── deep_learning_experiments.json # neural-model comparison evidence
│   └── audit_log.jsonl           # runtime audit trail (generated)
├── tests/test_risk_engine.py    # feature leakage, policy, failure-handling tests
├── data/                         # generated CSVs (raw + processed)
├── requirements.txt
├── .env.example
└── .gitignore
```
