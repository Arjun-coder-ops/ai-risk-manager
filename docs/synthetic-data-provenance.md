# Synthetic simulation provenance

## Scope and boundary

`model/simulator.py` creates deterministic **synthetic merchant-day
aggregates** for offline development and evaluation. It does not model real
Razorpay merchants, fraud rates, payment outcomes, or operational data-delivery
behaviour. It must never be cited as real-world validation and its event
manifest must not be returned from dashboard or tenant APIs.

The current default simulation is `synthetic-simulator-v2`, configured through
the `SimulationConfig` dataclass. Run it reproducibly with:

```powershell
python model/generate_data.py
python model/features.py
python model/train.py
python evaluation/evaluate.py
```

The generator writes the data CSV and `data/raw/simulation_manifest.json`.
The manifest records the seed, full parameter set, output totals, and
simulator-only events. A custom JSON config may be supplied with
`python model/generate_data.py --config path/to/config.json`; unknown config
keys fail rather than being silently ignored.

## Behaviour generation (label-independent)

Before any fraud labels are assigned, the simulator samples small, medium, and
large merchant profiles with merchant-specific transaction scale, failure rate,
amount distribution, user mix, weekly/monthly seasonality, and both growth and
decline trends. It also creates the following label-negative conditions:

- lifecycle variation: new-merchant cold starts and random merchant churn;
- legitimate holiday demand and merchant-specific campaign windows;
- unusually quiet periods;
- occasional missing merchant-day observations;
- delayed-record metadata (`record_delay_days` and `record_available_date`).

Missing observations are omitted from the merchant-day CSV. Delayed-record
metadata describes simulated arrival time but is deliberately **not** a model
feature and does not make the current batch pipeline availability-aware. That
gap remains an ingestion/stream-processing concern, not a claim solved by this
simulator.

Legitimate campaigns intentionally increase signals such as volume, new-user
mix, devices, regions, and sometimes amount. They keep
`label_fraud_spike=0`, providing fraud-like but label-negative examples. This
does not mean campaigns are a representative sample of all false positives.

## Ground-truth labels

`inject_fraud_spikes` runs only after normal behaviour is materialized. It
selects non-overlapping, observed merchant-day windows with prior history and
applies one of these synthetic perturbations:

- volume burst;
- failed-payment spike;
- new-user/device burst;
- device reuse anomaly;
- region anomaly;
- amount anomaly.

Only rows modified by this stage receive `label_fraud_spike=1`. The amount
anomaly uses one signed, bounded multiplier; this corrects the older
generator's redundant double/random amount mutation. All generated rows are
bounded to the same count relationships enforced by `model/validation.py`.

The label and manifest are offline evaluation artifacts. The model uses only
the declared feature columns in `model/features.py`; simulator event types and
delivery-delay metadata are not training features and are not included in
production response schemas.

## What this does and does not evaluate

The simulator increases scenario coverage and makes synthetic experiments
repeatable. It does not establish external validity, calibration for real
merchant populations, detection of actual fraud, or resilience to a real
delayed-event stream. Phase 2.4 must separately investigate data provenance
and either build a suitable external adapter or create and label an independent
out-of-distribution benchmark.
