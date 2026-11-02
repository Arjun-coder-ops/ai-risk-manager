# External-data provenance and OOD evaluation

## Decision record

Reviewed: 2026-11-02.

The product predicts risk at the **merchant-day** grain using eight aggregate
signals, including failures, user/device diversity, and region spread. A
transaction-fraud label by itself is not enough to validate that task. For an
external dataset to support the current model without fabricating inputs, it
needs all of the following:

1. a usable transaction time or calendar-day key;
2. a stable merchant identifier;
3. enough transaction attributes to calculate the required merchant-day
   fields without inventing failures, users, devices, or regions;
4. a documented fraud-label meaning; and
5. licensing/access terms that allow the intended evaluation use.

No reviewed public dataset meets the complete contract. No external file is
downloaded, redistributed, or silently transformed by this repository.

| Candidate | Original source / access | Time and merchant coverage | Labels and available attributes | License / access finding | Decision |
|---|---|---|---|---|---|
| ULB/Worldline Credit Card Fraud | Worldline + ULB MLG dataset, published through Kaggle | Two days; elapsed-seconds `Time`; no merchant ID | 284,807 transactions, 492 frauds; `Amount`, anonymized PCA components, `Class` | Public dataset page; obtain under the host’s current terms before use | Not adaptable to merchant-day risk: no merchant key or required behavioural attributes |
| IEEE-CIS Fraud Detection | IEEE-CIS/Vesta Kaggle competition | `TransactionDT` is a delta from an undisclosed reference time; no merchant ID | 590,540 labelled train transactions; amount/card/address/e-mail/device features and `isFraud` | Competition-hosted data; access subject to current competition/platform terms | Not adaptable to merchant-day risk: no actual date or merchant grouping |
| PaySim | Lopez-Rojas, Elmir & Axelsson mobile-money simulator; code repository | Hourly simulation step; origin/destination accounts, not a stable merchant contract | Synthetic transaction type, amount, balances, simulated fraud label | Simulator code is GPL-3.0; any separate CSV distribution must be licensed and verified independently | Not real-world validation; insufficient for current failure/device/region aggregates |
| BankSim | Lopez-Rojas & Axelsson bank-payment simulator | 180 simulated days and a merchant field | Synthetic customer/merchant/category/amount/fraud records (594,643 transactions, 7,200 frauds reported by the authors) | Common public copies are hosted by third parties; their exact terms must be checked at acquisition | Closest structural match, but still synthetic and missing failures, device/user diversity, and region-spread signals; no full-model adapter without fabrication |

The output label in BankSim/PaySim is a simulator truth label, not confirmed
payment-fraud evidence. Neither can raise this project’s real-world validation
claim.

## Selected fallback: independent synthetic OOD benchmark

Because a compatible external real dataset is not available, the repository
uses `evaluation/ood_benchmark.py`. It is intentionally separate from
`model/simulator.py` and never trains or tunes the committed model. It creates
a new 2027 merchant population with heavy-tailed volumes and amounts,
negative-binomial counts, normal structural business shifts, recurring
promotions/outages, and independent attack mechanisms:

- credential stuffing;
- mule-network activity;
- account takeover.

It produces only an offline JSON report at
`evaluation/ood_benchmark_results.json`. Its output states
`real_world_validation=false` and `training_data_used=false`. It must be
reported separately from the held-out in-distribution synthetic result, never
combined with it or labelled external validation.

Run it with:

```powershell
python evaluation/ood_benchmark.py
```

The benchmark relies on the committed model artifact. Regenerate the model
pipeline first when changing the in-distribution simulator, feature schema, or
model artifact. It does not download data or require credentials.

## Source records

- [ULB/Worldline Credit Card Fraud dataset record](https://berd-platform.de/records/qcqqe-g6q16)
- [IEEE-CIS Fraud Detection dataset description](https://www.kaggle.com/c/ieee-fraud-detection/data)
- [PaySim source repository and GPL-3.0 license](https://github.com/EdgarLopezPhD/PaySim)
- [PaySim original paper](https://www.msc-les.org/proceedings/emss/2016/EMSS2016_249.pdf)
- [BankSim original paper](https://www.researchgate.net/publication/265736405_BankSim_A_Bank_Payment_Simulation_for_Fraud_Detection_Research)

These source records support provenance review, not an endorsement of a
particular downstream copy of a dataset. Re-check the current access and
license terms before any future download or adapter implementation.
