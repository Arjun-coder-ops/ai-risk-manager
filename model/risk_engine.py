"""
Risk decision engine.

Pipeline:
  merchant-day feature row -> ML model -> risk_score
                                       -> policy engine -> LOW / REVIEW / HIGH
                                       -> feature contributions -> merchant-facing evidence

The ML model is the sole source of the risk score. The policy engine only
maps that score (plus data-sufficiency checks) to an action -- it never
overrides or second-guesses the score itself. This keeps "is this risky"
(ML) and "what do we do about it" (business policy) as separate, auditable
concerns, per the project's Phase 7 requirement.

FAILURE HANDLING: if the input is missing required fields, has an invalid
value (negative counts, more failed than total transactions, etc.), or the
merchant doesn't have enough trailing history for the rolling-baseline
features to be reliable, this engine returns an explicit
"insufficient evidence" result rather than a confident LOW/REVIEW/HIGH
label. It never silently guesses.
"""

import json
from dataclasses import dataclass, field
from typing import Optional

import joblib
import os
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from features import FEATURE_COLUMNS, build_features, MIN_HISTORY_DAYS

MODEL_PATH = os.path.join(REPO_ROOT, "model/artifacts/model.joblib")
META_PATH = os.path.join(REPO_ROOT, "model/artifacts/model_meta.json")

REQUIRED_INPUT_FIELDS = [
    "merchant_id", "date", "n_transactions", "n_failed", "avg_amount",
    "amount_std", "n_unique_users", "n_new_users", "n_devices", "distinct_regions_seen",
]

# Human-readable, non-technical descriptions of what each feature means,
# used to turn a raw feature contribution into merchant-facing evidence.
FEATURE_EXPLANATIONS = {
    "txn_volume_zscore": "Transaction volume today is unusually {direction} compared to this merchant's normal daily pattern.",
    "failed_rate_delta": "The share of failed/declined payments today is unusually {direction} compared to normal.",
    "amount_zscore": "The average transaction amount today is unusually {direction} compared to this merchant's normal ticket size.",
    "amount_volatility_ratio": "The spread of transaction amounts today is unusually {direction} compared to normal.",
    "new_user_ratio_delta": "The share of transactions from new/first-time users today is unusually {direction} compared to normal.",
    "device_diversity_ratio": "The ratio of devices to users today is unusually {direction} compared to normal.",
    "region_spread_zscore": "The number of distinct regions transacting today is unusually {direction} compared to this merchant's usual geographic footprint.",
    "rolling_volume_trend": "Recent 3-day transaction momentum is unusually {direction} compared to the 14-day trend.",
}


class RiskEngineError(Exception):
    pass


@dataclass
class RiskResult:
    status: str  # "ok" or "insufficient_evidence" or "error"
    merchant_id: Optional[str] = None
    date: Optional[str] = None
    risk_score: Optional[float] = None
    risk_level: Optional[str] = None
    signals: list = field(default_factory=list)
    recommended_action: Optional[str] = None
    model_version: Optional[str] = None
    reason: Optional[str] = None

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


class RiskEngine:
    def __init__(self, model_path=MODEL_PATH, meta_path=META_PATH):
        try:
            self.model = joblib.load(model_path)
            with open(meta_path) as f:
                self.meta = json.load(f)
        except FileNotFoundError as e:
            raise RiskEngineError(
                "Model artifacts not found. Run `python model/train.py` first."
            ) from e
        self.review_threshold = self.meta["review_threshold"]
        self.high_threshold = self.meta["high_threshold"]
        self.model_version = self.meta["model_version"]

        # SHAP explainer — optional, absent gracefully if shap is not installed.
        #
        # Output semantics (verified with shap==0.49.1, HistGradientBoostingClassifier):
        #   explainer.shap_values(X) → ndarray shape (n_samples, n_features).
        #   Row i = SHAP contributions for sample i (NOT a class index).
        #   Values are in the model's raw log-odds margin space.
        #   Positive value → feature pushed prediction toward fraud (class 1).
        #   Negative value → feature pushed prediction toward normal (class 0).
        #   Consistent with model.predict_proba(X)[:, 1] being the fraud probability
        #   (model.classes_ == [0, 1] confirmed experimentally).
        #   SHAP is purely post-hoc — it never alters risk scores or predictions.
        self.explainer = None
        try:
            import shap as _shap
            self.explainer = _shap.TreeExplainer(self.model)
        except Exception:
            # shap not installed, or model unsupported — heuristic fallback used.
            pass

    def _validate_row(self, row: dict) -> Optional[str]:
        missing = [f for f in REQUIRED_INPUT_FIELDS if f not in row or row[f] is None]
        if missing:
            return f"Missing required fields: {missing}"
        try:
            n_txn = float(row["n_transactions"])
            n_failed = float(row["n_failed"])
        except (TypeError, ValueError):
            return "Non-numeric transaction counts."
        if n_txn < 0 or n_failed < 0:
            return "Transaction counts cannot be negative."
        if n_failed > n_txn:
            return "n_failed cannot exceed n_transactions."
        if float(row.get("n_new_users", 0)) > float(row.get("n_unique_users", 0)):
            return "n_new_users cannot exceed n_unique_users."
        return None

    def score_merchant_day(self, history_df: pd.DataFrame, target_date: str, merchant_id: str) -> RiskResult:
        """
        history_df: raw merchant-day rows (same schema as generate_data.py output)
                    for this merchant, including the target date and enough
                    trailing history to compute rolling baselines.
        """
        if history_df is None or len(history_df) == 0:
            return RiskResult(status="error", reason="No transaction history provided.")

        merchant_hist = history_df[history_df["merchant_id"] == merchant_id].copy()
        if len(merchant_hist) == 0:
            return RiskResult(status="error", reason=f"No data found for merchant {merchant_id}.")

        target_row = merchant_hist[merchant_hist["date"] == target_date]
        if len(target_row) == 0:
            return RiskResult(status="error", reason=f"No transaction row found for {merchant_id} on {target_date}.")

        raw_target = target_row.iloc[0].to_dict()
        validation_error = self._validate_row(raw_target)
        if validation_error:
            return RiskResult(
                status="error", merchant_id=merchant_id, date=str(target_date),
                reason=f"Invalid input: {validation_error}",
            )

        try:
            featured = build_features(merchant_hist)
        except Exception as e:
            return RiskResult(
                status="error", merchant_id=merchant_id, date=str(target_date),
                reason=f"Feature computation failed: {e}",
            )

        target_features = featured[featured["date"] == target_date]
        if len(target_features) == 0:
            return RiskResult(status="error", merchant_id=merchant_id, date=str(target_date),
                               reason="Could not compute features for target date.")

        target_features = target_features.iloc[0]
        if not bool(target_features["has_sufficient_history"]):
            return RiskResult(
                status="insufficient_evidence",
                merchant_id=merchant_id,
                date=str(target_date),
                reason=(
                    f"Insufficient trailing history for a reliable baseline "
                    f"(need >= {MIN_HISTORY_DAYS} prior days). "
                    f"Manual review recommended until more history accumulates."
                ),
                recommended_action="MANUAL_REVIEW_INSUFFICIENT_DATA",
            )

        X = target_features[FEATURE_COLUMNS].to_frame().T.astype(float)
        try:
            risk_score = float(self.model.predict_proba(X)[0, 1])
        except Exception as e:
            return RiskResult(status="error", merchant_id=merchant_id, date=str(target_date),
                               reason=f"Model inference failed: {e}")

        risk_level, action = self._apply_policy(risk_score)

        # Compute SHAP for this single row if the explainer is available.
        # shap_values(X) → shape (1, 8); take row 0 (the only sample) to get a 1-D (8,) array.
        shap_row = None
        if self.explainer is not None:
            try:
                shap_matrix = self.explainer.shap_values(X)  # (1, 8)
                shap_row = shap_matrix[0]  # 1-D (8,) — SHAP for this sample
            except Exception:
                pass  # fall back to heuristic silently

        signals = self._build_signals(target_features, shap_row)

        return RiskResult(
            status="ok",
            merchant_id=merchant_id,
            date=str(target_date),
            risk_score=round(risk_score, 4),
            risk_level=risk_level,
            signals=signals,
            recommended_action=action,
            model_version=self.model_version,
        )

    def _apply_policy(self, risk_score: float):
        if risk_score >= self.high_threshold:
            return "HIGH", "Alert merchant and recommend immediate investigation."
        elif risk_score >= self.review_threshold:
            return "REVIEW", "Flag for analyst review; do not auto-alert merchant yet."
        else:
            return "LOW", "Allow / continue routine monitoring."

    def _explain(self, feature_row: pd.Series, top_k: int = 3) -> list:
        """
        Heuristic explanation: rank features by absolute deviation magnitude.
        Preserved as the fallback when SHAP is unavailable.
        """
        contributions = []
        for col in FEATURE_COLUMNS:
            val = float(feature_row[col])
            # normalize magnitude roughly onto a comparable scale for ranking
            if col in ("amount_volatility_ratio", "rolling_volume_trend", "device_diversity_ratio"):
                magnitude = abs(val - 1.0)
            else:
                magnitude = abs(val)
            contributions.append((col, val, magnitude))

        contributions.sort(key=lambda x: x[2], reverse=True)
        signals = []
        for col, val, magnitude in contributions[:top_k]:
            if magnitude < 0.15:
                continue  # not meaningfully deviated, don't manufacture a reason
            direction = "higher" if val > (1.0 if col in ("amount_volatility_ratio", "rolling_volume_trend") else 0) else "lower"
            template = FEATURE_EXPLANATIONS.get(col, "{col} deviated from baseline.")
            signals.append({
                "feature": col,
                "value": round(val, 3),
                "description": template.format(direction=direction),
            })
        if not signals:
            signals.append({
                "feature": None, "value": None,
                "description": "No individual signal deviated sharply; risk score reflects a combination of smaller shifts.",
            })
        return signals

    @staticmethod
    def _impact_label(shap_val: float) -> str:
        """Map a SHAP log-odds value to a human-readable risk impact label."""
        mag = abs(shap_val)
        if shap_val > 0:
            if mag > 2.0:
                return "Strongly increased risk"
            elif mag > 0.5:
                return "Increased risk"
            else:
                return "Slightly increased risk"
        else:
            if mag > 2.0:
                return "Strongly reduced risk"
            elif mag > 0.5:
                return "Reduced risk"
            else:
                return "Slightly reduced risk"

    def _explain_shap(self, feature_row: pd.Series, shap_row: np.ndarray, top_k: int = 3) -> list:
        """
        SHAP-based explanation for a single merchant-day.

        Parameters
        ----------
        feature_row : pd.Series
            The feature values for this merchant-day (used for the `value` field).
        shap_row : np.ndarray, shape (n_features,)
            The pre-computed SHAP values for this sample, already indexed from the
            (n_samples, n_features) batch output so this is a 1-D array of length 8.
            Values are in log-odds space: positive = pushed toward fraud, negative = normal.
        top_k : int
            Number of top contributors to surface.
        """
        # Pair each feature with its value and SHAP contribution
        pairs = []
        for i, col in enumerate(FEATURE_COLUMNS):
            pairs.append((col, float(feature_row[col]), float(shap_row[i])))

        # Sort by absolute SHAP magnitude descending
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        signals = []
        for col, val, shap_val in pairs[:top_k]:
            if abs(shap_val) <= 0.1:
                # Below noise threshold — skip to avoid manufacturing reasons
                continue
            direction = "higher" if shap_val > 0 else "lower"
            template = FEATURE_EXPLANATIONS.get(col, "{col} deviated from baseline.")
            signals.append({
                "feature": col,
                "value": round(val, 3),
                "shap_value": round(shap_val, 4),
                "impact": self._impact_label(shap_val),
                "description": template.format(direction=direction),
            })

        if not signals:
            signals.append({
                "feature": None,
                "value": None,
                "shap_value": None,
                "impact": None,
                "description": "No individual signal deviated sharply; risk score reflects a combination of smaller shifts.",
            })
        return signals

    def _build_signals(self, feature_row: pd.Series, shap_row: Optional[np.ndarray] = None) -> list:
        """
        Dispatcher: use SHAP explanation if available, fall back to heuristic.
        shap_row must be a 1-D ndarray of shape (n_features,) if provided.
        """
        if shap_row is not None:
            try:
                return self._explain_shap(feature_row, shap_row)
            except Exception:
                pass  # fall through to heuristic
        return self._explain(feature_row)

    def score_portfolio_day(self, raw_target_day: pd.DataFrame, featured_target_day: pd.DataFrame, target_date: str) -> list[RiskResult]:
        """
        Score all merchants on a specific date in bulk.
        raw_target_day and featured_target_day must be filtered to target_date.
        Uses vectorized predict_proba for speed.
        """
        if len(featured_target_day) == 0:
            return []

        results = []
        raw_rows = raw_target_day.to_dict(orient="records")
        feat_rows = featured_target_day.to_dict(orient="records")
        
        X_list = []
        valid_items = []
        
        for raw_row, feat_row in zip(raw_rows, feat_rows):
            merchant_id = raw_row["merchant_id"]
                
            validation_error = self._validate_row(raw_row)
            if validation_error:
                results.append(RiskResult(
                    status="error", merchant_id=merchant_id, date=str(target_date),
                    reason=f"Invalid input: {validation_error}"
                ))
                continue
                
            if not bool(feat_row["has_sufficient_history"]):
                results.append(RiskResult(
                    status="insufficient_evidence",
                    merchant_id=merchant_id,
                    date=str(target_date),
                    reason=(
                        f"Insufficient trailing history for a reliable baseline "
                        f"(need >= {MIN_HISTORY_DAYS} prior days). "
                        f"Manual review recommended until more history accumulates."
                    ),
                    recommended_action="MANUAL_REVIEW_INSUFFICIENT_DATA",
                ))
                continue
                
            valid_items.append((merchant_id, feat_row))
            features = [feat_row[col] for col in FEATURE_COLUMNS]
            X_list.append(features)
            
        if len(X_list) > 0:
            X = pd.DataFrame(X_list, columns=FEATURE_COLUMNS).astype(float)
            try:
                risk_scores = self.model.predict_proba(X)[:, 1]
            except Exception as e:
                for m_id, _ in valid_items:
                    results.append(RiskResult(status="error", merchant_id=m_id, date=str(target_date), reason=f"Model inference failed: {e}"))
                return results

            # Batch SHAP: compute all (n_valid, 8) SHAP values in one call.
            # shap_values(X) returns shape (n_samples, n_features).
            # Row idx corresponds exactly to valid_items[idx] — same ordering.
            # This is O(n) rather than n separate O(1) calls, and takes ~25ms for 150 rows.
            shap_matrix = None
            if self.explainer is not None:
                try:
                    shap_matrix = self.explainer.shap_values(X)  # (n_valid, 8)
                except Exception:
                    shap_matrix = None  # fall back to heuristic per merchant

            for idx, (merchant_id, feat_row) in enumerate(valid_items):
                risk_score = float(risk_scores[idx])
                risk_level, action = self._apply_policy(risk_score)
                row_series = pd.Series(feat_row)

                # Index into pre-computed batch SHAP — row idx belongs to this merchant.
                shap_row = shap_matrix[idx] if shap_matrix is not None else None
                signals = self._build_signals(row_series, shap_row)

                results.append(RiskResult(
                    status="ok",
                    merchant_id=merchant_id,
                    date=str(target_date),
                    risk_score=round(risk_score, 4),
                    risk_level=risk_level,
                    signals=signals,
                    recommended_action=action,
                    model_version=self.model_version,
                ))

        def sort_key(r):
            if r.status == "ok" and r.risk_score is not None:
                return (2, r.risk_score)
            elif r.status == "insufficient_evidence":
                return (1, 0)
            else:
                return (0, 0)

        results.sort(key=sort_key, reverse=True)
        return results


if __name__ == "__main__":
    raw = pd.read_csv(os.path.join(REPO_ROOT, "data/raw/merchant_daily_transactions.csv"), parse_dates=["date"])
    raw["date"] = raw["date"].dt.strftime("%Y-%m-%d")
    engine = RiskEngine()

    # demo: score one known-spike day and one known-normal day
    labeled = pd.read_csv(os.path.join(REPO_ROOT, "data/raw/merchant_daily_transactions.csv"), parse_dates=["date"])
    spike_example = labeled[labeled.label_fraud_spike == 1].iloc[10]
    normal_example = labeled[labeled.label_fraud_spike == 0].iloc[500]

    for label, ex in [("SPIKE-labeled example", spike_example), ("NORMAL-labeled example", normal_example)]:
        result = engine.score_merchant_day(raw, ex["date"].strftime("%Y-%m-%d"), ex["merchant_id"])
        print(f"\n--- {label}: {ex['merchant_id']} on {ex['date'].date()} ---")
        print(json.dumps(result.to_dict(), indent=2))
