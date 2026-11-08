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

from features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, build_features, MIN_HISTORY_DAYS
from explanations import (
    EXPLANATION_SCHEMA_VERSION,
    heuristic_deviation_signal,
    normalize_tree_shap_values,
    tree_attribution_signal,
)
from policy import PolicyDecision, PolicyValidationError, load_policy
from validation import (
    ObservationValidationError,
    normalize_date,
    normalize_merchant_id,
    validate_observations,
)

MODEL_PATH = os.path.join(REPO_ROOT, "model/artifacts/model.joblib")
META_PATH = os.path.join(REPO_ROOT, "model/artifacts/model_meta.json")
POLICY_PATH = os.path.join(REPO_ROOT, "model/artifacts/policy.json")

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
    policy_version: Optional[str] = None
    policy_decision: Optional[dict] = None
    reason: Optional[str] = None

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


class RiskEngine:
    def __init__(self, model_path=MODEL_PATH, meta_path=META_PATH, policy_path=POLICY_PATH):
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
        try:
            self.policy = load_policy(policy_path)
        except PolicyValidationError as exc:
            raise RiskEngineError(f"Risk policy could not be loaded: {exc}") from exc
        if self.policy.model_version != self.model_version:
            raise RiskEngineError(
                "Risk policy model version does not match the loaded model artifact. "
                "Publish a compatible policy artifact before scoring."
            )
        if (
            not np.isclose(self.policy.review_threshold, self.review_threshold)
            or not np.isclose(self.policy.high_threshold, self.high_threshold)
        ):
            raise RiskEngineError(
                "Risk policy thresholds do not match the model artifact metadata. "
                "Regenerate or explicitly version compatible artifacts."
            )
        if self.meta.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise RiskEngineError(
                "Model feature schema does not match the runtime feature contract. "
                "Run `python model/train.py` to regenerate compatible artifacts."
            )
        if self.meta.get("feature_columns") != FEATURE_COLUMNS:
            raise RiskEngineError(
                "Model feature columns do not match the runtime feature contract. "
                "Run `python model/train.py` to regenerate compatible artifacts."
            )

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

    def score_merchant_day(self, history_df: pd.DataFrame, target_date: str, merchant_id: str) -> RiskResult:
        """
        history_df: raw merchant-day rows (same schema as generate_data.py output)
                    for this merchant, including the target date and enough
                    trailing history to compute rolling baselines.
        """
        try:
            history_df = validate_observations(history_df)
            target_date = normalize_date(target_date)
            merchant_id = normalize_merchant_id(merchant_id)
        except ObservationValidationError as exc:
            return RiskResult(status="error", reason=f"Invalid input: {exc}")

        merchant_hist = history_df[history_df["merchant_id"] == merchant_id].copy()
        if len(merchant_hist) == 0:
            return RiskResult(status="error", reason=f"No data found for merchant {merchant_id}.")

        target_row = merchant_hist[merchant_hist["date"] == target_date]
        if len(target_row) == 0:
            return RiskResult(status="error", reason=f"No transaction row found for {merchant_id} on {target_date}.")

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
            return self._insufficient_evidence_result(merchant_id, target_date)

        X = target_features[FEATURE_COLUMNS].to_frame().T.astype(float)
        try:
            risk_score = float(self.model.predict_proba(X)[0, 1])
        except Exception as e:
            return RiskResult(status="error", merchant_id=merchant_id, date=str(target_date),
                               reason=f"Model inference failed: {e}")

        policy_decision = self._apply_policy(risk_score)

        # Compute SHAP for this single row if the explainer is available.
        # Normalize supported SHAP return conventions to one (1, n_features) row.
        shap_row = None
        if self.explainer is not None:
            try:
                shap_matrix = self.explainer.shap_values(X)
                shap_row = normalize_tree_shap_values(
                    shap_matrix, expected_rows=1, expected_features=len(FEATURE_COLUMNS)
                )[0]
            except Exception:
                pass  # fall back to heuristic silently

        signals = self._build_signals(target_features, shap_row)

        return RiskResult(
            status="ok",
            merchant_id=merchant_id,
            date=str(target_date),
            risk_score=round(risk_score, 4),
            risk_level=policy_decision.risk_level,
            signals=signals,
            recommended_action=policy_decision.action_description,
            model_version=self.model_version,
            policy_version=policy_decision.policy_version,
            policy_decision=policy_decision.to_dict(),
        )

    def _apply_policy(self, risk_score: float) -> PolicyDecision:
        """Map a model estimate to a versioned business decision only."""
        return self.policy.decide(risk_score)

    def _insufficient_evidence_result(self, merchant_id: str, target_date: str) -> RiskResult:
        """Do not make a numeric risk-policy claim when history is insufficient."""
        action_code = "MANUAL_REVIEW_INSUFFICIENT_DATA"
        return RiskResult(
            status="insufficient_evidence",
            merchant_id=merchant_id,
            date=str(target_date),
            reason=(
                f"Insufficient trailing history for a reliable baseline "
                f"(need >= {MIN_HISTORY_DAYS} prior days). "
                f"Manual review recommended until more history accumulates."
            ),
            recommended_action=action_code,
            policy_version=self.policy.policy_version,
            policy_decision={
                "risk_level": None,
                "action_code": action_code,
                "action_description": "Manual review recommendation due to insufficient evidence, not a model risk tier.",
                "policy_version": self.policy.policy_version,
                "threshold_band": "insufficient_evidence",
            },
        )

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
            signals.append(heuristic_deviation_signal(
                col,
                feature_row,
                model_version=self.model_version,
                feature_schema_version=FEATURE_SCHEMA_VERSION,
            ))
        if not signals:
            signals.append({
                "feature": None, "value": None,
                "description": "No individual signal deviated sharply; risk score reflects a combination of smaller shifts.",
                "explanation_provenance": {
                    "schema_version": EXPLANATION_SCHEMA_VERSION,
                    "method": "heuristic_deviation_fallback",
                    "attribution_space": None,
                    "model_version": self.model_version,
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "is_model_attribution": False,
                },
            })
        return signals

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
            signals.append(tree_attribution_signal(
                col,
                feature_row,
                shap_val,
                model_version=self.model_version,
                feature_schema_version=FEATURE_SCHEMA_VERSION,
            ))

        if not signals:
            signals.append({
                "feature": None,
                "value": None,
                "shap_value": None,
                "impact": None,
                "description": "No individual signal deviated sharply; risk score reflects a combination of smaller shifts.",
                "explanation_provenance": {
                    "schema_version": EXPLANATION_SCHEMA_VERSION,
                    "method": "tree_shap",
                    "attribution_space": "model_raw_margin_log_odds",
                    "model_version": self.model_version,
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "is_model_attribution": True,
                },
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
        try:
            target_date = normalize_date(target_date)
        except ObservationValidationError as exc:
            return [RiskResult(status="error", date=str(target_date), reason=f"Invalid input: {exc}")]

        # Keep the established API behavior for dates outside the available
        # data range. There is no malformed observation to validate here.
        if raw_target_day.empty and featured_target_day.empty:
            return []

        try:
            raw_target_day = validate_observations(raw_target_day)
        except ObservationValidationError as exc:
            return [RiskResult(status="error", date=str(target_date), reason=f"Invalid input: {exc}")]

        feature_required = {"merchant_id", "date", "has_sufficient_history", *FEATURE_COLUMNS}
        missing_feature_columns = sorted(feature_required - set(featured_target_day.columns))
        if missing_feature_columns:
            return [RiskResult(
                status="error", date=str(target_date),
                reason=f"Feature input is missing required fields: {missing_feature_columns}",
            )]

        features = featured_target_day.copy()
        try:
            features["merchant_id"] = features["merchant_id"].map(normalize_merchant_id)
            features["date"] = features["date"].map(normalize_date)
        except ObservationValidationError as exc:
            return [RiskResult(status="error", date=str(target_date), reason=f"Invalid feature input: {exc}")]
        if features.duplicated(["merchant_id", "date"]).any():
            return [RiskResult(
                status="error", date=str(target_date),
                reason="Feature input contains duplicate merchant-day keys.",
            )]
        try:
            values = features[FEATURE_COLUMNS].astype(float)
            if not np.isfinite(values.to_numpy()).all():
                raise ValueError("Feature values must be finite.")
        except (TypeError, ValueError) as exc:
            return [RiskResult(status="error", date=str(target_date), reason=f"Invalid feature input: {exc}")]

        raw_target_day = raw_target_day[raw_target_day["date"] == target_date].copy()
        features = features[features["date"] == target_date].copy()
        if raw_target_day.empty:
            return []

        results = []
        keyed_features = features.set_index(["merchant_id", "date"], drop=False)
        
        X_list = []
        valid_items = []
        
        for raw_row in raw_target_day.to_dict(orient="records"):
            merchant_id = raw_row["merchant_id"]
            key = (merchant_id, target_date)
            if key not in keyed_features.index:
                results.append(RiskResult(
                    status="error", merchant_id=merchant_id, date=str(target_date),
                    reason="No feature row found for this merchant-day key.",
                ))
                continue
            feat_row = keyed_features.loc[key].to_dict()
                
            if not bool(feat_row["has_sufficient_history"]):
                results.append(self._insufficient_evidence_result(merchant_id, target_date))
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
                    shap_matrix = normalize_tree_shap_values(
                        self.explainer.shap_values(X),
                        expected_rows=len(X),
                        expected_features=len(FEATURE_COLUMNS),
                    )
                except Exception:
                    shap_matrix = None  # fall back to heuristic per merchant

            for idx, (merchant_id, feat_row) in enumerate(valid_items):
                risk_score = float(risk_scores[idx])
                policy_decision = self._apply_policy(risk_score)
                row_series = pd.Series(feat_row)

                # Index into pre-computed batch SHAP — row idx belongs to this merchant.
                shap_row = shap_matrix[idx] if shap_matrix is not None else None
                signals = self._build_signals(row_series, shap_row)

                results.append(RiskResult(
                    status="ok",
                    merchant_id=merchant_id,
                    date=str(target_date),
                    risk_score=round(risk_score, 4),
                    risk_level=policy_decision.risk_level,
                    signals=signals,
                    recommended_action=policy_decision.action_description,
                    model_version=self.model_version,
                    policy_version=policy_decision.policy_version,
                    policy_decision=policy_decision.to_dict(),
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
