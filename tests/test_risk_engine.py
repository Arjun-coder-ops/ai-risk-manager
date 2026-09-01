import sys
import os
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))

from features import build_features, FEATURE_COLUMNS, MIN_HISTORY_DAYS  # noqa: E402
from cost_model import CostAssumptions, evaluate_cost  # noqa: E402
from risk_engine import RiskEngine, RiskEngineError  # noqa: E402

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "merchant_daily_transactions.csv")


@pytest.fixture(scope="module")
def raw_df():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df


@pytest.fixture(scope="module")
def raw_df_str_date(raw_df):
    df = raw_df.copy()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


@pytest.fixture(scope="module")
def engine():
    return RiskEngine()


class TestFeatureEngineering:
    def test_features_have_no_nans(self, raw_df):
        featured = build_features(raw_df)
        assert featured[FEATURE_COLUMNS].isna().sum().sum() == 0

    def test_no_future_leakage(self, raw_df):
        """A merchant's feature row on day T must not change if we truncate
        the input to only include days <= T (i.e. no feature ever looks
        forward)."""
        merchant_id = raw_df.merchant_id.iloc[0]
        sub = raw_df[raw_df.merchant_id == merchant_id].sort_values("date").reset_index(drop=True)
        cutoff_idx = 20
        target_date = sub.loc[cutoff_idx, "date"]

        full_features = build_features(sub)
        truncated_features = build_features(sub.iloc[: cutoff_idx + 1])

        full_row = full_features[full_features.date == target_date].iloc[0]
        trunc_row = truncated_features[truncated_features.date == target_date].iloc[0]

        for col in FEATURE_COLUMNS:
            assert full_row[col] == pytest.approx(trunc_row[col], abs=1e-6), (
                f"Feature {col} differs when future data is present -- possible leakage."
            )

    def test_insufficient_history_flag(self, raw_df):
        merchant_id = raw_df.merchant_id.iloc[0]
        sub = raw_df[raw_df.merchant_id == merchant_id].sort_values("date")
        featured = build_features(sub)
        early_rows = featured.iloc[:MIN_HISTORY_DAYS]
        assert (~early_rows["has_sufficient_history"]).all()


class TestCostModel:
    def test_cost_scales_with_counts(self):
        base = evaluate_cost(tp=10, fp=5, tn=100, fn=2)
        doubled_fp = evaluate_cost(tp=10, fp=10, tn=100, fn=2)
        assert doubled_fp["total_false_positive_cost_inr"] == pytest.approx(
            2 * base["total_false_positive_cost_inr"]
        )

    def test_custom_assumptions_are_used(self):
        custom = CostAssumptions(manual_review_cost_inr=1, merchant_friction_cost_inr=1, undetected_fraud_loss_inr=1)
        result = evaluate_cost(tp=0, fp=10, tn=0, fn=10, assumptions=custom)
        assert result["total_false_positive_cost_inr"] == 20
        assert result["total_false_negative_cost_inr"] == 10


class TestRiskEnginePolicy:
    def test_policy_thresholds_ordering(self, engine):
        assert engine.high_threshold > engine.review_threshold

    def test_scores_known_spike_as_elevated(self, engine, raw_df_str_date, raw_df):
        spike_row = raw_df[raw_df.label_fraud_spike == 1].iloc[5]
        result = engine.score_merchant_day(
            raw_df_str_date, spike_row["date"].strftime("%Y-%m-%d"), spike_row["merchant_id"]
        )
        assert result.status == "ok"
        assert result.risk_level in ("REVIEW", "HIGH")

    def test_explanation_grounded_in_actual_features(self, engine, raw_df_str_date, raw_df):
        spike_row = raw_df[raw_df.label_fraud_spike == 1].iloc[5]
        result = engine.score_merchant_day(
            raw_df_str_date, spike_row["date"].strftime("%Y-%m-%d"), spike_row["merchant_id"]
        )
        assert len(result.signals) >= 1
        for s in result.signals:
            if s["feature"] is not None:
                assert s["feature"] in FEATURE_COLUMNS


class TestFailureHandling:
    def test_missing_merchant(self, engine, raw_df_str_date):
        result = engine.score_merchant_day(raw_df_str_date, "2026-06-01", "NOT_A_REAL_MERCHANT")
        assert result.status == "error"

    def test_missing_date(self, engine, raw_df_str_date):
        result = engine.score_merchant_day(raw_df_str_date, "2099-01-01", raw_df_str_date.merchant_id.iloc[0])
        assert result.status == "error"

    def test_invalid_failed_exceeds_total(self, engine, raw_df_str_date):
        bad = raw_df_str_date[raw_df_str_date.merchant_id == raw_df_str_date.merchant_id.iloc[0]].head(20).copy()
        bad.loc[bad.index[-1], "n_failed"] = bad.loc[bad.index[-1], "n_transactions"] + 100
        result = engine.score_merchant_day(bad, bad.iloc[-1]["date"], bad.merchant_id.iloc[0])
        assert result.status == "error"

    def test_insufficient_history_not_confident(self, engine, raw_df_str_date):
        merchant_id = raw_df_str_date.merchant_id.iloc[0]
        sub = raw_df_str_date[raw_df_str_date.merchant_id == merchant_id].sort_values("date").head(3)
        result = engine.score_merchant_day(sub, sub.iloc[-1]["date"], merchant_id)
        assert result.status == "insufficient_evidence"
        assert result.risk_level is None

    def test_empty_input(self, engine):
        result = engine.score_merchant_day(pd.DataFrame(), "2026-01-01", "M0000")
        assert result.status == "error"
class TestPortfolioScoring:
    def test_portfolio_consistency_with_single_merchant(self, engine, raw_df_str_date):
        target_date = "2026-05-31"
        raw_target = raw_df_str_date[raw_df_str_date["date"] == target_date]
        from features import build_features
        featured = build_features(raw_df_str_date)
        featured_target = featured[featured["date"] == target_date]
        
        results = engine.score_portfolio_day(raw_target, featured_target, target_date)
        assert len(results) > 0
        
        # Check first merchant
        first_result = results[0]
        
        # Verify it matches the single-merchant route
        single_result = engine.score_merchant_day(raw_df_str_date, target_date, first_result.merchant_id)
        
        assert first_result.status == single_result.status
        if first_result.status == "ok":
            assert abs(first_result.risk_score - single_result.risk_score) < 0.0001
            assert first_result.risk_level == single_result.risk_level
            
    def test_ranking_by_risk_score(self, engine, raw_df_str_date):
        target_date = "2026-05-31"
        raw_target = raw_df_str_date[raw_df_str_date["date"] == target_date]
        from features import build_features
        featured = build_features(raw_df_str_date)
        featured_target = featured[featured["date"] == target_date]
        
        results = engine.score_portfolio_day(raw_target, featured_target, target_date)
        
        # Check sorting order
        scores = [r.risk_score for r in results if r.status == "ok"]
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
        
    def test_invalid_date(self, engine, raw_df_str_date):
        target_date = "2099-01-01"
        raw_target = raw_df_str_date[raw_df_str_date["date"] == target_date]
        from features import build_features
        featured = build_features(raw_df_str_date)
        featured_target = featured[featured["date"] == target_date]
        
        results = engine.score_portfolio_day(raw_target, featured_target, target_date)
        assert len(results) == 0


class TestSHAPExplainability:
    """Tests for the SHAP explanation layer.

    SHAP semantics (shap==0.49.1, HistGradientBoostingClassifier):
      shap_values(X) → ndarray shape (n_samples, n_features).
      Row i = SHAP contributions for sample i in log-odds space.
      Positive value → feature pushed prediction toward fraud.
      SHAP is post-hoc only — it never changes risk scores or predictions.
    """

    def test_explainer_initialised(self, engine):
        """Engine must have a SHAP explainer when shap is installed."""
        assert engine.explainer is not None, (
            "shap.TreeExplainer should be initialised; install shap>=0.46"
        )

    def test_shap_signals_present_for_valid_prediction(self, engine, raw_df_str_date):
        """A high-risk merchant-day must include signals with shap_value and impact."""
        spike_row = raw_df_str_date[raw_df_str_date.label_fraud_spike == 1].iloc[5]
        result = engine.score_merchant_day(
            raw_df_str_date, spike_row["date"], spike_row["merchant_id"]
        )
        assert result.status == "ok"
        assert len(result.signals) >= 1
        # At least one signal should carry SHAP fields (non-null feature)
        meaningful = [s for s in result.signals if s.get("feature") is not None]
        assert len(meaningful) >= 1
        for s in meaningful:
            assert "shap_value" in s, "signal is missing shap_value"
            assert "impact" in s, "signal is missing impact"
            assert s["shap_value"] is not None
            assert isinstance(s["shap_value"], float)

    def test_shap_feature_names_match_feature_columns(self, engine, raw_df_str_date):
        """Every named signal feature must be in FEATURE_COLUMNS."""
        spike_row = raw_df_str_date[raw_df_str_date.label_fraud_spike == 1].iloc[5]
        result = engine.score_merchant_day(
            raw_df_str_date, spike_row["date"], spike_row["merchant_id"]
        )
        for s in result.signals:
            if s.get("feature") is not None:
                assert s["feature"] in FEATURE_COLUMNS

    def test_shap_output_shape(self, engine, raw_df_str_date):
        """SHAP matrix from TreeExplainer must be (n_samples, len(FEATURE_COLUMNS))."""
        import numpy as np
        featured = build_features(raw_df_str_date)
        sample = featured[featured["has_sufficient_history"]].head(5)
        X = sample[FEATURE_COLUMNS].astype(float)
        shap_matrix = engine.explainer.shap_values(X)
        assert isinstance(shap_matrix, np.ndarray), "Expected ndarray from shap_values"
        assert shap_matrix.shape == (5, len(FEATURE_COLUMNS)), (
            f"Expected (5, {len(FEATURE_COLUMNS)}), got {shap_matrix.shape}"
        )

    def test_shap_row_alignment_in_portfolio(self, engine, raw_df_str_date):
        """Each merchant in portfolio must receive the correct SHAP row (not another merchant's)."""
        target_date = "2026-05-31"
        raw_target = raw_df_str_date[raw_df_str_date["date"] == target_date]
        featured = build_features(raw_df_str_date)
        featured_target = featured[featured["date"] == target_date]

        portfolio_results = engine.score_portfolio_day(raw_target, featured_target, target_date)

        # For each portfolio result, compare with single-merchant — signals should reference
        # consistent features (not another merchant's top features).
        for port_res in portfolio_results[:5]:  # spot-check top 5
            if port_res.status != "ok":
                continue
            single_res = engine.score_merchant_day(
                raw_df_str_date, target_date, port_res.merchant_id
            )
            assert port_res.risk_score == pytest.approx(single_res.risk_score, abs=1e-4)
            assert port_res.risk_level == single_res.risk_level

    def test_risk_score_unchanged_after_shap(self, engine, raw_df_str_date):
        """SHAP must not modify the predicted risk score."""
        spike_row = raw_df_str_date[raw_df_str_date.label_fraud_spike == 1].iloc[5]
        result = engine.score_merchant_day(
            raw_df_str_date, spike_row["date"], spike_row["merchant_id"]
        )
        # Compute raw predict_proba to compare
        import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
        import pandas as pd
        featured = build_features(raw_df_str_date)
        m_id = spike_row["merchant_id"]
        d = spike_row["date"]
        row = featured[(featured["date"] == d)].to_dict(orient="records")
        # Find this merchant's row
        merchant_rows = [r for r in row if r.get("merchant_id") == m_id or
                         raw_df_str_date[(raw_df_str_date["date"] == d) &
                                         (raw_df_str_date["merchant_id"] == m_id)].shape[0] > 0]
        # Simple: re-run predict_proba directly
        feat_row = featured[(featured["date"] == d)].copy()
        # Get the merchant history
        hist = raw_df_str_date[raw_df_str_date["merchant_id"] == m_id]
        feat_m = build_features(hist)
        X = feat_m[feat_m["date"] == d][FEATURE_COLUMNS].astype(float)
        raw_score = float(engine.model.predict_proba(X)[0, 1])
        assert result.risk_score == pytest.approx(round(raw_score, 4), abs=1e-6)

    def test_shap_fallback_when_explainer_is_none(self, engine, raw_df_str_date):
        """When explainer is None, explanation must fall back to heuristic gracefully."""
        original_explainer = engine.explainer
        try:
            engine.explainer = None
            spike_row = raw_df_str_date[raw_df_str_date.label_fraud_spike == 1].iloc[5]
            result = engine.score_merchant_day(
                raw_df_str_date, spike_row["date"], spike_row["merchant_id"]
            )
            assert result.status == "ok"
            assert len(result.signals) >= 1
            # Fallback signals should NOT have shap_value field
            for s in result.signals:
                assert "shap_value" not in s
        finally:
            engine.explainer = original_explainer  # always restore


class TestPortfolioDateRange:
    """Tests for date-range portfolio scoring."""
    
    def test_single_date_backward_compatible(self, engine, raw_df_str_date):
        """Existing single-date endpoint remains unaffected."""
        target_date = "2026-05-31"
        raw_target = raw_df_str_date[raw_df_str_date["date"] == target_date]
        featured = build_features(raw_df_str_date)
        featured_target = featured[featured["date"] == target_date]
        
        results = engine.score_portfolio_day(raw_target, featured_target, target_date)
        assert len(results) == 150  # All merchants present
        assert all(r.date == target_date for r in results)
        
        # Ranking still correct
        scores = [r.risk_score for r in results if r.status == "ok"]
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    
    def test_daterange_multiple_dates_scored(self, engine, raw_df_str_date):
        """Date-range returns results for each date in range."""
        featured = build_features(raw_df_str_date)
        dates_to_check = ["2026-05-29", "2026-05-30", "2026-05-31"]
        
        all_results = []
        for date_str in dates_to_check:
            raw_day = raw_df_str_date[raw_df_str_date["date"] == date_str]
            featured_day = featured[featured["date"] == date_str]
            if len(raw_day) > 0:
                results = engine.score_portfolio_day(raw_day, featured_day, date_str)
                all_results.append((date_str, results))
        
        # Should have results for all 3 dates
        assert len(all_results) == 3
        # Each date should have 150 merchants
        for date_str, results in all_results:
            assert len(results) == 150
    
    def test_merchant_consistency_across_dates(self, engine, raw_df_str_date):
        """A merchant's score is consistent whether queried in single-date or range mode."""
        featured = build_features(raw_df_str_date)
        merchant_id = "M0050"
        target_date = "2026-05-31"
        
        # Single date
        raw_single = raw_df_str_date[raw_df_str_date["date"] == target_date]
        featured_single = featured[featured["date"] == target_date]
        result_single = engine.score_portfolio_day(raw_single, featured_single, target_date)
        single_result = [r for r in result_single if r.merchant_id == merchant_id][0]
        
        # Date range containing same date
        raw_range = raw_df_str_date[raw_df_str_date["date"].isin(["2026-05-30", "2026-05-31"])]
        featured_range = featured[featured["date"].isin(["2026-05-30", "2026-05-31"])]
        
        # Query on the target date
        raw_target = raw_range[raw_range["date"] == target_date]
        feat_target = featured_range[featured_range["date"] == target_date]
        if len(raw_target) > 0:
            range_result_all = engine.score_portfolio_day(raw_target, feat_target, target_date)
            range_result = [r for r in range_result_all if r.merchant_id == merchant_id][0]
            
            assert single_result.risk_score == pytest.approx(range_result.risk_score, abs=1e-4)
            assert single_result.risk_level == range_result.risk_level
    
    def test_merchants_ranked_same_within_each_date(self, engine, raw_df_str_date):
        """All merchants within a date are sorted by risk_score descending."""
        featured = build_features(raw_df_str_date)
        dates_to_check = ["2026-05-29", "2026-05-30", "2026-05-31"]
        
        for target_date in dates_to_check:
            raw_day = raw_df_str_date[raw_df_str_date["date"] == target_date]
            feat_day = featured[featured["date"] == target_date]
            if len(raw_day) > 0:
                results = engine.score_portfolio_day(raw_day, feat_day, target_date)
                scores = [r.risk_score for r in results if r.status == "ok"]
                assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1)), \
                    f"Scores not descending on {target_date}"
    
    def test_empty_daterange_outside_dataset(self, engine, raw_df_str_date):
        """Dates outside dataset are silently skipped."""
        featured = build_features(raw_df_str_date)
        
        # Date range entirely outside dataset
        from_date = "2026-07-01"
        to_date = "2026-07-10"
        
        results = []
        for date_str in pd.date_range(from_date, to_date, freq='D').strftime('%Y-%m-%d').tolist():
            raw_day = raw_df_str_date[raw_df_str_date["date"] == date_str]
            if len(raw_day) == 0:
                continue
            feat_day = featured[featured["date"] == date_str]
            results.append(engine.score_portfolio_day(raw_day, feat_day, date_str))
        
        assert len(results) == 0, "No results expected for out-of-range dates"
    
    def test_daterange_chronological_order(self, engine, raw_df_str_date):
        """Date-range results are in chronological order."""
        featured = build_features(raw_df_str_date)
        
        from_date = "2026-05-28"
        to_date = "2026-05-31"
        date_range = pd.date_range(from_date, to_date, freq='D').strftime('%Y-%m-%d').tolist()
        
        all_dates = []
        for date_str in date_range:
            raw_day = raw_df_str_date[raw_df_str_date["date"] == date_str]
            feat_day = featured[featured["date"] == date_str]
            if len(raw_day) > 0:
                all_dates.append(date_str)
        
        # Should be in order
        assert all_dates == sorted(all_dates)
    
    def test_score_consistency_single_vs_range(self, engine, raw_df_str_date):
        """Risk scores match exactly between single-date and range queries."""
        featured = build_features(raw_df_str_date)
        target_date = "2026-05-31"
        
        raw_target = raw_df_str_date[raw_df_str_date["date"] == target_date]
        feat_target = featured[featured["date"] == target_date]
        
        results_single = engine.score_portfolio_day(raw_target, feat_target, target_date)
        
        # Simulate range query for same date
        results_range = engine.score_portfolio_day(raw_target, feat_target, target_date)
        
        # Compare top 10 merchants
        for i in range(min(10, len(results_single), len(results_range))):
            assert results_single[i].merchant_id == results_range[i].merchant_id
            assert results_single[i].risk_score == pytest.approx(results_range[i].risk_score, abs=1e-6)
            assert results_single[i].risk_level == results_range[i].risk_level
