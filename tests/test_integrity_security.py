"""Regression coverage for Milestone 1 data-integrity and API hardening."""

from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))

import app as backend_app  # noqa: E402
from features import FEATURE_COLUMNS, build_features  # noqa: E402
from risk_engine import RiskEngine, RiskResult  # noqa: E402
from validation import ObservationValidationError  # noqa: E402


DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "merchant_daily_transactions.csv"
FRONTEND_PATH = Path(__file__).parent.parent / "frontend" / "index.html"


@pytest.fixture(scope="module")
def raw_df():
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


@pytest.fixture(scope="module")
def engine():
    return RiskEngine()


@pytest.fixture
def client():
    backend_app.app.config["TESTING"] = True
    with backend_app.app.test_client() as test_client:
        yield test_client


class TestObservationValidation:
    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("n_transactions", -1, "cannot be negative"),
            ("n_failed", np.nan, "must be finite"),
            ("n_unique_users", 1.5, "must be an integer count"),
            ("avg_amount", np.inf, "must be finite"),
        ],
    )
    def test_rejects_invalid_numeric_values(self, engine, raw_df, field, value, message):
        history = raw_df[raw_df["merchant_id"] == "M0000"].head(20).copy()
        if field == "n_unique_users":
            history[field] = history[field].astype(float)
        history.loc[history.index[-1], field] = value
        result = engine.score_merchant_day(history, history.iloc[-1]["date"], "M0000")
        assert result.status == "error"
        assert message in result.reason

    def test_rejects_impossible_relationships(self, engine, raw_df):
        history = raw_df[raw_df["merchant_id"] == "M0000"].head(20).copy()
        history.loc[history.index[-1], "n_new_users"] = history.loc[history.index[-1], "n_unique_users"] + 1
        result = engine.score_merchant_day(history, history.iloc[-1]["date"], "M0000")
        assert result.status == "error"
        assert "Impossible count relationship" in result.reason

    def test_rejects_missing_fields_invalid_dates_and_duplicate_keys(self, engine, raw_df):
        history = raw_df[raw_df["merchant_id"] == "M0000"].head(20).copy()
        missing = history.drop(columns=["amount_std"])
        missing_result = engine.score_merchant_day(missing, history.iloc[-1]["date"], "M0000")
        assert missing_result.status == "error"
        assert "Missing required fields" in missing_result.reason

        invalid_date = history.copy()
        invalid_date["date"] = invalid_date["date"].astype("object")
        invalid_date.loc[invalid_date.index[-1], "date"] = "2026-99-99"
        date_result = engine.score_merchant_day(invalid_date, "2026-99-99", "M0000")
        assert date_result.status == "error"
        assert "date" in date_result.reason

        duplicate = pd.concat([history, history.tail(1)], ignore_index=True)
        duplicate_result = engine.score_merchant_day(duplicate, history.iloc[-1]["date"], "M0000")
        assert duplicate_result.status == "error"
        assert "Duplicate merchant-day" in duplicate_result.reason
        with pytest.raises(ObservationValidationError, match="Duplicate merchant-day"):
            build_features(duplicate)


class TestKeyedPortfolioScoring:
    def test_features_preserve_unique_merchant_day_keys(self, raw_df):
        featured = build_features(raw_df)
        assert {"merchant_id", "date"}.issubset(featured.columns)
        assert not featured.duplicated(["merchant_id", "date"]).any()

    def test_independently_shuffled_inputs_match_single_merchant_scores_and_signals(self, engine, raw_df):
        raw = raw_df.copy()
        raw["date"] = raw["date"].dt.strftime("%Y-%m-%d")
        target_date = "2026-05-31"
        featured = build_features(raw)
        merchant_ids = raw.loc[raw["date"] == target_date, "merchant_id"].iloc[:8].tolist()
        raw_target = raw[(raw["date"] == target_date) & raw["merchant_id"].isin(merchant_ids)]
        featured_target = featured[(featured["date"] == target_date) & featured["merchant_id"].isin(merchant_ids)]

        batch = engine.score_portfolio_day(
            raw_target.sample(frac=1, random_state=11),
            featured_target.sample(frac=1, random_state=29),
            target_date,
        )
        batch_by_merchant = {result.merchant_id: result for result in batch}
        assert set(batch_by_merchant) == set(merchant_ids)
        for merchant_id in merchant_ids:
            single = engine.score_merchant_day(
                raw[raw["merchant_id"] == merchant_id], target_date, merchant_id
            )
            batch_result = batch_by_merchant[merchant_id]
            assert batch_result.status == single.status
            assert batch_result.risk_score == pytest.approx(single.risk_score, abs=1e-4)
            assert batch_result.risk_level == single.risk_level
            assert batch_result.signals == single.signals

    def test_missing_feature_key_becomes_explicit_error(self, engine, raw_df):
        raw = raw_df.copy()
        raw["date"] = raw["date"].dt.strftime("%Y-%m-%d")
        target_date = "2026-05-31"
        featured = build_features(raw)
        raw_target = raw[raw["date"] == target_date].head(2)
        featured_target = featured[featured["date"] == target_date]
        missing_merchant = raw_target.iloc[0]["merchant_id"]
        featured_target = featured_target[featured_target["merchant_id"] != missing_merchant]
        results = engine.score_portfolio_day(raw_target, featured_target, target_date)
        error = next(result for result in results if result.merchant_id == missing_merchant)
        assert error.status == "error"
        assert "No feature row" in error.reason


class TestApiAndFrontendSecurity:
    def test_malformed_and_non_object_json_are_controlled_client_errors(self, client):
        malformed = client.post("/api/risk/analyze", data="{not-json", content_type="application/json")
        assert malformed.status_code == 400
        assert malformed.get_json()["status"] == "error"

        non_object = client.post("/api/risk/analyze", json=[])
        assert non_object.status_code == 400
        assert non_object.get_json()["status"] == "error"

    def test_invalid_markup_identifier_is_rejected_before_audit(self, client):
        response = client.post(
            "/api/risk/analyze",
            json={"merchant_id": "<img src=x onerror=alert(1)>", "date": "2026-05-31"},
        )
        assert response.status_code == 422
        assert "Invalid request" in response.get_json()["reason"]

    def test_cors_is_allowlisted_and_labels_are_not_in_dashboard_responses(self, client, monkeypatch):
        allowed = client.get("/health", headers={"Origin": "http://localhost:3000"})
        assert allowed.headers["Access-Control-Allow-Origin"] == "http://localhost:3000"
        denied = client.get("/health", headers={"Origin": "https://untrusted.example"})
        assert "Access-Control-Allow-Origin" not in denied.headers

        merchants = client.get("/api/risk/merchants").get_json()["merchants"]
        assert merchants and "spike_days" not in merchants[0]

        class FastTestEngine:
            model_version = "test-model"

            def score_merchant_day(self, _history, date, merchant_id):
                return RiskResult(
                    status="ok",
                    merchant_id=merchant_id,
                    date=date,
                    risk_score=0.1,
                    risk_level="LOW",
                )

        monkeypatch.setattr(backend_app, "engine", FastTestEngine())
        points = client.get("/api/risk/timeseries/M0000").get_json()["points"]
        assert points and all("actual_label" not in point for point in points)

    def test_frontend_uses_no_innerhtml_sink_for_dynamic_api_data(self):
        frontend = FRONTEND_PATH.read_text(encoding="utf-8")
        assert "innerHTML" not in frontend
        assert "textContent" in frontend
