"""
Backend API for the AI Risk Manager.

Engineering decision: implemented in Flask (Python), sitting directly on
top of the sklearn model, rather than a separate Node/Express service that
calls out to Python. Rationale: the risk engine's core logic (feature
computation + inference) is Python/pandas/sklearn; adding a second runtime
and an internal network hop between Node and Python would add latency and
a second point of failure for a hackathon-scale prototype without a real
benefit. If this were productionized behind Razorpay's actual stack, this
service would sit behind a thin gateway (which could be Node/Express) --
that boundary is a config change, not a rewrite, because the API is a
plain JSON HTTP contract.

Endpoints:
  POST /api/risk/analyze   -- score one merchant-day, returns structured risk result
  GET  /api/risk/audit      -- list recent audit records
  GET  /api/risk/merchants  -- list known merchants (for the dashboard dropdown)
  GET  /api/risk/timeseries/<merchant_id> -- historical scores for a merchant (for charting)
  GET  /api/risk/evaluation -- the last held-out test evaluation report
  GET  /health
"""

import json
import os
import sys
import uuid
from datetime import datetime, timezone

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from risk_engine import RiskEngine, RiskEngineError, build_features  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "merchant_daily_transactions.csv")
AUDIT_LOG_PATH = os.path.join(BASE_DIR, "evaluation", "audit_log.jsonl")
EVAL_RESULTS_PATH = os.path.join(BASE_DIR, "evaluation", "test_results.json")

app = Flask(__name__)
CORS(app)

try:
    engine = RiskEngine()
    ENGINE_LOAD_ERROR = None
except RiskEngineError as e:
    engine = None
    ENGINE_LOAD_ERROR = str(e)

_raw_cache = None
_featured_cache = None


def get_raw_data() -> pd.DataFrame:
    global _raw_cache
    if _raw_cache is None:
        df = pd.read_csv(DATA_PATH, parse_dates=["date"])
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        _raw_cache = df
    return _raw_cache

def get_featured_data(raw: pd.DataFrame) -> pd.DataFrame:
    global _featured_cache
    if _featured_cache is None:
        _featured_cache = build_features(raw)
    return _featured_cache

def write_audit_record(record: dict):
    os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


@app.route("/health", methods=["GET"])
def health():
    if engine is None:
        return jsonify({"status": "degraded", "reason": ENGINE_LOAD_ERROR}), 503
    return jsonify({"status": "ok", "model_version": engine.model_version})


@app.route("/api/risk/analyze", methods=["POST"])
def analyze():
    if engine is None:
        return jsonify({"status": "error", "reason": f"Risk engine unavailable: {ENGINE_LOAD_ERROR}"}), 503

    body = request.get_json(silent=True)
    if not body:
        return jsonify({"status": "error", "reason": "Request body must be valid JSON."}), 400

    merchant_id = body.get("merchant_id")
    date = body.get("date")
    if not merchant_id or not date:
        return jsonify({"status": "error", "reason": "merchant_id and date are required."}), 400

    try:
        raw = get_raw_data()
    except FileNotFoundError:
        return jsonify({"status": "error", "reason": "Transaction history data not found on server."}), 500

    result = engine.score_merchant_day(raw, date, merchant_id)
    result_dict = result.to_dict()
    result_dict["request_id"] = str(uuid.uuid4())
    result_dict["timestamp"] = datetime.now(timezone.utc).isoformat()

    write_audit_record({
        "request_id": result_dict["request_id"],
        "timestamp": result_dict["timestamp"],
        "merchant_id": merchant_id,
        "date": date,
        "status": result.status,
        "risk_score": result.risk_score,
        "risk_level": result.risk_level,
        "signals": result.signals,
        "model_version": result.model_version,
        "recommended_action": result.recommended_action,
        "reason": result.reason,
    })

    status_code = 200 if result.status in ("ok", "insufficient_evidence") else 422
    return jsonify(result_dict), status_code


@app.route("/api/risk/portfolio", methods=["GET"])
def portfolio():
    if engine is None:
        return jsonify({"status": "error", "reason": f"Risk engine unavailable: {ENGINE_LOAD_ERROR}"}), 503

    # Parse query parameters
    date = request.args.get("date")
    from_date = request.args.get("from_date")
    to_date = request.args.get("to_date")

    # Validate parameter combinations
    has_single = date is not None
    has_range = (from_date is not None) or (to_date is not None)

    if has_single and has_range:
        return jsonify({"status": "error", "reason": "Provide either 'date' OR 'from_date'+'to_date', not both."}), 400

    if has_single:
        return _portfolio_single_date(date)
    elif has_range:
        if not from_date or not to_date:
            return jsonify({"status": "error", "reason": "Both 'from_date' and 'to_date' are required for date-range queries."}), 400
        return _portfolio_daterange(from_date, to_date)
    else:
        return jsonify({"status": "error", "reason": "Provide either 'date' for single date or 'from_date'+'to_date' for a range."}), 400


def _portfolio_single_date(date: str):
    """Handle single-date portfolio query (unchanged from original)."""
    try:
        raw = get_raw_data()
    except FileNotFoundError:
        return jsonify({"status": "error", "reason": "Transaction history data not found on server."}), 500

    try:
        featured = get_featured_data(raw)
    except Exception as e:
        return jsonify({"status": "error", "reason": f"Feature computation failed: {e}"}), 500

    raw_target = raw[raw["date"] == date]
    featured_target = featured[featured["date"] == date]

    if len(raw_target) == 0:
        return jsonify({"status": "ok", "date": date, "merchants": []})

    results = engine.score_portfolio_day(raw_target, featured_target, date)
    
    return jsonify({
        "status": "ok",
        "date": date,
        "merchants": [r.to_dict() for r in results]
    })


def _portfolio_daterange(from_date_str: str, to_date_str: str):
    """Handle date-range portfolio query (new feature)."""
    # Parse and validate dates
    try:
        from_date = pd.to_datetime(from_date_str).date()
        to_date = pd.to_datetime(to_date_str).date()
    except Exception:
        return jsonify({"status": "error", "reason": "Invalid date format. Use YYYY-MM-DD."}), 400

    # Validate range ordering
    if from_date > to_date:
        return jsonify({"status": "error", "reason": f"from_date ({from_date_str}) cannot be after to_date ({to_date_str})."}), 400

    # Enforce maximum range
    days_in_range = (to_date - from_date).days
    if days_in_range > 60:
        return jsonify({"status": "error", "reason": "Date range cannot exceed 60 days. Use multiple requests to analyze longer periods."}), 400

    # Load data
    try:
        raw = get_raw_data()
        featured = get_featured_data(raw)
    except FileNotFoundError:
        return jsonify({"status": "error", "reason": "Transaction history data not found on server."}), 500

    # Generate date range
    date_range = pd.date_range(from_date, to_date, freq='D').strftime('%Y-%m-%d').tolist()

    # Score each date
    date_results = []
    for target_date in date_range:
        raw_target = raw[raw["date"] == target_date]
        featured_target = featured[featured["date"] == target_date]

        if len(raw_target) == 0:
            # No data for this date, skip it
            continue

        # Call existing single-date engine
        results = engine.score_portfolio_day(raw_target, featured_target, target_date)

        # Extract aggregated data for this date
        high_count = sum(1 for r in results if r.risk_level == "HIGH")
        review_count = sum(1 for r in results if r.risk_level == "REVIEW")
        low_count = sum(1 for r in results if r.risk_level == "LOW")

        # Get top-risk merchants (only essential fields, no full signals)
        top_merchants = []
        for r in results:
            if r.status == "ok":
                top_signal = None
                if r.signals and len(r.signals) > 0:
                    top_sig = r.signals[0]
                    top_signal = {
                        "feature": top_sig.get("feature"),
                        "impact": top_sig.get("impact"),
                    }
                top_merchants.append({
                    "merchant_id": r.merchant_id,
                    "risk_score": r.risk_score,
                    "risk_level": r.risk_level,
                    "recommended_action": r.recommended_action,
                    "top_signal": top_signal,
                })

        date_results.append({
            "date": target_date,
            "total_monitored": len(results),
            "high_count": high_count,
            "review_count": review_count,
            "low_count": low_count,
            "top_merchants": top_merchants,
        })

    # Calculate summary
    summary = {
        "dates_scored": len(date_results),
        "total_merchant_days": sum(d["total_monitored"] for d in date_results),
        "high_risk_total": sum(d["high_count"] for d in date_results),
        "review_risk_total": sum(d["review_count"] for d in date_results),
        "low_risk_total": sum(d["low_count"] for d in date_results),
    }

    return jsonify({
        "status": "ok",
        "date_range": {"from": from_date_str, "to": to_date_str},
        "summary": summary,
        "dates": date_results
    })


@app.route("/api/risk/audit", methods=["GET"])
def audit():
    limit = int(request.args.get("limit", 50))
    if not os.path.exists(AUDIT_LOG_PATH):
        return jsonify({"records": []})
    with open(AUDIT_LOG_PATH) as f:
        lines = f.readlines()
    records = [json.loads(line) for line in lines[-limit:]]
    records.reverse()
    return jsonify({"records": records})


@app.route("/api/risk/merchants", methods=["GET"])
def merchants():
    raw = get_raw_data()
    summary = (
        raw.groupby("merchant_id")
        .agg(archetype=("archetype", "first"), region=("home_region", "first"),
             days=("date", "count"), spike_days=("label_fraud_spike", "sum"))
        .reset_index()
    )
    return jsonify({"merchants": summary.to_dict(orient="records")})


@app.route("/api/risk/timeseries/<merchant_id>", methods=["GET"])
def timeseries(merchant_id):
    raw = get_raw_data()
    sub = raw[raw.merchant_id == merchant_id].sort_values("date")
    if len(sub) == 0:
        return jsonify({"status": "error", "reason": f"Unknown merchant {merchant_id}"}), 404

    if engine is None:
        return jsonify({"status": "error", "reason": f"Risk engine unavailable: {ENGINE_LOAD_ERROR}"}), 503

    points = []
    for _, row in sub.tail(60).iterrows():
        result = engine.score_merchant_day(raw, row["date"], merchant_id)
        points.append({
            "date": row["date"],
            "n_transactions": int(row["n_transactions"]),
            "actual_label": int(row["label_fraud_spike"]),
            "status": result.status,
            "risk_score": result.risk_score,
            "risk_level": result.risk_level,
        })
    return jsonify({"merchant_id": merchant_id, "points": points})


@app.route("/api/risk/evaluation", methods=["GET"])
def evaluation():
    if not os.path.exists(EVAL_RESULTS_PATH):
        return jsonify({"status": "error", "reason": "No evaluation report found. Run evaluation/evaluate.py first."}), 404
    with open(EVAL_RESULTS_PATH) as f:
        return jsonify(json.load(f))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
