"""
app.py — Monitored Flask API for California Housing Predictor
=============================================================
Every prediction is:
  1. Validated
  2. Run through pipeline.predict() (StandardScaler + RandomForest)
  3. Checked for feature drift and confidence bands by DriftMonitor
  4. Logged to prediction_log.jsonl
  5. Returned with full monitoring metadata

Endpoints:
  GET  /                     health check + model info
  POST /predict              single prediction
  POST /batch_predict        bulk predictions
  GET  /monitoring/report    full drift report + retrain status
  POST /monitoring/reset     clear prediction log (careful!)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

from monitor import DriftMonitor

# ── App Setup ──────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

MODEL_PATH = "production_model.pkl"
STATS_PATH = "training_stats.json"
LOG_PATH   = "prediction_log.jsonl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"'{MODEL_PATH}' not found — run train_and_save.py first.")
if not os.path.exists(STATS_PATH):
    raise FileNotFoundError(f"'{STATS_PATH}' not found — run train_and_save.py first.")

pipeline = joblib.load(MODEL_PATH)
monitor  = DriftMonitor(STATS_PATH, LOG_PATH, window=200)

FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]

print("[✓] production_model.pkl loaded")
print("[✓] DriftMonitor initialised  "
      f"(replayed {monitor._total_predictions} existing predictions)")


# ── Health ────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    rpt = monitor.report()
    return jsonify({
        "service":          "California Housing Price Predictor",
        "status":           "running",
        "model_version":    monitor.baseline.get("model_version"),
        "trained_at":       monitor.baseline.get("trained_at"),
        "monitoring":       rpt.get("status", "unknown"),
        "total_predictions": monitor._total_predictions,
        "retrain_recommended": rpt.get("retrain_recommended", False),
        "features":         FEATURES,
    })


# ── Single prediction ─────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": "Missing features", "missing": missing}), 400

    try:
        query_df = pd.DataFrame([{f: data[f] for f in FEATURES}])
    except Exception as e:
        return jsonify({"error": f"Data conversion failed: {e}"}), 400

    try:
        raw_pred  = pipeline.predict(query_df)
        pred_100k = float(raw_pred[0])
        price_usd = pred_100k * 100_000
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    # ── Run drift check & log ─────────────────────────────────────
    alert = monitor.record(data, pred_100k)

    return jsonify({
        # Core prediction
        "prediction_100k":  round(pred_100k, 4),
        "prediction_usd":   round(price_usd, 2),
        "prediction_label": f"${price_usd:,.0f}",

        # Confidence band from monitor
        "confidence":       alert["confidence"],
        "confidence_band": {
            "low_below_usd":  round(alert["confidence_band"]["low_below"] * 100_000, 2),
            "p5_usd":         round(alert["confidence_band"]["p5"]        * 100_000, 2),
            "p95_usd":        round(alert["confidence_band"]["p95"]       * 100_000, 2),
            "low_above_usd":  round(alert["confidence_band"]["low_above"] * 100_000, 2),
        },

        # Drift metadata
        "monitoring": {
            "drift_detected":      alert["drift_detected"],
            "alerts":              alert["alerts"],
            "feature_drift":       alert["feature_drift"],
            "retrain_recommended": alert["retrain_recommended"],
        },

        "model": "RandomForestRegressor + StandardScaler Pipeline",
    })


# ── Batch prediction ──────────────────────────────────────────────
@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    data = request.get_json(silent=True)
    if not data or "records" not in data:
        return jsonify({"error": "Body must have a 'records' key"}), 400

    records = data["records"]
    if not isinstance(records, list) or len(records) == 0:
        return jsonify({"error": "'records' must be a non-empty list"}), 400

    try:
        df    = pd.DataFrame(records)[FEATURES]
        preds = pipeline.predict(df)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    results = []
    for i, (row, p) in enumerate(zip(records, preds)):
        alert = monitor.record(row, float(p))
        results.append({
            "record_index":    i,
            "prediction_100k": round(float(p), 4),
            "prediction_usd":  round(float(p) * 100_000, 2),
            "confidence":      alert["confidence"],
            "drift_detected":  alert["drift_detected"],
            "alerts":          alert["alerts"],
        })

    return jsonify({
        "predictions":       results,
        "count":             len(results),
        "drift_any":         any(r["drift_detected"] for r in results),
        "low_confidence_any": any(r["confidence"] == "low" for r in results),
    })


# ── Monitoring report ─────────────────────────────────────────────
@app.route("/monitoring/report", methods=["GET"])
def monitoring_report():
    """
    Full drift report over the rolling window.
    Returns feature z-scores, prediction distribution shift,
    and retrain_recommended flag.
    """
    return jsonify(monitor.report())


# ── Reset log (dev/testing only) ──────────────────────────────────
@app.route("/monitoring/reset", methods=["POST"])
def monitoring_reset():
    """Clear prediction log and reset rolling window. Use with care."""
    try:
        if os.path.exists(LOG_PATH):
            os.remove(LOG_PATH)
        monitor._window_features.clear()
        monitor._window_predictions.clear()
        monitor._total_predictions = 0
        monitor._total_alerts      = 0
        return jsonify({"status": "ok", "message": "Prediction log cleared."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Run ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nEndpoints:")
    print("  GET  /                   health + model info")
    print("  POST /predict            single prediction + drift check")
    print("  POST /batch_predict      bulk predictions + drift check")
    print("  GET  /monitoring/report  full drift report")
    print("  POST /monitoring/reset   clear log (dev only)\n")
    app.run(host="0.0.0.0", port=5000, debug=True)