"""
DAY 20+: Train, Evaluate & Serialize — with Monitoring Baseline
===============================================================
Saves production_model.pkl AND training_stats.json (the drift baseline).
Run once before starting app.py.
"""

import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timezone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

SEP = "=" * 62
print(SEP)
print("  California Housing — Pipeline + Drift Baseline")
print(SEP)

# ── 1. Data ────────────────────────────────────────────────────────
data = fetch_california_housing()
X    = pd.DataFrame(data.data, columns=data.feature_names)
y    = data.target
print(f"\n[1] Loaded  {X.shape[0]:,} rows × {X.shape[1]} features")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"[2] Split   Train: {len(X_train):,}  Test: {len(X_test):,}")

# ── 2. Pipeline ────────────────────────────────────────────────────
pipeline = Pipeline([
    ('scaler',    StandardScaler()),
    ('regressor', RandomForestRegressor(
                      n_estimators=100, min_samples_leaf=2,
                      n_jobs=-1, random_state=42))
])
print("\n[3] Training …")
pipeline.fit(X_train, y_train)
print("    Done!")

# ── 3. Metrics ─────────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2     = float(r2_score(y_test, y_pred))
mae    = float(np.mean(np.abs(y_test - y_pred)))
print(f"\n[4] R²={r2:.4f}  RMSE=${rmse*1e5:,.0f}  MAE=${mae*1e5:,.0f}")

# ── 4. Training baseline snapshot ─────────────────────────────────
#  For every feature: mean/std/min/max/p5/p95
#  These are the rulers live predictions are measured against.
rf           = pipeline.named_steps['regressor']
feats        = list(X.columns)
train_preds  = pipeline.predict(X_train)

feature_stats = {}
for f in feats:
    col = X_train[f]
    feature_stats[f] = {
        "mean": round(float(col.mean()), 6),
        "std":  round(float(col.std()),  6),
        "min":  round(float(col.min()),  6),
        "max":  round(float(col.max()),  6),
        "p5":   round(float(col.quantile(0.05)), 6),
        "p95":  round(float(col.quantile(0.95)), 6),
    }

prediction_stats = {
    "mean": round(float(train_preds.mean()), 6),
    "std":  round(float(train_preds.std()),  6),
    "p5":   round(float(np.percentile(train_preds, 5)),  6),
    "p95":  round(float(np.percentile(train_preds, 95)), 6),
    "low_confidence_below": round(float(np.percentile(train_preds, 5)) * 0.5, 6),
    "low_confidence_above": round(float(np.percentile(train_preds, 95)) * 1.5, 6),
}

stats = {
    "trained_at":       datetime.now(timezone.utc).isoformat(),
    "model_version":    "1.0.0",
    "training_samples": int(len(X_train)),
    "test_samples":     int(len(X_test)),
    "model_metrics":    {"r2": round(r2,4), "rmse": round(rmse,6), "mae": round(mae,6)},
    "features":         feats,
    "feature_stats":    feature_stats,
    "prediction_stats": prediction_stats,
    "feature_importances": {
        f: round(float(i), 6)
        for f, i in zip(feats, rf.feature_importances_)
    },
    "retrain_triggers": {
        "min_r2_before_alert":       0.75,
        "max_mae_drift_pct":         0.20,
        "feature_drift_z_threshold": 3.0,
        "min_predictions_for_check": 50,
    }
}

with open("training_stats.json", "w") as fh:
    json.dump(stats, fh, indent=2)

joblib.dump(pipeline, "production_model.pkl")

print(f"\n[5] Saved  production_model.pkl")
print(f"    Saved  training_stats.json  (drift baseline)")
print(f"\n    Prediction window: p5={prediction_stats['p5']:.3f} "
      f"mean={prediction_stats['mean']:.3f} p95={prediction_stats['p95']:.3f}")
print(SEP)