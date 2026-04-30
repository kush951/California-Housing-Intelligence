"""
monitor.py — Production Drift Detection & Monitoring Engine
===========================================================
Baked in, not bolted on.

Responsibilities:
  1. Log every prediction to a rolling JSON-Lines file
  2. Compute per-feature drift (z-score vs training baseline)
  3. Track prediction distribution shift over time
  4. Emit retraining alerts when thresholds are breached
  5. Expose a /monitoring/report endpoint via Flask

Usage:
  from monitor import DriftMonitor
  mon = DriftMonitor("training_stats.json", "prediction_log.jsonl")
  alert = mon.record(features_dict, prediction_100k)
"""

import json
import os
import math
import threading
from datetime import datetime, timezone
from collections import deque
from typing import Optional


class DriftMonitor:
    """
    Thread-safe, zero-dependency drift monitor.

    Records each prediction, maintains a sliding window, and
    compares live feature distributions against the training baseline.
    """

    def __init__(
        self,
        stats_path: str = "training_stats.json",
        log_path:   str = "prediction_log.jsonl",
        window:     int = 200,          # rolling window for drift calculations
    ):
        self.log_path = log_path
        self.window   = window
        self._lock    = threading.Lock()

        # ── Load training baseline ────────────────────────────────
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"'{stats_path}' not found — run train_and_save.py first."
            )
        with open(stats_path) as f:
            self.baseline = json.load(f)

        self.features        = self.baseline["features"]
        self.feature_stats   = self.baseline["feature_stats"]
        self.prediction_stats = self.baseline["prediction_stats"]
        self.triggers        = self.baseline["retrain_triggers"]
        self.model_metrics   = self.baseline["model_metrics"]

        # ── Rolling in-memory window (deque caps automatically) ───
        self._window_features    = deque(maxlen=window)   # list of feature dicts
        self._window_predictions = deque(maxlen=window)   # list of prediction floats
        self._total_predictions  = 0
        self._total_alerts       = 0

        # Replay existing log into the window so a restart doesn't lose state
        self._replay_log()

    # ── Public API ────────────────────────────────────────────────

    def record(self, features: dict, prediction_100k: float) -> dict:
        """
        Call this on every prediction request.
        Returns an alert dict — callers attach it to the API response.

        alert = {
            "drift_detected": bool,
            "alerts": [...],          # human-readable issues
            "confidence": "high"|"medium"|"low",
            "feature_drift": {feat: z_score, ...},
            "retrain_recommended": bool,
        }
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        alert     = self._analyse(features, prediction_100k)

        record = {
            "ts":             timestamp,
            "features":       {f: features.get(f) for f in self.features},
            "prediction":     round(prediction_100k, 6),
            "prediction_usd": round(prediction_100k * 100_000, 2),
            "alert":          alert,
        }

        with self._lock:
            self._window_features.append(record["features"])
            self._window_predictions.append(prediction_100k)
            self._total_predictions += 1
            if alert["drift_detected"]:
                self._total_alerts += 1
            self._append_log(record)

        return alert

    def report(self) -> dict:
        """
        Full monitoring report — served by GET /monitoring/report.
        """
        with self._lock:
            n     = len(self._window_predictions)
            preds = list(self._window_predictions)
            feats = list(self._window_features)

        if n == 0:
            return {
                "status":             "no_data",
                "total_predictions":  self._total_predictions,
                "message":            "No predictions recorded yet.",
            }

        # ── Window prediction stats ───────────────────────────────
        pred_mean = sum(preds) / n
        pred_std  = math.sqrt(sum((p - pred_mean)**2 for p in preds) / max(n-1, 1))
        pred_min  = min(preds)
        pred_max  = max(preds)

        # ── Per-feature drift ─────────────────────────────────────
        feature_drift = {}
        for feat in self.features:
            vals = [row[feat] for row in feats if row.get(feat) is not None]
            if not vals:
                continue
            live_mean = sum(vals) / len(vals)
            baseline_mean = self.feature_stats[feat]["mean"]
            baseline_std  = self.feature_stats[feat]["std"]
            z = (live_mean - baseline_mean) / max(baseline_std, 1e-9)
            live_std = math.sqrt(sum((v - live_mean)**2 for v in vals) / max(len(vals)-1, 1))
            feature_drift[feat] = {
                "z_score":        round(z, 3),
                "live_mean":      round(live_mean, 4),
                "baseline_mean":  round(baseline_mean, 4),
                "live_std":       round(live_std, 4),
                "baseline_std":   round(baseline_std, 4),
                "drift_flag":     abs(z) > self.triggers["feature_drift_z_threshold"],
            }

        # ── Prediction distribution shift ─────────────────────────
        base_mean = self.prediction_stats["mean"]
        base_std  = self.prediction_stats["std"]
        pred_z    = (pred_mean - base_mean) / max(base_std, 1e-9)

        # ── MAE proxy on window (we don't have ground truth,
        #    so we use deviation from training mean as a proxy signal) ──
        mae_proxy = sum(abs(p - base_mean) for p in preds) / n
        mae_drift_pct = (mae_proxy - self.model_metrics["mae"]) / max(self.model_metrics["mae"], 1e-9)

        # ── Build alerts list ─────────────────────────────────────
        alerts = []
        drifted_features = [f for f, d in feature_drift.items() if d["drift_flag"]]

        if drifted_features:
            alerts.append({
                "type":     "FEATURE_DRIFT",
                "severity": "HIGH" if len(drifted_features) >= 3 else "MEDIUM",
                "message":  f"Feature distribution drift detected: {', '.join(drifted_features)}",
                "features": drifted_features,
            })

        if abs(pred_z) > 2.0:
            direction = "HIGHER" if pred_z > 0 else "LOWER"
            alerts.append({
                "type":     "PREDICTION_SHIFT",
                "severity": "HIGH" if abs(pred_z) > 3.0 else "MEDIUM",
                "message":  f"Live predictions trending {direction} than training baseline "
                            f"(z={pred_z:+.2f})",
                "pred_z":   round(pred_z, 3),
            })

        if mae_drift_pct > self.triggers["max_mae_drift_pct"]:
            alerts.append({
                "type":     "ERROR_DRIFT",
                "severity": "HIGH",
                "message":  f"Prediction error proxy drifted {mae_drift_pct*100:.1f}% "
                            f"above training baseline (threshold: "
                            f"{self.triggers['max_mae_drift_pct']*100:.0f}%)",
                "drift_pct": round(mae_drift_pct * 100, 2),
            })

        retrain = (
            n >= self.triggers["min_predictions_for_check"]
            and (
                len(drifted_features) >= 2
                or abs(pred_z) > 3.0
                or mae_drift_pct > self.triggers["max_mae_drift_pct"]
            )
        )

        # ── Health summary ────────────────────────────────────────
        if not alerts:
            health = "healthy"
        elif any(a["severity"] == "HIGH" for a in alerts):
            health = "critical"
        else:
            health = "warning"

        return {
            "status":                health,
            "generated_at":          datetime.now(timezone.utc).isoformat(),
            "model_version":         self.baseline.get("model_version", "unknown"),
            "trained_at":            self.baseline.get("trained_at", "unknown"),
            "total_predictions":     self._total_predictions,
            "total_alerts_fired":    self._total_alerts,
            "window_size":           n,
            "window_limit":          self.window,
            "retrain_recommended":   retrain,
            "alerts":                alerts,
            "prediction_distribution": {
                "live_mean":      round(pred_mean, 4),
                "live_std":       round(pred_std,  4),
                "live_min":       round(pred_min,  4),
                "live_max":       round(pred_max,  4),
                "baseline_mean":  round(base_mean, 4),
                "baseline_std":   round(base_std,  4),
                "z_score":        round(pred_z,    3),
            },
            "feature_drift":       feature_drift,
            "baseline_metrics":    self.model_metrics,
            "retrain_triggers":    self.triggers,
        }

    # ── Private helpers ───────────────────────────────────────────

    def _analyse(self, features: dict, prediction_100k: float) -> dict:
        """Per-request lightweight drift check (runs synchronously)."""
        feature_drift = {}
        alerts        = []

        for feat in self.features:
            val = features.get(feat)
            if val is None:
                continue
            mean = self.feature_stats[feat]["mean"]
            std  = self.feature_stats[feat]["std"]
            p5   = self.feature_stats[feat]["p5"]
            p95  = self.feature_stats[feat]["p95"]
            z    = (val - mean) / max(std, 1e-9)
            feature_drift[feat] = round(z, 3)

            if abs(z) > self.triggers["feature_drift_z_threshold"]:
                alerts.append(f"{feat} z={z:+.2f} (>{self.triggers['feature_drift_z_threshold']:.0f}σ from training mean)")
            elif val < p5 or val > p95:
                alerts.append(f"{feat}={val:.3g} outside training p5–p95 [{p5:.3g}, {p95:.3g}]")

        # Prediction confidence band
        lo  = self.prediction_stats["low_confidence_below"]
        hi  = self.prediction_stats["low_confidence_above"]
        p5  = self.prediction_stats["p5"]
        p95 = self.prediction_stats["p95"]

        if prediction_100k < lo or prediction_100k > hi:
            confidence = "low"
            alerts.append(f"Prediction {prediction_100k:.3f} is outside low-confidence bounds [{lo:.3f}, {hi:.3f}]")
        elif prediction_100k < p5 or prediction_100k > p95:
            confidence = "medium"
        else:
            confidence = "high"

        return {
            "drift_detected":     len(alerts) > 0,
            "alerts":             alerts,
            "confidence":         confidence,
            "feature_drift":      feature_drift,
            "retrain_recommended": len(alerts) >= 3,
            "confidence_band": {
                "low_below":  round(lo,  4),
                "low_above":  round(hi,  4),
                "p5":         round(p5,  4),
                "p95":        round(p95, 4),
            },
        }

    def _append_log(self, record: dict):
        """Append one JSON-Lines record to the prediction log."""
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except IOError:
            pass   # never crash the API because of log IO

    def _replay_log(self):
        """Warm up the rolling window from the existing log file on startup."""
        if not os.path.exists(self.log_path):
            return
        try:
            with open(self.log_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self._window_features.append(rec.get("features", {}))
                    self._window_predictions.append(rec.get("prediction", 0))
                    self._total_predictions += 1
                    if rec.get("alert", {}).get("drift_detected"):
                        self._total_alerts += 1
        except (IOError, json.JSONDecodeError):
            pass   # corrupted log → start fresh
