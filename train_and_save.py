"""
DAY 20: Train & Serialize the California Housing Pipeline
=========================================================
Builds a robust sklearn Pipeline (Scaler + RandomForest),
trains it, evaluates it, and saves it as production_model.pkl
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ── 1. LOAD DATA ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  California Housing — End-to-End Pipeline (Day 20)")
print("=" * 60)

data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target   # Median house value in $100,000s

print(f"\n[1] Dataset loaded  →  {X.shape[0]:,} rows × {X.shape[1]} features")
print(f"    Features : {list(X.columns)}")
print(f"    Target   : Median House Value (×$100k)")

# ── 2. TRAIN / TEST SPLIT ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n[2] Split  →  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── 3. BUILD PIPELINE ────────────────────────────────────────────────────────
#    StandardScaler  →  RandomForestRegressor
#    One object = scaler + model, no leakage possible
pipeline = Pipeline([
    ('scaler',     StandardScaler()),
    ('regressor',  RandomForestRegressor(
                        n_estimators=100,
                        max_depth=None,
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=42))
])
print("\n[3] Pipeline architecture:")
for name, step in pipeline.steps:
    print(f"    └─ {name:12s} → {step.__class__.__name__}")

# ── 4. TRAIN ─────────────────────────────────────────────────────────────────
print("\n[4] Training …  (this may take ~10 s)")
pipeline.fit(X_train, y_train)
print("    ✓ Training complete!")

# ── 5. EVALUATE ──────────────────────────────────────────────────────────────
y_pred   = pipeline.predict(X_test)
rmse     = np.sqrt(mean_squared_error(y_test, y_pred))
r2       = r2_score(y_test, y_pred)
mae      = np.mean(np.abs(y_test - y_pred))

print(f"\n[5] Evaluation on hold-out test set:")
print(f"    R²   Score : {r2:.4f}   (1.0 = perfect)")
print(f"    RMSE       : ${rmse * 100_000:,.0f}")
print(f"    MAE        : ${mae  * 100_000:,.0f}")

# Feature importances (from the RF inside the pipeline)
rf       = pipeline.named_steps['regressor']
fi_df    = pd.Series(rf.feature_importances_, index=X.columns) \
             .sort_values(ascending=False)
print("\n    Feature Importances (top 8):")
for feat, imp in fi_df.items():
    bar = "█" * int(imp * 60)
    print(f"    {feat:12s} {bar} {imp:.4f}")

# ── 6. SAVE ──────────────────────────────────────────────────────────────────
joblib.dump(pipeline, 'production_model.pkl')
print("\n[6] ✓ Pipeline saved  →  production_model.pkl")
print("    Load anywhere with:  pipeline = joblib.load('production_model.pkl')")
print("=" * 60)
