"""
DAY 20: Flask REST API — California Housing Price Predictor
===========================================================
Run after train_and_save.py has created production_model.pkl

    python app.py

Then POST to  http://localhost:5000/predict
"""

from flask import Flask, request, jsonify
from flask_cors import CORS          # ← allows browser (index.html) to call the API
import joblib
import pandas as pd
import numpy as np
import os

# ── App Setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # permits cross-origin requests from index.html opened in a browser

MODEL_PATH = 'production_model.pkl'

# Guard: fail fast if model file is missing
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"'{MODEL_PATH}' not found. Run train_and_save.py first!"
    )

# Load once at startup — not on every request
pipeline = joblib.load(MODEL_PATH)
print(f"[✓] Model loaded from '{MODEL_PATH}'")

# California Housing expected feature names (same order as training)
FEATURES = [
    'MedInc',       # Median income in block group
    'HouseAge',     # Median house age
    'AveRooms',     # Average number of rooms per household
    'AveBedrms',    # Average number of bedrooms per household
    'Population',   # Block group population
    'AveOccup',     # Average number of household members
    'Latitude',     # Block group latitude
    'Longitude',    # Block group longitude
]

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def home():
    """Health-check endpoint."""
    return jsonify({
        'service':  'California Housing Price Predictor',
        'status':   'running',
        'version':  '1.0.0',
        'endpoint': 'POST /predict',
        'features': FEATURES
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts a JSON body with all 8 features and returns a price prediction.

    Example body:
    {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984,
        "AveBedrms": 1.023,
        "Population": 322.0,
        "AveOccup": 2.555,
        "Latitude": 37.88,
        "Longitude": -122.23
    }
    """
    # ── Validate input ───────────────────────────────────────────────────────
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({'error': 'Request body must be valid JSON'}), 400

    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({
            'error':    'Missing required features',
            'missing':  missing,
            'required': FEATURES
        }), 400

    # ── Build DataFrame in correct column order ──────────────────────────────
    try:
        query_df = pd.DataFrame([{f: data[f] for f in FEATURES}])
    except Exception as e:
        return jsonify({'error': f'Data conversion failed: {str(e)}'}), 400

    # ── Run Pipeline (scales + predicts automatically) ───────────────────────
    try:
        raw_pred = pipeline.predict(query_df)          # in $100k units
        price_usd = float(raw_pred[0]) * 100_000       # convert to USD
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    # ── Return structured response ───────────────────────────────────────────
    return jsonify({
        'prediction_100k':   round(float(raw_pred[0]), 4),
        'prediction_usd':    round(price_usd, 2),
        'prediction_label':  f"${price_usd:,.0f}",
        'input_features':    data,
        'model':             'RandomForestRegressor + StandardScaler Pipeline'
    })


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Accepts a list of records and returns predictions for all of them.

    Example body:
    {
        "records": [
            {"MedInc": 8.3, "HouseAge": 41, ...},
            {"MedInc": 3.1, "HouseAge": 25, ...}
        ]
    }
    """
    data = request.get_json(silent=True)
    if not data or 'records' not in data:
        return jsonify({'error': "Body must have a 'records' key with a list"}), 400

    records = data['records']
    if not isinstance(records, list) or len(records) == 0:
        return jsonify({'error': "'records' must be a non-empty list"}), 400

    try:
        df    = pd.DataFrame(records)[FEATURES]          # ensure correct order
        preds = pipeline.predict(df)
        results = [
            {'record_index': i,
             'prediction_100k': round(float(p), 4),
             'prediction_usd':  round(float(p) * 100_000, 2)}
            for i, p in enumerate(preds)
        ]
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'predictions': results, 'count': len(results)})


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Starting California Housing Price Predictor API...")
    print("Endpoints:  GET /  |  POST /predict  |  POST /batch_predict")
    app.run(host='0.0.0.0', port=5000, debug=True)
