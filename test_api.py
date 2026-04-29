"""
DAY 20: Integration Test — California Housing Prediction API
============================================================
Run AFTER starting app.py in another terminal:

    python app.py           # Terminal 1
    python test_api.py      # Terminal 2
"""

import requests
import json

BASE_URL = "http://localhost:5000"

# ── Helpers ──────────────────────────────────────────────────────────────────
def section(title):
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print('─' * 55)

def pp(obj):
    print(json.dumps(obj, indent=2))

# ── 1. Health Check ───────────────────────────────────────────────────────────
section("TEST 1 — Health Check  (GET /)")
resp = requests.get(f"{BASE_URL}/")
print(f"Status: {resp.status_code}")
pp(resp.json())

# ── 2. Single Prediction ─────────────────────────────────────────────────────
section("TEST 2 — Single Prediction  (POST /predict)")

# This is the FIRST sample from the California Housing dataset
# Actual target = $4.526 (×$100k) = ~$452,600
sample_house = {
    "MedInc":      8.3252,   # High income neighborhood
    "HouseAge":    41.0,     # Older home
    "AveRooms":    6.9841,   # ~7 rooms
    "AveBedrms":   1.0238,   # ~1 bedroom per room
    "Population":  322.0,    # Small block
    "AveOccup":    2.5556,   # ~2.5 people per household
    "Latitude":    37.88,    # Bay Area (San Francisco)
    "Longitude":  -122.23
}

print("Sending payload:")
pp(sample_house)

resp = requests.post(f"{BASE_URL}/predict", json=sample_house)
print(f"\nStatus: {resp.status_code}")
print("Response:")
pp(resp.json())

# ── 3. A Cheaper House ────────────────────────────────────────────────────────
section("TEST 3 — Affordable Inland House  (POST /predict)")

cheap_house = {
    "MedInc":      2.5,      # Lower income area
    "HouseAge":    20.0,
    "AveRooms":    4.5,
    "AveBedrms":   1.1,
    "Population":  1200.0,
    "AveOccup":    3.0,
    "Latitude":    35.5,     # Central California (inland)
    "Longitude":  -119.0
}

print("Sending payload:")
pp(cheap_house)

resp = requests.post(f"{BASE_URL}/predict", json=cheap_house)
print(f"\nStatus: {resp.status_code}")
print("Response:")
pp(resp.json())

# ── 4. Batch Prediction ───────────────────────────────────────────────────────
section("TEST 4 — Batch Prediction  (POST /batch_predict)")

batch_payload = {
    "records": [
        {**sample_house},                                  # Bay Area luxury
        {**cheap_house},                                   # Inland affordable
        {   # Mid-range suburban
            "MedInc": 4.5, "HouseAge": 15.0,
            "AveRooms": 5.5, "AveBedrms": 1.05,
            "Population": 800.0, "AveOccup": 2.8,
            "Latitude": 34.05, "Longitude": -118.25      # Los Angeles area
        }
    ]
}

resp = requests.post(f"{BASE_URL}/batch_predict", json=batch_payload)
print(f"Status: {resp.status_code}")
print("Response:")
pp(resp.json())

# ── 5. Error Handling Test ────────────────────────────────────────────────────
section("TEST 5 — Missing Feature Error Handling  (POST /predict)")

bad_payload = {"MedInc": 5.0, "HouseAge": 30.0}  # Only 2 of 8 features
resp = requests.post(f"{BASE_URL}/predict", json=bad_payload)
print(f"Status: {resp.status_code}  (expected 400)")
pp(resp.json())

# ── Summary ───────────────────────────────────────────────────────────────────
section("ALL TESTS COMPLETE ✓")
print("""
WHY SAVE A PIPELINE INSTEAD OF SEPARATE FILES?
───────────────────────────────────────────────
❌  Two-file approach (bad):
    scaler.pkl + model.pkl
    → You must remember to call scaler.transform() BEFORE model.predict()
    → Easy to forget in production → wrong predictions silently
    → The scaler fit on training data must be manually managed

✅  Pipeline approach (correct):
    production_model.pkl  (contains BOTH scaler + model)
    → pipeline.predict(X) does scaling + predicting atomically
    → Impossible to skip the scaling step
    → One file to version-control, deploy, and share
    → Prevents DATA LEAKAGE by design
""")
