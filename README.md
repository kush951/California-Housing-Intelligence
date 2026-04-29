
#  Day 20 Report: Production ML API + Interactive UI (Homewise AI)

---

##  Objective
The goal of Day 20 was to move beyond a standalone ML model and build a **complete production system**, including:
- End-to-end ML pipeline
- Flask REST API
- Real-time integration
- Interactive frontend UI

---

##  1. ML Pipeline (Backend Intelligence)

A robust pipeline was created using `sklearn.pipeline.Pipeline`:

### 🔧 Components:
- **StandardScaler** → Feature normalization  
- **RandomForestRegressor** → Prediction model  

###  Benefits:
- Prevents data leakage  
- Ensures consistent preprocessing  
- Simplifies deployment  
- Encapsulates full ML workflow  

---

##  2. Dataset

- California Housing Dataset  
- Total Samples: **20,640**  
- Features: 8  

---

##  3. Model Performance

- **R² Score:** 0.815  
- **RMSE:** ~$52,800  
- Model captures geographic and economic patterns effectively  

---

##  4. Model Serialization

The trained pipeline was saved as:

```

production_model.pkl

```

👉 This file acts as the **core prediction engine** for the system.

---

##  5. Flask API (System Core)

The ML model was deployed using Flask as a REST API.

### 🔹 Endpoints:

- `GET /` → Health check  
- `POST /predict` → Single prediction  
- `POST /batch_predict` → Batch predictions  

###  Features:
- JSON-based communication  
- Error handling (400 responses)  
- Structured outputs  

---

##  6. Interactive UI (Homewise AI) 🔥

A modern frontend interface was built to interact with the API in real time.

###  UI Highlights:
- Elegant dark-themed design (premium look)
- Fully interactive input form (8 features)
- Real-time prediction display
- Formatted price output (e.g., **$362,120**)
- API response preview panel
- System status indicator (**Flask: Online**)

---

###  Key UI Sections:

#### 🔹 1. Pipeline Overview
- Explains the 3-step system:
  1. Train model  
  2. Serve via API  
  3. Test integration  

---

#### 🔹 2. Prediction Interface
- Users input:
  - Income  
  - Age  
  - Rooms  
  - Location  
- Click **"Run Prediction"**
- Instantly get output from backend  

---

#### 🔹 3. Result Display
- Shows:
  - Predicted price  
  - Confidence range  
  - Feature summary  
  - Raw API response  

---

#### 🔹 4. System Dashboard
- Displays:
  - Model R² score  
  - Dataset size  
  - Pipeline steps  
  - Median value insights  

---

##  7. System Architecture

```

Frontend UI (Homewise AI)
↓
Flask REST API
↓
ML Pipeline (.pkl)
↓
Prediction Response (JSON)

```

---

##  8. Integration Testing

Performed using `test_api.py`:

### ✔ Results:
- API responded with **Status 200**
- Predictions returned successfully
- Batch processing verified
- Error handling validated

---

##  9. Reflection

Saving a Pipeline object is better than saving the scaler and model separately because it ensures that preprocessing and prediction are always applied in the correct sequence.

With separate files:
- Risk of skipping transformations  
- Inconsistent predictions  

With Pipeline:
- Fully automated workflow  
- Single-file deployment  
- Prevents data leakage  
- Reliable for production systems  

---

##  10. Cross-Team Integration

This project supports all roles:

- **Backend (Node.js)** → consumes `/predict` API  
- **Data Analysts** → visualize outputs in dashboards  
- **DevOps** → deploy scalable service  

 The ML model becomes a **shared service**, not an isolated component.

---

##  11. Completion Checklist

- ✔ Built ML Pipeline  
- ✔ Serialized model (`.pkl`)  
- ✔ Developed Flask API  
- ✔ Implemented batch prediction  
- ✔ Added validation & error handling  
- ✔ Tested API via script  
- ✔ Built interactive UI  

