import json
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from typing import List, Dict, Any

app = Flask(__name__)

# ---- Paths to your saved artifacts ----
MODEL_PATH = "best_churn_model.joblib"
FEATURES_PATH = "model_features.json"

# ---- Load model ----
model = joblib.load(MODEL_PATH)

# ---- Robust feature loader: supports many JSON shapes ----
def load_feature_names(path: str) -> List[str]:
    """
    Supports these model_features.json shapes:
      1) ["age","tenure", ...]                                # list
      2) {"features": ["age","tenure", ...]}                  # dict w/ features key
      3) {"feature_names": ["age","tenure", ...]}             # dict w/ feature_names key
      4) {"columns": ["age","tenure", ...]}                   # dict w/ columns key
      5) {"feature_names_in": [...]} / {"feature_names_in_": [...]}
      6) {"age":"num","gender":"cat", ...}                    # dict of name -> type/metadata
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Case 1: already a list
    if isinstance(data, list):
        return data

    # Case 2: dict with common keys
    if isinstance(data, dict):
        for key in ["features", "feature_names", "columns", "feature_names_in", "feature_names_in_"]:
            if key in data and isinstance(data[key], list):
                return data[key]
        # Fallback: treat dict keys as feature names (e.g., {"age":"num", ...})
        return list(data.keys())

    raise ValueError("model_features.json must be a list or a dict containing a feature list")

FEATURES: List[str] = load_feature_names(FEATURES_PATH)

# ---- Helpers ----
def df_from_one_payload(x: Dict[str, Any]) -> pd.DataFrame:
    """Validate keys and build a 1-row DataFrame in the exact FEATURE order."""
    missing = [k for k in FEATURES if k not in x]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    return pd.DataFrame([[x[k] for k in FEATURES]], columns=FEATURES)

# ---- Endpoints ----
@app.get("/health")
def health():
    return jsonify(status="ok", n_features=len(FEATURES), features_example=FEATURES[:5])

@app.post("/predict")
def predict():
    """
    Accepts JSON bodies:
      - {"features": {...}}                 # single row
      - {"instances": [ {...}, {...} ]}     # multiple rows
    """
    data = request.get_json(silent=True) or {}
    try:
        if "features" in data and isinstance(data["features"], dict):
            df = df_from_one_payload(data["features"])
        elif "instances" in data and isinstance(data["instances"], list) and data["instances"]:
            rows = [df_from_one_payload(row) for row in data["instances"]]
            df = pd.concat(rows, ignore_index=True)
        else:
            return jsonify(error="Provide 'features' (dict) or 'instances' (list of dicts)."), 400

        preds = model.predict(df).tolist()
        resp = {"predictions": preds}

        # Include probabilities when available
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(df)
                # If binary classifier, return positive-class column
                if proba.ndim == 2 and proba.shape[1] == 2:
                    resp["probabilities"] = proba[:, 1].tolist()
                else:
                    resp["probabilities"] = proba.tolist()
            except Exception:
                # If predict_proba fails, just return predictions
                pass

        return jsonify(resp)
    except ValueError as ve:
        return jsonify(error=str(ve)), 400
    except Exception as e:
        return jsonify(error=f"Prediction failed: {e}"), 500

if __name__ == "__main__":
    # Dev server (ok for local testing)
    app.run(host="0.0.0.0", port=5000, debug=True)
