import json
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = "best_churn_model.joblib"
FEATURES_PATH = "model_features.json"

# Load model & feature order once at startup
model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    FEATURES = json.load(f)  # e.g. ["age","tenure","monthly_charges",...]

def df_from_payload(x: dict) -> pd.DataFrame:
    missing = [k for k in FEATURES if k not in x]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    return pd.DataFrame([[x[k] for k in FEATURES]], columns=FEATURES)

@app.get("/health")
def health():
    return jsonify(status="ok", n_features=len(FEATURES))

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    try:
        if "features" in data:
            df = df_from_payload(data["features"])
        elif "instances" in data and isinstance(data["instances"], list):
            frames = [df_from_payload(row) for row in data["instances"]]
            df = pd.concat(frames, ignore_index=True)
        else:
            return jsonify(error="Provide 'features' (dict) or 'instances' (list of dicts)."), 400

        preds = model.predict(df).tolist()
        resp = {"predictions": preds}

        # Add probabilities if the model supports it
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(df)
                if proba.ndim == 2 and proba.shape[1] == 2:
                    resp["probabilities"] = proba[:, 1].tolist()
                else:
                    resp["probabilities"] = proba.tolist()
            except Exception:
                pass

        return jsonify(resp)
    except ValueError as ve:
        return jsonify(error=str(ve)), 400
    except Exception as e:
        return jsonify(error=f"Prediction failed: {e}"), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
