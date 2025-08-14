# Churn Model Flask API

A tiny Flask API that serves your trained churn model saved as `best_churn_model.joblib`.
It expects the feature order listed in `model_features.json` (a JSON array).

## Run locally

### 1) Create & activate a virtual environment
```bash
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Start the API
```bash
python app.py
```

Visit http://localhost:5000/health

## Endpoints
- `GET /health` – quick status
- `POST /predict`
  - Single row: `{"features": { ... }}`
  - Batch rows:  `{"instances": [ {...}, {...} ]}`

## Notes
- Make sure `model_features.json` has *every* feature name, in the exact order your model expects.
- If your model uses preprocessing (one-hot, scaling), save it inside the pipeline before exporting the joblib.
