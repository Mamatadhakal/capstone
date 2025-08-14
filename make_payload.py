# make_payload.py
# Usage:
#   python make_payload.py
#
# This reads model_features.json in the current folder and creates:
#   - sample_single_payload.json (one row with placeholders)
#   - sample_batch_payload.json (two rows with placeholders)
#
# It supports these model_features.json formats:
#   1) ["age","tenure", ...]                  # simple list of feature names
#   2) {"features": ["age","tenure", ...]}    # object with a 'features' array
#   3) {"age":"num","gender":"cat", ...}      # dict of feature -> type (num/cat)
#
# After generation, edit any categorical fields in sample_single_payload.json
# and replace 0 with strings your model expects (e.g., "Male", "Yes").
import json
from collections.abc import Mapping, Sequence

def load_features(path="model_features.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Case 1: It's already a list/tuple of names
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        return list(data)

    # Case 2: Dict with "features" key
    if isinstance(data, Mapping):
        if "features" in data and isinstance(data["features"], Sequence):
            return list(data["features"])
        # Case 3: Dict of feature->type or feature->metadata
        return list(data.keys())

    raise ValueError("Unrecognized model_features.json format")

def build_single_payload(feature_names):
    return {"features": {name: 0 for name in feature_names}}

def build_batch_payload(feature_names):
    row1 = {name: 0 for name in feature_names}
    row2 = {name: 1 for name in feature_names}
    return {"instances": [row1, row2]}

def main():
    features = load_features()
    print(f"Detected {len(features)} features.")
    with open("sample_single_payload.json", "w", encoding="utf-8") as f:
        json.dump(build_single_payload(features), f, indent=2, ensure_ascii=False)
    with open("sample_batch_payload.json", "w", encoding="utf-8") as f:
        json.dump(build_batch_payload(features), f, indent=2, ensure_ascii=False)
    print("Wrote sample_single_payload.json and sample_batch_payload.json")
    print("\nNext steps:")
    print(" 1) Open sample_single_payload.json and replace 0 with strings for categorical fields (e.g., \"Male\", \"Yes\").")
    print(" 2) Send it to your running server:")
    print("    PowerShell:")
    print("      $body = Get-Content .\\sample_single_payload.json -Raw")
    print("      Invoke-RestMethod -Uri http://localhost:5000/predict -Method Post -ContentType 'application/json' -Body $body | ConvertTo-Json -Depth 5")

if __name__ == '__main__':
    main()
