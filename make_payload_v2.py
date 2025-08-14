# make_payload_v2.py
# Usage:
#   python make_payload_v2.py
#
# Reads model_features.json and creates:
#   - sample_single_payload.json
#   - sample_batch_payload.json
#
# Supports any of these shapes:
#   1) ["age","tenure", ...]
#   2) {"features": ["age","tenure", ...]}
#   3) {"feature_names": ["age","tenure", ...]}
#   4) {"columns": ["age","tenure", ...]}
#   5) {"feature_names_in": ["age","tenure", ...]} or {"feature_names_in_":[...]}
#   6) {"age":"num","gender":"cat", ...}  # dict of name->type
import json
from collections.abc import Mapping, Sequence

CANDIDATE_KEYS = ["features", "feature_names", "columns", "feature_names_in", "feature_names_in_"]

def load_features(path="model_features.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1) Already a list/tuple of names
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        return list(data)

    # 2) Dict with one of the well-known keys
    if isinstance(data, Mapping):
        for key in CANDIDATE_KEYS:
            if key in data and isinstance(data[key], Sequence):
                return list(data[key])
        # 3) Dict of feature->type or metadata
        return list(data.keys())

    raise ValueError("Unrecognized model_features.json format")

def build_single_payload(feature_names):
    return {"features": {str(name): 0 for name in feature_names}}

def build_batch_payload(feature_names):
    row1 = {str(name): 0 for name in feature_names}
    row2 = {str(name): 1 for name in feature_names}
    return {"instances": [row1, row2]}

def main():
    features = load_features()
    print(f"Detected {len(features)} features.")
    with open("sample_single_payload.json", "w", encoding="utf-8") as f:
        json.dump(build_single_payload(features), f, indent=2, ensure_ascii=False)
    with open("sample_batch_payload.json", "w", encoding="utf-8") as f:
        json.dump(build_batch_payload(features), f, indent=2, ensure_ascii=False)
    print("Wrote sample_single_payload.json and sample_batch_payload.json")

if __name__ == "__main__":
    main()
