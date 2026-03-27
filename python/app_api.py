import os
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

load_dotenv(ROOT / ".env")

app = Flask(__name__)


def load_artifacts():
    model = joblib.load(MODELS_DIR / "best_model.pkl")
    features = joblib.load(MODELS_DIR / "feature_columns.pkl")
    return model, features


@app.get("/health")
def health():
    alpha_key_present = bool(os.getenv("ALPHAVANTAGE_API_KEY"))
    return jsonify({"status": "ok", "alpha_vantage_key_loaded": alpha_key_present})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    if "features" not in payload:
        return jsonify({"error": "Request must contain 'features' object"}), 400

    model, feature_columns = load_artifacts()
    features = payload["features"]
    row = {col: features.get(col) for col in feature_columns}

    if any(v is None for v in row.values()):
        missing = [k for k, v in row.items() if v is None]
        return jsonify({"error": f"Missing feature(s): {missing}"}), 400

    X = pd.DataFrame([row])
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else None

    return jsonify(
        {
            "prediction_direction": pred,
            "prediction_label": "UP" if pred == 1 else "DOWN",
            "probability_up": proba,
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
