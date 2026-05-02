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


def load_bundle():
    path = MODELS_DIR / "unsupervised_bundle.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run python/train_model.py first.")
    return joblib.load(path)


@app.get("/health")
def health():
    alpha_key_present = bool(os.getenv("ALPHAVANTAGE_API_KEY"))
    bundle_ok = (MODELS_DIR / "unsupervised_bundle.pkl").exists()
    return jsonify(
        {
            "status": "ok",
            "alpha_vantage_key_loaded": alpha_key_present,
            "unsupervised_bundle_loaded": bundle_ok,
        }
    )


@app.post("/predict")
def predict():
    """Assign the latest feature vector to a K-Means cluster (unsupervised regime)."""
    payload = request.get_json(silent=True) or {}
    if "features" not in payload:
        return jsonify({"error": "Request must contain 'features' object"}), 400

    bundle = load_bundle()
    scaler = bundle["scaler"]
    kmeans = bundle["kmeans"]
    feature_columns = bundle["feature_columns"]

    features = payload["features"]
    row = {col: features.get(col) for col in feature_columns}

    if any(v is None for v in row.values()):
        missing = [k for k, v in row.items() if v is None]
        return jsonify({"error": f"Missing feature(s): {missing}"}), 400

    X = scaler.transform(pd.DataFrame([row]))
    cluster_id = int(kmeans.predict(X)[0])
    distances = kmeans.transform(X)[0].tolist()
    dist_to_assigned = float(distances[cluster_id])

    bullish = set(bundle.get("bullish_clusters") or [])
    signal_long = cluster_id in bullish

    out = {
        "cluster": cluster_id,
        "distance_to_own_centroid": dist_to_assigned,
        "distances_to_all_centroids": [float(d) for d in distances],
        "bullish_clusters": sorted(bullish),
        "signal_long": bool(signal_long),
    }

    pca = bundle.get("pca")
    if pca is not None:
        xy = pca.transform(X)[0].tolist()
        out["pca_2d"] = [float(xy[0]), float(xy[1])] if len(xy) >= 2 else [float(xy[0])]

    return jsonify(out)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
