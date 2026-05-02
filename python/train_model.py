import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from data_pipeline import main as run_data_pipeline


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = ROOT / "models"

FEATURES = [
    "return_1d",
    "return_5d",
    "ma_10",
    "ma_20",
    "volatility_10",
    "volume_change_1d",
]

N_CLUSTERS = 3
RANDOM_STATE = 42


def train_test_split_time(df: pd.DataFrame, split_ratio: float = 0.8):
    split_idx = int(len(df) * split_ratio)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def _bullish_clusters_from_returns(
    train: pd.DataFrame, cluster_col: str, return_col: str
) -> set[int]:
    """Pick clusters whose mean next-day return on the train split is positive."""
    means = train.groupby(cluster_col, observed=True)[return_col].mean()
    return set(int(k) for k in means.index if means[k] > 0)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    featured_path = OUTPUTS_DIR / "featured_data.csv"
    if not featured_path.exists():
        run_data_pipeline()

    df = pd.read_csv(featured_path)
    if "future_return_1d" not in df.columns:
        run_data_pipeline()
        df = pd.read_csv(featured_path)
    missing_cols = [col for col in FEATURES if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in featured data: {missing_cols}")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES).copy()

    if "future_return_1d" not in df.columns:
        raise ValueError("featured_data must include future_return_1d for evaluation exports.")

    train, test = train_test_split_time(df)

    X_train = train[FEATURES].to_numpy(dtype=float)
    X_test = test[FEATURES].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init="auto",
    )
    kmeans.fit(X_train_s)

    train_clusters = kmeans.predict(X_train_s)
    test_clusters = kmeans.predict(X_test_s)

    train = train.copy()
    test = test.copy()
    train["cluster"] = train_clusters
    test["cluster"] = test_clusters

    silhouette_train = float(silhouette_score(X_train_s, train_clusters))
    silhouette_test = float(silhouette_score(X_test_s, test_clusters))
    inertia = float(kmeans.inertia_)

    bullish = _bullish_clusters_from_returns(train, "cluster", "future_return_1d")
    if not bullish:
        means = train.groupby("cluster", observed=True)["future_return_1d"].mean()
        best = int(means.idxmax())
        bullish = {best}

    train["signal_long"] = train["cluster"].isin(bullish).astype(int)
    test["signal_long"] = test["cluster"].isin(bullish).astype(int)

    pca = PCA(n_components=min(2, X_train_s.shape[1]), random_state=RANDOM_STATE)
    pca.fit(X_train_s)

    bundle = {
        "scaler": scaler,
        "kmeans": kmeans,
        "feature_columns": FEATURES,
        "n_clusters": N_CLUSTERS,
        "pca": pca,
        "bullish_clusters": sorted(bullish),
    }
    joblib.dump(bundle, MODELS_DIR / "unsupervised_bundle.pkl")
    joblib.dump(FEATURES, MODELS_DIR / "feature_columns.pkl")

    cluster_train_means = (
        train.groupby("cluster", observed=True)["future_return_1d"].mean().to_dict()
    )
    cluster_train_counts = train["cluster"].value_counts().sort_index().to_dict()
    cluster_train_counts = {int(k): int(v) for k, v in cluster_train_counts.items()}

    metrics_rows = [
        {"metric": "silhouette_train", "value": silhouette_train},
        {"metric": "silhouette_test", "value": silhouette_test},
        {"metric": "inertia_train", "value": inertia},
        {"metric": "n_clusters", "value": float(N_CLUSTERS)},
    ]
    pd.DataFrame(metrics_rows).to_csv(OUTPUTS_DIR / "unsupervised_metrics_python.csv", index=False)

    test_out = test.copy()
    test_out.to_csv(OUTPUTS_DIR / "test_clusters_python.csv", index=False)

    summary = {
        "method": "kmeans_unsupervised",
        "n_clusters": N_CLUSTERS,
        "silhouette_train": silhouette_train,
        "silhouette_test": silhouette_test,
        "inertia": inertia,
        "bullish_clusters": sorted(bullish),
        "cluster_train_mean_future_return_1d": {str(k): float(v) for k, v in cluster_train_means.items()},
        "cluster_train_counts": {str(k): v for k, v in cluster_train_counts.items()},
        "explained_variance_ratio_pca": [float(x) for x in pca.explained_variance_ratio_.tolist()],
    }
    with open(OUTPUTS_DIR / "model_summary_python.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"Unsupervised training complete (KMeans k={N_CLUSTERS}). "
        f"Silhouette (train/test): {silhouette_train:.4f} / {silhouette_test:.4f}. "
        f"Bullish clusters (train mean return > 0): {sorted(bullish)}"
    )


if __name__ == "__main__":
    main()
