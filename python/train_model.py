import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

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


def train_test_split_time(df: pd.DataFrame, split_ratio: float = 0.8):
    split_idx = int(len(df) * split_ratio)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    featured_path = OUTPUTS_DIR / "featured_data.csv"
    if not featured_path.exists():
        run_data_pipeline()

    df = pd.read_csv(featured_path)
    train, test = train_test_split_time(df)

    X_train = train[FEATURES]
    y_train = train["target_direction"].astype(int)
    X_test = test[FEATURES]
    y_test = test["target_direction"].astype(int)

    baseline_pred = np.zeros(len(y_test), dtype=int)
    baseline_acc = accuracy_score(y_test, baseline_pred)

    logreg = LogisticRegression(max_iter=200)
    logreg.fit(X_train, y_train)
    logreg_pred = logreg.predict(X_test)
    logreg_acc = accuracy_score(y_test, logreg_pred)

    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)

    best_model = rf if rf_acc >= logreg_acc else logreg
    best_name = "random_forest" if rf_acc >= logreg_acc else "logistic_regression"

    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    joblib.dump(FEATURES, MODELS_DIR / "feature_columns.pkl")

    eval_table = pd.DataFrame(
        [
            {"model": "baseline_always_down", "accuracy": baseline_acc},
            {"model": "logistic_regression", "accuracy": logreg_acc},
            {"model": "random_forest", "accuracy": rf_acc},
        ]
    )
    eval_table.to_csv(OUTPUTS_DIR / "model_scores_python.csv", index=False)

    test_out = test.copy()
    test_out["pred_logreg"] = logreg_pred
    test_out["pred_rf"] = rf_pred
    test_out.to_csv(OUTPUTS_DIR / "test_predictions_python.csv", index=False)

    summary = {
        "best_model": best_name,
        "scores": {
            "baseline": baseline_acc,
            "logistic_regression": logreg_acc,
            "random_forest": rf_acc,
        },
        "classification_report_best": classification_report(
            y_test,
            rf_pred if best_name == "random_forest" else logreg_pred,
            output_dict=True,
        ),
    }
    with open(OUTPUTS_DIR / "model_summary_python.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Training complete. Best model: {best_name}")


if __name__ == "__main__":
    main()
