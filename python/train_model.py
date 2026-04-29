import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

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
    missing_cols = [col for col in FEATURES + ["target_direction"] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in featured data: {missing_cols}")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES + ["target_direction"]).copy()
    train, test = train_test_split_time(df)

    X_train = train[FEATURES]
    y_train = train["target_direction"].astype(int)
    X_test = test[FEATURES]
    y_test = test["target_direction"].astype(int)
    y_train_close = train["target_next_close"].astype(float) if "target_next_close" in train.columns else None
    y_test_close = test["target_next_close"].astype(float) if "target_next_close" in test.columns else None

    baseline_pred = np.zeros(len(y_test), dtype=int)
    baseline_acc = accuracy_score(y_test, baseline_pred)

    logreg = LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear", random_state=42)
    logreg.fit(X_train, y_train)
    logreg_pred = logreg.predict(X_test)
    logreg_acc = accuracy_score(y_test, logreg_pred)

    linreg_pred_direction = np.zeros(len(y_test), dtype=int)
    linreg_acc = 0.0
    if y_train_close is not None and y_test_close is not None and "Close" in test.columns:
        linreg = LinearRegression()
        linreg.fit(X_train, y_train_close)
        linreg_next_close = linreg.predict(X_test)
        linreg_pred_direction = (linreg_next_close > test["Close"].astype(float).to_numpy()).astype(int)
        linreg_acc = accuracy_score(y_test, linreg_pred_direction)

    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=3,
        max_depth=12,
        class_weight="balanced_subsample",
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)

    best_model = rf if rf_acc >= logreg_acc else logreg
    best_name = "random_forest" if rf_acc >= logreg_acc else "logistic_regression"

    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    joblib.dump(FEATURES, MODELS_DIR / "feature_columns.pkl")

    baseline_precision = precision_score(y_test, baseline_pred, zero_division=0)
    baseline_recall = recall_score(y_test, baseline_pred, zero_division=0)
    baseline_f1 = f1_score(y_test, baseline_pred, zero_division=0)
    logreg_precision = precision_score(y_test, logreg_pred, zero_division=0)
    logreg_recall = recall_score(y_test, logreg_pred, zero_division=0)
    logreg_f1 = f1_score(y_test, logreg_pred, zero_division=0)
    rf_precision = precision_score(y_test, rf_pred, zero_division=0)
    rf_recall = recall_score(y_test, rf_pred, zero_division=0)
    rf_f1 = f1_score(y_test, rf_pred, zero_division=0)
    linreg_precision = precision_score(y_test, linreg_pred_direction, zero_division=0)
    linreg_recall = recall_score(y_test, linreg_pred_direction, zero_division=0)
    linreg_f1 = f1_score(y_test, linreg_pred_direction, zero_division=0)

    eval_table = pd.DataFrame(
        [
            {
                "model": "baseline_always_down",
                "accuracy": baseline_acc,
                "precision_up_class": baseline_precision,
                "recall_up_class": baseline_recall,
                "f1_up_class": baseline_f1,
            },
            {
                "model": "logistic_regression",
                "accuracy": logreg_acc,
                "precision_up_class": logreg_precision,
                "recall_up_class": logreg_recall,
                "f1_up_class": logreg_f1,
            },
            {
                "model": "random_forest",
                "accuracy": rf_acc,
                "precision_up_class": rf_precision,
                "recall_up_class": rf_recall,
                "f1_up_class": rf_f1,
            },
            {
                "model": "linear_regression",
                "accuracy": linreg_acc,
                "precision_up_class": linreg_precision,
                "recall_up_class": linreg_recall,
                "f1_up_class": linreg_f1,
            },
        ]
    )
    eval_table.to_csv(OUTPUTS_DIR / "model_scores_python.csv", index=False)

    test_out = test.copy()
    test_out["pred_logreg"] = logreg_pred
    test_out["pred_rf"] = rf_pred
    test_out["pred_linreg"] = linreg_pred_direction
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
