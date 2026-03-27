from pathlib import Path

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"

app = Dash(__name__)
app.title = "Stock Trend Dashboard"

scores_path = OUTPUTS_DIR / "model_scores_python.csv"
preds_path = OUTPUTS_DIR / "test_predictions_python.csv"
featured_path = OUTPUTS_DIR / "featured_data.csv"

if not (scores_path.exists() and preds_path.exists() and featured_path.exists()):
    raise FileNotFoundError(
        "Missing output files. Run python/data_pipeline.py and python/train_model.py first."
    )

scores = pd.read_csv(scores_path)
preds = pd.read_csv(preds_path)
featured = pd.read_csv(featured_path)

for df in (preds, featured):
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

app.layout = html.Div(
    [
        html.H1("Stock Market Trend Prediction Dashboard"),
        html.P("Interactive overview of model quality and recent predictions."),
        dcc.Dropdown(
            id="prediction-model",
            options=[
                {"label": "Logistic Regression", "value": "pred_logreg"},
                {"label": "Random Forest", "value": "pred_rf"},
            ],
            value="pred_rf",
            clearable=False,
            style={"maxWidth": "360px"},
        ),
        dcc.Graph(id="price-chart"),
        dcc.Graph(id="scores-chart"),
        dcc.Graph(id="prediction-distribution"),
    ],
    style={"padding": "20px", "fontFamily": "Arial"},
)


@app.callback(
    Output("price-chart", "figure"),
    Output("scores-chart", "figure"),
    Output("prediction-distribution", "figure"),
    Input("prediction-model", "value"),
)
def update_charts(model_column):
    price_fig = px.line(featured, x="Date", y="Close", title="Close Price Over Time")
    score_fig = px.bar(scores, x="model", y="accuracy", title="Model Accuracy Comparison")
    pred_fig = px.histogram(
        preds,
        x=model_column,
        title=f"Prediction Distribution ({model_column})",
        labels={model_column: "Predicted Class (0=DOWN, 1=UP)"},
    )
    return price_fig, score_fig, pred_fig


if __name__ == "__main__":
    app.run(debug=True, port=8050)
