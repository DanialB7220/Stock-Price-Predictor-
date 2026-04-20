from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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


def _format_pct(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def _downsample_price_frame(df: pd.DataFrame, lookback_days: int, granularity: str) -> pd.DataFrame:
    end_date = df["Date"].max()
    cutoff = end_date - pd.Timedelta(days=lookback_days)
    filtered = df.loc[df["Date"] >= cutoff, ["Date", "Close"]].copy()
    if granularity == "weekly":
        return (
            filtered.set_index("Date")
            .resample("W")
            .last()
            .dropna()
            .reset_index()
        )
    if granularity == "monthly":
        return (
            filtered.set_index("Date")
            .resample("ME")
            .last()
            .dropna()
            .reset_index()
        )
    return filtered


latest_close = float(featured["Close"].iloc[-1]) if len(featured) else float("nan")
close_5d_ago = float(featured["Close"].iloc[-6]) if len(featured) > 5 else float("nan")
return_5d = (latest_close / close_5d_ago - 1) if pd.notna(close_5d_ago) and close_5d_ago != 0 else float("nan")
best_model = scores.sort_values("accuracy", ascending=False).iloc[0]["model"] if len(scores) else "N/A"
best_accuracy = float(scores.sort_values("accuracy", ascending=False).iloc[0]["accuracy"]) if len(scores) else float("nan")

app.layout = html.Div(
    [
        html.H1("Stock Market Trend Prediction Dashboard"),
        html.P("Optimized analytics view for model quality, signals, and price action."),
        html.Div(
            [
                html.Div(
                    [
                        html.H4("Latest Close", style={"marginBottom": "4px"}),
                        html.H2(f"${latest_close:.2f}" if pd.notna(latest_close) else "N/A", style={"marginTop": "0px"}),
                    ],
                    style={"padding": "12px", "background": "#f5f7fb", "borderRadius": "10px", "minWidth": "220px"},
                ),
                html.Div(
                    [
                        html.H4("5-Day Return", style={"marginBottom": "4px"}),
                        html.H2(_format_pct(return_5d), style={"marginTop": "0px"}),
                    ],
                    style={"padding": "12px", "background": "#f5f7fb", "borderRadius": "10px", "minWidth": "220px"},
                ),
                html.Div(
                    [
                        html.H4("Best Model", style={"marginBottom": "4px"}),
                        html.H2(str(best_model), style={"marginTop": "0px", "textTransform": "capitalize"}),
                        html.P(f"Accuracy: {_format_pct(best_accuracy)}", style={"margin": "0px"}),
                    ],
                    style={"padding": "12px", "background": "#f5f7fb", "borderRadius": "10px", "minWidth": "220px"},
                ),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "16px"},
        ),
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
        html.Div(
            [
                html.Label("Price Lookback Window (days)", style={"fontWeight": "bold"}),
                dcc.Slider(id="lookback-days", min=180, max=2500, step=30, value=900),
            ],
            style={"maxWidth": "640px", "marginTop": "14px"},
        ),
        html.Div(
            [
                html.Label("Price Aggregation", style={"fontWeight": "bold"}),
                dcc.RadioItems(
                    id="price-granularity",
                    options=[
                        {"label": "Daily", "value": "daily"},
                        {"label": "Weekly", "value": "weekly"},
                        {"label": "Monthly", "value": "monthly"},
                    ],
                    value="daily",
                    inline=True,
                ),
            ],
            style={"marginTop": "10px", "marginBottom": "4px"},
        ),
        dcc.Graph(id="price-chart"),
        dcc.Graph(id="scores-chart"),
        dcc.Graph(id="strategy-chart"),
        dcc.Graph(id="prediction-distribution"),
    ],
    style={"padding": "20px", "fontFamily": "Arial"},
)


@app.callback(
    Output("price-chart", "figure"),
    Output("scores-chart", "figure"),
    Output("strategy-chart", "figure"),
    Output("prediction-distribution", "figure"),
    Input("prediction-model", "value"),
    Input("lookback-days", "value"),
    Input("price-granularity", "value"),
)
def update_charts(model_column, lookback_days, granularity):
    chart_df = _downsample_price_frame(featured, lookback_days, granularity)
    price_fig = go.Figure(
        data=[
            go.Scattergl(
                x=chart_df["Date"],
                y=chart_df["Close"],
                mode="lines",
                name="Close",
                line={"width": 2},
            )
        ]
    )
    price_fig.update_layout(title=f"Close Price ({granularity.capitalize()} Aggregation)")

    score_metrics = ["accuracy", "f1_up_class"] if "f1_up_class" in scores.columns else ["accuracy"]
    score_long = scores.melt(id_vars="model", value_vars=score_metrics, var_name="metric", value_name="value")
    score_fig = px.bar(
        score_long,
        x="model",
        y="value",
        color="metric",
        barmode="group",
        title="Model Quality Comparison",
    )
    score_fig.update_yaxes(tickformat=".0%")

    temp = preds.sort_values("Date").copy()
    temp["signal"] = temp[model_column].astype(int)
    temp["actual_return"] = temp.get("future_return_1d", 0.0).astype(float)
    temp["strategy_return"] = temp["actual_return"] * (temp["signal"] * 2 - 1)
    temp["cum_market"] = (1 + temp["actual_return"]).cumprod() - 1
    temp["cum_strategy"] = (1 + temp["strategy_return"]).cumprod() - 1
    strategy_fig = go.Figure()
    strategy_fig.add_trace(go.Scatter(x=temp["Date"], y=temp["cum_market"], mode="lines", name="Market Buy & Hold"))
    strategy_fig.add_trace(go.Scatter(x=temp["Date"], y=temp["cum_strategy"], mode="lines", name="Model Strategy"))
    strategy_fig.update_layout(title="Cumulative Return: Market vs Model Signal")
    strategy_fig.update_yaxes(tickformat=".0%")

    pred_fig = px.histogram(
        preds,
        x=model_column,
        title=f"Prediction Distribution ({model_column})",
        labels={model_column: "Predicted Class (0=DOWN, 1=UP)"},
    )
    return price_fig, score_fig, strategy_fig, pred_fig


if __name__ == "__main__":
    app.run(debug=True, port=8050)
