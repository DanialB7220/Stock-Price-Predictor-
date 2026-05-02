from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"

app = Dash(__name__)
app.title = "Stock Regime Dashboard (Unsupervised)"

metrics_path = OUTPUTS_DIR / "unsupervised_metrics_python.csv"
clusters_path = OUTPUTS_DIR / "test_clusters_python.csv"
featured_path = OUTPUTS_DIR / "featured_data.csv"

if not (metrics_path.exists() and clusters_path.exists() and featured_path.exists()):
    raise FileNotFoundError(
        "Missing output files. Run python/data_pipeline.py and python/train_model.py first."
    )

metrics = pd.read_csv(metrics_path)
clusters = pd.read_csv(clusters_path)
featured = pd.read_csv(featured_path)

for df in (clusters, featured):
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

ticker_col = "ticker" if "ticker" in featured.columns else None
ticker_options = (
    sorted(featured[ticker_col].dropna().astype(str).unique().tolist()) if ticker_col else []
)
if not ticker_options:
    ticker_options = ["__all__"]
default_ticker = ticker_options[0]

signal_column = "signal_long" if "signal_long" in clusters.columns else None
if signal_column is None:
    raise ValueError("test_clusters_python.csv must include signal_long")


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


def _business_days_between(start_date: pd.Timestamp, end_date: pd.Timestamp) -> int:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end <= start:
        return 0
    return len(pd.bdate_range(start=start + pd.Timedelta(days=1), end=end))


def _forecast_price(
    close_series: pd.Series, return_series: pd.Series, model_bias: float, horizon_days: int
) -> tuple[float, float]:
    last_close = float(close_series.iloc[-1])
    if horizon_days <= 0:
        return last_close, 0.0

    recent_ret = return_series.tail(30).dropna()
    base_daily = float(recent_ret.mean()) if len(recent_ret) else 0.0
    adjusted_daily = base_daily + ((model_bias - 0.5) * 0.0025)
    adjusted_daily = max(min(adjusted_daily, 0.08), -0.08)

    future_price = last_close * ((1.0 + adjusted_daily) ** horizon_days)
    total_return = (future_price / last_close) - 1.0 if last_close != 0 else 0.0
    return future_price, total_return


def _forecast_path(
    close_series: pd.Series, return_series: pd.Series, model_bias: float, target_date: pd.Timestamp
) -> pd.DataFrame:
    anchor_date = pd.Timestamp(close_series.index[-1]) if isinstance(close_series.index, pd.DatetimeIndex) else None
    if anchor_date is None:
        return pd.DataFrame(columns=["Date", "Close"])

    target = pd.Timestamp(target_date).normalize()
    if target <= anchor_date.normalize():
        return pd.DataFrame(columns=["Date", "Close"])

    future_days = pd.bdate_range(start=anchor_date.normalize() + pd.Timedelta(days=1), end=target)
    if len(future_days) == 0:
        return pd.DataFrame(columns=["Date", "Close"])

    recent_ret = return_series.tail(30).dropna()
    base_daily = float(recent_ret.mean()) if len(recent_ret) else 0.0
    adjusted_daily = max(min(base_daily + ((model_bias - 0.5) * 0.0025), 0.08), -0.08)
    last_close = float(close_series.iloc[-1])

    prices = []
    running_price = last_close
    for day in future_days:
        running_price = running_price * (1.0 + adjusted_daily)
        prices.append({"Date": day, "Close": running_price})
    return pd.DataFrame(prices)


latest_close = float(featured["Close"].iloc[-1]) if len(featured) else float("nan")
close_5d_ago = float(featured["Close"].iloc[-6]) if len(featured) > 5 else float("nan")
return_5d = (latest_close / close_5d_ago - 1) if pd.notna(close_5d_ago) and close_5d_ago != 0 else float("nan")

sil_row = metrics.loc[metrics["metric"] == "silhouette_test", "value"]
silhouette_test = float(sil_row.iloc[0]) if len(sil_row) else float("nan")
last_data_date = featured["Date"].max().date() if len(featured) else pd.Timestamp.today().date()
default_forecast_date = (pd.Timestamp(last_data_date) + pd.offsets.BDay(5)).date()

app.layout = html.Div(
    [
        html.H1("Stock Market Regime Analysis (Unsupervised K-Means)"),
        html.P(
            "Clusters are learned from price/volume features only. "
            "Long/short overlay uses train-split clusters with positive mean next-day return."
        ),
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
                        html.H4("Silhouette (test)", style={"marginBottom": "4px"}),
                        html.H2(f"{silhouette_test:.3f}" if pd.notna(silhouette_test) else "N/A", style={"marginTop": "0px"}),
                        html.P("Higher is better (separation)", style={"margin": "0px", "color": "#4b5563"}),
                    ],
                    style={"padding": "12px", "background": "#f5f7fb", "borderRadius": "10px", "minWidth": "220px"},
                ),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "16px"},
        ),
        html.P("Signal: K-Means regime → long when cluster had positive mean future return on the train split."),
        html.Div(
            [
                html.Label("Stock (Ticker)", style={"fontWeight": "bold"}),
                dcc.Dropdown(
                    id="ticker-filter",
                    options=[{"label": t, "value": t} for t in ticker_options],
                    value=default_ticker,
                    clearable=False,
                    style={"maxWidth": "360px"},
                ),
            ],
            style={"marginTop": "12px", "display": "none" if ticker_col is None else "block"},
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
        html.Div(
            [
                html.H3("Future Price Projection", style={"marginBottom": "8px"}),
                html.P(
                    "Heuristic path from recent returns and recent regime signal strength (not a model forecast).",
                    style={"marginTop": "0px", "color": "#4b5563"},
                ),
                dcc.DatePickerSingle(
                    id="forecast-date",
                    date=default_forecast_date,
                    min_date_allowed=last_data_date,
                    display_format="YYYY-MM-DD",
                ),
                html.Div(
                    id="forecast-output",
                    style={
                        "marginTop": "10px",
                        "padding": "10px",
                        "background": "#f5f7fb",
                        "borderRadius": "8px",
                        "maxWidth": "560px",
                    },
                ),
            ],
            style={"marginTop": "12px", "marginBottom": "10px"},
        ),
        dcc.Graph(id="price-chart"),
        dcc.Graph(id="scores-chart"),
        dcc.Graph(id="strategy-chart"),
        dcc.Graph(id="cluster-chart"),
    ],
    style={"padding": "20px", "fontFamily": "Arial"},
)


@app.callback(
    Output("price-chart", "figure"),
    Output("scores-chart", "figure"),
    Output("strategy-chart", "figure"),
    Output("cluster-chart", "figure"),
    Output("forecast-output", "children"),
    Input("lookback-days", "value"),
    Input("price-granularity", "value"),
    Input("ticker-filter", "value"),
    Input("forecast-date", "date"),
)
def update_charts(lookback_days, granularity, selected_ticker, forecast_date):
    featured_view = featured
    if ticker_col and selected_ticker and str(selected_ticker) != "__all__":
        featured_view = featured.loc[featured[ticker_col].astype(str) == str(selected_ticker)].copy()
    chart_df = _downsample_price_frame(featured_view, lookback_days, granularity)
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
    ticker_title = (
        f" - {selected_ticker}" if selected_ticker and str(selected_ticker) != "__all__" else ""
    )
    price_fig.update_layout(title=f"Close Price{ticker_title} ({granularity.capitalize()} Aggregation)")

    score_fig = px.bar(
        metrics,
        x="metric",
        y="value",
        title="Unsupervised quality metrics",
    )

    temp = clusters.sort_values("Date").copy()
    temp["signal"] = temp[signal_column].astype(int)
    if "future_return_1d" in temp.columns:
        temp["actual_return"] = temp["future_return_1d"].astype(float)
    elif "Close" in temp.columns:
        temp["actual_return"] = temp["Close"].astype(float).pct_change().fillna(0.0)
    else:
        temp["actual_return"] = pd.Series(0.0, index=temp.index, dtype=float)
    temp["strategy_return"] = temp["actual_return"] * (temp["signal"] * 2 - 1)
    temp["cum_market"] = (1 + temp["actual_return"]).cumprod() - 1
    temp["cum_strategy"] = (1 + temp["strategy_return"]).cumprod() - 1
    strategy_fig = go.Figure()
    strategy_fig.add_trace(go.Scatter(x=temp["Date"], y=temp["cum_market"], mode="lines", name="Market Buy & Hold"))
    strategy_fig.add_trace(go.Scatter(x=temp["Date"], y=temp["cum_strategy"], mode="lines", name="Regime signal"))
    strategy_fig.update_layout(title="Cumulative Return: Market vs regime-based long/short (test split)")
    strategy_fig.update_yaxes(tickformat=".0%")

    cluster_fig = px.histogram(
        clusters,
        x="cluster",
        title="Test-period cluster assignments (K-Means)",
        labels={"cluster": "Cluster id"},
    )

    if len(featured_view) == 0:
        forecast_text = "No data available for selected ticker."
    else:
        featured_view = featured_view.sort_values("Date").copy()
        anchor_date = pd.Timestamp(featured_view["Date"].max())
        target_date = pd.Timestamp(forecast_date) if forecast_date else anchor_date
        horizon_days = _business_days_between(anchor_date, target_date)
        model_bias = float(clusters[signal_column].tail(120).mean()) if signal_column in clusters.columns else 0.5
        predicted_price, predicted_return = _forecast_price(
            featured_view["Close"], featured_view.get("return_1d", pd.Series(dtype=float)), model_bias, horizon_days
        )
        close_series = featured_view.set_index("Date")["Close"]
        return_series = featured_view.get("return_1d", pd.Series(dtype=float))
        forecast_df = _forecast_path(close_series, return_series, model_bias, target_date)
        if len(forecast_df):
            price_fig.add_trace(
                go.Scatter(
                    x=forecast_df["Date"],
                    y=forecast_df["Close"],
                    mode="lines",
                    name="Projection",
                    line={"width": 2, "dash": "dash", "color": "#ef4444"},
                )
            )
        if horizon_days <= 0:
            forecast_text = "Choose a date after the latest available data date to project."
        else:
            ticker_label = (
                str(selected_ticker)
                if selected_ticker and str(selected_ticker) != "__all__"
                else "selected stock"
            )
            forecast_text = (
                f"{ticker_label}: {horizon_days} business days ahead | "
                f"Projected price: ${predicted_price:.2f} | "
                f"Projected return: {_format_pct(predicted_return)}"
            )

    return price_fig, score_fig, strategy_fig, cluster_fig, forecast_text


if __name__ == "__main__":
    app.run(debug=True, port=8050)
