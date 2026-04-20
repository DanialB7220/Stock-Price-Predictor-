from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "stock data.csv"
OUTPUTS_DIR = ROOT / "outputs"
DEFAULT_TICKER = "AAL"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.sort_values("Date").copy()
    featured["return_1d"] = featured["Close"].pct_change(1)
    featured["return_5d"] = featured["Close"].pct_change(5)
    featured["ma_10"] = featured["Close"].rolling(10).mean()
    featured["ma_20"] = featured["Close"].rolling(20).mean()
    featured["volatility_10"] = featured["return_1d"].rolling(10).std()
    featured["volume_change_1d"] = featured["Volume"].pct_change(1)
    featured["future_return_1d"] = featured["Close"].shift(-1) / featured["Close"] - 1
    featured["target_direction"] = (featured["future_return_1d"] > 0).astype(int)

    featured = featured.dropna(
        subset=[
            "return_1d",
            "return_5d",
            "ma_10",
            "ma_20",
            "volatility_10",
            "volume_change_1d",
            "future_return_1d",
        ]
    ).copy()
    return featured


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing input data file: {DATA_PATH}")

    raw = pd.read_csv(DATA_PATH)
    raw.columns = [c.strip() for c in raw.columns]
    raw = raw.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    raw["Date"] = pd.to_datetime(raw["Date"], format="%d/%m/%Y", errors="coerce")
    raw = raw.dropna(subset=["Date", "Close", "Volume"])

    if "Name" not in raw.columns:
        raise ValueError("Input file must include a 'Name' ticker column.")

    ticker = DEFAULT_TICKER
    if ticker not in set(raw["Name"]):
        ticker = str(raw["Name"].mode().iloc[0])

    ticker_df = raw.loc[raw["Name"] == ticker, ["Date", "Open", "High", "Low", "Close", "Volume"]]
    featured = build_features(ticker_df)
    featured.to_csv(OUTPUTS_DIR / "featured_data.csv", index=False)

    print(f"Saved {len(featured)} rows to {OUTPUTS_DIR / 'featured_data.csv'} for ticker {ticker}")


if __name__ == "__main__":
    main()