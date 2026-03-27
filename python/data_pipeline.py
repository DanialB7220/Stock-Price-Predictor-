import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
ARCHIVE_DIR = ROOT / "archive"


def fetch_yfinance_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No yfinance data returned for {ticker}.")

    # yfinance can return MultiIndex columns (e.g., Price/Ticker); flatten them.
    if isinstance(df.columns, pd.MultiIndex):
        flat_cols = []
        for col in df.columns:
            if isinstance(col, tuple):
                flat_cols.append(str(col[0]))
            else:
                flat_cols.append(str(col))
        df.columns = flat_cols

    df = df.reset_index()
    if "Date" not in df.columns:
        # Some yfinance versions return lowercase date column after reset.
        date_col = [c for c in df.columns if str(c).lower() == "date"]
        if date_col:
            df = df.rename(columns={date_col[0]: "Date"})

    # Normalize expected OHLCV names.
    rename_map = {
        "adj close": "Adj Close",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    for c in list(df.columns):
        key = str(c).strip().lower()
        if key in rename_map:
            df = df.rename(columns={c: rename_map[key]})
    return df


def fetch_archive_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    ticker = ticker.upper()
    candidates = [
        ARCHIVE_DIR / "stocks" / f"{ticker}.csv",
        ARCHIVE_DIR / "etfs" / f"{ticker}.csv",
        ARCHIVE_DIR / "index" / f"{ticker}.csv",
    ]

    archive_file = next((p for p in candidates if p.exists()), None)
    if archive_file is None:
        raise FileNotFoundError(f"No archive CSV found for ticker '{ticker}'.")

    df = pd.read_csv(archive_file)
    expected_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = expected_cols.difference(set(df.columns))
    if missing:
        raise ValueError(f"Archive file {archive_file} missing columns: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]
    if df.empty:
        raise ValueError(f"Archive data exists for {ticker}, but no rows in selected date range.")
    return df.reset_index(drop=True)


def fetch_alpha_vantage_indicators(ticker: str, api_key: str) -> pd.DataFrame:
    base = "https://www.alphavantage.co/query"
    params = {
        "function": "SMA",
        "symbol": ticker,
        "interval": "daily",
        "time_period": 14,
        "series_type": "close",
        "apikey": api_key,
    }
    response = requests.get(base, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    ts_key = "Technical Analysis: SMA"
    if ts_key not in payload:
        return pd.DataFrame(columns=["Date", "alpha_sma_14"])

    records = []
    for date_str, value in payload[ts_key].items():
        records.append({"Date": pd.to_datetime(date_str), "alpha_sma_14": float(value["SMA"])})
    return pd.DataFrame(records)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.sort_values("Date").reset_index(drop=True)

    out["return_1d"] = out["Close"].pct_change()
    out["return_5d"] = out["Close"].pct_change(5)
    out["ma_10"] = out["Close"].rolling(10).mean()
    out["ma_20"] = out["Close"].rolling(20).mean()
    out["volatility_10"] = out["return_1d"].rolling(10).std()
    out["volume_change_1d"] = out["Volume"].pct_change()

    out["target_next_close"] = out["Close"].shift(-1)
    out["target_direction"] = np.where(out["target_next_close"] > out["Close"], 1, 0)
    out = out.dropna().reset_index(drop=True)
    return out


def main() -> None:
    load_dotenv(ROOT / ".env")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    ticker = os.getenv("DEFAULT_TICKER", "AAPL")
    start_date = os.getenv("DEFAULT_START_DATE", "2018-01-01")
    end_date = os.getenv("DEFAULT_END_DATE", "2025-12-31")
    api_key = os.getenv("ALPHAVANTAGE_API_KEY", "")

    # Use local archive data first; fallback to yfinance if ticker file is missing.
    try:
        raw = fetch_archive_data(ticker=ticker, start_date=start_date, end_date=end_date)
        print(f"Loaded local archive data for {ticker}.")
    except (FileNotFoundError, ValueError):
        raw = fetch_yfinance_data(ticker=ticker, start_date=start_date, end_date=end_date)
        print(f"Loaded yfinance data for {ticker}.")
    raw.to_csv(DATA_DIR / f"{ticker}_raw.csv", index=False)

    if api_key:
        indicators = fetch_alpha_vantage_indicators(ticker=ticker, api_key=api_key)
        if not indicators.empty:
            raw["Date"] = pd.to_datetime(raw["Date"])
            raw = raw.merge(indicators, on="Date", how="left")

    featured = build_features(raw)
    featured.to_csv(OUTPUTS_DIR / "featured_data.csv", index=False)
    print(f"Saved featured dataset with {len(featured)} rows.")


if __name__ == "__main__":
    main()
