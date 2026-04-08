import yfinance as yf
import numpy as np
import pandas as pd

# Pull SPY daily data
df = yf.download("SPY", start="2018-01-01", end="2025-12-31")
df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

# --- Feature Engineering ---
df["return_1d"] = df["Close"].pct_change(1)
df["return_5d"] = df["Close"].pct_change(5)
df["return_20d"] = df["Close"].pct_change(20)

df["ma_10"] = df["Close"].rolling(10).mean()
df["ma_20"] = df["Close"].rolling(20).mean()
df["ma_ratio"] = df["ma_10"] / df["ma_20"]  # normalized, not raw

df["volatility_10d"] = df["return_1d"].rolling(10).std()

# RSI-14
delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
df["rsi_14"] = 100 - (100 / (1 + gain / loss))

# Volume z-score (20-day)
df["volume_zscore"] = (
    (df["Volume"] - df["Volume"].rolling(20).mean()) 
    / df["Volume"].rolling(20).std()
)

df["day_of_week"] = df.index.dayofweek

# --- Label: ±1% threshold over 5 days ---
future_return = (df["Close"].shift(-5) - df["Close"]) / df["Close"]
df["target"] = np.where(future_return > 0.01, 1,
               np.where(future_return < -0.01, -1,
               np.nan))

df = df.dropna()  # drops NaN from rolling windows AND flat zone

print(df["target"].value_counts())
print(df.shape)