"""
fetch_bitcoin_data.py

Downloads full daily Bitcoin (BTC-USD) price history and prepares the
derived series needed for the finance case study (Section: Case Study
- Financial Tail Risk): daily log-returns and the loss series used for
the POT/GPD and Block-Maxima/GEV analyses.

Run locally (not in a sandboxed/restricted-network environment):

    pip install yfinance pandas
    python fetch_bitcoin_data.py

Outputs three CSV files in the current directory:
    bitcoin_daily_raw.csv        -- raw OHLCV data
    bitcoin_daily_returns.csv    -- date, close, log_return
    bitcoin_daily_losses.csv     -- date, loss (= -log_return), for POT/GEV fitting
"""

import numpy as np
import pandas as pd
import yfinance as yf

# ------------------------------------------------------------------
# 1. Download full history. BTC-USD has data on Yahoo Finance back to
#    2014-09-17. Adjust `end` to today's date when you run this.
# ------------------------------------------------------------------
TICKER = "BTC-USD"
START = "2014-09-17"
END = None  # None = up to the most recent available date

print(f"Downloading {TICKER} daily data from {START} to {END or 'today'}...")
raw = yf.download(TICKER, start=START, end=END, interval="1d", auto_adjust=False)

if raw.empty:
    raise RuntimeError(
        "No data returned. Check your internet connection and that "
        "yfinance is up to date (`pip install --upgrade yfinance`)."
    )

raw.to_csv("bitcoin_daily_raw.csv")
print(f"Saved {len(raw)} rows to bitcoin_daily_raw.csv")

# ------------------------------------------------------------------
# 2. Compute daily log-returns from the closing price.
# ------------------------------------------------------------------
close = raw["Close"].squeeze()
log_ret = np.log(close / close.shift(1)).dropna()

returns_df = pd.DataFrame({
    "date": log_ret.index,
    "close": close.loc[log_ret.index].values,
    "log_return": log_ret.values,
})
returns_df.to_csv("bitcoin_daily_returns.csv", index=False)
print(f"Saved {len(returns_df)} rows to bitcoin_daily_returns.csv")

# ------------------------------------------------------------------
# 3. Loss series (negative returns) -- this is what feeds the POT/GPD
#    (Pareto II) and Block-Maxima/GEV fits in the case study.
# ------------------------------------------------------------------
losses_df = returns_df.copy()
losses_df["loss"] = -losses_df["log_return"]
losses_df = losses_df[["date", "loss"]]
losses_df.to_csv("bitcoin_daily_losses.csv", index=False)
print(f"Saved {len(losses_df)} rows to bitcoin_daily_losses.csv")

# ------------------------------------------------------------------
# 4. Quick diagnostics useful for choosing a POT threshold and
#    checking how many exceedances / monthly blocks you'll have.
# ------------------------------------------------------------------
n = len(losses_df)
years = n / 252  # approx trading days; BTC trades every day so use 365 if preferred
print(f"\nTotal observations: {n} (~{n/365:.1f} calendar years)")

for q in [0.90, 0.95, 0.975, 0.99]:
    thresh = losses_df["loss"].quantile(q)
    n_exceed = (losses_df["loss"] > thresh).sum()
    print(f"  Threshold at {q*100:.1f}th pct = {thresh:.4f} -> {n_exceed} exceedances")

n_months = losses_df["date"].dt.to_period("M").nunique() if hasattr(losses_df["date"], "dt") else None
print(f"\nApprox. number of monthly blocks available for Block-Maxima/GEV fit: "
      f"{pd.to_datetime(losses_df['date']).dt.to_period('M').nunique()}")
