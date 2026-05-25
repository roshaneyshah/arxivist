#!/usr/bin/env bash
# Download price data for the three universes + benchmarks via yfinance.
# Run from repo root:  bash data/download.sh

set -euo pipefail

OUT=data/cache
mkdir -p "$OUT"

python - <<'PY'
import os, yfinance as yf, pandas as pd
from pathlib import Path

START = "2003-01-02"
END   = "2026-03-13"
OUT = Path("data/cache")
OUT.mkdir(parents=True, exist_ok=True)

# Benchmarks + market proxies + VIX (always available)
EXTRA = ["QQQ", "EWJ", "FEZ", "^VIX", "^GSPC"]

# Current constituents (ARXIVIST FALLBACK — Bloomberg historical membership unavailable)
NDX = ["AAPL","MSFT","GOOGL","GOOG","AMZN","META","NVDA","TSLA","ADBE","COST",
       "PEP","CSCO","TMUS","CMCSA","TXN","QCOM","AMGN","INTC","INTU","HON",
       "AMD","BKNG","ISRG","SBUX","GILD","MDLZ","ADP","REGN","VRTX","LRCX"]
NKY = ["7203.T","6758.T","9984.T","9432.T","8306.T","6861.T","6098.T","8058.T",
       "8035.T","6594.T","4063.T","6981.T","7974.T","9433.T","8316.T","4502.T"]
SX5E = ["ASML.AS","SAP.DE","LIN","TTE.PA","SIE.DE","SAN.PA","ITX.MC","ALV.DE",
        "OR.PA","MC.PA","AIR.PA","BNP.PA","BAS.DE","DTE.DE","IBE.MC"]

all_tickers = list(set(EXTRA + NDX + NKY + SX5E))
print(f"Downloading {len(all_tickers)} tickers from {START} to {END}…")

# Batch download (yfinance handles ~30-40 at a time well)
data = yf.download(all_tickers, start=START, end=END, auto_adjust=True, progress=True)
prices = data["Close"] if "Close" in data.columns.get_level_values(0) else data
prices.to_parquet(OUT / "prices.parquet")
print(f"Saved {prices.shape} to {OUT / 'prices.parquet'}")

# Persist universe lists
import json
(OUT / "universes.json").write_text(json.dumps({
    "ndx": NDX, "nky": NKY, "sx5e": SX5E,
    "benchmarks": {"ndx": "QQQ", "nky": "EWJ", "sx5e": "FEZ"},
    "vix": "^VIX",
    "_note": "Current constituents only — survivorship-biased fallback."
}, indent=2))
print("Done.")
PY
