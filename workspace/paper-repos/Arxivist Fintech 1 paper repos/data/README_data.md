# Data Requirements

## Overview

This paper uses **proprietary historical order book data** from the INET Electronic Communication
Network (ECN), which is a NASDAQ trading venue. This data is **not publicly available**.

## What Data You Need

- **Source**: INET ECN historical order book records
- **Stocks**: AMZN (Amazon), NVDA (NVIDIA), QCOM (Qualcomm)
- **Period**: 1.5 years of millisecond-resolution data (paper uses ~2003–2004 era data)
- **Format**: Full limit order book depth (all outstanding bid/ask prices and volumes)
- **Size**: Several GB per stock

## Alternatives

### 1. Modern Publicly Available Order Book Data

While the exact INET data is not available, similar millisecond-resolution limit order book data
can be obtained from:

- **LOBSTER** (https://lobsterdata.com) — NASDAQ order book reconstructions, paid service
- **Nasdaq TotalView-ITCH** — Available via Nasdaq data products (subscription required)
- **Interactive Brokers Historical Data API** — For recent data
- **Alpaca Markets API** — Free tier available for US equities

### 2. Synthetic Data (for development)

Use the built-in `SyntheticOrderBookGenerator` for development and testing:

```python
from rl_trade_execution.data.loader import SyntheticOrderBookGenerator

gen = SyntheticOrderBookGenerator(seed=42)
episodes = gen.generate_episodes(n_episodes=1000, T=8, stock="SYN")
```

Or run training in debug mode:
```bash
python train.py --config configs/config.yaml --debug
```

**Note**: Synthetic data results will NOT match the paper's reported numbers.

## Expected CSV Format

If you have real order book data, format it as CSV with these columns:

```
timestamp_ms, bid_p1, bid_v1, bid_p2, bid_v2, ..., bid_p10, bid_v10,
              ask_p1, ask_v1, ask_p2, ask_v2, ..., ask_p10, ask_v10,
              signed_volume_15s
```

- `timestamp_ms`: Unix timestamp in milliseconds
- `bid_pN`, `bid_vN`: Price and volume of Nth best bid level
- `ask_pN`, `ask_vN`: Price and volume of Nth best ask level
- `signed_volume_15s`: Net signed trade volume in last 15 seconds
  (positive = buyer-initiated, negative = seller-initiated)

Place files at: `data/raw/{STOCK}.csv` (e.g., `data/raw/AMZN.csv`)

## Directory Structure

```
data/
├── raw/
│   ├── AMZN.csv    ← Place your order book data here
│   ├── NVDA.csv
│   └── QCOM.csv
└── README_data.md  ← This file
```
