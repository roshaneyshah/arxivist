# Data Requirements for FS-GCLSTM Replication

## Required Data Sources

### 1. LSEG Value-Chain Data (Proprietary)
- **Source**: LSEG (London Stock Exchange Group) value-chain dataset
- **Access**: Requires paid LSEG subscription
- **Content**: Supplier-customer relationships with confidence scores and timestamps
- **Format needed**: Parquet file with columns `[source, target, confidence, date]`
- **Save to**: `data/value_chain.parquet`

### 2. Historical Daily Stock Prices
- **Eurostoxx 600**: All constituents + network-linked partners, 2000-01-01 to 2024-12-31
- **S&P 500**: All constituents + network-linked partners, 2000-01-01 to 2024-12-31
- **Sources**: Bloomberg, Refinitiv, Yahoo Finance (yfinance), or WRDS CRSP
- **Format needed**: Parquet file with columns `[date, ticker, close_price]`
- **Save to**: `data/prices.parquet`

## Placeholder Directory Structure
```
data/
├── prices.parquet          ← daily closing prices
├── value_chain.parquet     ← LSEG supplier-customer edges
└── README_data.md          ← this file
```

## Testing Without Real Data
Use the `--synthetic` flag to generate synthetic data that mimics the paper's graph statistics:
```bash
python train.py --config configs/config.yaml --synthetic
```
