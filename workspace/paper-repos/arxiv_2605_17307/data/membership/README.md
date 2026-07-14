# Index membership data

The paper relies on **Bloomberg Terminal** for time-varying historical membership of NASDAQ-100, Nikkei 225 and Euro Stoxx 50. Bloomberg is paywalled and cannot be redistributed.

## Default behaviour (survivorship-biased fallback)

If no membership CSVs are provided here, `src/portfolio_rl/data/membership.py` falls back to **current membership** for every historical date. This **introduces survivorship bias** and will likely inflate measured returns. A warning is printed at startup.

## How to supply historical membership

Drop CSVs under:

```
data/membership/
├── ndx/membership.csv
├── nky/membership.csv
└── sx5e/membership.csv
```

Each CSV must follow this schema:

| column   | dtype       | example      |
|----------|-------------|--------------|
| `ticker` | str         | `AAPL`       |
| `start`  | YYYY-MM-DD  | `2003-01-02` |
| `end`    | YYYY-MM-DD  | `2026-03-13` (or empty = still active) |

A ticker is considered active on date `d` iff `start ≤ d < end` (or `end` is empty).

The membership loader is at [`src/portfolio_rl/data/membership.py`](../../src/portfolio_rl/data/membership.py).
