# Data — SpotV2Net (arXiv:2401.06249)

## Raw data

The paper uses 1-second TAQ Millisecond Consolidated Trades data for the 30 DJIA
constituents, June 1 2020 – May 10 2023 (Sec. 7.1), sourced via **Wharton Research
Data Services (WRDS)** — a proprietary, institutionally-licensed dataset. It is
**not redistributed** in this repository.

Run `python data/download.py` for step-by-step WRDS instructions, or
`python data/download.py --synthetic` to generate a small synthetic dataset that
exercises the full pipeline (Fourier estimation → graph construction → training)
without real market data.

Expected layout once real data is obtained:
```
data/raw/{TICKER}.csv   # columns: timestamp, price (1-second, NYSE hours 09:30-16:00)
```

## Fourier cutting-frequency calibration (⚠ High-risk assumption)

The paper computes Fourier spot/co-spot volatility and volatility-of-volatility
estimates (Sec. 6) using cutting frequencies (Nc, Mc, Sc, Lc and per-asset Nvi,
Mvi, Svi, Lvi) selected via the external **Sanfelici & Toscano (2024) FMVol
MATLAB library** — exact numeric values are **not given in the paper text**.

`configs/config.yaml`'s `data.fourier.*` block ships placeholder defaults. For a
faithful reproduction:
1. Obtain the FMVol library (Sanfelici & Toscano, 2024, *Mathematics and
   Computers in Simulation*, 226:338-353) and use its frequency-selection
   routine, **or**
2. Use the asymptotic MSE-minimizing frequency formulas of Mancino & Recchioni
   (2015) as a principled alternative.

Update `configs/config.yaml` accordingly before treating results as reproducing
the paper's reported MSE/QLIKE numbers (Tables 2 and 5).

## Processed panels

`src/spotv2net/data/dataset.py` expects four processed panels under
`data/processed/`:

| File | Contents |
|---|---|
| `vol.parquet` | Spot volatility, columns = 30 tickers, index = 30-min timestamps |
| `covol.parquet` | Spot co-volatility, columns = `"TICKER_A__TICKER_B"` pairs |
| `vov.parquet` | Spot volatility-of-volatility, columns = 30 tickers |
| `covov.parquet` | Spot co-volatility-of-volatility, columns = `"TICKER_A__TICKER_B"` pairs |

Produce these from `data/raw/*.csv` using
`src/spotv2net/data/fourier_estimators.py`'s `FourierSpotEstimator` (jump-filter
raw returns first via `src/spotv2net/data/transforms.py:JumpFilter`, β=0.5, α=0.5,
Sec. 7.1 footnote 7).

## Train/validation/test split (Table 1)

| Split | Start | End | # 30-min obs | Proportion |
|---|---|---|---|---|
| Train | 2020-06-01 | 2022-07-20 | 7518 | 73% |
| Validation | 2022-07-21 | 2022-10-14 | 840 | 8% |
| Test | 2022-10-15 | 2023-05-10 | 1960 | 19% |
