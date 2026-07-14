# Dissecting Characteristics Nonparametrically
### Freyberger, Neuhierl & Weber (2017) — NBER Working Paper 23227

Reproducible implementation of the adaptive group LASSO framework for nonparametric cross-sectional return prediction.

---

## What this paper does

The paper proposes a nonparametric method to determine which firm characteristics provide **independent information** for the cross section of expected returns. It uses the **adaptive group LASSO** of Huang, Horowitz & Wei (2010) to simultaneously:
1. Select which of 36 characteristics have incremental predictive power
2. Estimate how those characteristics affect expected returns (without assuming linearity)

Key findings:
- Only **7–15** of 36 characteristics provide independent information (depending on specification)
- Nonlinear functional forms matter — Sharpe ratios **50% higher** than the linear model out-of-sample (3.42 vs 2.26)
- Many well-known return predictors (Q, ROA, Investment, Profitability) lose predictive power once you condition on other characteristics

---

## Repository Structure

```
├── src/dcnp/
│   ├── data/
│   │   ├── loader.py            # CRSP/Compustat loading & Fama-French timing
│   │   ├── transforms.py        # Rank normalization (Section III.C)
│   │   └── synthetic_generator.py  # Synthetic DGP for testing (no CRSP needed)
│   ├── models/
│   │   ├── spline_basis.py      # Quadratic spline basis (Section III.D, Eq. 4)
│   │   ├── group_lasso.py       # Two-step adaptive group LASSO (Eqs. 5–7)
│   │   └── nonparametric.py     # Main AdaptiveGroupLASSOModel
│   ├── estimation/
│   │   ├── bic_selector.py      # BIC lambda selection (Yuan & Lin 2006)
│   │   └── confidence_bands.py  # Uniform confidence bands (Section III.E)
│   └── evaluation/
│       ├── portfolio.py         # Hedge portfolio & rolling OOS evaluation
│       └── metrics.py           # Sharpe ratio, R², FF3 alpha
├── configs/config.yaml          # All hyperparameters with confidence annotations
├── scripts/
│   ├── run_insample.py          # Replicates Table 4
│   ├── run_oos.py               # Replicates Table 5
│   └── run_rolling.py           # Replicates Figures 12–15
├── notebooks/reproduce_paper.ipynb  # Interactive walkthrough
└── docker/Dockerfile
```

---

## Quick Start (Synthetic Data — No CRSP Required)

```bash
# Install
pip install -r requirements.txt
pip install -e src/

# Run in-sample estimation with synthetic nonlinear DGP (Section III.B)
python scripts/run_insample.py --config configs/config.yaml --use-synthetic

# Run OOS evaluation
python scripts/run_oos.py --config configs/config.yaml --use-synthetic

# Interactive notebook
jupyter notebook notebooks/reproduce_paper.ipynb
```

### Docker
```bash
docker build -f docker/Dockerfile -t dcnp .
docker run dcnp
```

---

## Full Replication (CRSP + Compustat)

CRSP and Compustat data require access via **WRDS** (Wharton Research Data Services).

1. Download CRSP monthly stock file → save as `data/crsp_monthly.parquet`
2. Download Compustat annual fundamentals → save as `data/compustat_annual.parquet`
3. Download Fama-French 3 factors from [Ken French's library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) → save as `data/ff3_factors.csv`

```bash
python scripts/run_insample.py --config configs/config.yaml --data-dir data/
python scripts/run_oos.py --config configs/config.yaml --data-dir data/
```

### Expected Results

| Result | Paper | Notes |
|--------|-------|-------|
| In-sample Sharpe (14 knots, all stocks) | **2.98** | Table 4, col 1 |
| OOS Sharpe NP (9 knots, EW) | **3.42** | Table 5, col 1 |
| OOS Sharpe Linear (9 knots, EW) | **2.26** | Table 5, col 3 |
| N selected (OOS, all stocks) | **8** | Table 4, col 7 |
| Firm-level R² (NP) | **3.11%** | Section V.C |
| Forecast slope (NP) | **0.78** | Section V.C |

---

## Key Implementation Decisions

| Decision | Assumption | Paper evidence | Confidence |
|----------|-----------|----------------|------------|
| BIC criterion | Yuan & Lin (2006) group BIC | Explicitly cited | 0.88 |
| Confidence band n_sims | 10,000 draws | Not specified | **0.65** ⚠ |
| Pooled OLS re-estimation | Pooled time-series cross-section | "pooled … regression" | 0.80 |
| Knot placement | t_l = l/L (equally-spaced quantiles) | Explicitly stated | 0.99 |
| HC covariance | White (1980) | "heteroscedasticity-consistent" | 0.95 |

Items marked ⚠ are assumed — see `sir-registry/.../sir.json` for full ambiguity log.

---

## Citation

```bibtex
@techreport{freyberger2017dissecting,
  title={Dissecting Characteristics Nonparametrically},
  author={Freyberger, Joachim and Neuhierl, Andreas and Weber, Michael},
  institution={National Bureau of Economic Research},
  type={Working Paper},
  number={23227},
  year={2017}
}
```
