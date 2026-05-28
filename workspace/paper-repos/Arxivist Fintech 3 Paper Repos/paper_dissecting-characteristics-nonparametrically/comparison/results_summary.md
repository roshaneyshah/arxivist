# Results Comparison: Expected vs Actual
## Paper: Freyberger, Neuhierl & Weber (2017) — NBER WP 23227

---

## Metrics to Validate

Run `python scripts/run_insample.py --use-synthetic` and
`python scripts/run_oos.py --use-synthetic` to populate the "Actual (Synthetic)"
column. Run with real CRSP data to populate "Actual (CRSP)".

| Metric | Paper Value | Source | Actual (Synthetic) | Actual (CRSP) | Status |
|--------|------------|---------|-------------------|---------------|--------|
| In-sample Sharpe (14 knots, all stocks) | 2.98 | Table 4, col 1 | TBD | TBD | ⏳ |
| In-sample Sharpe (9 knots, all stocks) | 3.02 | Table 4, col 2 | TBD | TBD | ⏳ |
| OOS Sharpe NP (9 knots, EW, 1991–2014) | 3.42 | Table 5, col 1 | TBD | TBD | ⏳ |
| OOS Sharpe NP (9 knots, VW, 1991–2014) | 1.24 | Table 5, col 2 | TBD | TBD | ⏳ |
| OOS Sharpe Linear (9 knots, EW) | 2.26 | Table 5, col 3 | TBD | TBD | ⏳ |
| OOS Sharpe Linear (9 knots, VW) | 1.01 | Table 5, col 4 | TBD | TBD | ⏳ |
| N selected NP (OOS, all stocks) | 8 | Table 5, col 1 | TBD | TBD | ⏳ |
| N selected Linear (OOS, all stocks) | 21 | Table 5, col 3 | TBD | TBD | ⏳ |
| Firm-level R² (NP, 1991–2014) | 3.11% | Section V.C | TBD | TBD | ⏳ |
| Forecast slope (NP, 1991–2014) | 0.78 | Section V.C | TBD | TBD | ⏳ |
| OOS SR (NP, size > q10, EW) | 1.33 | Table 5, col 8 | TBD | TBD | ⏳ |

---

## Section III.B Simulation Results (Checkable Without CRSP)

| Metric | Paper Value | Source |
|--------|------------|--------|
| Linear Sharpe (nonlinear DGP) | ~0.74 | Table 2, Section III.B |
| NP Sharpe (nonlinear DGP) | ~1.19 | Table 2, Section III.B |
| Linear Return | 0.1154 | Table 2 |
| NP Return | 0.1863 | Table 2 |
| Return Std (both) | ~0.1576 | Table 2 |

Note: Section III.B simulations use 2,000 stocks × 240 periods,
averaging over 1,000 replications. This repo uses a single run for tractability.

---

## Selected Characteristics (Table 4, col 1 — 14 knots, all stocks, full sample)

Paper selects (15 characteristics):
`A2ME, AT, Beta, D2A, E2P, FC2Y, Idio_vol, LME, Lturnover, Rel_to_High,
r12_2, r2_1, r36_13, SGA2M, SUV`

Never selected (in any specification):
`ATO, CTO, DPI2A, Investment, Lev, NOA, OL, PCM, PM, Prof, Q, ROA, ROE, S2P`

---

## Known Limitations of this Replication

1. **Proprietary data**: Full replication requires CRSP/Compustat via WRDS
2. **Simulation averaging**: Paper averages over 1,000 simulated datasets; this repo runs once
3. **n_confidence_sims**: Assumed 10,000; paper does not specify (SIR confidence: 0.65)
4. **Compustat merge**: Simplified merge stub; production requires CRSP-Compustat linking table
5. **Coordinate descent**: Group LASSO implementation uses block coordinate descent;
   paper likely uses a specialized solver (possibly MATLAB `fmincon` or custom code)
