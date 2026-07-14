# Architecture Plan: Dissecting Characteristics Nonparametrically
## Paper: Freyberger, Neuhierl & Weber (2017)

---

## 1. Framework Selection

**Primary Framework**: Python 3.10+ with NumPy/SciPy/pandas/scikit-learn  
**Reasoning**: This is an econometric estimation method (adaptive group LASSO with splines), not a neural network. PyTorch is unnecessary — the core computations are matrix algebra, convex optimization, and statistical inference. scikit-learn provides the closest available tools; the group LASSO itself requires a custom implementation or `group-lasso` package.  
**CUDA**: Not required  
**Config**: YAML + argparse  
**Visualization**: matplotlib  

---

## 2. Module Hierarchy

```
paper-repos/paper_dissecting-characteristics-nonparametrically/
├── src/
│   └── dcnp/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py           # CRSP/Compustat data loading & merging
│       │   └── transforms.py       # Rank normalization, winsorization
│       ├── models/
│       │   ├── __init__.py
│       │   ├── spline_basis.py     # Quadratic spline basis expansion
│       │   ├── group_lasso.py      # Two-step adaptive group LASSO estimator
│       │   └── nonparametric.py    # Main AdaptiveGroupLASSOModel class
│       ├── estimation/
│       │   ├── __init__.py
│       │   ├── bic_selector.py     # BIC-based lambda selection
│       │   └── confidence_bands.py # Uniform confidence band computation
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── portfolio.py        # Hedge portfolio construction & Sharpe ratio
│       │   └── metrics.py          # R², forecast slope, FF3 alpha
│       └── utils/
│           ├── __init__.py
│           └── config.py           # Config dataclass
├── configs/
│   └── config.yaml
├── data/
│   └── download_ff_factors.py
├── docker/
│   └── Dockerfile
├── notebooks/
│   └── reproduce_paper.ipynb
├── scripts/
│   ├── run_insample.py
│   ├── run_oos.py
│   └── run_rolling.py
└── README.md
```

### Module Specifications

#### `data/loader.py`
- **Class**: `PanelDataLoader`
- **Methods**:
  - `load_crsp(path: str) -> pd.DataFrame`: Load monthly stock returns
  - `load_compustat(path: str) -> pd.DataFrame`: Load accounting variables
  - `merge_crsp_compustat(crsp, compustat) -> pd.DataFrame`: Merge with FF timing convention
  - `apply_filters(df) -> pd.DataFrame`: Price > $5, common shares, US incorporated

#### `data/transforms.py`
- **Class**: `RankNormalizer`
- **Methods**:
  - `fit_transform(X: np.ndarray, dates: np.ndarray) -> np.ndarray`: Cross-sectional rank in (0,1)
  - `transform(X: np.ndarray, dates: np.ndarray) -> np.ndarray`: Apply fitted transformation
- **Function**: `winsorize(X, quantiles=(0.01, 0.99)) -> np.ndarray`

#### `models/spline_basis.py`
- **Class**: `QuadraticSplineBasis`
- **Constructor params**: `n_knots: int` (L, number of interior knots)
- **Methods**:
  - `fit(X_tilde: np.ndarray) -> None`: Set knots at quantiles t_l = l/L
  - `transform(X_tilde: np.ndarray) -> np.ndarray`: Expand [N x S] → [N x S*(L+2)]
  - `basis_for_char(c: np.ndarray, s: int) -> np.ndarray`: Basis for single characteristic

#### `models/group_lasso.py`
- **Class**: `AdaptiveGroupLASSO`
- **Constructor params**: `groups: List[List[int]]`, `lambda1: float`, `lambda2: float`
- **Methods**:
  - `fit_stage1(X, y) -> np.ndarray`: Group LASSO Stage 1 coefficients
  - `compute_adaptive_weights(beta_tilde) -> np.ndarray`: w_s weights
  - `fit_stage2(X, y, weights) -> np.ndarray`: Stage 2 with adaptive weights
  - `selected_groups() -> List[int]`: Indices of non-zero characteristic groups

#### `models/nonparametric.py`
- **Class**: `AdaptiveGroupLASSOModel` (main model)
- **Constructor params**: `n_knots: int`, `n_chars: int`, `bic_grid: List[float]`
- **Methods**:
  - `fit(X_tilde, y, dates) -> None`: Full two-stage estimation pipeline
  - `predict(X_tilde) -> np.ndarray`: Predicted expected returns
  - `selected_characteristics() -> List[str]`: Names of selected chars
  - `get_conditional_mean_function(char_idx: int, grid: np.ndarray) -> np.ndarray`: m_ts(c)

#### `estimation/bic_selector.py`
- **Class**: `BICSelector`
- **Methods**:
  - `select_lambda(X, y, groups, lambda_grid) -> float`: BIC-optimal lambda
  - `compute_bic(X, y, beta, n_selected) -> float`: Yuan-Lin BIC

#### `estimation/confidence_bands.py`
- **Class**: `UniformConfidenceBand`
- **Methods**:
  - `fit(X_selected, y, beta_hat) -> None`: Compute HC covariance Sigma_hat
  - `critical_value(alpha=0.05, n_sims=10000) -> float`: Simulated d_ts
  - `band(char_idx, grid) -> Tuple[np.ndarray, np.ndarray]`: Lower and upper bounds

#### `evaluation/portfolio.py`
- **Class**: `HedgePortfolioEvaluator`
- **Methods**:
  - `form_decile_portfolios(returns, predicted, dates) -> pd.DataFrame`
  - `compute_sharpe(portfolio_returns, annualize=True) -> float`
  - `rolling_oos_evaluation(model, data, estimation_window=120) -> pd.DataFrame`

---

## 3. Data Flow Specification

```
PIPELINE: In-Sample Estimation
  raw_panel: [N*T x S+1]     ← raw characteristics + returns
  filtered_panel              → apply price/exchange filters
  ranked_panel: [N*T x S]    ← rank-normalized C_tilde in (0,1)
  X_spline: [N*T x S*(L+2)]  ← quadratic spline basis expansion
  beta_tilde: [(L+2)*S]      ← Stage 1 group LASSO (lambda1 by BIC)
  weights: [S]                ← adaptive weights from Stage 1
  beta_breve: [(L+2)*S]      ← Stage 2 adaptive group LASSO (lambda2 by BIC)
  S_selected: int             ← number of non-zero characteristic groups
  X_selected: [N*T x S_sel*(L+2)]  ← spline matrix for selected chars only
  beta_hat: [(L+2)*S_sel]    ← OLS re-estimation on selected chars
  R_hat: [N*T]               ← predicted expected returns
  hedge_portfolio             ← long top 10%, short bottom 10% (by R_hat)
  Sharpe_ratio: float

PIPELINE: Out-of-Sample Rolling
  For each month t from Jan 1991 to Dec 2014:
    X_window: [120*N_t x S*(L+2)]  ← 10-year estimation window
    beta_hat_t ← fit AdaptiveGroupLASSOModel on X_window
    R_hat_{t+1} ← predict using beta_hat_t and C_{t}
    portfolio_return_{t+1} ← long/short decile portfolio
  SR_OOS = annualized_sharpe(portfolio_returns)
```

---

## 4. Configuration Schema

```yaml
# config.yaml
model:
  n_knots: 14              # Primary in-sample; use 9 for OOS (conf: 0.99)
  n_characteristics: 36    # Number of candidate characteristics (conf: 0.99)

lasso:
  lambda1_grid: [0.001, 0.01, 0.1, 1.0, 10.0]  # BIC-selected (conf: 0.88)
  lambda2_grid: [0.001, 0.01, 0.1, 1.0, 10.0]
  bic_criterion: "yuan_lin_2006"

estimation:
  rolling_window_months: 120   # conf: 0.99
  hedge_decile: 0.10           # conf: 0.99
  weighting: "equal"           # equal or value
  n_confidence_sims: 10000     # ASSUMED: standard practice, conf: 0.65

data:
  sample_start: "1963-07"
  sample_end: "2015-06"
  oos_start: "1991-01"
  model_selection_end: "1990-12"
  min_price: 5.0
  exchanges: ["NYSE", "AMEX", "NASDAQ"]
  min_compustat_years: 2

evaluation:
  alpha_bands: 0.05
  annualization_factor: 12
  compute_ff3_alpha: true
```

---

## 5. Dependencies

### requirements.txt
```
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.3.0
group-lasso>=1.5.0
matplotlib>=3.7.0
seaborn>=0.12.0
statsmodels>=0.14.0
pyyaml>=6.0
tqdm>=4.65.0
```

### requirements-dev.txt
```
pytest>=7.3.0
pytest-cov>=4.1.0
black>=23.3.0
flake8>=6.0.0
mypy>=1.3.0
jupyter>=1.0.0
ipykernel>=6.23.0
```

---

## 6. Entrypoints

### `scripts/run_insample.py`
```
CLI args:
  --config     Path to config.yaml
  --data-dir   Path to CRSP/Compustat data
  --n-knots    Override knot count [4,9,14,19]
  --size-filter  NYSE percentile filter [None,10,20,50]
  --output-dir   Results output directory
```

### `scripts/run_oos.py`
```
CLI args:
  --config       Path to config.yaml
  --data-dir     Path to data
  --oos-start    Start of OOS period (default: 1991-01)
  --model        [nonparametric, linear] (can specify both)
  --n-knots      Spline knots
  --output-dir   Results directory
```

### `scripts/run_rolling.py`
```
CLI args:
  --config       Path to config.yaml
  --data-dir     Path to data
  --char-list    Space-separated list of characteristic names to plot over time
  --output-dir   Results and figures directory
```

---

## 7. Docker Specification

```
Base image: python:3.10-slim
System deps: gcc, g++, libgomp1 (for scipy/sklearn parallelism)
WORKDIR: /app
COPY: requirements.txt, src/, configs/, scripts/
RUN: pip install -r requirements.txt && pip install -e src/
VOLUME: /app/data (user mounts CRSP/Compustat data here)
CMD: python scripts/run_insample.py --config configs/config.yaml
```

---

## 8. Risk Assessment

| Risk | Severity | Description | Mitigation |
|------|----------|-------------|------------|
| Proprietary data (CRSP/Compustat) | High | Core data requires paid subscriptions; cannot include in repo | Provide synthetic data generator for testing; document download steps |
| Group LASSO convergence | Medium | Two-step adaptive LASSO may not converge in all lambda configurations | Implement warm-starting; add convergence tolerance config; fall back to coordinate descent |
| BIC lambda grid sensitivity | Medium | Results may depend on lambda grid resolution; paper does not specify grid | Test multiple grid resolutions; expose grid as config parameter |
| Confidence band simulation count | Low | Paper does not specify n_sims; assumed 10,000 | Make configurable; default 10,000 |
| Exact spline differentiability at knots | Low | Quadratic splines with continuity constraint require careful implementation | Unit-test basis functions against paper's p_k formulas |
| Reproducibility of rolling window | Medium | Exact fiscal year timing conventions for each of 36 characteristics require careful implementation | Implement each characteristic's timing per appendix; add unit tests |
