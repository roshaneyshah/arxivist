"""
data/synthetic_generator.py
============================
Synthetic panel data generator for testing without CRSP/Compustat access.

Generates data that mirrors the structure of Section III.B simulations:
    R_it = -0.3 + 0.3*Phi((C1 - 0.1)/0.1) + 0.3*Phi((C2 - 0.9)/0.1) + epsilon

(The "hard" nonlinear DGP from Section III.B where the linear model fails.)

Paper reference: Section III.B, Figure 1
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Optional


def generate_synthetic_panel(
    n_stocks: int = 500,
    n_periods: int = 240,
    n_chars: int = 36,
    dgp: str = "nonlinear",
    noise_std: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic panel mimicking CRSP structure for testing.

    Three DGP options matching paper simulations (Section III.B):
      'linear'    : R = -0.2 + 0.3*sqrt(C1) + 0.25*C2^2 + epsilon
      'nonlinear' : R = -0.3 + 0.3*Phi((C1-0.1)/0.1) + 0.3*Phi((C2-0.9)/0.1) + epsilon
      'quadratic' : R = (C1 - 0.5)^2 + epsilon  (non-monotone case)

    All other characteristics are noise (uniform [0,1]).

    Args:
        n_stocks: Number of firms per period
        n_periods: Number of monthly periods
        n_chars: Total number of characteristics (paper uses 36)
        dgp: Data-generating process ('linear', 'nonlinear', 'quadratic')
        noise_std: Standard deviation of idiosyncratic noise
        seed: Random seed

    Returns:
        DataFrame with columns: permno, date, ret, char_0, ..., char_{n_chars-1}
    """
    rng = np.random.default_rng(seed)

    # Build panel
    permnos = np.arange(n_stocks)
    dates = pd.date_range("1963-07-01", periods=n_periods, freq="MS")

    rows = []
    for t_idx, date in enumerate(dates):
        # Draw characteristics ~ Uniform[0,1] (post rank-normalization, Section III.B)
        C = rng.uniform(0, 1, size=(n_stocks, n_chars))
        C1 = C[:, 0]
        C2 = C[:, 1]

        # Signal from DGP
        if dgp == "linear":
            # Section III.B, first example: m1(c)=0.3*sqrt(c), m2(c)=0.25*c^2
            signal = -0.2 + 0.3 * np.sqrt(C1) + 0.25 * C2 ** 2
        elif dgp == "nonlinear":
            # Section III.B, second example (hard case, Figure 1)
            signal = -0.3 + 0.3 * norm.cdf((C1 - 0.1) / 0.1) + 0.3 * norm.cdf((C2 - 0.9) / 0.1)
        elif dgp == "quadratic":
            # Non-monotone case: 10-1 portfolio return = 0 if sorted on C1
            signal = (C1 - 0.5) ** 2
        else:
            raise ValueError(f"Unknown dgp='{dgp}'. Choose: 'linear', 'nonlinear', 'quadratic'")

        epsilon = rng.normal(0, noise_std, n_stocks)
        ret = signal + epsilon

        df_t = pd.DataFrame(C, columns=[f"char_{s}" for s in range(n_chars)])
        df_t["ret"] = ret
        df_t["permno"] = permnos
        df_t["date"] = date
        rows.append(df_t)

    panel = pd.concat(rows, ignore_index=True)
    return panel[["permno", "date", "ret"] + [f"char_{s}" for s in range(n_chars)]]


def generate_readme_synthetic() -> str:
    """Return README text explaining the synthetic data generator."""
    return """
# Synthetic Data Generator

Since CRSP and Compustat require paid WRDS subscriptions, this module provides
synthetic panel data for testing the DCNP pipeline end-to-end.

## DGPs Available

| DGP          | Signal function                                         | Paper reference |
|------------- |---------------------------------------------------------|-----------------|
| `linear`     | -0.2 + 0.3*sqrt(C1) + 0.25*C2^2                        | Section III.B, Table 1 (first simulation) |
| `nonlinear`  | -0.3 + 0.3*Phi((C1-0.1)/0.1) + 0.3*Phi((C2-0.9)/0.1) | Section III.B, Figure 1, Table 2 (second simulation) |
| `quadratic`  | (C1 - 0.5)^2                                            | Section III.B (non-monotone discussion) |

## Expected Results (nonlinear DGP)

Per paper (Section III.B):
- Linear model Sharpe ratio ≈ 0.74
- Nonparametric model Sharpe ratio ≈ 1.19

## Usage

```python
from dcnp.data.synthetic_generator import generate_synthetic_panel

panel = generate_synthetic_panel(n_stocks=2000, n_periods=240, dgp='nonlinear')
```
"""
