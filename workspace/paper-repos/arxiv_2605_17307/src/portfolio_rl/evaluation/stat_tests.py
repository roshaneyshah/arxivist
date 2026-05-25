"""HAC (Newey-West) and stationary block bootstrap tests (paper §6.4)."""
from __future__ import annotations

import numpy as np


class StatisticalTests:
    @staticmethod
    def hac_pvalue(strategy_returns, benchmark_returns) -> float:
        """Two-sided HAC-robust t-test on mean(r_strategy - r_benchmark) = 0."""
        import statsmodels.api as sm
        import pandas as pd

        diff = pd.Series(strategy_returns) - pd.Series(benchmark_returns)
        diff = diff.dropna()
        X = np.ones((len(diff), 1))
        model = sm.OLS(diff.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
        return float(model.pvalues[0])

    @staticmethod
    def block_bootstrap_pvalue(strategy_returns, benchmark_returns, n_boot: int = 10_000, block_size: int = 20) -> float:
        """Politis stationary block bootstrap p-value for mean-difference > 0."""
        import pandas as pd

        diff = (pd.Series(strategy_returns) - pd.Series(benchmark_returns)).dropna().values
        n = len(diff)
        if n == 0:
            return 1.0
        observed = diff.mean()
        rng = np.random.default_rng(0)
        boot_means = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            # geometric block lengths
            sample = []
            i = 0
            while len(sample) < n:
                start = rng.integers(0, n)
                length = max(1, int(rng.geometric(1.0 / block_size)))
                for j in range(length):
                    if len(sample) >= n:
                        break
                    sample.append(diff[(start + j) % n])
            boot_means[b] = float(np.mean(sample) - observed)
        # two-sided p-value
        return float(np.mean(np.abs(boot_means) >= abs(observed)))
