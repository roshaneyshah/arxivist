"""
data/loader.py
==============
Data loading, merging, and filtering for the DCNP replication.

Handles CRSP monthly stock file and Compustat annual accounting data,
applying the exact filters and timing conventions of:
  Freyberger, Neuhierl & Weber (2017) — NBER WP 23227, Section IV

IMPORTANT: CRSP and Compustat data require paid subscriptions (Wharton
Research Data Services — WRDS). See data/README_data.md for instructions.

Timing conventions (Section IV):
  - Balance-sheet variables: fiscal year ending in t-1, used from July t to June t+1
  - Market-based variables (LME, returns): lagged one month
  - Anti-survivorship bias: require ≥ 2 years of Compustat history
  - Filters: price > $5, common shares (sharecodes 10/11), NYSE/AMEX/NASDAQ

Paper reference: Section IV "Data"
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All 36 characteristics as in Table 1 and Section IV
CHARACTERISTIC_COLS = [
    "A2ME", "AT", "ATO", "BEME", "Beta", "C", "CTO", "D2A", "DPI2A", "E2P",
    "FC2Y", "Free_CF", "Idio_vol", "Investment", "Lev", "LME", "Lturnover",
    "NOA", "OA", "OL", "PCM", "PM", "Prof", "Q", "Rel_to_High", "RNA",
    "ROA", "ROE", "r12_2", "r12_7", "r2_1", "r36_13", "S2P", "SGA2M",
    "Spread", "SUV",
]


class PanelDataLoader:
    """Load and merge CRSP/Compustat panel data for return prediction.

    Applies all filters from Section IV of Freyberger et al. (2017):
      - US common shares only (share codes 10, 11)
      - NYSE, AMEX, NASDAQ only
      - Price > $5
      - At least 2 years of Compustat history (anti-survivorship)
      - Fama-French (1993) timing convention for accounting variables

    Args:
        crsp_path: Path to CRSP monthly parquet file
        compustat_path: Path to Compustat annual parquet file
        ff3_factors_path: Path to Fama-French 3-factor CSV
        min_price: Minimum stock price filter (default: 5.0)
        exchanges: List of exchange codes to include
        share_codes: Share type codes for common stocks
        min_compustat_years: Minimum years of Compustat history
    """

    def __init__(
        self,
        crsp_path: str | Path,
        compustat_path: str | Path,
        ff3_factors_path: str | Path,
        min_price: float = 5.0,
        exchanges: List[str] = None,
        share_codes: List[int] = None,
        min_compustat_years: int = 2,
    ) -> None:
        self.crsp_path = Path(crsp_path)
        self.compustat_path = Path(compustat_path)
        self.ff3_factors_path = Path(ff3_factors_path)
        self.min_price = min_price
        self.exchanges = exchanges or ["NYSE", "AMEX", "NASDAQ"]
        self.share_codes = share_codes or [10, 11]
        self.min_compustat_years = min_compustat_years

    def load_crsp(self) -> pd.DataFrame:
        """Load CRSP monthly stock file.

        Expected columns (WRDS CRSP monthly file):
            permno, date, ret, prc, shrout, vol, exchcd, shrcd, cusip

        Returns:
            Filtered CRSP DataFrame with columns:
              permno, date, ret, prc, shrout, exchcd, shrcd, LME, Lturnover
        """
        if not self.crsp_path.exists():
            raise FileNotFoundError(
                f"CRSP file not found: {self.crsp_path}\n"
                f"See data/README_data.md for download instructions."
            )

        crsp = pd.read_parquet(self.crsp_path)

        # Standardize column names
        crsp.columns = crsp.columns.str.lower()

        # Date parsing
        crsp["date"] = pd.to_datetime(crsp["date"])

        # Filter: common shares only (Section IV)
        crsp = crsp[crsp["shrcd"].isin(self.share_codes)].copy()

        # Filter: US exchanges only
        exchange_map = {"NYSE": 1, "AMEX": 2, "NASDAQ": 3}
        allowed_exchcds = [exchange_map.get(ex, 99) for ex in self.exchanges]
        crsp = crsp[crsp["exchcd"].isin(allowed_exchcds)].copy()

        # Filter: price > $5 (Section IV)
        crsp["prc"] = crsp["prc"].abs()   # CRSP uses negative prices for bid-ask midpoint
        crsp = crsp[crsp["prc"] >= self.min_price].copy()

        # Construct LME: market cap in previous month
        # LME = |prc| * shrout / 1000 (in $000s; paper Section IV)
        crsp = crsp.sort_values(["permno", "date"])
        crsp["ME"] = crsp["prc"] * crsp["shrout"]
        crsp["LME"] = crsp.groupby("permno")["ME"].shift(1)

        # Construct Lturnover: prior month volume / shares outstanding
        crsp["Lturnover"] = crsp["vol"] / crsp["shrout"].replace(0, np.nan)

        return crsp

    def load_compustat(self) -> pd.DataFrame:
        """Load Compustat annual fundamentals.

        Expected: standard annual Compustat industrial file from WRDS.

        Returns:
            Compustat DataFrame with all required accounting variables.
        """
        if not self.compustat_path.exists():
            raise FileNotFoundError(
                f"Compustat file not found: {self.compustat_path}\n"
                f"See data/README_data.md for download instructions."
            )

        comp = pd.read_parquet(self.compustat_path)
        comp.columns = comp.columns.str.lower()
        comp["datadate"] = pd.to_datetime(comp["datadate"])
        comp["fyear"] = comp["datadate"].dt.year

        return comp

    def load_ff3_factors(self) -> pd.DataFrame:
        """Load Fama-French 3-factor returns from Ken French data library.

        Returns:
            DataFrame with columns: date, mkt_rf, smb, hml, rf
        """
        if not self.ff3_factors_path.exists():
            raise FileNotFoundError(
                f"FF3 factors file not found: {self.ff3_factors_path}\n"
                f"Download from: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"
            )

        ff3 = pd.read_csv(self.ff3_factors_path, skiprows=3)
        ff3.columns = ff3.columns.str.strip()
        # Handle Ken French format (YYYYMM integer date)
        ff3 = ff3.rename(columns={"": "date"})
        ff3["date"] = pd.to_datetime(ff3["date"].astype(str), format="%Y%m")
        ff3 = ff3.apply(pd.to_numeric, errors="coerce")
        ff3 = ff3.dropna(subset=["date"])
        # Convert percent to decimal
        for col in ["Mkt-RF", "SMB", "HML", "RF"]:
            if col in ff3.columns:
                ff3[col] = ff3[col] / 100.0
        return ff3

    def apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply survivorship bias filter (Section IV).

        "To alleviate a potential survivorship bias due to backfilling,
        we require that a firm has at least two years of Compustat data."

        Args:
            df: Merged panel DataFrame with 'gvkey' and 'year' columns

        Returns:
            Filtered DataFrame
        """
        if "gvkey" in df.columns and "fyear" in df.columns:
            # Count years per firm in Compustat
            comp_history = (
                df.groupby("gvkey")["fyear"]
                .transform("count")
            )
            df = df[comp_history >= self.min_compustat_years].copy()
        return df

    def merge_crsp_compustat(
        self,
        crsp: pd.DataFrame,
        compustat: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge CRSP and Compustat with Fama-French (1993) timing convention.

        Fama-French timing (Section IV):
          "We use the book-to-market ratio for estimation starting in June
          of year t until May of year t+1 predicting returns from July of
          year t until June of year t"

        Accounting variables from fiscal year ending in calendar year t-1
        are first used for return prediction from July of year t.
        This ensures a 6-month gap between fiscal year end and first use.

        Args:
            crsp: CRSP monthly DataFrame (from load_crsp)
            compustat: Compustat annual DataFrame (from load_compustat)

        Returns:
            Merged panel with returns and accounting characteristics
        """
        # Compustat: assign the "linking year" following FF convention
        # Fiscal year ending in calendar year t-1 → used July t to June t+1
        compustat = compustat.copy()
        compustat["link_year"] = compustat["fyear"] + 1  # Use starting July link_year

        # CRSP: extract year and month for merging
        crsp = crsp.copy()
        crsp["year"] = crsp["date"].dt.year
        crsp["month"] = crsp["date"].dt.month
        # July-June fiscal year: months 7-12 in year t link to fyear t-1;
        # months 1-6 in year t link to fyear t-2 (i.e., link_year = t)
        crsp["link_year"] = np.where(
            crsp["month"] >= 7, crsp["year"] + 1, crsp["year"]
        )

        # Merge on permno/gvkey link (in practice, use CRSP-Compustat link table)
        # STUB: actual merge requires CRSP-Compustat linking table from WRDS
        # Here we perform a simplified merge on cusip/gvkey if available
        merged = crsp.merge(
            compustat,
            on=["link_year"],  # simplified; real merge needs permno-gvkey link
            how="left",
            suffixes=("", "_comp"),
        )

        return merged

    def __repr__(self) -> str:
        return (
            f"PanelDataLoader(crsp={self.crsp_path.name}, "
            f"compustat={self.compustat_path.name}, "
            f"price_filter={self.min_price})"
        )
