"""Per-timestamp fully-connected graph construction (Eq. 1-2, Sec. 5).

Turns Fourier-estimate panels (spot volatility, co-volatility, volatility-of-volatility,
co-volatility-of-volatility) into node and edge feature tensors for a single graph
snapshot at time tau_b.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class GraphSnapshotBuilder:
    """Builds SpotV2Net's fully-connected graph inputs from Fourier estimate panels.

    All panels are indexed ``[timestamp, asset]`` (or ``[timestamp, asset_pair]`` for
    co-volatility panels).
    """

    def fully_connected_edge_index(self, num_nodes: int) -> np.ndarray:
        """Directed edge list for a fully connected graph (Sec. 5: A[i,j]=1 for all i,j).

        Self-loops are excluded (edges only for i != j, matching Fig. 3's pairwise
        edge features x^e_ij).

        Returns:
            edge_index, shape ``[2, N*(N-1)]`` (source, target).
        """
        src, dst = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    src.append(i)
                    dst.append(j)
        return np.array([src, dst], dtype=np.int64)

    def build_node_features(
        self, vol_df: pd.DataFrame, covol_df: pd.DataFrame, num_lags: int, timestamp_idx: int
    ) -> np.ndarray:
        """Build node feature matrix x (Eq. 1) at a given timestamp index.

        Args:
            vol_df: Spot volatility panel, columns = asset tickers, index = timestamps.
            covol_df: Spot co-volatility panel, columns = "ASSET_A__ASSET_B" pairs.
            num_lags: L, the number of lags (contemporaneous + L historical values).
            timestamp_idx: Integer row index b into ``vol_df`` / ``covol_df``.

        Returns:
            Node feature matrix, shape ``[N, M]`` where
            ``M = (num_lags + 1) * N`` (own vol + (N-1) co-vols, all lagged).
        """
        assets = list(vol_df.columns)
        n_assets = len(assets)
        if timestamp_idx < num_lags:
            raise ValueError(
                f"timestamp_idx={timestamp_idx} must be >= num_lags={num_lags} to build lag features"
            )

        node_feats = np.zeros((n_assets, (num_lags + 1) * n_assets), dtype=np.float32)
        for i, asset_i in enumerate(assets):
            cols = []
            # own volatility, contemporaneous + L lags
            for lag in range(num_lags + 1):
                cols.append(vol_df[asset_i].iloc[timestamp_idx - lag])
            # co-volatility with every other asset, contemporaneous + L lags
            for asset_j in assets:
                if asset_j == asset_i:
                    continue
                pair_key = f"{asset_i}__{asset_j}" if f"{asset_i}__{asset_j}" in covol_df.columns else f"{asset_j}__{asset_i}"
                for lag in range(num_lags + 1):
                    cols.append(covol_df[pair_key].iloc[timestamp_idx - lag])
            node_feats[i, : len(cols)] = cols
        return node_feats

    def build_edge_features(
        self, vov_df: pd.DataFrame, covov_df: pd.DataFrame, num_lags: int, timestamp_idx: int
    ) -> np.ndarray:
        """Build edge feature matrix x^e (Eq. 2) aligned with a fully-connected edge_index.

        Args:
            vov_df: Spot volatility-of-volatility panel, columns = asset tickers.
            covov_df: Spot co-volatility-of-volatility panel, columns = "A__B" pairs.
            num_lags: L, number of lags.
            timestamp_idx: Integer row index b.

        Returns:
            Edge feature matrix, shape ``[N*(N-1), E]`` where
            ``E = 3 * (num_lags + 1)`` — matches the ordering of
            ``fully_connected_edge_index``.
        """
        assets = list(vov_df.columns)
        n_assets = len(assets)
        edge_feats = []
        for i, asset_i in enumerate(assets):
            for j, asset_j in enumerate(assets):
                if i == j:
                    continue
                pair_key = f"{asset_i}__{asset_j}" if f"{asset_i}__{asset_j}" in covov_df.columns else f"{asset_j}__{asset_i}"
                row = []
                for lag in range(num_lags + 1):
                    row.append(vov_df[asset_i].iloc[timestamp_idx - lag])
                for lag in range(num_lags + 1):
                    row.append(vov_df[asset_j].iloc[timestamp_idx - lag])
                for lag in range(num_lags + 1):
                    row.append(covov_df[pair_key].iloc[timestamp_idx - lag])
                edge_feats.append(row)
        return np.array(edge_feats, dtype=np.float32)
