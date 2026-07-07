"""SpotVolGraphDataset — per-timestamp graph snapshot dataset (Table 1 splits).

Expects preprocessed Fourier-estimate panels (parquet/csv) under
``{data_dir}/processed/`` with columns matching the schema documented in
``data/README_data.md``. Splits follow Table 1's date ranges exactly.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from spotv2net.data.graph_builder import GraphSnapshotBuilder

# Table 1 split date ranges
_SPLIT_DATES = {
    "train": ("2020-06-01", "2022-07-20"),
    "validation": ("2022-07-21", "2022-10-14"),
    "test": ("2022-10-15", "2023-05-10"),
}


class SpotVolGraphDataset(Dataset):
    """Yields per-timestamp graph snapshots (x, edge_index, edge_attr, y).

    Args:
        data_dir: Root data directory containing ``processed/`` panels
            (vol.parquet, covol.parquet, vov.parquet, covov.parquet).
        split: One of ``"train"``, ``"validation"``, ``"test"``.
        num_lags: L, number of lags for node/edge feature construction (Eq. 1-2).
        horizon: 1 for single-step (Sec. 7.2), 14 for multi-step functional
            forecast (Sec. 7.4).
        use_edge_features: If False, still builds edge tensors (needed by the
            model API) but downstream SpotV2Net-NE zeroes them internally.
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        num_lags: int = 42,
        horizon: int = 1,
        use_edge_features: bool = True,
    ) -> None:
        if split not in _SPLIT_DATES:
            raise ValueError(f"split must be one of {list(_SPLIT_DATES)}, got '{split}'")

        self.data_dir = data_dir
        self.split = split
        self.num_lags = num_lags
        self.horizon = horizon
        self.use_edge_features = use_edge_features
        self.builder = GraphSnapshotBuilder()

        processed_dir = os.path.join(data_dir, "processed")
        self.vol_df = self._load_panel(processed_dir, "vol")
        self.covol_df = self._load_panel(processed_dir, "covol")
        self.vov_df = self._load_panel(processed_dir, "vov")
        self.covov_df = self._load_panel(processed_dir, "covov")

        start, end = _SPLIT_DATES[split]
        mask = (self.vol_df.index >= start) & (self.vol_df.index <= end)
        all_idx = np.where(mask)[0]
        # Only keep indices that have enough lag history AND enough horizon lookahead.
        self.valid_indices = [
            idx
            for idx in all_idx
            if idx >= num_lags and idx + horizon < len(self.vol_df)
        ]

        self.num_nodes = self.vol_df.shape[1]
        self.edge_index = torch.from_numpy(
            self.builder.fully_connected_edge_index(self.num_nodes)
        )

    @staticmethod
    def _load_panel(processed_dir: str, name: str) -> pd.DataFrame:
        path_parquet = os.path.join(processed_dir, f"{name}.parquet")
        path_csv = os.path.join(processed_dir, f"{name}.csv")
        if os.path.isfile(path_parquet):
            return pd.read_parquet(path_parquet)
        if os.path.isfile(path_csv):
            return pd.read_csv(path_csv, index_col=0, parse_dates=True)
        raise FileNotFoundError(
            f"Could not find processed panel '{name}' under {processed_dir}. "
            "Run data/download.py and the Fourier estimation pipeline first, "
            "or see data/README_data.md."
        )

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self.valid_indices[idx]

        x = self.builder.build_node_features(self.vol_df, self.covol_df, self.num_lags, t)
        edge_attr = self.builder.build_edge_features(self.vov_df, self.covov_df, self.num_lags, t)

        if self.horizon == 1:
            y = self.vol_df.iloc[t + 1].to_numpy(dtype=np.float32)[:, None]  # [N, 1]
        else:
            y = self.vol_df.iloc[t + 1 : t + 1 + self.horizon].to_numpy(dtype=np.float32).T  # [N, H]

        return {
            "x": torch.from_numpy(x).float(),
            "edge_index": self.edge_index,
            "edge_attr": torch.from_numpy(edge_attr).float(),
            "y": torch.from_numpy(y).float(),
        }
