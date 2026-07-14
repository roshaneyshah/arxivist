"""
evaluation/reporter.py
======================
Aggregates per-query results into per-method, per-dataset tables matching
Tables 1–4 of the paper.

Paper: "Chunking Methods on Retrieval-Augmented Generation" (arXiv:2606.00881)
Table 1: Accuracy@5 per (method, dataset)
Table 2: Recall@10 per (method, dataset)
Table 3: Chunking time per (method, dataset)
Table 4: LLM-as-a-judge score per (method, dataset)
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


FAILURE_T = "T"   # Timeout > 48h (paper marker)
FAILURE_S = "S"   # spaCy memory error (paper marker)


class ResultReporter:
    """
    Stores and aggregates benchmark results, outputting CSV and JSON tables
    matching Tables 1–4 from the paper.

    Args:
        output_dir: Directory to write result files.
    """

    def __init__(self, output_dir: str) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Structure: {metric: {method: {dataset: value or "T"/"S"}}}
        self._data: Dict[str, Dict[str, Dict[str, float | str]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        self._timing: Dict[str, Dict[str, float | str]] = defaultdict(dict)

    def record(
        self,
        method: str,
        dataset: str,
        metric: str,
        value: float | str,
    ) -> None:
        """
        Record a metric value for a (method, dataset) pair.

        Args:
            method: Chunker name (e.g. 'fixed_size').
            dataset: Dataset name (e.g. 'squad').
            metric: Metric name (e.g. 'accuracy_at_5', 'recall_at_10', 'llm_judge').
            value: Float result or failure marker ("T" or "S").
        """
        self._data[metric][method][dataset] = value

    def record_failure(self, method: str, dataset: str, marker: str) -> None:
        """
        Record a T (timeout) or S (spaCy memory) failure for all metrics.

        Args:
            method: Chunker name.
            dataset: Dataset name.
            marker: "T" or "S".
        """
        assert marker in (FAILURE_T, FAILURE_S), f"Unknown failure marker: {marker}"
        for metric in ["accuracy_at_5", "recall_at_10", "llm_judge"]:
            self._data[metric][method][dataset] = marker
        self._timing[method][dataset] = marker

    def record_timing(self, method: str, dataset: str, seconds: float) -> None:
        """
        Record chunking wall-clock time for Table 3 reproduction.

        Args:
            method: Chunker name.
            dataset: Dataset name.
            seconds: Elapsed seconds (converted to human-readable in export).
        """
        self._timing[method][dataset] = seconds

    def export_json(self, filename: str = "results.json") -> None:
        """Export all results to JSON."""
        out = {
            "metrics": {k: dict(v) for k, v in self._data.items()},
            "timing_seconds": dict(self._timing),
        }
        path = self._output_dir / filename
        with open(path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"Results written to {path}")

    def export_csv(self) -> None:
        """Export each metric table as a CSV matching Tables 1-4 layout."""
        import csv

        methods = sorted({
            m for metric_data in self._data.values() for m in metric_data
        })
        datasets = sorted({
            d for metric_data in self._data.values()
            for method_data in metric_data.values() for d in method_data
        })

        for metric, metric_data in self._data.items():
            path = self._output_dir / f"table_{metric}.csv"
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["dataset"] + methods)
                for dataset in datasets:
                    row = [dataset]
                    for method in methods:
                        val = metric_data.get(method, {}).get(dataset, "-")
                        if isinstance(val, float):
                            row.append(f"{val * 100:.2f}")
                        else:
                            row.append(str(val))
                    writer.writerow(row)
                # Average row
                avg_row = ["Average"]
                for method in methods:
                    vals = [
                        v for v in metric_data.get(method, {}).values()
                        if isinstance(v, float)
                    ]
                    avg_row.append(f"{sum(vals)/len(vals)*100:.2f}" if vals else "-")
                writer.writerow(avg_row)
            print(f"Table written to {path}")

        # Timing table
        path = self._output_dir / "table_timing.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["dataset"] + methods)
            for dataset in datasets:
                row = [dataset]
                for method in methods:
                    val = self._timing.get(method, {}).get(dataset, "-")
                    if isinstance(val, float):
                        row.append(_seconds_to_human(val))
                    else:
                        row.append(str(val))
                writer.writerow(row)
        print(f"Timing table written to {path}")

    def print_summary_table(self, metric: str = "accuracy_at_5") -> None:
        """Print a summary table for the given metric to stdout."""
        metric_data = self._data.get(metric, {})
        methods = sorted(metric_data.keys())
        datasets = sorted({d for m in metric_data.values() for d in m})

        col_w = max(20, max((len(m) for m in methods), default=10) + 2)
        header = f"{'dataset':<25}" + "".join(f"{m:<{col_w}}" for m in methods)
        print("\n" + "=" * len(header))
        print(f"Metric: {metric}")
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        for dataset in datasets:
            row = f"{dataset:<25}"
            for method in methods:
                val = metric_data.get(method, {}).get(dataset, "-")
                if isinstance(val, float):
                    row += f"{val*100:<{col_w}.2f}"
                else:
                    row += f"{str(val):<{col_w}}"
            print(row)
        print("=" * len(header))


def _seconds_to_human(s: float) -> str:
    if s < 1:
        return "<1s"
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s/60:.2f}m"
    return f"{s/3600:.2f}h"
