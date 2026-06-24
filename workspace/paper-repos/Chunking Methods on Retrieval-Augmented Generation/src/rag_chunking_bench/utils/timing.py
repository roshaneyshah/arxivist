"""
utils/timing.py
===============
Wall-clock Timer context manager used by ChunkingPipeline to measure and
reproduce the chunking execution times reported in Table 3 of:

Paper: "Chunking Methods on Retrieval-Augmented Generation" (arXiv:2606.00881)
Section 4: Table 3 — Chunking time per dataset and method.
"""

from __future__ import annotations

import time
from typing import Optional


class Timer:
    """
    Context manager for measuring wall-clock elapsed time.

    Used to reproduce Table 3 (chunking time per method/dataset).
    The paper enforces a 48-hour timeout per (method, dataset) pair;
    ChunkingPipeline uses this class to track elapsed time and raise
    TimeoutError when the limit is exceeded.

    Example:
        with Timer() as t:
            do_work()
        print(f"Elapsed: {t.elapsed_seconds():.2f}s")
    """

    def __init__(self) -> None:
        self._start: Optional[float] = None
        self._end: Optional[float] = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self._end = None
        return self

    def __exit__(self, *args) -> None:
        self._end = time.perf_counter()

    def elapsed_seconds(self) -> float:
        """Return elapsed wall-clock seconds. Can be called mid-run."""
        if self._start is None:
            raise RuntimeError("Timer has not been started")
        end = self._end if self._end is not None else time.perf_counter()
        return end - self._start

    def elapsed_human(self) -> str:
        """Return human-readable elapsed time (e.g. '2.10m', '3.09h', '<1s')."""
        s = self.elapsed_seconds()
        if s < 1:
            return "<1s"
        if s < 60:
            return f"{s:.0f}s"
        if s < 3600:
            return f"{s/60:.2f}m"
        return f"{s/3600:.2f}h"

    def __repr__(self) -> str:
        if self._start is None:
            return "Timer(not started)"
        return f"Timer(elapsed={self.elapsed_human()})"
