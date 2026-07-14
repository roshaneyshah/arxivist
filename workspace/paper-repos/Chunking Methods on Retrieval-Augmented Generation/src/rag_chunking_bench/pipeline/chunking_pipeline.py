"""
pipeline/chunking_pipeline.py
==============================
Orchestrates chunking → embedding → indexing for one (method, dataset) pair.

Paper: "Chunking Methods on Retrieval-Augmented Generation" (arXiv:2606.00881)
Section 3: "imposing a 48-hour time limit for each chunking process, as we
considered any method exceeding this duration to introduce excessive
computational overhead."

Failure markers (Tables 1-2):
  T = procedure exceeded 48-hour time limit
  S = spaCy library triggered error due to excessive temporary memory
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import signal
from pathlib import Path
from typing import List, Optional, Tuple

from rag_chunking_bench.chunkers.base import BaseChunker
from rag_chunking_bench.embedding.embedder import ChunkEmbedder
from rag_chunking_bench.retrieval.index import FAISSChunkIndex
from rag_chunking_bench.utils.timing import Timer

logger = logging.getLogger(__name__)

FAILURE_T = "T"   # Timeout marker
FAILURE_S = "S"   # spaCy memory error marker


class ChunkingTimeoutError(Exception):
    """Raised when chunking exceeds the configured time limit."""
    pass


class ChunkingMemoryError(Exception):
    """Raised when chunking fails due to memory constraints (S marker)."""
    pass


class ChunkingPipeline:
    """
    Runs the offline chunking → embedding → FAISS indexing pass for one
    (method, dataset) pair, enforcing the paper's 48-hour timeout per pair.

    Paper Section 3: timeout rationale and failure marker definitions.

    Args:
        chunker: Instantiated BaseChunker subclass.
        embedder: ChunkEmbedder for producing dense vectors.
        timeout_hours: Hard time limit for chunking (default 48h, paper value).
        index_dir: Optional directory to save/load the FAISS index.
    """

    def __init__(
        self,
        chunker: BaseChunker,
        embedder: ChunkEmbedder,
        timeout_hours: float = 48.0,  # SIR conf 0.99 — explicitly stated
        index_dir: Optional[str] = None,
    ) -> None:
        self._chunker = chunker
        self._embedder = embedder
        self._timeout_seconds = timeout_hours * 3600
        self._index_dir = Path(index_dir) if index_dir else None

    def run(
        self, documents: List[str]
    ) -> Tuple[FAISSChunkIndex, List[List[str]], float]:
        """
        Execute chunking + embedding + indexing with timeout enforcement.

        Args:
            documents: List of raw document text strings.

        Returns:
            Tuple of:
              - FAISSChunkIndex (ready for retrieval)
              - all_chunks: list of chunk lists (one per document)
              - elapsed_seconds: wall-clock time for the chunking step only

        Raises:
            ChunkingTimeoutError: If chunking exceeds timeout_hours (T marker).
            ChunkingMemoryError: If spaCy memory error occurs (S marker).
        """
        logger.info(
            f"[{self._chunker.name}] Starting chunking "
            f"({len(documents)} documents, timeout={self._timeout_seconds/3600:.1f}h)"
        )

        # --- Step 1: Chunking (timed, with timeout) ---
        with Timer() as chunk_timer:
            all_chunks = self._chunk_with_timeout(documents)

        elapsed = chunk_timer.elapsed_seconds()
        logger.info(
            f"[{self._chunker.name}] Chunking complete in "
            f"{chunk_timer.elapsed_human()} → "
            f"{sum(len(c) for c in all_chunks)} total chunks"
        )

        # --- Step 2: Flatten chunks and embed ---
        flat_chunks: List[str] = [c for doc_chunks in all_chunks for c in doc_chunks]
        if not flat_chunks:
            logger.warning(f"[{self._chunker.name}] No chunks produced — empty documents?")
            flat_chunks = documents  # fallback: treat each doc as one chunk

        logger.info(f"[{self._chunker.name}] Embedding {len(flat_chunks)} chunks...")
        embeddings = self._embedder.encode(flat_chunks, show_progress=True)

        # --- Step 3: Build FAISS index ---
        index = FAISSChunkIndex(embedding_dim=self._embedder._embedding_dim)
        index.build(flat_chunks, embeddings)
        logger.info(f"[{self._chunker.name}] FAISS index built: {index}")

        # --- Optionally save index ---
        if self._index_dir is not None:
            save_path = self._index_dir / self._chunker.name
            index.save(str(save_path))
            logger.info(f"[{self._chunker.name}] Index saved to {save_path}")

        return index, all_chunks, elapsed

    def _chunk_with_timeout(self, documents: List[str]) -> List[List[str]]:
        """
        Run chunking in a thread with timeout enforcement.

        Uses concurrent.futures.ThreadPoolExecutor with a hard wall-clock
        timeout. If chunking exceeds self._timeout_seconds, raises
        ChunkingTimeoutError (T marker per paper).

        Also catches MemoryError and spaCy-related memory errors, re-raising
        as ChunkingMemoryError (S marker per paper).

        Args:
            documents: List of raw text documents.

        Returns:
            List of chunk lists (one per document).

        Raises:
            ChunkingTimeoutError: Timeout exceeded (T marker).
            ChunkingMemoryError: Memory error from spaCy or system (S marker).
        """
        def _do_chunk():
            result = []
            for i, doc in enumerate(documents):
                try:
                    chunks = self._chunker.chunk(doc)
                    result.append(chunks if chunks else [doc])
                except MemoryError as e:
                    raise ChunkingMemoryError(
                        f"Memory error (S marker) chunking document {i}: {e}"
                    ) from e
                except Exception as e:
                    # Detect spaCy memory errors specifically (S marker)
                    if _is_spacy_memory_error(e):
                        raise ChunkingMemoryError(
                            f"spaCy memory error (S marker) on document {i}: {e}"
                        ) from e
                    raise
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_do_chunk)
            try:
                return future.result(timeout=self._timeout_seconds)
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise ChunkingTimeoutError(
                    f"[{self._chunker.name}] Timeout after "
                    f"{self._timeout_seconds/3600:.1f}h (T marker). "
                    "This matches the paper's reported T-marker behavior."
                )
            except ChunkingMemoryError:
                raise
            except Exception as e:
                if _is_spacy_memory_error(e):
                    raise ChunkingMemoryError(
                        f"spaCy memory error (S marker): {e}"
                    ) from e
                raise


def _is_spacy_memory_error(exc: Exception) -> bool:
    """
    Detect spaCy-related memory errors (S marker in paper Tables 1-2).

    Paper note: "The underlying spaCy library triggered an error due to
    excessive temporary memory requirements" on documents > 1,000,000 chars.
    """
    msg = str(exc).lower()
    return (
        "spacy" in msg
        or "memory" in msg
        or isinstance(exc, (MemoryError, OverflowError))
    )
