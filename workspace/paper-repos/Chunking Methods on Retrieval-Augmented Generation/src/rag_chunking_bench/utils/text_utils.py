"""
utils/text_utils.py
===================
Shared NLP utilities used across chunkers and evaluation.

Paper: "Chunking Methods on Retrieval-Augmented Generation" (arXiv:2606.00881)

Implements:
- Sentence splitting (used by SequentialHAC, MaxMin, TextTiling)
- Token counting (used to enforce 4000-token generation context limit)
- Answer span overlap check (implements EQ4: relevance label assignment)
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import List


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using NLTK punkt tokenizer.
    Falls back to regex splitting if NLTK is unavailable.

    Used by: SequentialHACChunker, MaxMinChunker, TextTilingChunker.

    Args:
        text: Raw document text.

    Returns:
        List of sentence strings (non-empty, stripped).
    """
    try:
        import nltk
        try:
            tokenizer = nltk.data.load("tokenizers/punkt_tab/english.pickle")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            tokenizer = nltk.data.load("tokenizers/punkt_tab/english.pickle")
        sentences = tokenizer.tokenize(text.strip())
    except Exception:
        # Fallback: split on sentence-ending punctuation
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_encoding(encoding_name: str):
    """Cached tiktoken encoding loader."""
    import tiktoken
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    """
    Count tokens in text using tiktoken.

    Used to enforce the 4000-token context limit in the generator
    (Section 3.2 of the paper — explicitly stated).

    Args:
        text: Input text string.
        encoding: Tiktoken encoding name (default: cl100k_base for GPT models).

    Returns:
        Integer token count.
    """
    enc = _get_encoding(encoding)
    return len(enc.encode(text))


def truncate_to_tokens(text: str, max_tokens: int, encoding: str = "cl100k_base") -> str:
    """
    Truncate text to at most max_tokens tokens.

    Args:
        text: Input text.
        max_tokens: Maximum number of tokens.
        encoding: Tiktoken encoding name.

    Returns:
        Truncated text string.
    """
    enc = _get_encoding(encoding)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


# ---------------------------------------------------------------------------
# Answer span overlap (EQ4)
# ---------------------------------------------------------------------------

def spans_overlap(chunk: str, answer_span: str) -> bool:
    """
    Check whether a chunk overlaps with an answer span.

    Implements EQ4 from the SIR:
        relevant(c, q) = 1[c ∩ answer_span(q) ≠ ∅]

    The paper (Section 3.1) defines a chunk as relevant if it overlaps
    with the span of the extractive answer. We implement this as a
    substring containment check (case-insensitive, whitespace-normalized).

    Args:
        chunk: A text chunk c_i.
        answer_span: The ground-truth extractive answer span.

    Returns:
        True if the answer span (or the chunk) is contained in the other.
    """
    if not answer_span or not chunk:
        return False
    # Normalize whitespace and lowercase for robust matching
    chunk_norm = " ".join(chunk.lower().split())
    span_norm = " ".join(answer_span.lower().split())
    return span_norm in chunk_norm or chunk_norm in span_norm


def find_relevant_chunks(
    chunks: List[str],
    answer_span: str,
    relevant_doc_id: str = "",
    all_doc_chunks: List[str] | None = None,
) -> List[str]:
    """
    Return the subset of chunks relevant to a query.

    Per Section 3.1 of the paper:
    - If answer_span is available: relevant = chunks overlapping with answer span (EQ4)
    - If no answer_span: all chunks from the relevant document are relevant

    Args:
        chunks: All candidate chunks.
        answer_span: Extractive answer text (may be empty).
        relevant_doc_id: Identifier of the relevant document (for fallback).
        all_doc_chunks: All chunks from the relevant document (for fallback).

    Returns:
        List of relevant chunks.
    """
    if answer_span:
        return [c for c in chunks if spans_overlap(c, answer_span)]
    # Fallback: all chunks from the relevant document
    if all_doc_chunks is not None:
        return list(all_doc_chunks)
    return []
