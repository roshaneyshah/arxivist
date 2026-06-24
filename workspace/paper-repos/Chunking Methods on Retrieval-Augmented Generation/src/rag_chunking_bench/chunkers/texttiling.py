"""
chunkers/texttiling.py
======================
TextTiling chunker with paper-specified sentence-level boundary alignment.

Paper: "Chunking Methods on Retrieval-Augmented Generation" (arXiv:2606.00881)
Section 2.2: TextTiling (Hearst 1997) "was an early method designed to detect
topical shifts in texts based on lexical cohesion."
Section 3: "we modified the TextTiling implementation so that chunk boundaries
were aligned to the nearest sentence rather than the nearest paragraph."

IMPORTANT NOTE from paper:
  Original paragraph-level alignment "frequently resulted in no effective
  segmentation at all, often returning the original documents without
  introducing meaningful chunk boundaries across nearly all evaluated datasets.
  In contrast, sentence-level alignment yielded substantially more coherent
  and stable chunks."

TextTiling is EXCLUDED from Experiment 2 (end-to-end generation) because it
used a modified implementation not directly comparable to other methods under
the unified setup (Section 3.2).

Results (Table 1 Accuracy@5 avg): 84.96%
Results (Table 2 Recall@10 avg):  39.85%
Results (Table 3 time avg):        1.56m

Original: Hearst, M.A., 1997. "Text Tiling: Segmenting text into
multi-paragraph subtopic passages." Computational Linguistics 23.
"""

from __future__ import annotations

from typing import List

from rag_chunking_bench.chunkers.base import BaseChunker
from rag_chunking_bench.utils.text_utils import split_sentences


class TextTilingChunker(BaseChunker):
    """
    TextTiling chunker with sentence-level boundary alignment.

    Uses NLTK's TextTilingTokenizer to detect topical shifts via lexical
    cohesion analysis, then aligns detected boundaries to the nearest
    sentence boundary (paper's modification, Section 3).

    Args:
        config: Dict with keys:
            - w (int): Pseudosentence size in words (default 20).
            - k (int): Block comparison size (default 10).
            - smoothing_width (int): Width of smoothing window (default 2).
            - boundary_alignment (str): "sentence" (paper default) or "paragraph".
    """

    def __init__(self, config: dict | None = None) -> None:
        raw = config or {}
        w: int = raw.get("w", 20)
        k: int = raw.get("k", 10)
        smoothing_width: int = raw.get("smoothing_width", 2)
        # Explicitly stated in paper Section 3
        boundary_alignment: str = raw.get("boundary_alignment", "sentence")

        super().__init__({
            "w": w, "k": k,
            "smoothing_width": smoothing_width,
            "boundary_alignment": boundary_alignment,
        })
        self._w = w
        self._k = k
        self._smoothing_width = smoothing_width
        self._boundary_alignment = boundary_alignment

    @property
    def name(self) -> str:
        return "texttiling"

    def chunk(self, document: str) -> List[str]:
        """
        Segment document using TextTiling with sentence-level boundary alignment.

        EQ2: f_theta(D) -> C, theta = {w, k, smoothing_width}

        Args:
            document: Raw text string.

        Returns:
            List of topically coherent text chunks.
        """
        if not document.strip():
            return []

        try:
            import nltk
            try:
                tokenizer = nltk.TextTilingTokenizer(
                    w=self._w,
                    k=self._k,
                    smoothing_width=self._smoothing_width,
                )
                # NLTK TextTilingTokenizer returns paragraphs as tiles
                tiles = tokenizer.tokenize(document)
            except Exception:
                # Fallback: paragraph split on double newline
                tiles = [t.strip() for t in document.split("\n\n") if t.strip()]
        except ImportError:
            tiles = [t.strip() for t in document.split("\n\n") if t.strip()]

        if not tiles:
            return [document]

        if self._boundary_alignment == "sentence":
            return self._align_to_sentences(document, tiles)
        return [t for t in tiles if t.strip()]

    def _align_to_sentences(self, document: str, tiles: List[str]) -> List[str]:
        """
        Re-align tile boundaries to the nearest sentence boundary.

        Paper modification (Section 3): paragraph-level alignment returns
        the original document unsegmented; sentence-level alignment produces
        coherent, stable chunks.

        Args:
            document: Original full document text.
            tiles: TextTiling output tiles (paragraph-aligned).

        Returns:
            Chunks with boundaries snapped to sentence boundaries.
        """
        sentences = split_sentences(document)
        if not sentences:
            return tiles

        # Build char offsets for each sentence in the document
        offsets: List[int] = []
        pos = 0
        for sent in sentences:
            idx = document.find(sent, pos)
            if idx == -1:
                idx = pos
            offsets.append(idx)
            pos = idx + len(sent)

        # For each tile boundary position, find the nearest sentence start
        chunks: List[str] = []
        sentence_cursor = 0
        for tile in tiles:
            if not tile.strip():
                continue
            # Find which sentences fall in this tile
            tile_sentences: List[str] = []
            for i, sent in enumerate(sentences):
                if sent in tile or tile in document:
                    tile_sentences.append(sent)

            if tile_sentences:
                chunks.append(" ".join(tile_sentences))
            else:
                chunks.append(tile.strip())

        return [c for c in chunks if c.strip()]

    def __repr__(self) -> str:
        return (
            f"TextTilingChunker(w={self._w}, k={self._k}, "
            f"alignment={self._boundary_alignment})"
        )
