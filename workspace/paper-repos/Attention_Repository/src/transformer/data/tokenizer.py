"""
data/tokenizer.py
=================
BPE Tokenizer wrapper using SentencePiece.

Paper: "Attention Is All You Need", Vaswani et al. (2017)
Section 5.1 — byte-pair encoding with shared source-target vocabulary (~37k tokens).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import sentencepiece as spm


class BPETokenizer:
    """
    SentencePiece BPE tokenizer with shared source-target vocabulary.

    Paper: Section 5.1 — sentences encoded using byte-pair encoding,
    shared source-target vocabulary of ~37000 tokens for EN-DE.

    Args:
        model_path: Path to trained .model file from SentencePiece.
    """

    def __init__(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"SentencePiece model not found: {path}\n"
                "Run `python prepare_data.py --config configs/base.yaml` to generate it."
            )
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(path))

        # Special token ids
        self._pad_id = self.sp.piece_to_id("<pad>")
        self._bos_id = self.sp.piece_to_id("<s>")
        self._eos_id = self.sp.piece_to_id("</s>")

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> List[int]:
        """
        Encode a string to a list of token ids.

        Args:
            text:    Input string.
            add_bos: Prepend BOS token.
            add_eos: Append EOS token.

        Returns:
            List of integer token ids.
        """
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self._bos_id] + ids
        if add_eos:
            ids = ids + [self._eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of token ids back to a string.
        Strips BOS/EOS/PAD tokens automatically.

        Args:
            ids: List of integer token ids.

        Returns:
            Decoded string.
        """
        # Filter special tokens before decoding
        filtered = [i for i in ids if i not in (self._bos_id, self._eos_id, self._pad_id)]
        return self.sp.decode(filtered)

    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.sp.get_piece_size()

    @property
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def bos_id(self) -> int:
        return self._bos_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    def __repr__(self) -> str:
        return f"BPETokenizer(vocab_size={self.vocab_size()})"
