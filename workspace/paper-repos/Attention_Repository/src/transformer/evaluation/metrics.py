"""
evaluation/metrics.py
=====================
BLEU evaluation and beam-search decoder.

Paper: "Attention Is All You Need", Vaswani et al. (2017)
Section 6.1 — beam search with beam_size=4, length penalty α=0.6,
max output length = input length + 50.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader

import sacrebleu

from transformer.models.transformer import Transformer
from transformer.data.tokenizer import BPETokenizer
from transformer.utils.config import EvalConfig, TransformerConfig
from transformer.utils.masking import MaskFactory


# ---------------------------------------------------------------------------
# Beam Search
# ---------------------------------------------------------------------------

def beam_search_decode(
    model: Transformer,
    src: Tensor,
    src_mask: Tensor,
    tokenizer: BPETokenizer,
    beam_size: int = 4,
    max_len_offset: int = 50,
    length_penalty_alpha: float = 0.6,
    device: torch.device = torch.device("cpu"),
) -> List[int]:
    """
    Beam search decoding for a single source sequence.

    Paper: Section 6.1 — beam_size=4, length penalty α=0.6,
    max output length = input length + 50.

    Args:
        model:               Trained Transformer model (in eval mode).
        src:                 [1, T_src] source token ids.
        src_mask:            [1, 1, 1, T_src] source padding mask.
        tokenizer:           BPETokenizer for special token ids.
        beam_size:           Number of beams.
        max_len_offset:      Max decode length = T_src + max_len_offset.
        length_penalty_alpha: Length penalty exponent α.
        device:              Torch device.

    Returns:
        Best hypothesis as a list of token ids (excluding BOS).
    """
    model.eval()
    with torch.no_grad():
        # Encode source once
        memory = model.encode(src, src_mask)  # [1, T_src, d_model]
        T_src = src.size(1)
        max_len = T_src + max_len_offset

        bos_id = tokenizer.bos_id
        eos_id = tokenizer.eos_id
        pad_id = tokenizer.pad_id

        # Expand memory for all beams: [beam_size, T_src, d_model]
        memory = memory.expand(beam_size, -1, -1)
        src_mask = src_mask.expand(beam_size, -1, -1, -1)

        # Each beam: (cumulative_log_prob, token_ids_list, is_finished)
        beams: List[Tuple[float, List[int], bool]] = [
            (0.0, [bos_id], False)
        ]
        completed: List[Tuple[float, List[int]]] = []

        for step in range(max_len):
            if all(b[2] for b in beams):
                break
            if len(beams) == 0:
                break

            active_beams = [(score, ids) for score, ids, done in beams if not done]
            done_beams = [(score, ids, True) for score, ids, done in beams if done]

            # Stack current sequences
            tgt_ids = torch.tensor(
                [ids for _, ids in active_beams], dtype=torch.long, device=device
            )  # [n_active, step+1]

            tgt_mask = MaskFactory.make_tgt_mask(tgt_ids, pad_id)
            mem_slice = memory[: len(active_beams)]
            mask_slice = src_mask[: len(active_beams)]

            dec_out = model.decode(tgt_ids, mem_slice, src_mask=mask_slice, tgt_mask=tgt_mask)
            logits = model.output_projection(dec_out[:, -1, :])  # [n_active, V]
            log_probs = torch.log_softmax(logits, dim=-1)         # [n_active, V]

            # Expand each beam by top-k candidates
            candidates: List[Tuple[float, List[int], bool]] = []
            V = log_probs.size(-1)

            for i, (score, ids) in enumerate(active_beams):
                top_lp, top_ids = log_probs[i].topk(beam_size)
                for lp, tok in zip(top_lp.tolist(), top_ids.tolist()):
                    new_ids = ids + [tok]
                    new_score = score + lp
                    finished = tok == eos_id
                    if finished:
                        # Apply length penalty: score / ((5 + len) / 6)^α
                        lp_factor = ((5 + len(new_ids)) / 6) ** length_penalty_alpha
                        completed.append((new_score / lp_factor, new_ids))
                    else:
                        candidates.append((new_score, new_ids, False))

            # Keep top beam_size active beams
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = done_beams + candidates[:beam_size]

            if len(beams) > beam_size:
                beams = sorted(beams, key=lambda x: x[0], reverse=True)[:beam_size]

        # Collect any unfinished beams
        for score, ids, _ in beams:
            lp_factor = ((5 + len(ids)) / 6) ** length_penalty_alpha
            completed.append((score / lp_factor, ids))

        # Return best hypothesis (strip BOS)
        completed.sort(key=lambda x: x[0], reverse=True)
        best_ids = completed[0][1][1:]  # remove BOS
        # Strip EOS if present
        if best_ids and best_ids[-1] == eos_id:
            best_ids = best_ids[:-1]
        return best_ids


# ---------------------------------------------------------------------------
# BLEU Evaluator
# ---------------------------------------------------------------------------

class BLEUEvaluator:
    """
    Compute corpus-level BLEU score using sacrebleu.

    Paper: Section 6.1 — reported BLEU scores on WMT 2014 EN-DE and EN-FR.
    Uses sacrebleu for standardized, reproducible BLEU computation.

    Args:
        config: EvalConfig with beam_size, length_penalty_alpha, etc.
    """

    def __init__(self, config: EvalConfig) -> None:
        self.config = config

    def compute_bleu(
        self,
        hypotheses: List[str],
        references: List[str],
    ) -> float:
        """
        Compute corpus BLEU score.

        Args:
            hypotheses: List of model output strings.
            references: List of reference strings.

        Returns:
            BLEU score (0–100).
        """
        assert len(hypotheses) == len(references), (
            f"Hypothesis count ({len(hypotheses)}) != reference count ({len(references)})"
        )
        result = sacrebleu.corpus_bleu(hypotheses, [references])
        return result.score

    def evaluate_dataset(
        self,
        model: Transformer,
        data_loader: DataLoader,
        tokenizer: BPETokenizer,
        device: torch.device,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run beam-search decoding over a dataset and return BLEU and perplexity.

        Args:
            model:       Trained Transformer in eval mode.
            data_loader: DataLoader over the evaluation split.
            tokenizer:   BPETokenizer.
            device:      Torch device.
            max_batches: If set, evaluate on at most this many batches (for quick checks).

        Returns:
            Dict with 'bleu' and 'ppl'.
        """
        model.eval()
        hypotheses: List[str] = []
        references_raw: List[str] = []
        total_loss = 0.0
        total_tokens = 0

        from transformer.training.losses import LabelSmoothedCrossEntropy
        criterion = LabelSmoothedCrossEntropy(
            vocab_size=tokenizer.vocab_size(),
            smoothing=0.0,  # no smoothing for perplexity
            ignore_index=data_loader.dataset.tokenizer.pad_id,
        )

        pad_idx = tokenizer.pad_id

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if max_batches is not None and i >= max_batches:
                    break

                src = batch["src"].to(device)
                tgt_in = batch["tgt_in"].to(device)
                tgt_out = batch["tgt_out"].to(device)

                src_mask = MaskFactory.make_padding_mask(src, pad_idx)
                tgt_mask = MaskFactory.make_tgt_mask(tgt_in, pad_idx)

                # Perplexity via teacher-forced forward pass
                logits = model(src, tgt_in, src_mask=src_mask, tgt_mask=tgt_mask)
                loss = criterion(logits, tgt_out)
                ntokens = (tgt_out != pad_idx).sum().item()
                total_loss += loss.item() * ntokens
                total_tokens += ntokens

                # BLEU via beam search (one example at a time for simplicity)
                for j in range(src.size(0)):
                    src_j = src[j : j + 1]
                    mask_j = MaskFactory.make_padding_mask(src_j, pad_idx)
                    pred_ids = beam_search_decode(
                        model, src_j, mask_j, tokenizer,
                        beam_size=self.config.beam_size,
                        max_len_offset=self.config.max_decode_len_offset,
                        length_penalty_alpha=self.config.length_penalty_alpha,
                        device=device,
                    )
                    hypotheses.append(tokenizer.decode(pred_ids))

                    # Decode reference (tgt_out row, strip pad)
                    ref_ids = tgt_out[j].tolist()
                    references_raw.append(tokenizer.decode(ref_ids))

        bleu = self.compute_bleu(hypotheses, references_raw) if hypotheses else 0.0
        ppl = math.exp(total_loss / max(total_tokens, 1))

        return {"bleu": bleu, "ppl": ppl}
