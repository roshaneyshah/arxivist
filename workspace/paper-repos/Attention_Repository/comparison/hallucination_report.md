# Hallucination Report

**Paper**: Attention Is All You Need  
**Paper ID**: arxiv_1706_03762  
**Report Date**: 2026-05-31  
**ArXivist SIR Version**: 1  
**Auditor**: ArXivist Stage 6 — Results Comparator  

---

## Overview

This report audits the generated implementation against the SIR for three classes
of hallucination: **structural** (invented components), **parametric** (assumed
hyperparameters that may be wrong), and **omission** (SIR components missing from
the code).

**Summary**:

| Hallucination Type | Count | Critical | Significant | Minor |
|---|---|---|---|---|
| Structural | 0 | — | — | — |
| Parametric | 4 | 0 | 1 | 3 |
| Omission | 0 | — | — | — |
| **Total** | **4** | **0** | **1** | **3** |

No critical hallucinations. No structural or omission hallucinations. The implementation
faithfully reproduces every SIR module, equation, and hyperparameter that was unambiguously
specified in the paper. The four parametric findings all stem from documented SIR ambiguities
(confidence < 0.85).

---

## Section 1 — Structural Hallucinations

> Structural hallucinations: components present in the generated code that are NOT in
> the SIR, suggesting the generator invented architecture elements.

**None found.**

Every class and module in the generated code maps directly to a named module in
`sir.json → architecture.modules`. No extraneous layers, normalisation variants,
or novel sub-layers were introduced. The implementation is structurally faithful
to Figure 1 and Sections 3.1–3.5 of the paper.

---

## Section 2 — Parametric Hallucinations

### P1 — 3-Way Weight Tying

**Severity**: Significant  
**Location**: `src/transformer/models/transformer.py`, `tie_weights()`, line ~75  
**SIR Confidence**: 0.82  
**Type**: parametric

**What was implemented**:
```python
# 3-way weight tying: encoder_embed = decoder_embed = output_projection
self.tgt_embedding.embedding.weight = self.src_embedding.embedding.weight
self.output_projection.weight = self.src_embedding.embedding.weight
```

**Paper text** (Section 3.4):
> "we share the same weight matrix between the two embedding layers and the
> pre-softmax linear transformation"

**Issue**: "two embedding layers" could mean (a) encoder input + decoder input, with
both tied to output projection (3-way), or (b) decoder input + output projection only
(2-way), with the encoder having its own embedding. Reading (b) is also plausible
because encoder and decoder vocabularies are shared but their usage contexts differ.

**Evidence against the implementation**: The tensor2tensor codebase (the paper's
original implementation) uses 3-way tying when `shared_embedding=True`, matching
our implementation. This is **corroborating evidence**, not proof.

**Evidence for a potential error**: For language pairs with separate source/target
vocabularies (e.g., EN-FR with 32k word-piece vocab), tying encoder and decoder
embeddings implicitly assumes a shared vocabulary — which Section 5.1 confirms for
EN-DE ("shared source-target vocabulary") but is less clear for EN-FR ("word-piece").

**Impact on results**: 
- If 2-way tying is correct: encoder embedding is a separate 37k×512 matrix (~19M params),
  encoder gets independent gradient updates, decoder output distribution is unaffected.
- Expected BLEU impact: ±0.1–0.3 BLEU (minor).

**Suggested fix**: Add a config flag `encoder_decoder_embed_sharing: true/false` to
ablate. Default `true` (current behavior) matches tensor2tensor.

---

### P2 — Attention Dropout

**Severity**: Minor  
**Location**: `src/transformer/models/attention.py`, `ScaledDotProductAttention.forward()`, line ~68  
**SIR Confidence**: 0.78  
**Type**: parametric

**What was implemented**:
```python
# Attention dropout (see SIR ambiguity note in docstring above)
attn_weights = self.dropout(attn_weights)
```

**Paper text** (Section 5.4):
> "We apply dropout to the output of each sub-layer, before it is added to the
> sub-layer input and normalized. In addition, we apply dropout to the sums of
> the embeddings and the positional encodings."

**Issue**: The paper lists two specific dropout locations. Attention weight dropout
is not mentioned. It is an additional regulariser present in many reproductions
(fairseq, annotated-transformer) but not specified in the paper.

**Impact on results**: 
- Removing attention dropout (setting `attn_weights = attn_weights` with no dropout)
  reduces regularisation by one location. With P_drop=0.1, impact is minor.
- Expected BLEU impact: ±0.0–0.2 BLEU.

**Suggested fix**: Expose a separate `attention_dropout` config field:
```yaml
model:
  dropout: 0.1            # residual + embedding dropout
  attention_dropout: 0.1  # NEW — set to 0.0 to match paper's explicit description
```

---

### P3 — Xavier Uniform Weight Initialization

**Severity**: Minor  
**Location**: All `_init_weights()` methods in `attention.py`, `feedforward.py`, `transformer.py`  
**SIR Confidence**: 0.70  
**Type**: parametric

**What was implemented**: `nn.init.xavier_uniform_` for all linear layers.

**Paper text**: No mention of weight initialization anywhere in the paper.

**Issue**: Xavier uniform is a reasonable default but is an assumption. The original
tensor2tensor code uses a combination of `initializers.glorot_uniform` (≈ Xavier
uniform) and `initializers.uniform_unit_scaling`. PyTorch's default for `nn.Linear`
is Kaiming uniform (He initialization), which would be activated if our `_init_weights`
call were removed.

**Impact on results**: Initialization affects early training dynamics but not final
converged performance given 100k steps. Expected BLEU impact: 0.0 at convergence.

**Suggested fix**: No action required for full training. For ablation or debugging,
add a `weight_init: xavier_uniform | kaiming_uniform | normal_scaled` config option.

---

### P4 — Positional Encoding Addition Ordering

**Severity**: Minor  
**Location**: `src/transformer/models/embeddings.py`, `PositionalEncoding.forward()`, line ~79  
**SIR Confidence**: 0.88  
**Type**: parametric

**What was implemented**:
```python
x = x + self.pe[:, :x.size(1), :]   # add PE
return self.dropout(x)               # then dropout
```

**Paper text** (Section 5.4):
> "we apply dropout to the sums of the embeddings and the positional encodings"

**Issue**: "sums of the embeddings and the positional encodings" strongly implies that
dropout is applied *after* the PE is added to the embedding — which is exactly what
the implementation does. This is rated Minor only because the SIR logged it as an
ambiguity (confidence 0.88); the implementation matches the most natural reading.

**Impact on results**: Effectively zero — the alternative (dropout before PE addition)
would be unusual and not supported by any published reproduction.

**Suggested fix**: No action required.

---

## Section 3 — Omission Hallucinations

> Omission hallucinations: components present in the SIR but absent or stubbed
> in the generated code.

**None found.**

All 15 SIR architecture modules are present and fully implemented (no stubs):

| SIR Module | Implementation Location | Status |
|---|---|---|
| InputEmbedding | `models/embeddings.py::TokenEmbedding` | ✓ Full |
| OutputEmbedding | `models/embeddings.py::TokenEmbedding` (shared) | ✓ Full |
| PositionalEncoding | `models/embeddings.py::PositionalEncoding` | ✓ Full |
| EncoderStack | `models/encoder.py::Encoder` | ✓ Full |
| EncoderLayer | `models/encoder.py::EncoderLayer` | ✓ Full |
| MultiHeadSelfAttention_Encoder | `models/attention.py::MultiHeadAttention` | ✓ Full |
| FeedForward_Encoder | `models/feedforward.py::PositionwiseFeedForward` | ✓ Full |
| DecoderStack | `models/decoder.py::Decoder` | ✓ Full |
| DecoderLayer | `models/decoder.py::DecoderLayer` | ✓ Full |
| MaskedMultiHeadSelfAttention_Decoder | `models/attention.py::MultiHeadAttention` + mask | ✓ Full |
| MultiHeadCrossAttention_Decoder | `models/attention.py::MultiHeadAttention` | ✓ Full |
| FeedForward_Decoder | `models/feedforward.py::PositionwiseFeedForward` | ✓ Full |
| LayerNorm | `torch.nn.LayerNorm` (inline) | ✓ Full |
| PreSoftmaxLinear | `models/transformer.py::output_projection` | ✓ Full |
| Softmax | Applied in `evaluation/metrics.py` (inference only) | ✓ Full |

All 6 SIR equations are implemented:

| Equation | Paper Reference | Implementation |
|---|---|---|
| Scaled dot-product attention | Eq. 1, §3.2.1 | `attention.py::ScaledDotProductAttention` |
| Multi-head attention | §3.2.2 | `attention.py::MultiHeadAttention` |
| Position-wise FFN | Eq. 2, §3.3 | `feedforward.py::PositionwiseFeedForward` |
| Sinusoidal PE | §3.5 | `embeddings.py::PositionalEncoding` |
| LR schedule | Eq. 3, §5.3 | `training/trainer.py::WarmupRsqrtScheduler` |
| Residual sublayer | §3.1 | Inline in `EncoderLayer` / `DecoderLayer` |

---

## Overall Assessment

The generated implementation is **high-fidelity**. All structural components are
present, all specified equations are correctly implemented (verified by sanity checks),
and all deviations are documented with `# ASSUMED` comments and confidence scores
from the SIR. The four parametric findings represent the irreducible uncertainty
in reproducing a paper that does not fully specify its implementation.

**Estimated impact on BLEU** if all assumptions are wrong simultaneously: −0.4 to −0.8 BLEU.
With correct hyperparameters and full training (100k steps, 8×P100, 25k tokens/batch),
achieving 27.0–27.5 BLEU on EN-DE newstest2014 with the base model is a reasonable
expectation.
