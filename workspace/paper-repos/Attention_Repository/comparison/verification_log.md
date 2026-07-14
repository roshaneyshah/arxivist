# Verification Log — Audit Trail

**Paper ID**: arxiv_1706_03762  
**Paper Title**: Attention Is All You Need  
**Comparison Run**: 2026-05-31T00:00:00Z  
**ArXivist Version**: 1.0  
**Run Mode**: PRE-TRAINING DIAGNOSTIC (no user results)

---

## 1. Input Inventory

| Artifact | Path | Version | SHA / Status |
|---|---|---|---|
| SIR | `sir-registry/arxiv_1706_03762/sir.json` | v1 | Loaded OK |
| Architecture Plan | `sir-registry/arxiv_1706_03762/architecture_plan.json` | v1 | Loaded OK |
| User Results | — | — | **NOT SUBMITTED** |
| Config (base) | `configs/base.yaml` | — | Present |
| Config (big) | `configs/big.yaml` | — | Present |

---

## 2. Paper Metrics Extracted from SIR

Source: `sir.json → evaluation_protocol.reported_results`

| # | Metric | Dataset | Split | Model | Value | is_primary |
|---|---|---|---|---|---|---|
| 1 | BLEU | WMT 2014 EN-DE | newstest2014 | Transformer (big) | 28.4 | true |
| 2 | BLEU | WMT 2014 EN-DE | newstest2014 | Transformer (base) | 27.3 | false |
| 3 | BLEU | WMT 2014 EN-FR | newstest2014 | Transformer (big) | 41.8 | true |
| 4 | BLEU | WMT 2014 EN-FR | newstest2014 | Transformer (base) | 38.1 | false |
| 5 | F1 | WSJ Penn Treebank | Section 23 | Transformer (4L, semi-sup) | 92.7 | false |
| 6 | F1 | WSJ Penn Treebank | Section 23 | Transformer (4L, WSJ only) | 91.3 | false |

**Total paper metrics**: 6  
**User results matched**: 0 (no user results submitted)

---

## 3. Automated Sanity Check Log

Executed: 2026-05-31  
Environment: Python 3.10, PyTorch (CPU)  
Working directory: `paper-repos/arxiv_1706_03762/`

```
=== SANITY CHECK RESULTS ===
  ✓ ScaledDotProductAttention             PASS
  ✓ MultiHeadAttention                    PASS
  ✓ PositionwiseFeedForward               PASS
  ✓ PositionalEncoding_formula            PASS
  ✓ Encoder_N6                            PASS
  ✓ Decoder_N6                            PASS
  ✓ Transformer_forward                   PASS
  ✓ WeightTying_3way                      PASS
  ✓ LabelSmoothing                        PASS
  ✓ LRSchedule_Eq3                        PASS
  ✓ CausalMask                            PASS
  ✓ GradientFlow                          PASS
  ✓ ParamCount_sanity(45.0M)             PASS

13/13 checks passed
```

Notable checks:
- **LRSchedule_Eq3**: Peak LR at step 4000 matches `d_model^{-0.5} * 4000^{-0.5}` to 1e-8.
  Monotonically increasing over all 4000 warmup steps verified.
- **CausalMask**: Verified strictly lower-triangular (zero upper triangle to 1e-6 after softmax).
- **PositionalEncoding_formula**: PE(0,0)=sin(0)=0, PE(0,1)=cos(0)=1, PE(1,0)=sin(1)≈0.8415
  all match to 1e-4 precision.
- **GradientFlow**: Zero dead-gradient parameters across all named parameters.
- **WeightTying_3way**: `src_embedding.weight is output_projection.weight` = True (Python identity).

---

## 4. Hallucination Audit Summary

Source: `sir.json`, `architecture_plan.json`, all source files in `src/`

| Category | Checked | Found |
|---|---|---|
| Structural (invented modules) | All 13 arch plan modules vs SIR modules | 0 |
| Parametric (assumed hyperparams) | All 8 SIR assumptions + 4 SIR ambiguities | 4 |
| Omission (missing SIR components) | All 15 SIR modules, 6 SIR equations | 0 |

Parametric findings (detailed in `hallucination_report.md`):

| ID | Name | Severity | Confidence |
|---|---|---|---|
| P1 | 3-Way weight tying | Significant | 0.82 |
| P2 | Attention dropout | Minor | 0.78 |
| P3 | Xavier uniform init | Minor | 0.70 |
| P4 | PE addition ordering | Minor | 0.88 |

---

## 5. SIR Assumption Coverage

All 8 implementation assumptions from `sir.json` are reflected in the code:

| # | Assumption | Code location | `# ASSUMED` comment present |
|---|---|---|---|
| 1 | Xavier uniform init | `_init_weights()` in all modules | ✓ Yes |
| 2 | Attention dropout | `ScaledDotProductAttention.forward()` | ✓ Yes |
| 3 | PyTorch framework | All imports | ✓ Yes (in configs) |
| 4 | Padding mask applied | `MaskFactory.make_padding_mask()` | ✓ Yes (TODO comment) |
| 5 | Shared vocab encoder/decoder | `DataConfig.shared_vocab=True` | ✓ Yes (in YAML comment) |
| 6 | Cross-entropy with label smoothing | `LabelSmoothedCrossEntropy` | ✓ Yes |
| 7 | Autoregressive teacher forcing | `train_step()` in trainer | ✓ Yes (in comments) |
| 8 | Token-based batching 25k tokens | `TokenBatchSampler` | ✓ Yes |

---

## 6. User Config Modifications

No user results submitted. No config modifications reported.

---

## 7. Reproducibility Score Computation

Not computable (no user results). Formula for when results are submitted:

```
base_score = mean(1 - min(abs(pct_deviation_i) / 50, 1.0)  for all i matched)
sir_penalty = (1 - mean_sir_confidence) × 0.15
            = (1 - 0.93) × 0.15 = 0.0105
unmatched_penalty = (unmatched / total_paper_metrics) × 0.2

reproducibility_score = max(0, base_score - sir_penalty - unmatched_penalty)
```

SIR confidence penalty pre-computed: **−0.0105** (low penalty, high SIR confidence)

---

## 8. Audit Decisions Log

| Decision | Rationale |
|---|---|
| Run mode set to PRE-TRAINING DIAGNOSTIC | User confirmed no experimental results available |
| Sanity checks run on CPU | GPU not available in ArXivist environment; checks are shape/math only, not compute-dependent |
| Hallucination scan: structural = 0 | All generated modules match SIR 1:1; no extraneous components found |
| Parametric P1 rated Significant not Critical | Both 2-way and 3-way tying are plausible readings of Section 3.4; corroborating evidence from tensor2tensor supports 3-way |
| Parametric P3 (init) rated Minor not Significant | Paper makes no claim about initialisation; Xavier uniform is the community consensus for Transformers |

---

## 9. Next Actions

1. **User**: Run `prepare_data.py`, then `train.py --config configs/base.yaml`
2. **User**: Run `evaluate.py` after training completes
3. **User**: Submit results (BLEU + PPL + steps trained + hardware) to activate full comparison
4. **ArXivist Stage 6**: Will compute reproducibility score, fill metric table, produce root cause analysis for any deviations

---

## 10. Reproducibility Prediction

Based on sanity checks, hallucination audit, and literature context, the implementation
is predicted to reproduce the paper's base model result within:

| Scenario | Expected BLEU (EN-DE) | Confidence |
|---|---|---|
| Full training, correct config (8×P100, 100k steps, 25k tok/batch) | 26.8–27.5 | Medium-High |
| Partial training (50k steps) | 25.5–26.5 | Medium |
| Single GPU, fixed batch_size=32 | 20–24 | Low (under-training) |
| Weight tying off (2-way only) | 26.7–27.3 | Medium |

*Confidence is Medium-High rather than High because hardware, tokenization quality,
and exact data preprocessing can shift results by ±0.5 BLEU.*
