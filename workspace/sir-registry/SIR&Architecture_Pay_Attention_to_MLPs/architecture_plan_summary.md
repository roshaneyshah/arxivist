# Architecture Plan — "Pay Attention to MLPs" (gMLP)
**paper_id**: `arxiv_2105_08050` | **Plan v1** | 2026-06-11

---

## 1. Framework

| Choice | Value | Reason |
|--------|-------|--------|
| Framework | PyTorch ≥ 2.1.0 | Paper uses TF/TPU; PyTorch is standard for open reproductions |
| Python | 3.10+ | |
| CUDA | ≥ 11.8 | Required for bfloat16 and flash-attention compat |
| HuggingFace | Yes (Transformers + Datasets) | C4/GLUE/SQuAD data loading; SentencePiece tokenizer |
| Config | YAML + dataclass | Lightweight, no Hydra dependency |

---

## 2. Module Hierarchy

```
gmlp/
├── models/
│   ├── toeplitz.py       ← ToeplitzLinear: W parameterized as vector w∈R^{2n-1}
│   ├── tiny_attn.py      ← TinyAttention: single-head, d_attn ∈ {64,128}
│   ├── sgu.py            ← SpatialGatingUnit + aMLP_SGU (hybrid)
│   ├── gmlp_block.py     ← gMLPBlock: full residual block
│   ├── patch_embed.py    ← PatchEmbedding: ViT 16×16 stem (vision)
│   └── gmlp.py           ← gMLP: top-level model (NLP + Vision unified)
├── data/
│   ├── mlm_dataset.py    ← C4 + BERT 15% masking
│   ├── glue_dataset.py   ← SST-2 / MNLI
│   ├── squad_dataset.py  ← SQuAD v1.1 / v2.0
│   ├── imagenet_dataset.py
│   └── transforms.py     ← AutoAugment, Mixup, CutMix
├── training/
│   ├── trainer_nlp.py    ← MLM pretraining loop
│   ├── trainer_vision.py ← ImageNet training loop
│   ├── finetuner.py      ← GLUE + SQuAD finetuning
│   ├── lr_schedulers.py
│   └── losses.py
├── evaluation/
│   ├── metrics.py        ← perplexity, accuracy, F1, MNLI-acc
│   └── evaluator.py      ← median-of-5-runs wrapper
└── utils/
    ├── config.py         ← gMLPConfig dataclass
    ├── checkpointing.py
    ├── logging_utils.py
    └── distributed.py
```

---

## 3. Core Forward Pass (gMLP NLP)

```
input_ids [B, n]
    ↓ token embedding
x [B, n, d_model]
    ↓ × L blocks:
    ┌─ shortcut = x
    │  x_pre = LayerNorm(x)          [B, n, d_model]
    │  x = x_pre @ U                 [B, n, d_ffn]       ← expand
    │  x = GeLU(x)
    │  ── SGU ──
    │  z1, z2 = split(x)             [B, n, d_ffn/2] each
    │  z2 = LayerNorm(z2)
    │  z2 = ToeplitzLinear(z2)       [B, n, d_ffn/2]     ← W∈R^{n×n} (Toeplitz)
    │  x = z1 * z2                   [B, n, d_ffn/2]     ← gate
    │  ── end SGU ──
    │  x = x @ V                     [B, n, d_model]     ← contract
    └─ x = x + shortcut
    ↓ LayerNorm
    ↓ lm_head (weight-tied)
logits [B, n, vocab_size]
```

**aMLP delta** (inside SGU, replaces plain ToeplitzLinear gate):
```
spatial_gate = ToeplitzLinear(z2)          [B, n, d_ffn/2]
attn_gate    = TinyAttention(x_pre) @ Wo   [B, n, d_ffn/2]
gate         = spatial_gate + attn_gate    ← additive fusion ⚠ TODO:verify
x            = z1 * gate
```

**Vision** replaces token embedding with PatchEmbedding(16×16) and the final head with global avg pool → Linear(1000). Toeplitz constraint is **disabled** for vision (W is free [196×196]).

---

## 4. Key Implementation Notes

### ToeplitzLinear
- Parameterize as `w ∈ R^{2n-1}`
- At forward: construct W via padding+tile+reshape (see paper Appendix C TF code → translate to PyTorch)
- W init: `normal(std=0.002)` ⚠ **ASSUMED** — not stated in paper
- b init: `ones`
- Shared across all channels (same W for every channel of Z2)

### SGU Initialization Goal
At the start of training: `f_{W,b}(Z) ≈ 1` → `s(Z) ≈ Z` → each block acts like a plain FFN. This gradually relaxes as W learns. Critical for stable training.

### Stochastic Depth
- Vision only: per-block DropPath with survival_prob from Table 1
- NLP: disabled (survival_prob=1.0)

### Padding Tokens (NLP)
- No masking of `<pad>` tokens in gMLP blocks needed — model naturally ignores them
- Padding mask still passed for loss computation (don't compute loss on pad tokens)

---

## 5. Model Size Presets

### NLP (Full BERT Setup)
| Model | L | d_model | d_ffn | Params | d_attn |
|-------|---|---------|-------|--------|--------|
| gMLPbase | 48 | 512 | 3072 | 130M | — |
| aMLPbase | 36 | 512 | 3072 | 109M | 64 |
| gMLPlarge | 96 | 768 | 3072 | 365M | — |
| aMLPlarge | 72 | 768 | 3072 | 316M | 128 |
| gMLPxlarge | 144 | 1024 | 4096 | 941M | — |

### Vision (ImageNet)
| Model | L | d_model | d_ffn | Params | Stoch Depth |
|-------|---|---------|-------|--------|-------------|
| gMLP-Ti | 30 | 128 | 768 | 5.9M | 1.00 |
| gMLP-S | 30 | 256 | 1536 | 19.5M | 0.95 |
| gMLP-B | 30 | 512 | 3072 | 73.4M | 0.80 |

---

## 6. Risk Register (summary)

| # | Severity | Issue | Mitigation |
|---|----------|-------|------------|
| R1 | 🔴 High | C4 full dataset is ~300GB | Use streaming mode; ablation config uses RealNews subset |
| R2 | 🔴 High | Full training needs 100+ A100-GPU-hours | Ablation config (125K steps, seq=128) is default |
| R3 | 🟡 Medium | aMLP fusion mechanism ambiguous (conf=0.75) | Config flag `fusion_mode`; default `add`; TODO:verify |
| R4 | 🟡 Medium | W init std unspecified (conf=0.65) | Default 0.002; monitor gate norms early training |
| R5 | 🟡 Medium | W is O(n²) per layer | Toeplitz constraint mandatory for NLP; warn at n>512 |
| R6 | 🟡 Medium | Vision pooling ambiguous (conf=0.70) | Default global avg pool; CLS token as config fallback |
| R7 | 🟢 Low | SentencePiece vocab not open-sourced | Use T5/mT5 32K SentencePiece tokenizer as proxy |
| R8 | 🟢 Low | Stochastic depth NLP behavior | Disabled (p=0) for all NLP configs |

---

## 7. Entrypoints

```bash
python train.py        --config configs/gmlp_base_mlm.yaml       # NLP pretrain
python train_vision.py --config configs/gmlp_s_imagenet.yaml     # Vision
python finetune.py     --pretrained_checkpoint ckpt/ --task sst2  # Finetune
python evaluate.py     --checkpoint ckpt/ --task squad_v2        # Eval
python inference.py    --checkpoint ckpt/ --task mlm --input "The [MASK] sat on the mat."
```

---

## 8. Docker

```
Base:  pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
CMD:   python train.py --config configs/gmlp_base_mlm.yaml
Vols:  /data → /workspace/gmlp/data
       /outputs → /workspace/gmlp/outputs
```

---

*Overall plan confidence: **0.93** — Ready for Stage 4 (Code Generator)*
