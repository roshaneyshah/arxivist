# Architecture Plan: DeepVol

**Paper**: DeepVol: volatility forecasting from high-frequency data with dilated causal convolutions  
**Paper ID**: `paper_deepvol_dilated_causal_convolutions`  
**Generated**: 2026-05-30

---

## 1. Framework Selection

- **Framework**: PyTorch (via PyTorch-Lightning) — explicitly stated in paper
- **Python**: 3.10+
- **CUDA**: Required (NVIDIA GPU; paper used Titan Xp). CPU fallback supported for inference.
- **Config management**: YAML + OmegaConf/dataclasses
- **HuggingFace**: Not applicable

---

## 2. Module Hierarchy

```
paper-repos/paper_deepvol_dilated_causal_convolutions/
├── src/
│   └── deepvol/
│       ├── __init__.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── deepvol.py              ← DeepVol top-level model (DeepVol, DeepVolRM)
│       │   ├── dcc_block.py            ← DilatedCausalConvBlock (single residual block)
│       │   ├── attention.py            ← Bahdanau attention mechanism
│       │   └── output_head.py          ← Output MLP head
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py              ← VolatilityDataset, intraday return sequences
│       │   └── transforms.py           ← log-return computation, normalisation utilities
│       ├── training/
│       │   ├── __init__.py
│       │   ├── losses.py               ← QLIKE, RMSE, MAE, SMAPE, ME, MedAE
│       │   └── trainer.py              ← LightningModule wrapping DeepVol
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── metrics.py              ← all 6 metrics + MCS test wrapper
│       └── utils/
│           ├── __init__.py
│           └── config.py               ← config loading, seed utility, device setup
├── configs/
│   └── config.yaml
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── data/
│   ├── README_data.md
│   └── download.py
├── notebooks/
│   ├── reproduce_paper_deepvol_dilated_causal_convolutions.ipynb
│   └── explore_paper_deepvol_dilated_causal_convolutions.ipynb
├── results/
├── comparison/
├── train.py
├── evaluate.py
├── inference.py
├── requirements.txt
├── requirements-dev.txt
├── environment.yaml
├── setup.py
└── README.md
```

---

## 3. Tensor Flow

### DeepVol Forward Pass

```
INPUT:  x [B, 1, T*J]   float32  ← raw intraday log-returns (T days × J intervals)
                                    T=1, J=78 for optimal 5-min, 1-day config

x = InputProjection(x)              → [B, residual_channels=32, T*J]

skip_outputs = []
for block in range(num_blocks=2):
    for layer in range(num_layers=6):
        d = 2 ** layer               # dilation doubles each layer
        residual, skip = DilatedCausalConvBlock(x, dilation=d)
        x = x + residual             # residual connection [B, 32, T*J]
        skip_outputs.append(skip)    # [B, skip_channels=128, T*J]

skip_sum = sum(skip_outputs)         → [B, 128, T*J]

context = BahdanauAttention(skip_sum) → [B, 128]   ← collapses time dimension

logit = OutputHead(context)           → [B, 1]     ← two-layer MLP

OUTPUT: sigma2_hat [B, 1]   float32  ← day-ahead realised variance forecast
```

---

## 4. Configuration Schema (key parameters from Table 1)

| Parameter | Value | Confidence | Note |
|---|---|---|---|
| learning_rate | 1e-3 | 1.0 | Table 1 |
| batch_size | 512 | 1.0 | Table 1 |
| num_epochs | 1000 | 1.0 | Table 1 |
| early_stopping_patience | 50 | 1.0 | Table 1 |
| kernel_size | 3 | 1.0 | Table 1 |
| sampling_freq | 5min | 1.0 | Table 1 |
| conditioning_range | 1 day | 1.0 | Table 1 |
| num_blocks | 2 | 1.0 | Table 1 |
| num_layers | 6 | 1.0 | Table 1 |
| residual_channels | 32 | 1.0 | Table 1 |
| dilation_channels | 64 | 1.0 | Table 1 |
| skip_channels | 128 | 1.0 | Table 1 |
| end_channels | 64 | 1.0 | Table 1 |
| loss_function | QLIKE | 1.0 | Table 1 |
| gated_activation | tanh*sigmoid | 0.78 | ASSUMED: WaveNet style |
| weight_decay | 0.0 | 0.80 | ASSUMED: not stated |

---

## 5. Risk Assessment

| Risk | Severity | Description | Mitigation |
|---|---|---|---|
| Proprietary data | High | NASDAQ-100 HF data not publicly available | Provide synthetic data generator for testing; add download README |
| Output head ambiguity | Medium | Exact post-attention MLP structure inferred, not stated | Implement as configurable; add TODO comment |
| Gated activation type | Low-Medium | WaveNet tanh*sigmoid assumed | Make activation configurable via config |
| Multi-asset collation | Low | Exact batching strategy across 90 stocks not described | Use asset-agnostic dataset with stock index as optional metadata |

---

## 6. Dependencies

**Core**: torch, pytorch-lightning, numpy, pandas, scipy, statsmodels, arch (GARCH baselines), omegaconf, tqdm  
**Dev**: pytest, black, isort, mypy, jupyter, ipywidgets, matplotlib, seaborn
