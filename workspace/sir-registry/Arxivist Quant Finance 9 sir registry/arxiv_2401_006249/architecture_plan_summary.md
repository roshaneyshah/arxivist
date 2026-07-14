# SpotV2Net — Architecture Plan Summary

**Paper ID:** `arxiv_2401_006249`
**Framework:** PyTorch 2.1 + PyTorch Geometric 2.4 (config via plain YAML)

## Why PyTorch Geometric
The paper's core contribution is an edge-feature-aware Graph Attention Network. PyG's
`MessagePassing` base class is the domain-standard way to implement custom attention that
consumes both node and edge tensors, so it was selected even though the paper itself does not
name a framework (confidence 0.6, flagged as an assumption).

## Module Map

| File | Implements (SIR module) |
|---|---|
| `models/gat_layer.py` | Edge-aware GAT attention (`EdgeAwareGATLayer`) |
| `models/spotv2net.py` | Full 2-hidden-layer GAT stack + prediction head, with `use_edge_features` toggle for the SpotV2Net-NE ablation |
| `models/har_spot.py` | Panel HAR-Spot baseline (App. A.1) |
| `models/lstm_baseline.py` | LSTM baseline (App. A.3) |
| `data/fourier_estimators.py` | Malliavin-Mancino Fourier spot vol / co-vol / vol-of-vol / co-vol-of-vol estimators (Sec. 6) |
| `data/graph_builder.py` | Turns Fourier estimate panels into per-timestamp node/edge feature tensors (Eq. 1-2) |
| `data/dataset.py` | `SpotVolGraphDataset` — train/val/test splits per Table 1 |
| `data/transforms.py` | Feature standardization + jump filter (footnote 7, β=0.5, α=0.5) |
| `training/losses.py` | MSE, `QLIKELoss` (Eq. 4/5/7/8) |
| `training/trainer.py` | Training loop, checkpointing, seeding |
| `evaluation/metrics.py` | MSE/QLIKE aggregation, Diebold-Mariano test |
| `evaluation/explainer.py` | GNNExplainer wrapper (Sec. 7.3) |
| `utils/config.py` | YAML config loader + global seeding utility |

## Forward Pass (single-step)

```
x: [30, M] float32           ← node features (contemporaneous + L=42 lags)
edge_attr: [870, E] float32  ← edge features (contemporaneous + L=42 lags)
x = EdgeAwareGATLayer_1(x, edge_index, edge_attr) → [30, 400], ReLU   (heads concatenated)
x = EdgeAwareGATLayer_2(x, edge_index, edge_attr_proj) → [30, 200], ReLU (heads averaged)
y_hat = Linear(200 → 1)(x) → [30, 1]
```

Multi-step (functional) forecast is identical except the final linear layer outputs `[30, 14]`
directly — no recursion, matching the paper's stated non-recursive multi-step capability.

## Config System
Plain YAML, five top-level blocks: `model`, `training`, `data`, `evaluation`, `hardware`.
Every hyperparameter pulled from Table 8 (SpotV2Net optimal values) is set directly; anything the
paper does not disclose (LR schedule, gradient clipping, AdamW betas/weight-decay, random seed)
is set to a PyTorch/library default and annotated `# ASSUMED` in the generated `config.yaml`.

## Dependencies
Core: `torch`, `torch-geometric`, `numpy`, `pandas`, `scikit-learn`, `xgboost`, `scipy`, `optuna`,
`pyyaml`, `matplotlib`. Dev: `pytest`, `black`, `flake8`, `jupyter`.

## Entrypoints
- `train.py --config configs/config.yaml [--resume] [--seed] [--debug] [--dry-run]`
- `evaluate.py --config ... --checkpoint ... [--split test|validation]`
- `inference.py --config ... --checkpoint ... --input snapshot.npz`

## Docker
Base image `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime`; `git`, `wget` system deps;
default CMD prints `train.py --help`.

## Top Risks
1. **[High]** Exact Fourier cutting frequencies are deferred to an external MATLAB library not
   reproduced numerically in the paper — estimators are shipped fully configurable with
   literature-typical defaults and a calibration note in `data/README_data.md`.
2. **[High]** Raw TAQ tick data (WRDS) is proprietary — `data/download.py` checks for a
   user-supplied extract and falls back to a synthetic data generator for smoke tests.
3. **[Medium]** LR schedule / gradient clipping / AdamW betas undisclosed — config-driven with
   `# ASSUMED` comments, PyTorch defaults used.
4. **[Medium]** Multi-head concat vs. per-head dimensionality is ambiguous relative to the
   Table 8 `[400, 200]` hidden-dim list — treated as *total* post-concat/post-average dim per
   layer (standard PyG `GATConv` convention), documented in `spotv2net.py`.
5. **[Low]** GNNExplainer's internal mask-optimization hyperparameters are undisclosed — PyG
   defaults used, documented in `explainer.py`.
