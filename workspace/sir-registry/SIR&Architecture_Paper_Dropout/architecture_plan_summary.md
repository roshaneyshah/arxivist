# Architecture Plan — Dropout Reproduction
**Paper:** Dropout: A Simple Way to Prevent Neural Networks from Overfitting (Srivastava et al., JMLR 2014)  
**paper_id:** `paper_dropout_srivastava2014`  
**Plan version:** 1 | **Generated:** 2026-06-19

---

## 1. Framework & Environment

| Decision | Choice | Reason |
|---|---|---|
| Framework | **PyTorch ≥ 2.1.0** | Contemporary standard for DL; CUDA-C original replaced |
| Python | 3.10+ | Type hints, dataclasses, modern stdlib |
| CUDA | 11.8 (min) | RTX 3050 compatible; paper required GPU |
| Config mgmt | YAML + dataclasses | Sufficient scope; no Hydra overhead |

> **⚠ Critical convention note:** PyTorch `nn.Dropout(p)` treats `p` as the DROP probability and scales activations by `1/(1-p)` at training time (inverted dropout). The paper defines `p` as the RETENTION probability and scales weights DOWN at test time. These are mathematically equivalent, but `nn.Dropout` must receive `(1 - p_paper)` as its argument. This is the #1 potential bug in any Dropout paper implementation.

---

## 2. Module Hierarchy

```
paper-repos/paper_dropout_srivastava2014/
├── train.py                          ← CLI: training entrypoint
├── evaluate.py                       ← CLI: evaluation on test set
├── inference.py                      ← CLI: single-image prediction
├── requirements.txt
├── requirements-dev.txt
├── environment.yaml
│
├── configs/
│   ├── mnist_3layer_1024.yaml        ← PRIMARY TARGET (Table 2, 1.06% error)
│   ├── mnist_ablation_dropout_rate.yaml
│   ├── mnist_ablation_dataset_size.yaml
│   └── mnist_regularizer_comparison.yaml
│
├── src/dropout_repro/
│   ├── models/
│   │   ├── dropout_net.py            ← DropoutNet (primary model)
│   │   └── dropout_rbm.py            ← DropoutRBM (Section 8, bonus)
│   ├── data/
│   │   ├── dataset.py                ← MNISTDataModule
│   │   └── transforms.py             ← normalization pipelines
│   ├── training/
│   │   ├── trainer.py                ← Trainer (SGD, max-norm, checkpointing)
│   │   └── losses.py                 ← cross_entropy_loss
│   ├── evaluation/
│   │   └── metrics.py                ← error_rate, sparsity_stats
│   └── utils/
│       ├── config.py                 ← DropoutConfig dataclass
│       └── max_norm.py               ← apply_max_norm_constraint()
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── data/                             ← MNIST downloads here
├── checkpoints/                      ← saved model weights
├── logs/                             ← training metrics (JSONL)
└── notebooks/
    └── dropout_repro.ipynb           ← Stage 5 output
```

---

## 3. Tensor Flow

### Training Forward Pass (DropoutNet, 3-layer, 1024 units)

```
x:  [B, 784]  ← raw MNIST pixels, normalized to mean=0.1307, std=0.3081

→ InputDropout (p_retain=0.8, drops 20%)
x:  [B, 784]

→ Linear(784→1024) + ReLU
x:  [B, 1024]

→ HiddenDropout_0 (p_retain=0.5, drops 50%)
x:  [B, 1024]

→ Linear(1024→1024) + ReLU
x:  [B, 1024]

→ HiddenDropout_1 (p_retain=0.5)
x:  [B, 1024]

→ Linear(1024→1024) + ReLU
x:  [B, 1024]

→ HiddenDropout_2 (p_retain=0.5)
x:  [B, 1024]

→ Linear(1024→10)   [NO dropout on output]
logits: [B, 10]

→ CrossEntropyLoss(logits, y)
loss: scalar
```

### After Every optimizer.step() — Max-Norm Projection
```
For each hidden Linear layer (not output):
  W: [D_out, D_in]
  norms = L2_norm(W, dim=1)        → [D_out]
  scale = clamp(c=2.0 / norms, max=1.0)
  W = W * scale.unsqueeze(1)       [in-place]
```

### Test Time (model.eval())
```
All nn.Dropout masks disabled automatically.
No manual weight scaling needed (inverted dropout handles this).
Forward pass identical to training structure, but no masking.
```

---

## 4. Key Hyperparameters (from `configs/mnist_3layer_1024.yaml`)

| Param | Value | SIR Confidence | Source |
|---|---|---|---|
| `hidden_dims` | [1024, 1024, 1024] | 0.95 | Table 2, Appendix B.1 |
| `p_hidden` | 0.5 | 0.95 | Appendix B.1, Section 7.3 |
| `p_input` | 0.8 | 0.95 | Appendix B.1 |
| `max_norm_c` | 2.0 | 0.95 | Appendix B.1 |
| `momentum` | 0.95 | 0.95 | Appendix B.1 |
| `n_weight_updates` | 1,000,000 | 0.95 | Appendix B.1 |
| `activation` | relu | 0.95 | Table 2 (ReLU row) |
| `learning_rate` | 0.01 | **0.65 ⚠ ASSUMED** | Not stated for MNIST |
| `batch_size` | 128 | **0.65 ⚠ ASSUMED** | Not stated in paper |

---

## 5. Reproduction Targets

| Experiment | Config File | Paper Result | Section |
|---|---|---|---|
| **PRIMARY: 3-layer 1024 ReLU + max-norm** | `mnist_3layer_1024.yaml` | **1.06% error** | Table 2 |
| Dropout rate sweep (Fig 9a) | `mnist_ablation_dropout_rate.yaml` | U-shaped curve, flat 0.4≤p≤0.8 | Sec 7.3 |
| Dataset size effect (Fig 10) | `mnist_ablation_dataset_size.yaml` | Dropout helps most at mid-range sizes | Sec 7.4 |
| Regularizer comparison (Table 9) | `mnist_regularizer_comparison.yaml` | Dropout+max-norm = 1.05% best | Sec 6.5 |

---

## 6. Risk Register

| ID | Severity | Issue | Mitigation |
|---|---|---|---|
| RISK-01 | **Medium** | LR not stated for MNIST (assumed 0.01) | YAML param; include lr sweep |
| RISK-02 | **Medium** | Batch size not stated (assumed 128) | YAML param; document assumption |
| RISK-03 | Low | PyTorch dropout convention inversion | Unit test + prominent docstring |
| RISK-04 | Low | 1M updates = long training on CPU | `--quick-run` flag for 100K updates |
| RISK-05 | Low | 0.79% result requires DBM pretraining | Out of scope; primary target is 1.06% |
| RISK-06 | Low | Max-norm for conv layers | FC-only for primary target; extensible design |

---

## 7. Test Plan

| Test | Type | Assertion |
|---|---|---|
| `test_dropout_mask_scale` | Unit | Train mode output mean ≈ eval mode output mean (law of large numbers) |
| `test_max_norm_projection` | Unit | All hidden weight row norms ≤ c after projection |
| `test_forward_shape` | Unit | forward(rand(32,784)) → shape [32,10] |
| `test_no_dropout_baseline` | Unit | use_dropout=False → identical outputs on two passes |
| `test_mnist_convergence_smoke` | Integration | 5K updates → test error < 5% |

---

## 8. Docker Spec

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
# System deps: git, wget
# pip install -r requirements.txt
# WORKDIR /workspace
# CMD: python train.py --config configs/mnist_3layer_1024.yaml
```

---

*Ready for Stage 4 — Code Generator*
