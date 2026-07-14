# benchmark_comparison.md
**ArXivist Stage 6 — Benchmark Comparison**  
**Paper:** Srivastava et al. (2014) "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"  
**paper_id:** `paper_dropout_srivastava2014`  
**Generated:** 2026-06-20

---

## Table 2 — MNIST Classification Error (Permutation-Invariant Setting)

| Model | Architecture | Paper Error % | Repro Error % | Δ (pp) | Status |
|---|---|---|---|---|---|
| Standard NN (Simard et al. 2003) | 2 layers, 800 units | 1.60 | — | — | Out of scope |
| SVM (Gaussian kernel) | — | 1.40 | — | — | Out of scope |
| Dropout NN, logistic | 3 layers, 1024 units | 1.35 | — | — | Out of scope |
| Dropout NN, ReLU | 3 layers, 1024 units | 1.25 | — | — | Out of scope |
| **Dropout NN + max-norm, ReLU** | **3 layers, 1024 units** | **1.06** | **1.12** | **+0.06** | **✓ PASS** |
| Dropout NN + max-norm, ReLU | 2 layers, 4096 units | 1.01 | — | — | Out of scope |
| Dropout NN + max-norm, ReLU | 2 layers, 8192 units | 0.95 | — | — | Out of scope |
| Dropout NN + max-norm, Maxout | 2 layers, 5×240 units | 0.94 | — | — | Out of scope |
| DBN + dropout finetuning | 500-500-2000 | 0.92 | — | — | Out of scope (pretraining) |
| DBM + dropout finetuning | 500-500-2000 | 0.79 | — | — | Out of scope (pretraining) |

**Primary target result: 1.12% reproduced vs 1.06% paper — Δ = +0.06pp ✓ PASS**

---

## Table 9 — Regularizer Comparison on MNIST (Section 6.5)

Architecture: 784-1024-1024-2048-10, ReLU

| Method | Paper Error % | Repro Error % | Δ (pp) | Status |
|---|---|---|---|---|
| L2 | 1.62 | — | — | Not run |
| L2 + L1 | 1.60 | — | — | Not run |
| L2 + KL-sparsity | 1.55 | — | — | Not run |
| Max-norm | 1.35 | — | — | Not run |
| Dropout + L2 | 1.25 | — | — | Not run |
| Dropout + Max-norm | 1.05 | — | — | Not run |

> Run `python train.py --config configs/mnist_regularizer_comparison.yaml` to populate this table.

---

## Other Paper Benchmarks (All Out of Scope for This Reproduction)

| Dataset | Model | Paper Result | Status |
|---|---|---|---|
| SVHN | Conv Net + dropout all layers | 2.55% error | Out of scope |
| CIFAR-10 | Conv Net + dropout all layers | 12.61% error | Out of scope |
| CIFAR-100 | Conv Net + dropout all layers | 37.20% error | Out of scope |
| ImageNet ILSVRC-2010 | Conv Net + dropout (AlexNet) | 17.0% top-5 error | Out of scope |
| ImageNet ILSVRC-2012 | Avg 5 Conv Nets + dropout | 16.4% top-5 error | Out of scope |
| TIMIT | DBN-pretrained NN + dropout | 19.7% phone error | Out of scope |
| Reuters-RCV1 | Dropout NN | 29.62% error | Out of scope |
| Alternative Splicing | Dropout NN | 567 code quality | Out of scope |

---

## Summary

| Metric | Value |
|---|---|
| Metrics reproduced | 1 / 9 (primary MNIST target) |
| Primary result | **1.12% vs 1.06% paper (+0.06pp)** |
| Tolerance | ±0.30pp |
| Primary target status | **✓ PASS** |
| Training steps | 1,000,000 / 1,000,000 |
| Training time | 851.3 min on RTX 3050 6GB + i5-13420HX |
| Framework | PyTorch 2.5.1+cu121, CUDA 12.1 |
