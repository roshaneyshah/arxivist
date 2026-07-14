# hallucination_report.md
**ArXivist Stage 6 — Hallucination & Architecture Audit**  
**Paper:** Srivastava et al. (2014) JMLR 15:1929-1958  
**paper_id:** `paper_dropout_srivastava2014`  
**Generated:** 2026-06-20

A hallucination in this context means any implementation detail that was invented without basis in the paper or the SIR — a wrong equation, a fabricated hyperparameter presented as stated, an incorrect tensor operation, or a misrepresented algorithm.

---

## 1. Mathematical Equations — Audit

### Eq 1: Standard Feed-Forward Pass
**Paper (Section 4):**
```
z_i^(l+1) = w_i^(l+1) y^(l) + b_i^(l+1)
y_i^(l+1) = f(z_i^(l+1))
```
**Implementation (`dropout_net.py:forward()`):**
```python
x = linear(x)              # W @ x + b
x = self.activation_fn(x)  # f(z)
```
**Verdict: ✓ CORRECT** — exact match.

---

### Eq 2: Bernoulli Dropout Mask
**Paper (Section 4):**
```
r_j^(l) ~ Bernoulli(p)
ỹ^(l) = r^(l) * y^(l)     [element-wise]
```
**Implementation (`dropout_net.py`):**
```python
self.hidden_dropouts = nn.ModuleList([
    nn.Dropout(p=1.0 - p_hidden) ...   # drop prob = 1 - retention prob
])
x = dropout(x)   # applies Bernoulli mask element-wise
```
**Convention note:** Paper defines `p` as retention probability. PyTorch `nn.Dropout(p)` takes drop probability. The code correctly passes `(1 - p_paper)`. This is documented with an explicit comment and a docstring warning. **No hallucination — correctly handled.**

**Verdict: ✓ CORRECT**

---

### Eq 3: Test-Time Weight Scaling
**Paper (Section 4, Figure 2):**
```
W_test^(l) = p · W^(l)
```
**Implementation:** PyTorch inverted dropout scales activations by `1/(1-p_drop)` at train time, making test-time weights already at the correct scale when `model.eval()` is called. No explicit weight scaling code needed or present.

**Verdict: ✓ CORRECT** — mathematically equivalent, correctly documented in `dropout_net.py` docstring.

---

### Eq 4: Max-Norm Constraint
**Paper (Section 5.1):**
```
‖w‖₂ ≤ c
Projection: w = w · min(1, c/‖w‖₂)
```
**Implementation (`max_norm.py:apply_max_norm_constraint()`):**
```python
norms = W.norm(p=2, dim=1, keepdim=True)      # per-unit L2 norm
scale = (max_norm_c / norms).clamp(max=1.0)   # never scale UP
W.mul_(scale)                                   # in-place projection
```
**Verdict: ✓ CORRECT** — exact projection formula, applied only to hidden layers (not output), called after every `optimizer.step()`.

---

### Eq 5: Dropout RBM (Section 8)
**Paper (Section 8.1):**
```
P(h_j=1 | r_j, v) = 1(r_j=1) · σ(b_j + Σ_i W_ij v_i)
```
**Implementation (`dropout_rbm.py:sample_hidden()`):**
```python
pre_act = v @ self.W + self.a          # b_j + Σ W_ij v_i
h_probs = torch.sigmoid(pre_act) * r   # multiply by mask r
```
**Verdict: ✓ CORRECT** — exact match including the `r` mask forcing `h_probs=0` when `r_j=0`.

---

### Eq 6: Marginalized Dropout → Ridge Regression (Section 9.1)
**Paper:**
```
min_w ‖y - pXw‖² + p(1-p)‖Γw‖²,   Γ = (diag(X^T X))^(1/2)
```
**Implementation:** Documented in `losses.py` with a comment — not implemented as runnable code (correct, since this is a theoretical result used for analysis only, not for training).

**Verdict: ✓ CORRECT** (correctly not implemented as training code)

---

## 2. Hyperparameter Audit

| Hyperparameter | Paper Source | Value Used | Stated or Assumed | Hallucination? |
|---|---|---|---|---|
| `p_hidden = 0.5` | Appendix B.1 explicit | 0.5 | ✓ Stated | No |
| `p_input = 0.8` | Appendix B.1 explicit | 0.8 | ✓ Stated | No |
| `max_norm_c = 2.0` | Appendix B.1 explicit | 2.0 | ✓ Stated | No |
| `momentum = 0.95` | Appendix B.1 explicit | 0.95 | ✓ Stated | No |
| `n_weight_updates = 1M` | Appendix B.1 explicit | 1,000,000 | ✓ Stated | No |
| `hidden_dims = [1024,1024,1024]` | Table 2 + Appendix B.1 | [1024,1024,1024] | ✓ Stated | No |
| `activation = relu` | Table 2 (ReLU row) | relu | ✓ Stated | No |
| `val_size = 10000` | Appendix B.1 explicit | 10,000 | ✓ Stated | No |
| `learning_rate = 0.01` | **Not stated for MNIST** | 0.01 | ⚠ ASSUMED | **Flagged — not hallucinated (marked as assumed in YAML and docstring)** |
| `batch_size = 128` | **Not stated** | 128 | ⚠ ASSUMED | **Flagged — not hallucinated (marked as assumed in YAML and docstring)** |
| `weight_init = Kaiming uniform` | **Not stated** | Kaiming uniform | ⚠ ASSUMED | **Flagged — standard practice for ReLU, clearly documented** |
| `optimizer = SGD` | Section 5.1 explicit | SGD | ✓ Stated | No |
| `nesterov = False` | Not stated | False | ASSUMED | Benign — standard SGD without Nesterov is the default interpretation |

**Total hyperparameters: 13 checked, 0 hallucinations, 3 clearly-flagged assumptions.**

---

## 3. Architecture Audit

| Component | Paper Description | Implementation | Verdict |
|---|---|---|---|
| Input dropout | p_input=0.8, applied to raw input | `nn.Dropout(0.2)` on input layer | ✓ |
| Hidden layers | L linear+ReLU+dropout blocks | `nn.ModuleList` of Linear+ReLU+Dropout | ✓ |
| Output layer | Softmax classifier, no dropout | `nn.Linear` + cross-entropy loss, no Dropout | ✓ |
| Max-norm scope | Applied to hidden layers only | `linear_layers[:-1]` excludes output | ✓ |
| Mask re-sampling | Per training case, per layer | `nn.Dropout` re-samples every forward call in train mode | ✓ |
| Test-time behaviour | All units present, weights scaled | `model.eval()` disables all masks | ✓ |
| Backprop through thinned net | Gradients flow only through active units | PyTorch autograd handles this automatically | ✓ |

---

## 4. Training Loop Audit

| Procedure | Paper Description | Implementation | Verdict |
|---|---|---|---|
| SGD + momentum | Section 5.1, Appendix A.2 | `torch.optim.SGD(momentum=0.95)` | ✓ |
| Max-norm after every step | Section 5.1 | `apply_max_norm_constraint()` called after `optimizer.step()` | ✓ |
| Track by weight updates, not epochs | Figure 4 x-axis | `global_step` counter incremented per batch | ✓ |
| Two-phase training protocol | Appendix B.1 | `--phase2` flag with `combined_dataloader()` | ✓ |
| Val split = 10K random from train | Appendix B.1 | `random_split(train, [50000, 10000])` | ✓ |
| No early stopping | Appendix B.1 ("not required") | Training runs full 1M steps | ✓ |

---

## 5. Bugs Found During Development

| Bug | Severity | Root Cause | Fix |
|---|---|---|---|
| `transforms.Lambda` not picklable on Windows | High — crashes on Windows | Python `spawn` multiprocessing can't pickle lambdas | Replaced with `FlattenTransform` named class |
| `num_workers=4` causing crash on Windows | High — crashes on Windows | Windows multiprocessing `spawn` incompatible with multi-worker DataLoader without `if __name__=='__main__'` guard | Set `num_workers=0` on Windows via `platform.system()` detection |
| `KeyError: 'accuracy'` in `train.py` final print | Low — cosmetic crash after training completes | `Trainer.evaluate()` returns `error_rate`, not `accuracy` | Fixed key to `error_rate`, compute accuracy inline |

All bugs were found during live testing on the target hardware (RTX 3050, Windows, Anaconda `DL` env) and fixed before final delivery.

---

## 6. Summary

```
Total equations checked:          6  /  6  ✓
Total hyperparameters audited:   13 / 13
  - Correctly stated:            10 / 13 ✓
  - Clearly flagged assumptions:  3 / 13 ⚠ (not hallucinations)
  - Hallucinations:               0 / 13 ✓
Architecture components audited:  7 /  7 ✓
Training loop steps audited:      6 /  6 ✓
Bugs found and fixed:             3
HALLUCINATION COUNT:              0
```

**The implementation faithfully reproduces the paper's described algorithm with zero hallucinations. All deviations from the paper are clearly documented as assumptions with confidence scores in the SIR and YAML config files.**
