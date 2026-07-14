# Architecture Plan — Physics Informed Neural Networks (PINNs)
**Paper**: Raissi, Perdikaris, Karniadakis (arXiv:1711.10561v1)
**Generated**: 2026-06-29 | Plan v1

---

## Framework
**PyTorch ≥ 2.0**, Python 3.10+, CUDA 11.8+
- `torch.autograd.grad` replaces TF1's `tf.gradients` for PDE derivative computation
- `torch.optim.LBFGS` with `strong_wolfe` line search is the primary optimizer
- `scipy.polynomial.legendre.leggauss` computes Gauss-Legendre RK tableau nodes

---

## File Structure

```
paper-repos/paper_pinns_raissi2017/
├── src/pinns/
│   ├── models/
│   │   ├── mlp.py                  ← Shared MLP backbone (Xavier init, tanh)
│   │   ├── continuous_pinn.py      ← ContinuousPINN: u(t,x) + autograd residual f(t,x)
│   │   └── discrete_pinn.py        ← DiscretePINN: multi-output MLP + implicit RK
│   ├── pde/
│   │   └── operators.py            ← BurgersOperator, SchrodingerOperator, AllenCahnOperator
│   ├── data/
│   │   ├── sampling.py             ← LatinHypercubeSampler, PDEDataset
│   │   └── exact_solutions.py      ← Reference solution loaders/generators
│   ├── training/
│   │   ├── losses.py               ← ContinuousPINNLoss, DiscretePINNLoss
│   │   └── trainer.py              ← Trainer (L-BFGS closure loop)
│   ├── evaluation/
│   │   └── metrics.py              ← RelativeL2Error
│   └── utils/
│       ├── rk_tableau.py           ← GaussLegendreTableau ⚠ HIGH RISK
│       ├── autograd_helpers.py     ← DerivativeComputer
│       └── config.py               ← YAML config loader
├── configs/
│   ├── burgers_continuous.yaml
│   ├── burgers_discrete.yaml
│   ├── schrodinger.yaml
│   └── allen_cahn.yaml
├── data/
│   └── generate_reference.py       ← Spectral reference solutions (Schrodinger, Allen-Cahn)
├── train.py
├── evaluate.py
├── requirements.txt
├── environment.yaml
├── docker/Dockerfile
└── README.md
```

---

## The Two PINN Variants

### Variant 1 — Continuous-time PINN

```
(t, x) → [concat] → MLP(2→1, 9L×20N, tanh) → u_pred
                                  ↓ autograd
                            u_t, u_x, u_xx
                                  ↓
                         BurgersOperator → f_pred
                                  ↓
              MSE_u (data fit) + MSE_f (physics) → L-BFGS
```

**Key insight**: `f_pred` has no new parameters — it's a deterministic function of the MLP weights via the autograd graph. MSE_f acts as physics regularization.

### Variant 2 — Discrete-time PINN (Implicit RK)

```
x → MLP(1→q+1, 4L×50N, tanh) → [u^{n+c1},...,u^{n+cq}, u^{n+1}]
                                          ↓
                              apply N[·] per stage (autograd w.r.t. x)
                                          ↓
                              apply RK equations (9) with Gauss-Legendre A, b
                                          ↓
                           u^n_pred [Nn, q+1]  vs  u_data [Nn, 1]
                                          ↓
                                    SSE_n + SSE_b → L-BFGS
```

**Key insight**: Network takes only `x` as input (no `t`). The timestep Δt and RK tableau encode all temporal dynamics. At q=500 stages, theoretical error is O(Δt^1000) — essentially machine-precision temporal integration.

---

## Benchmark Configurations

| Experiment | Variant | Architecture | Data | Target L2 |
|---|---|---|---|---|
| Burgers (continuous) | Continuous | 9L × 20N | Nu=100, Nf=10k | 6.7×10⁻⁴ |
| Schrödinger | Continuous | 5L × 100N, out_dim=2 | N0=50, Nb=50, Nf=20k | 1.97×10⁻³ |
| Burgers (discrete) | Discrete RK | 4L × 50N, q=500 | Nn=250, Δt=0.8 | 8.2×10⁻⁴ |
| Allen-Cahn | Discrete RK | 4L × 200N, q=100 | Nn=200, Δt=0.8 | 6.99×10⁻³ |

---

## Risk Summary

| ID | Severity | Component | Issue |
|---|---|---|---|
| R1 | 🔴 High | `GaussLegendreTableau` | q=500 Vandermonde system — numerical conditioning |
| R2 | 🔴 High | `DiscretePINN` autograd | 500 derivative passes — must vectorize with `vmap` |
| R3 | 🟡 Medium | `torch.optim.LBFGS` | Subtle differences from scipy L-BFGS-B |
| R4 | 🟡 Medium | Weight init | Not specified (confidence 0.72); Xavier assumed |
| R5 | 🟡 Medium | Schrödinger reference | Chebfun (MATLAB) → scipy port; tiny differences possible |
| R6 | 🟢 Low | `create_graph=True` | Silently breaks if omitted; centralized in DerivativeComputer |

---

## Implementation Order Recommendation

1. `utils/rk_tableau.py` → unit-test Gauss-Legendre nodes at q=1,2,4 against known values
2. `models/mlp.py` → test forward pass shapes
3. `utils/autograd_helpers.py` → unit-test derivatives on sin(x)
4. `pde/operators.py` → validate Burgers residual = 0 on exact solution
5. `training/losses.py` + `training/trainer.py` → run Burgers continuous first (simplest)
6. `models/discrete_pinn.py` → start with q=1 (trapezoidal rule), validate, then scale to q=500
7. Schrodinger + Allen-Cahn after Burgers passes
