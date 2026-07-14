# Architecture Plan Summary — arxiv_2605_17307

**Paper:** *Deep Reinforcement Learning Framework for Diversified Portfolio Management Across Global Equity Markets*
**Authors:** Kamil Kashif, Robert Ślepaczuk
**Plan version:** 1 · **Created:** 2026-05-24

---

## 1. Framework choice

- **Language:** Python 3.10
- **DL framework:** **PyTorch 2.x** (custom SAC + Dirichlet policy heads + LSTM/Transformer encoders).
- **RL library:** Custom SAC implementation (stable-baselines3 lacks first-class support for Dirichlet action distributions).
- **Environment:** [`gymnasium`](https://gymnasium.farama.org/) `Env` subclass.
- **Config:** plain YAML (one file per `(model_config, market)` combination).
- **CUDA:** required for training; CPU OK for inference demos.

## 2. Repository layout

```
workspace/paper-repos/arxiv_2605_17307/
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── environment.yaml          # conda alternative
├── pyproject.toml            # build/install config
├── train.py                  # main entrypoint
├── evaluate.py               # metrics + statistical tests
├── inference.py              # single-date weight generation
├── configs/                  # one YAML per (config × market)
│   ├── lstm_1_ndx.yaml
│   ├── lstm_2_ndx.yaml
│   ├── lstm_nc_1_ndx.yaml
│   ├── lstm_nc_2_ndx.yaml
│   ├── transformers_ndx.yaml
│   └── ... (× nky, sx5e)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── data/
│   ├── download.sh           # yfinance bulk download
│   └── membership/
│       └── README.md         # how to drop historical-membership CSVs
├── scripts/
│   └── run_walk_forward.py   # multi-fold orchestrator
├── notebooks/
│   └── reproduce_arxiv_2605_17307.ipynb
├── src/portfolio_rl/
│   ├── __init__.py
│   ├── data/                 # download, features, top-k, membership
│   ├── envs/                 # PortfolioEnv (Gymnasium)
│   ├── models/
│   │   ├── encoders/         # LSTM, Transformer, cross-sectional attention
│   │   ├── policies/         # FlatDirichlet, HierarchicalDirichlet
│   │   └── critics/          # TwinCritic
│   ├── agents/               # SACAgent + ReplayBuffer
│   ├── training/             # walk-forward, adaptive retraining
│   ├── evaluation/           # performance metrics, HAC, bootstrap
│   └── utils/                # config, logging, seeding
├── runs/                     # training outputs (gitignored)
├── checkpoints/              # model weights (gitignored)
├── results/                  # CSV/JSON evaluation outputs
└── comparison/               # reproduced vs reported (Stage 6, future)
```

## 3. Tensor flow (one env step)

```
raw_prices (N_total, T_full)
    │
    ▼  FeatureExtractor (RSI, MACD, %B, momentum, volatility, beta, …)
asset_features (N_t, T=60, F_asset≈15)        global_features (F_global≈7)
    │                                              │
    ▼  TopKSelector(k ∈ {10,20,30})                │
topk_features (k, 60, F_asset)                     │
    │                                              │
    ▼  LSTMEncoder | TransformerEncoder            │
asset_embeddings (B, k, H)                         │
    │                                              ▼
    └──► CrossSectionalAttention ◄── GlobalFeatureProjector
              │
              ▼
       state_repr (B, H_state)
              │
              ▼  FlatDirichlet | HierarchicalDirichlet
       action_weights (B, k+1)   # last slot = cash
              │
              ▼  PortfolioEnv.step
       reward (Eq. 16 or 17), next_state
```

## 4. Hyperparameters (Table 4–6 of paper)

| Group | Param | Value |
|---|---|---|
| SAC | actor LR | 3e-4 |
| SAC | critic LR | 5e-4 |
| SAC | entropy LR | 3e-4 |
| SAC | γ | 0.99 |
| SAC | τ | 0.005 |
| SAC | α (entropy) | 0.2 (fixed) |
| SAC | batch | 128 |
| SAC | replay capacity | 20 000 |
| SAC | warm-up steps | 500 |
| SAC | gradient steps / env step | 2 |
| Encoder | LSTM hidden | 64 or 128 |
| Encoder | Transformer hidden | 128 (2 layers) |
| Env | lookback | 60 days |
| Env | txn cost | 2 bps |
| Env | λ_turnover | 0.003 |
| Env | λ_concentration | {0.0, 0.1, 0.5} |
| Env | epochs | 50 |
| Env | early-stop patience | 8 |
| WFO | train / val / test | 5 / 1 / 1 years |
| WFO | folds | 16 (2009-04 → 2026-03) |
| WFO | adaptive retrain m | 5 |
| WFO | max folds w/o retrain | 3 |

## 5. Key risks

1. **Index membership data is paywalled (Bloomberg).** Reproduction falls back to current constituents → survivorship bias. See `data/membership/README.md`.
2. **Compute cost is huge** (≈3 400+ GPU-h on NVIDIA L4 for full grid). Use `--quick-test` flag for development.
3. **Hierarchical policy equity-vs-cash distribution** assumed Beta; configurable.
4. **Critic architecture** assumed `[256, 256]` MLP (SAC default).
5. **Cross-sectional attention** variant assumed scaled dot-product (single layer).

## 6. Entrypoints

```bash
# Full training (one config × one market, ~14-23h on L4)
python train.py --config configs/lstm_2_ndx.yaml --seed 42

# Quick smoke test
python train.py --config configs/lstm_2_ndx.yaml --quick-test

# Evaluation after training
python evaluate.py --run-dir runs/lstm_2_ndx_seed42 --benchmark QQQ

# Inference
python inference.py --checkpoint checkpoints/lstm_2_ndx_fold16.pt --date 2026-03-13
```

## 7. Overall confidence: **0.85**

Lower confidence regions (will be flagged in code with `# ASSUMED:` comments):
- `models/policies/hierarchical.py` — equity/cash split distribution
- `models/encoders/attention.py` — cross-sectional attention variant
- `models/critics/twin_critic.py` — MLP depth/width
- `data/membership/__init__.py` — survivorship-bias fallback
