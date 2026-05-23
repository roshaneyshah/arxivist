# RL for Optimized Trade Execution

**ArXivist-generated reproduction repository**

> Nevmyvaka, Feng, Kearns — *"Reinforcement Learning for Optimized Trade Execution"*  
> ICML 2006 | [Paper PDF](https://icml.cc/imls/conferences/2006/proceedings/papers/166.pdf)

---

## What This Paper Does

Given a large block of shares (V) to sell (or buy) within a fixed time horizon H, find a policy that minimizes **trading cost** — the gap between achieved execution prices and the mid-spread at episode start (measured in basis points).

The key contribution is a **tabular backward-induction Q-learning** algorithm that:
1. Decomposes state into **private variables** (elapsed time `t`, remaining inventory `i`) and **market variables** (order book features)
2. Exploits the approximate independence between the two — market variables evolve without regard to our actions — making training O(T×I×L) **independent of the number of market features R**
3. Demonstrated on 1.5 years of millisecond-scale NASDAQ order book data for AMZN, NVDA, QCOM

**Results:** 27–50%+ improvement in execution cost over optimized Submit-and-Leave strategies.

---

## Repository Structure

```
rl_trade_execution/
├── src/rl_trade_execution/
│   ├── env/
│   │   ├── order_book.py        # Limit order book simulation (Section 2)
│   │   ├── market_env.py        # RL environment: state encoding, step(), reset()
│   │   └── market_features.py   # Market variable extraction + discretization (Section 4.2)
│   ├── agent/
│   │   ├── q_table.py           # Q-cost table with incremental averaging update
│   │   └── policy.py            # Greedy policy from trained Q-table
│   ├── training/
│   │   └── trainer.py           # Backward induction trainer (Algorithm, Section 3)
│   ├── baselines/
│   │   ├── submit_and_leave.py  # S&L baseline with optimized fixed limit offset
│   │   └── market_order.py      # Immediate market order baseline
│   ├── evaluation/
│   │   └── metrics.py           # Trading cost (bps), relative improvement
│   ├── data/
│   │   └── loader.py            # INET data loader + SyntheticOrderBookGenerator
│   └── utils/
│       └── config.py            # ExperimentConfig from YAML
├── configs/config.yaml          # All hyperparameters (annotated with confidence)
├── notebooks/
│   └── reproduce_paper_rl_trade_execution.ipynb   # Step-by-step walkthrough
├── comparison/
│   └── results_comparator.md    # Expected results and how to compare
├── data/
│   └── README_data.md           # Instructions for obtaining INET / LOBSTER data
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── train.py                     # Main training entrypoint
├── evaluate.py                  # Evaluation entrypoint
├── requirements.txt
└── setup.py
```

---

## Quick Start

### 1. Install

```bash
git clone <this-repo>
cd paper_rl_trade_execution
pip install -e .
```

### 2. Run with Synthetic Data (no real data needed)

```bash
python train.py --config configs/config.yaml --debug
```

This runs the full backward induction loop on 200 synthetic episodes (~seconds). Results will not match the paper but verify the algorithm works.

### 3. Run with Real Data

See `data/README_data.md` for data acquisition. Then:

```bash
# Place your data at data/raw/AMZN.csv, NVDA.csv, QCOM.csv
python train.py --config configs/config.yaml --stock AMZN
python evaluate.py --policy models/policy_final.pkl --config configs/config.yaml
```

### 4. Jupyter Notebook

```bash
pip install -r requirements-dev.txt
jupyter notebook notebooks/reproduce_paper_rl_trade_execution.ipynb
```

---

## Core Algorithm (Section 3)

**Update rule** (backward in time from t=T to t=0):

```
c(x, a) = n/(n+1) * c(x, a) + 1/(n+1) * [c_im(x, a) + min_p c(y, p)]
```

**Pseudocode:**
```
for t = T down to 0:
    for each order_book_snapshot in training_data:
        o1..oR = discretize_market_features(snapshot)
        for i = 0 to I:            # all inventory levels
            for a = 0 to L:        # all actions
                state = encode(t, i, o1..oR)
                immediate_cost, next_i = simulate_execution(snapshot, i, a)
                best_future = min_action Q[t+1, next_i, ...]
                Q[t, i, o1..oR, a] += incremental_update(immediate_cost + best_future)
```

**Complexity:** O(T × I × L) passes over data — does **not** depend on R (number of market variables).

---

## Key Configuration (configs/config.yaml)

```yaml
state_space:
  T: 8           # time resolution (decision points)
  I: 8           # inventory levels
  market_variables: [bid_ask_spread, immediate_market_order_cost]
  n_bins_market: 3   # ASSUMED: confidence 0.80

action_space:
  L: 21          # ASSUMED: confidence 0.60 (inferred from Figures 5/7)
  action_min: -6
  action_max: 14
```

See all annotated assumptions in `configs/config.yaml` and `comparison/results_comparator.md`.

---

## Expected Results

| Config | Improvement vs S&L |
|--------|-------------------|
| Private vars only (T=4, I=4) | 27.16% |
| Private vars only (T=8, I=8) | 35.50% |
| + Spread + ImmCost + SignedVol | **≥50%** |

---

## ⚠️ Data Availability

The INET ECN historical data used in the paper is **proprietary and not publicly available**. To reproduce the exact results, you need millisecond-resolution NASDAQ order book data. Options include [LOBSTER](https://lobsterdata.com) and [Nasdaq TotalView-ITCH](https://www.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHspecification.pdf). See `data/README_data.md` for full details.

---

## SIR Confidence Summary

| Component | Confidence |
|-----------|-----------|
| Architecture | 0.88 |
| Math spec | 0.94 |
| Training pipeline | 0.75 |
| Evaluation protocol | 0.93 |
| **Overall** | **0.83** |

*Generated by ArXivist v1.0 — 2026-05-20*
