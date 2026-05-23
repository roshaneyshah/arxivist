# Results Comparator — ArXivist Stage 6
# Nevmyvaka, Feng, Kearns — ICML 2006

## Expected Results from Paper

### Section 4.1 — Private Variables Only

| Stock | H | V | S&L cost (bps) | RL T=4,I=4 | RL T=8,I=8 | Improvement range |
|-------|---|---|----------------|-----------|-----------|-------------------|
| AMZN | 8min | 5K | ~6–8 | lower | lower | — |
| AMZN | 2min | 5K | ~8–12 | lower | lower | — |
| NVDA | 8min | 5K | ~10–15 | lower | lower | — |
| QCOM | 8min | 5K | ~5–7 | lower | lower | — |

**Reported improvement: 27.16% – 35.50% (averaged across all settings)**

Exact per-stock/per-condition costs can be read from Figure 3 bars.

### Section 4.2 — Adding Market Variables (Table 1)

Additional improvement over private-variable-only RL (averaged across all stocks/sizes/horizons):

| Market Variable(s) | Additional Improvement |
|--------------------|----------------------|
| Bid-Ask Spread | 7.97% |
| Bid-Ask Volume Misbalance | 0.13% |
| Spread + Immediate Cost | 8.69% |
| Immediate Market Order Cost | 4.26% |
| Signed Transaction Volume | 2.81% |
| Spread + Imm Cost + Signed Vol | **12.85%** |

**Overall vs S&L (best config): ≥50% improvement**

---

## How to Use This Comparator

After a full training run, execute the evaluation script:

```bash
python evaluate.py --policy models/policy_final.pkl --config configs/config.yaml --output results/eval_results.json
```

Then compare `results/eval_results.json` against the targets above.

### Comparison Script

```python
import json

with open("results/eval_results.json") as f:
    results = json.load(f)

rl_vs_sl = results["rl_vs_sl"]
actual_improvement = rl_vs_sl["relative_improvement"]

# Paper benchmark: 0.2716 – 0.3550 for private-only
# Paper benchmark: >= 0.50 for best config with market vars
target_low, target_high = 0.2716, 0.3550

print(f"Actual RL vs S&L improvement: {actual_improvement:.2%}")
print(f"Paper target range:           {target_low:.2%} – {target_high:.2%}")

if actual_improvement >= target_low * 0.75:
    print("✓ PASS: Within 25% of paper's reported range")
else:
    print("✗ FAIL: Below 75% of paper's lower bound")
    print("  Likely causes:")
    print("  1. Synthetic data (not real INET) — expected")
    print("  2. Action space size L differs from paper assumption")
    print("  3. Insufficient training data or episodes")
    print("  4. Market variable binning differs from paper")
```

---

## Known Gaps vs Paper

| Gap | Severity | Impact |
|-----|----------|--------|
| INET data not available (synthetic used) | High | Cannot reproduce exact numbers |
| Action space size L assumed (conf. 0.60) | Medium | Could shift policy by a few bps |
| Market variable bin boundaries not specified | Medium | Affects feature quality |
| Exact episode partitioning algorithm not given | Low | Minor variation in train/test split |

---

## Qualitative Checks (should hold on any data)

These structural properties of the learned policy should reproduce regardless of data source:

1. **Aggressive-when-pressured**: Policy is more aggressive (higher action) when inventory is high AND time is low
2. **Monotone in inventory**: For fixed t, optimal action increases (more aggressive) as i increases
3. **Monotone in time**: For fixed i, optimal action increases as t decreases (less time remaining)
4. **Spread sensitivity**: Larger spread → more aggressive action (confirmed by Figure 6)
5. **Runtime independence**: Training time is nearly flat as R (market vars) increases — confirmed by Figure 2
6. **Liquidity ordering**: NVDA (least liquid) has highest costs; QCOM (most liquid) lowest

Run `notebooks/reproduce_paper_rl_trade_execution.ipynb` to verify checks 1-5 on synthetic data.
