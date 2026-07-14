# Results Comparison Template
## FS-GCLSTM — arXiv:2303.09406

Run `python train.py --synthetic` then fill in "Your Results" column.

| Metric | Paper (Eurostoxx) | Your Results | Match? |
|--------|-------------------|--------------|--------|
| Ann. Return % | 7.41 | TBD | ⏳ |
| Sharpe Ratio | 0.462 | TBD | ⏳ |
| Sortino Ratio | 0.592 | TBD | ⏳ |
| MSE | 4.236e-4 | TBD | ⏳ |

| Metric | Paper (S&P 500) | Your Results | Match? |
|--------|-----------------|--------------|--------|
| Ann. Return % | 9.79 | TBD | ⏳ |
| Sharpe Ratio | 0.608 | TBD | ⏳ |
| Sortino Ratio | 0.754 | TBD | ⏳ |

## Known Reproducibility Gaps
- LSEG value-chain data is proprietary (use `--synthetic` for testing)
- `hidden_dim` not stated in paper (assumed 64, conf: 0.45)
- MLP architecture not specified (conf: 0.50)
- Training loss not stated (assumed MSE, conf: 0.70)
