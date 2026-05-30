# FS-GCLSTM: Full-State Graph Convolutional LSTM for Stock Return Prediction
### Liu (2023/2025) — arXiv:2303.09406

Reproducible PyTorch implementation of FS-GCLSTM, which integrates value-chain (supplier-customer) graph structure into all components of an LSTM cell for stock return forecasting.

---

## What the Paper Does

Represents firms as graph nodes connected by LSEG supply-chain edges, then applies a Graph Convolutional LSTM where the key innovation is applying GCN to **all** LSTM inputs — including the cell state $c_{t-1}$ — rather than only the input features $X_t$.

**Results**: Highest annualized return (7.41% Eurostoxx, 9.79% S&P 500), Sharpe, and Sortino ratios across all baselines, despite not having the lowest MSE/MAE.

---

## Repository Structure

```
src/fsgclstm/
  models/
    gcn_layer.py          # GCN layer: Eq. (1)-(2), Section III.a
    fsgclstm_cell.py      # FS-GCLSTM cell: Eqs. (3)-(6), Section III.b  ← core
    fsgclstm_model.py     # Full model: 3 stacked cells + MLP, Section III.c
    baselines.py          # FCL and LSTM baselines, Section IV.D
  data/
    graph_builder.py      # Adjacency matrix from LSEG data
    dataset.py            # Rolling-window dataset + synthetic generator
  training/
    losses.py             # MSE loss (ASSUMED)
    trainer.py            # Adam + OneCycleLR + early stopping
  evaluation/
    metrics.py            # MSE, MAE, Sharpe, Sortino, portfolio return
  utils/
    config.py             # Config loading + reproducibility seeds
configs/config.yaml       # All hyperparameters with confidence annotations
```

---

## Quick Start (No LSEG Data Required)

```bash
pip install -r requirements.txt
python train.py --config configs/config.yaml --synthetic
jupyter notebook notebooks/reproduce_arxiv_2303_09406.ipynb
```

### Docker
```bash
docker-compose -f docker/docker-compose.yml run train
```

---

## Expected Results

| Dataset | Metric | FS-GCLSTM | Best Baseline | Benchmark |
|---------|--------|-----------|---------------|-----------|
| Eurostoxx 600 | Ann. Return | **7.41%** | 6.24% (FCL) | 5.03% (const.) |
| Eurostoxx 600 | Sharpe | **0.462** | 0.394 (GConvGRU) | 0.310 |
| S&P 500 | Ann. Return | **9.79%** | 8.95% (GConvGRU) | 8.52% (const.) |
| S&P 500 | Sharpe | **0.608** | 0.549 (LSTM) | 0.529 |

---

## Key Implementation Assumptions

| Parameter | Value | Confidence | Basis |
|-----------|-------|------------|-------|
| `hidden_dim` | 64 | ⚠️ 0.45 | Not stated — common default |
| MLP architecture | 2-layer | ⚠️ 0.50 | Not stated |
| Training loss | MSE | 0.70 | Paper evaluates MSE but doesn't state training loss |
| `n_gcn_layers` | 2 | ✓ 0.99 | Explicitly stated: "two GCN layers" |
| `n_lstm_layers` | 3 | ✓ 0.99 | Explicitly stated: "three stacked FS-GCLSTM cells" |
| lr / weight_decay | 0.001 / 1e-5 | ✓ 0.99 | Explicitly stated |

---

## Citation

```bibtex
@article{liu2023exploiting,
  title={Exploiting Supply Chain Interdependencies for Stock Return Prediction: A Full-State Graph Convolutional LSTM},
  author={Liu, Chang},
  journal={arXiv preprint arXiv:2303.09406},
  year={2023}
}
```
