# Landmark Paper Reproductions

A collection of mechanism-faithful, runnable reproductions of landmark ML papers,
generated and scored by [ArXivist](https://github.com/qosi-org/arxivist).

Each subfolder contains:
- A Jupyter notebook reproducing the paper's core mechanism
- ArXivist Stage-6 comparison reports (benchmark, hallucination, verification)
- Result images from the reproduction run
- A reproducibility score against scaled paper targets

---

## Papers

| Folder | Paper | Year | Score | Confidence |
|--------|-------|------|-------|------------|
| [ddpm_ho_et_al_2020_arxiv_2006_11239](./ddpm_ho_et_al_2020_arxiv_2006_11239/) | Denoising Diffusion Probabilistic Models — Ho, Jain, Abbeel | 2020 | **0.92** | Medium |

---

## Reproducibility Score Guide

| Score | Classification | Meaning |
|-------|---------------|---------|
| 0.90–1.00 | Excellent | Mechanism correct, results match scaled targets |
| 0.75–0.89 | Good | Minor deviations, likely training variance |
| 0.60–0.74 | Moderate | Implementation differences present |
| < 0.60 | Poor | Fundamental issue — review required |

## Confidence Levels

- **High** — ≥3 metrics matched, full training budget
- **Medium** — 1–2 metrics matched, or partial training
- **Low** — no direct metric match, or substantial config changes

---

## Adding a New Paper

Run ArXivist on any arXiv paper:

```
/arxivist https://arxiv.org/abs/YYMM.NNNNN
```

Then run the generated notebook, upload your results, and trigger Stage 6:

```
/arxivist i ran this
```

The Stage-6 output goes into a new subfolder here.
