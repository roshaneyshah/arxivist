# Verification Log — Stage 6 Results Comparator

**Paper ID**: paper_chunking_methods_rag_2026
**ArXiv**: 2606.00881
**Comparison Run Timestamp**: 2026-06-24T00:00:00Z
**ArXivist Stage**: 6 (Results Comparator)
**SIR Version Used**: 1
**Architecture Plan Version**: 1

---

## Input Artifacts

| Artifact | Path | SHA256 (first 16) |
|---|---|---|
| User results JSON | `results/sandbox_run_results.json` | `473dbf1f4611e5a1` |
| SIR | `sir-registry/paper_chunking_methods_rag_2026/sir.json` | — |
| Architecture plan | `sir-registry/.../architecture_plan.json` | — |
| Pipeline state | `sir-registry/.../pipeline_state.json` | — |

---

## Paper Metrics Retrieved from SIR

Source: `sir.json → evaluation_protocol.chunking_methods_evaluated`
Dataset used for comparison: SQuAD (paper Table 1 / Table 2)

| # | Method | Metric | Paper Value |
|---|---|---|---|
| 1 | fixed_size | accuracy_at_5 | 96.24% |
| 2 | fixed_size | recall_at_10 | 81.81% |
| 3 | recursive_semantic | accuracy_at_5 | 96.87% |
| 4 | recursive_semantic | recall_at_10 | 85.81% |
| 5 | sequential_hac | accuracy_at_5 | 94.60% |
| 6 | sequential_hac | recall_at_10 | 79.84% |
| 7 | max_min | accuracy_at_5 | 96.24% |
| 8 | max_min | recall_at_10 | 84.24% |

**Paper metrics found**: 8 / 8 targeted
**User results matched**: 8 / 8
**Unmatched**: 0

Note: Paper reports 88 total experimental configurations (11 datasets × 8 methods). This run covers 8 of 88 (SQuAD only, 4 robust methods). Coverage = 9.1%.

---

## User-Reported Configuration

| Parameter | User Value | Paper Value | Match? |
|---|---|---|---|
| Dataset | SQuAD validation (100 docs, 200 queries) | SQuAD validation (100 docs) | ✓ Same split |
| Embedding model | TF-IDF dim=512 | BAAI/bge-m3 | ✗ Substituted |
| Reranker | None | BAAI/bge-reranker-v2-m3 | ✗ Missing |
| Fixed-size chunk_size | 512 | 512 | ✓ |
| Fixed-size overlap | 50 | 50 | ✓ |
| HAC threshold | 0.85 (assumed) | Not stated in paper | ⚠ Assumed |
| Max-min alpha | 0.5 (assumed) | Not stated in paper | ⚠ Assumed |
| HAC/Max-min internal embedder | 3-sentence proxy | BAAI/bge-m3 | ✗ Substituted |
| Timeout | 48h | 48h | ✓ |
| Accuracy@k | k=5 | k=5 | ✓ |
| Recall@k | k=10 | k=10 | ✓ |
| Seed | 42 | Not stated | — |

**Config modifications declared by user**: Embedder and reranker substituted due to HuggingFace network restriction in sandbox environment.

---

## Metric Matching Log

All 8 comparisons matched by (method_name, metric_name, dataset) triple.
No fuzzy matching required. No ambiguous metric names encountered.

---

## Deviation Computation Audit

Formula: `pct_deviation = (user_value - paper_value) / paper_value × 100`
Severity thresholds: ≤2% Excellent | 2–5% Good | 5–15% Moderate | 15–30% Significant | >30% Critical

| Method | Metric | user | paper | abs_Δ | pct_Δ | severity |
|---|---|---|---|---|---|---|
| fixed_size | acc@5 | 82.50 | 96.24 | −13.74 | −14.28% | Moderate |
| fixed_size | rec@10 | 90.00 | 81.81 | +8.19 | +10.01% | Moderate |
| recursive_semantic | acc@5 | 85.00 | 96.87 | −11.87 | −12.25% | Moderate |
| recursive_semantic | rec@10 | 90.50 | 85.81 | +4.69 | +5.47% | Moderate |
| sequential_hac | acc@5 | 76.50 | 94.60 | −18.10 | −19.13% | Significant |
| sequential_hac | rec@10 | 85.50 | 79.84 | +5.66 | +7.09% | Moderate |
| max_min | acc@5 | 76.50 | 96.24 | −19.74 | −20.51% | Significant |
| max_min | rec@10 | 85.50 | 84.24 | +1.26 | +1.50% | **Excellent** |

---

## Reproducibility Score Computation

```
base_score              = mean(1 - min(|pct_dev|/50, 1.0)) over 8 metrics
                        = (0.7144 + 0.7998 + 0.7550 + 0.8906 + 0.6174 + 0.8582 + 0.5898 + 0.9700) / 8
                        = 0.7744

sir_confidence_penalty  = (1 - mean([0.99, 0.99, 0.99, 0.99, 0.90, 0.88, 0.83])) × 0.15
                        = (1 - 0.9386) × 0.15
                        = 0.0614 × 0.15
                        = 0.0092

unmatched_penalty       = (0 / 8) × 0.2 = 0.0000

reproducibility_score   = 0.7744 − 0.0092 − 0.0000 = 0.7652
```

**Final score: 0.765** | **Confidence: Medium**

Score confidence rationale: Medium (not High) because:
- Only 1 of 11 datasets run (9.1% coverage)
- Embedder substituted — primary retrieval component differs from paper
- Experiment 2 not run — Table 4 entirely unverified
- HAC and Max-min internal chunking used proxy logic

---

## Hallucination Check Summary

| Type | Found | Critical |
|---|---|---|
| Structural | 0 | 0 |
| Parametric | 2 (PH-01 HAC threshold, PH-02 Max-min alpha) | 0 |
| Omission | 3 (OH-01 stubs, OH-02 TextTiling approx, OH-03 EXP2 not run) | 0 |

No Critical hallucinations. All Significant parametric hallucinations have suggested fixes.

---

## Stage 6 Output Checklist

- [x] `benchmark_comparison.md` — full comparison table with root cause analysis
- [x] `reproducibility_score.json` — all fields populated, score computation documented
- [x] `hallucination_report.md` — all three hallucination types checked, 5 items found
- [x] `verification_log.md` — this file
- [x] `metadata.json` — will be updated: `has_comparison_report: true`
- [x] All Moderate/Significant deviations have root cause analysis
- [x] All Significant hallucinations have suggested fixes
- [x] No Critical hallucinations found — no mandatory re-implementation required
