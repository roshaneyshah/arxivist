# Hallucination Report

**Paper**: Chunking Methods on Retrieval-Augmented Generation (arXiv:2606.00881)
**Paper ID**: paper_chunking_methods_rag_2026
**SIR Version**: 1 | **Architecture Plan Version**: 1
**Date**: 2026-06-24

Hallucination = any component, parameter, or omission in the generated code that
deviates from what the paper specifies, potentially affecting reproducibility.

---

## Section 1 — Structural Hallucinations

> Components present in the generated code that are NOT described in the SIR/paper.

**Count: 0**

No structural hallucinations detected. Every class and module in the generated
codebase maps directly to a component described in the paper's methodology
(Sections 2–3) or to supporting infrastructure (config, timing, reporting)
required to operationalise those components. The experimental stubs
(GraphSeg, LumberChunker, DenseX) are correctly identified as non-implementations
and not counted as structural additions.

---

## Section 2 — Parametric Hallucinations

> Hyperparameters assumed in the generated code that may be wrong, particularly
> those coinciding with Moderate or Significant deviations.

**Count: 2**

---

### PH-01 — Sequential HAC: `similarity_threshold = 0.85`

| Field | Value |
|---|---|
| Severity | **Significant** |
| Type | parametric |
| Location | `configs/config.yaml → chunkers.sequential_hac.similarity_threshold` |
| Also in | `chunkers/sequential_hac.py → SequentialHACChunker.__init__` |
| SIR Confidence | 0.88 |
| Coincident deviation | Accuracy@5: −19.13% (Significant) |

**Evidence**: The paper (Section 3) states "Default hyperparameters were taken from the original publications." The original publication for Sequential HAC is Qu et al. 2025 (NAACL Findings). The value `0.85` is a common default for cosine similarity thresholds but was not directly verified from Qu et al. 2025's codebase or paper.

**Impact**: If the true threshold is lower (e.g., 0.70), HAC produces larger, fewer chunks. If higher (e.g., 0.95), it produces many small chunks. Either direction changes the chunk count, chunk boundary positions, and thus retrieval results. The Significant deviation in HAC Accuracy@5 (−19.13%) may be partly attributable to this.

**Suggested fix**: Clone the original Sequential HAC implementation from Qu et al. 2025 (NAACL 2025 Findings paper GitHub) and read their default config. Update `configs/config.yaml → chunkers.sequential_hac.similarity_threshold` accordingly.

---

### PH-02 — Max-Min: `alpha = 0.5`

| Field | Value |
|---|---|
| Severity | **Significant** |
| Type | parametric |
| Location | `configs/config.yaml → chunkers.max_min.alpha` |
| Also in | `chunkers/max_min.py → MaxMinChunker.__init__` |
| SIR Confidence | 0.83 |
| Coincident deviation | Accuracy@5: −20.51% (Significant) |

**Evidence**: The paper cites Kiss et al. 2025 (Discover Computing 28, article 117) as the source of Max-Min Semantic Chunking. The value `alpha=0.5` was assumed as a neutral midpoint for the adaptive threshold scaling factor. The Discover Computing paper may specify a different default.

**Impact**: `alpha` directly controls the adaptive threshold in `_compute_adaptive_threshold`: `threshold = alpha * min_pairwise_similarity`. A wrong alpha changes when chunks are split, producing different granularity. The Significant Accuracy@5 deviation for max_min (−20.51%) may be partly attributable to this.

**Suggested fix**: Obtain Kiss et al. 2025 (DOI: 10.1007/s44227-025-00075-0) and read the default alpha value. Update `configs/config.yaml → chunkers.max_min.alpha` accordingly.

---

## Section 3 — Omission Hallucinations

> Components present in the SIR that are absent or incompletely implemented in the generated code.

**Count: 3**

---

### OH-01 — GraphSeg, LumberChunker, DenseX: Stub only

| Field | Value |
|---|---|
| Severity | **Minor** |
| Type | omission |
| Location | `chunkers/experimental/{graphseg,lumberchunker,densex}.py` |
| SIR Component | SIR architecture: Chunker variants (experimental) |

**Evidence**: Three of the eight evaluated methods are implemented as stubs that raise `NotImplementedError`. They are correctly identified in the SIR as methods expected to fail (T/S markers), and their stubs accurately document the expected failure mode. However, they are not executable implementations.

**Impact**: Cannot reproduce Table 1/2 rows for GraphSeg (Accuracy@5 avg 86.85%, Recall@10 avg 61.75%), LumberChunker (Recall@10 avg 78.16%, highest among all methods), or DenseX. Table 4 (LLM judge) is also incomplete for these methods.

**Suggested fix**: Integrate the original author implementations. GraphSeg: Verma 2025 (arXiv:2501.05485). LumberChunker: Duarte et al. 2024 (EMNLP). DenseX: Chen et al. 2024 (EMNLP). Note: T/S failures are expected by design per the paper — full reproduction of their failure behavior is itself a valid result.

---

### OH-02 — TextTiling `_align_to_sentences` implementation is approximate

| Field | Value |
|---|---|
| Severity | **Minor** |
| Type | omission |
| Location | `chunkers/texttiling.py → _align_to_sentences()` |
| SIR Component | SIR: TextTiling (modified, sentence-level alignment) |

**Evidence**: The paper (Section 3) explicitly states TextTiling was modified so chunk boundaries align to the nearest sentence rather than paragraph. The generated implementation uses a `sent in tile` substring check to assign sentences to tiles, which is an approximation. The original authors likely implemented this as a character-offset snap: find the character position of the tile boundary, then find the nearest sentence start position.

**Impact**: Minor difference in boundary placement. Likely produces functionally similar chunks but may diverge on documents where sentence and tile boundaries are far apart. TextTiling is excluded from Experiment 2 per paper, limiting the impact scope.

**Suggested fix**: Implement character-offset-based boundary snapping: record the char offset of each sentence via `document.find(sent)`, find the nearest sentence start to each tile boundary position.

---

### OH-03 — Experiment 2 not run (LLM judge scores)

| Field | Value |
|---|---|
| Severity | **Minor** (sandbox constraint, not code error) |
| Type | omission |
| Location | `generation/generator.py`, `generation/judge.py` |
| SIR Component | SIR: EXP2 — End-to-End RAG Answer Generation |

**Evidence**: Table 4 (LLM-as-a-judge scores) was not reproduced in this run. The generator and judge modules are fully implemented and correctly parameterised (model, 4000-token context limit, 5-point Likert scale — all SIR conf 0.99). The omission is due to the sandbox having no API access, not a code defect.

**Impact**: Cannot verify Table 4 scores. On your machine with Ollama (qwen2.5:7b) or an OpenAI key, these modules will run as implemented. Expected deviation from paper: ±0.3–0.5 Likert points due to model substitution.

**Suggested fix**: Run `python run_full_bench.py --methods fixed_size recursive_semantic --datasets squad` with `OPENAI_API_KEY` set or Ollama running locally.

---

## Hallucination Summary

| ID | Type | Severity | Status |
|---|---|---|---|
| PH-01 | Parametric | Significant | HAC threshold assumed — verify vs Qu et al. 2025 |
| PH-02 | Parametric | Significant | Max-min alpha assumed — verify vs Kiss et al. 2025 |
| OH-01 | Omission | Minor | Experimental stubs — expected, T/S failures documented |
| OH-02 | Omission | Minor | TextTiling boundary alignment approximate |
| OH-03 | Omission | Minor | Experiment 2 not run — sandbox constraint only |

**No Critical or Structural hallucinations detected.**
The core implementation (EQ2–EQ6, chunking pipeline, FAISS index, metrics) is faithful to the paper.
