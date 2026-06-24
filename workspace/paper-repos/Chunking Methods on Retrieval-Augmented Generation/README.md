# RAG Chunking Benchmark

**Reproduction of:** "Chunking Methods on Retrieval-Augmented Generation – Effectiveness Evaluation Against Computational Cost and Limitations"  
**arXiv:** [2606.00881](https://arxiv.org/abs/2606.00881) | KES 2026  
**Authors:** Śmigielski et al., Wrocław University of Science and Technology

---

## Overview

This repository reproduces the evaluation framework from arXiv:2606.00881, which is the **first systematic comparison of a wide range of chunking methods** in RAG systems. The paper evaluates 8 chunking methods across 11 dataset configurations via two experiments:

- **Experiment 1 (Evidence Retrieval):** Accuracy@5 and Recall@10 using bge-m3 + bge-reranker-v2-m3
- **Experiment 2 (End-to-End RAG):** LLM-as-a-Judge (5-point Likert) using GPT-OSS-20B

**Key finding:** Simpler methods (Fixed-size, Recursive Semantic) match or exceed computationally expensive methods in practice, while being orders of magnitude faster and fully robust.

---

## Key Results to Reproduce

| Method | Accuracy@5 | Recall@10 | Avg Time | Robustness |
|---|---|---|---|---|
| **Recursive Semantic** | **89.36%** | 53.81% | 4.90m | ★★★ Full |
| Fixed-size | 87.71% | 44.75% | **<1s** | ★★★ Full |
| GraphSeg | 86.85% | **61.75%** | 3.09h | ★☆☆ S-failures |
| Max-min | 85.75% | 41.68% | 2.44m | ★★★ Full |
| TextTiling | 84.96% | 39.85% | 1.56m | ★★☆ EXP1 only |
| Sequential HAC | 80.09% | 33.85% | 2.10m | ★★★ Full |
| DenseX | 69.10% | 27.43% | 15.05h | ★☆☆ T-failures |

`T` = exceeded 48-hour time limit | `S` = spaCy memory error on large documents

---

## Quick Start

### 1. Environment Setup

```bash
# Option A: Conda (recommended)
conda env create -f environment.yaml
conda activate rag-chunking-bench

# Option B: pip
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab
python -m spacy download en_core_web_sm

# Option C: Docker
docker build -f docker/Dockerfile -t rag-chunking-bench .
docker run -v $(pwd)/data:/workspace/data -v $(pwd)/results:/workspace/results \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e OPENAI_API_BASE=$OPENAI_API_BASE \
  rag-chunking-bench
```

### 2. Install Package

```bash
pip install -e .
```

### 3. Configure API Access

The paper uses **GPT-OSS-20B** for generation (Experiment 2) and as judge. Substitute any OpenAI-compatible model:

```bash
export OPENAI_API_BASE="https://api.openai.com/v1"   # or your endpoint
export OPENAI_API_KEY="sk-..."
```

> **RISK-02 Note:** Exact reproduction of Table 4 requires access to GPT-OSS-20B. Any OpenAI-compatible model can be substituted via `configs/config.yaml → generation.model`.

### 4. Download Datasets

HuggingFace datasets (SQuAD, TriviaQA, NQ, Qasper) download automatically.

For manually-downloaded datasets (RISK-01):
```bash
python data/download_datasets.py --help
```

| Dataset | Source | Auto-download |
|---|---|---|
| SQuAD | HuggingFace | ✅ |
| TriviaQA | HuggingFace | ✅ |
| NQ | HuggingFace | ✅ |
| Qasper | HuggingFace | ✅ |
| PoQuAD | [PoQuAD GitHub](https://github.com/PoQuAD/PoQuAD) | ❌ Manual |
| GutenQA | [LumberChunker repo](https://github.com/avduarte333/LumberChunker) | ❌ Manual |
| LiteraryQA | [Babelscape/LiteraryQA](https://github.com/Babelscape/LiteraryQA) | ❌ Manual |
| NovelQA | [NovelQA GitHub](https://github.com/NovelQA/novelqa) | ❌ Manual |

### 5. Run Smoke Test (SQuAD only, 2 robust methods)

```bash
python run_full_bench.py \
  --config configs/config.yaml \
  --methods fixed_size recursive_semantic \
  --datasets squad \
  --skip_generation
```

Expected runtime: ~2-5 minutes on CPU.

### 6. Run Full Benchmark

```bash
# Experiment 1 only (no API key needed)
python run_full_bench.py --config configs/config.yaml --skip_generation

# Both experiments (requires API key)
python run_full_bench.py --config configs/config.yaml

# Resume interrupted run
python run_full_bench.py --config configs/config.yaml --resume
```

---

## Project Structure

```
paper_chunking_methods_rag_2026/
├── src/rag_chunking_bench/
│   ├── chunkers/          # 5 implementations + 3 experimental stubs
│   ├── embedding/         # bge-m3 embedder, bge-reranker-v2-m3
│   ├── retrieval/         # FAISS cosine index + retriever
│   ├── generation/        # GPT-OSS-20B generator + LLM judge
│   ├── evaluation/        # Accuracy@5, Recall@10, result reporter
│   ├── data/              # Dataset loaders (11 configurations)
│   ├── pipeline/          # Chunking (48h timeout) + eval orchestration
│   └── utils/             # Config, Timer, text utilities
├── configs/config.yaml    # All hyperparameters (SIR confidence annotated)
├── tests/                 # Unit tests for chunkers, metrics, utils
├── data/                  # Dataset files (populate via download scripts)
├── results/               # Benchmark outputs (Tables 1-4 equivalent)
├── run_full_bench.py      # Master entrypoint
├── requirements.txt
└── docker/Dockerfile
```

---

## Configuration

All hyperparameters are in `configs/config.yaml`. Parameters with `# ASSUMED` comments were inferred from cited papers — see the SIR for confidence levels.

**Hyperparameters explicitly stated in the paper (SIR conf ≥ 0.99):**

| Parameter | Value | Source |
|---|---|---|
| Fixed-size chunk_size | 512 | Paper Section 3 |
| Fixed-size overlap | 50 | Paper Section 3 |
| Embedding model | BAAI/bge-m3 | Paper Section 3.1 |
| Reranker model | BAAI/bge-reranker-v2-m3 | Paper Section 3.1 |
| Generator | GPT-OSS-20B | Paper Section 3.2 |
| Max context tokens | 4000 | Paper Section 3.2 |
| Top-k for generation | 5 | Paper Section 3.2 |
| Timeout | 48 hours | Paper Section 3 |
| Accuracy@k | k=5 | Paper Section 3.1 |
| Recall@k | k=10 | Paper Section 3.1 |

**Assumed hyperparameters (verify against original publications):**

| Parameter | Assumed Value | SIR Conf | Source |
|---|---|---|---|
| HAC similarity_threshold | 0.85 | 0.88 | Qu et al. 2025 (NAACL) |
| Max-min alpha | 0.5 | 0.83 | Kiss et al. 2025 (Discover Computing) |
| Generation temperature | 1.0 | 0.55 | Not stated in paper |

---

## Running Tests

```bash
pytest tests/ -v
pytest tests/ -v --timeout=60   # with timeout
pytest tests/test_chunkers.py   # chunkers only
pytest tests/test_metrics.py    # EQ4/EQ5/EQ6 only
```

---

## SIR Confidence Summary

Overall SIR confidence: **0.91**

| Section | Confidence |
|---|---|
| Architecture | 0.95 |
| Mathematical specification (EQ2-EQ7) | 0.93 |
| Evaluation protocol | 0.97 |
| Training/evaluation pipeline | 0.90 |
| Implementation assumptions | 0.85 |

---

## Known Deviations from Paper

1. **GPT-OSS-20B substitution** (RISK-02): If using a different model for generation/judge, Table 4 scores will differ. Experiment 1 (Accuracy@5, Recall@10) is unaffected.
2. **HAC threshold and Max-min alpha** (RISK-03/04): These are assumed defaults. If the cited papers use different values, chunking behavior will differ.
3. **Hardware** (RISK-07): The paper used H100×20 nodes. Timeout behavior (T-markers) depends on your hardware. Adjust `pipeline.timeout_hours` in config.
4. **TextTiling excluded from EXP2**: Explicitly stated in paper Section 3.2. This repo respects that exclusion.
5. **Experimental chunkers** (GraphSeg, LumberChunker, DenseX): Implemented as stubs. Expect `NotImplementedError`. See individual stub docstrings for original implementations.

---

## Citation

```bibtex
@article{smigielski2026chunking,
  title={Chunking Methods on Retrieval-Augmented Generation -- Effectiveness Evaluation Against Computational Cost and Limitations},
  author={Śmigielski, Mateusz and Rajkowski, Michał and Zbrocki, Mateusz and Bernacki-Janson, Michał and Kunicki, Karol and Godziszewska, Julianna and Piasecki, Maciej and Wojtasik, Konrad},
  journal={Procedia Computer Science},
  year={2026},
  note={arXiv:2606.00881}
}
```

---

## Original Benchmark Code

The authors provide their original evaluation framework at:  
[https://github.com/ApriiM/Chunking-Research](https://github.com/ApriiM/Chunking-Research)
