# Architecture Plan — Stage 3 Summary
## Paper: Chunking Methods on Retrieval-Augmented Generation (arXiv:2606.00881)
**Generated:** 2026-06-22 | **Plan Version:** 1 | **SIR Confidence:** 0.91

---

## 1. Framework Selection

| Decision | Choice | Reason |
|---|---|---|
| Language | Python 3.10+ | Paper uses spaCy, LangChain — Python ecosystem |
| Embedding | `sentence-transformers` (bge-m3) | Exact model cited in paper |
| Vector Index | FAISS (cosine) | Standard ANN; paper specifies cosine similarity (EQ3) |
| Chunking libs | LangChain text splitters + scipy + nltk | Covers all 4 robust methods directly |
| LLM interface | OpenAI-compatible REST API | GPT-OSS-20B used for generator + judge |
| Config | YAML + Python dataclasses | Lightweight; no Hydra needed |

> **Critical framing:** This is an **evaluation/benchmark paper**, not a novel model paper. There is no training loop. The "architecture" is a modular harness with interchangeable chunking front-ends. Stage 4 (Code Generator) will produce a benchmark framework, not a neural network.

---

## 2. Project Structure

```
paper_chunking_methods_rag_2026/
│
├── src/rag_chunking_bench/
│   ├── chunkers/                    ← Core: 4 robust + 1 modified + 3 experimental
│   │   ├── base.py                  ← Abstract BaseChunker: f_theta(D) -> C  [EQ2]
│   │   ├── fixed_size.py            ← chunk_size=512, overlap=50  ★ Fastest
│   │   ├── recursive_semantic.py    ← LangChain default  ★ Best Accuracy@5
│   │   ├── sequential_hac.py        ← Agglomerative merging  ★ Robust
│   │   ├── max_min.py               ← Greedy adaptive threshold  ★ Robust
│   │   ├── texttiling.py            ← Modified: sentence-level boundary alignment
│   │   └── experimental/
│   │       ├── graphseg.py          ← ⚠ spaCy memory errors on large docs
│   │       ├── lumberchunker.py     ← ⚠ Timeout on most datasets
│   │       └── densex.py            ← ⚠ Slowest (avg 15h); lowest Accuracy@5
│   │
│   ├── embedding/
│   │   ├── embedder.py              ← bge-m3: E(c), E(q) → R^1024  [EQ3]
│   │   └── reranker.py              ← bge-reranker-v2-m3 cross-encoder
│   │
│   ├── retrieval/
│   │   ├── index.py                 ← FAISSChunkIndex: cosine top-k  [EQ3]
│   │   └── retriever.py             ← RAGRetriever: embed → search → rerank
│   │
│   ├── generation/
│   │   ├── generator.py             ← GPT-OSS-20B, top-5 chunks, ≤4000 tokens
│   │   └── judge.py                 ← LLM-as-Judge: 5-point Likert scorer
│   │
│   ├── evaluation/
│   │   ├── metrics.py               ← Accuracy@5 [EQ5], Recall@10 [EQ6]
│   │   └── reporter.py              ← Tables 1–4 reproduction (CSV + JSON)
│   │
│   ├── data/
│   │   └── dataset_loader.py        ← 11 dataset configs incl. merged stress-tests
│   │
│   ├── pipeline/
│   │   ├── chunking_pipeline.py     ← 48h timeout enforcer; logs T/S failures
│   │   └── eval_pipeline.py         ← Orchestrates EXP1 + EXP2
│   │
│   └── utils/
│       ├── config.py                ← YAML → typed dataclasses
│       ├── timing.py                ← Wall-clock Timer context manager
│       └── text_utils.py            ← sentence split, token count, span overlap
│
├── configs/
│   └── config.yaml                  ← All hyperparameters (annotated with confidence)
│
├── data/
│   └── download_datasets.py         ← Dataset acquisition scripts
│
├── chunk.py                         ← CLI: run one chunker on one dataset
├── index.py                         ← CLI: build FAISS index from chunks
├── evaluate.py                      ← CLI: Experiment 1 (Accuracy@5, Recall@10)
├── generate_and_judge.py            ← CLI: Experiment 2 (generation + judge score)
├── run_full_bench.py                ← CLI: master entrypoint, all methods × datasets
├── report.py                        ← CLI: aggregate results → Tables 1–4
│
├── docker/Dockerfile
├── requirements.txt
├── requirements-dev.txt
├── environment.yaml
└── README.md
```

**Total source files: 39**

---

## 3. Data Flow Specification

### Pipeline A — Chunking + Indexing (Offline)
```
documents: list[str]   ← DatasetLoader.load_documents()
    │
    ▼  [48h timeout enforced by ChunkingPipeline]
chunker.chunk(doc)     ← EQ2: f_theta(D) -> C = {c1 ... cm}
    │                     θ = method-specific hyperparams
    ▼
all_chunks: list[str]  [N_total variable chunks]
    │
    ▼
embedder.encode(all_chunks)
    │                  ← bge-m3, normalize=True
    ▼
embeddings: np.ndarray [N_total, 1024]  dtype=float32
    │
    ▼
FAISSChunkIndex.build(chunks, embeddings)
    │                  ← cosine similarity index
    ▼
index + elapsed_seconds  ← Timer logs Table 3 timing
```

### Pipeline B — Experiment 1: Evidence Retrieval
```
query: str
    │
    ▼
embedder.encode_query(query) → query_emb: [1024]   ← E(q)
    │
    ▼  EQ3: argTopK cosine(E(q), E(c))
index.search(query_emb, k=10) → candidates: list[str]
    │
    ▼
reranker.rerank(query, candidates, top_k=10)
    │
    ├─→ Accuracy@5:  ∃ c ∈ top-5 s.t. relevant(c,q)=1   [EQ5]
    └─→ Recall@10:   |retrieved∩relevant| / |relevant|    [EQ6]
              ↑
    relevant(c,q) = 1[c ∩ answer_span(q) ≠ ∅]            [EQ4]
```

### Pipeline C — Experiment 2: End-to-End RAG
```
query: str + ground_truth_answer: str
    │
    ▼
retrieve top-5 (via Pipeline B, k=5)
    │
    ▼
generator._build_prompt(query, top5)  ← ≤ 4000 tokens hard limit
    │
    ▼
GPT-OSS-20B → answer: str
    │
    ▼
judge.score(query, answer, ground_truth) → int ∈ {1,2,3,4,5}
    │                                       ← Likert scale
    ▼
reporter.record(method, dataset, 'llm_judge', score)
```

---

## 4. Configuration Schema (configs/config.yaml)

```yaml
# ============================================================
# RAG Chunking Benchmark — Master Config
# SIR confidence values annotated as comments
# ============================================================

chunkers:
  enabled: [fixed_size, recursive_semantic, sequential_hac, max_min]
  optional: [texttiling, graphseg, lumberchunker, densex]

  fixed_size:
    chunk_size: 512       # SIR confidence: 0.99 — explicitly stated in paper
    overlap: 50           # SIR confidence: 0.99 — explicitly stated in paper

  recursive_semantic:
    chunk_size: 512       # SIR confidence: 0.92
    overlap: 50
    separators: ["\n\n", "\n", ". ", " "]
    # TODO: verify exact separator hierarchy from LangChain version used

  sequential_hac:
    similarity_threshold: 0.85  # SIR confidence: 0.88
    # ASSUMED: default from Qu et al. 2025 — cross-reference original publication
    max_chunk_tokens: 512

  max_min:
    alpha: 0.5            # SIR confidence: 0.83
    # ASSUMED: alpha from Kiss et al. 2025 — cross-reference original publication

  texttiling:
    w: 20                 # pseudosentence size (words)
    k: 10                 # block size comparison
    smoothing_width: 2
    boundary_alignment: sentence  # EXPLICITLY STATED in paper Section 3

embedding:
  model: BAAI/bge-m3       # SIR confidence: 0.99
  batch_size: 64
  device: auto             # 'cpu', 'cuda', or 'auto'
  normalize: true          # required for cosine similarity
  embedding_dim: 1024      # SIR confidence: 0.90 — from bge-m3 spec

reranker:
  model: BAAI/bge-reranker-v2-m3  # SIR confidence: 0.99
  device: auto

retrieval:
  top_k_index: 10          # retrieved before reranking
  top_k_accuracy: 5        # Accuracy@5 — EQ5
  top_k_recall: 10         # Recall@10 — EQ6
  top_k_generation: 5      # chunks passed to generator — EXP2

generation:
  model: gpt-oss-20b        # SIR confidence: 0.99
  api_base: ${OPENAI_API_BASE}
  api_key: ${OPENAI_API_KEY}
  max_context_tokens: 4000  # SIR confidence: 0.99 — explicitly stated

judge:
  model: gpt-oss-20b        # SIR confidence: 0.99
  api_base: ${OPENAI_API_BASE}
  api_key: ${OPENAI_API_KEY}
  scale: [1, 5]             # 1=poor, 5=highly accurate

datasets:
  data_dir: data/
  enabled:
    - squad
    - triviaqa
    - triviaqa_merged
    - poquad
    - poquad_merged
    - nq
    - qasper
    - gutenqa
    - gutenqa_merged
    - literaryqa
    - novelqa

pipeline:
  timeout_hours: 48.0      # SIR confidence: 0.99 — explicitly stated
  output_dir: results/
  resume: true             # skip already-completed (method, dataset) pairs
  log_failures: true       # record T and S markers
```

---

## 5. Dependencies

### requirements.txt
```
sentence-transformers==3.0.1   # bge-m3 embedder + bge-reranker
faiss-cpu==1.8.0               # cosine ANN index (swap for faiss-gpu if CUDA)
langchain-text-splitters==0.3.8 # FixedSize + RecursiveSemantic chunkers
nltk==3.8.1                    # punkt sentence tokenizer
spacy==3.7.4                   # sentence splitting; TextTiling; GraphSeg
scipy==1.13.0                  # agglomerative clustering for HAC
scikit-learn==1.5.0            # cosine_similarity utility
numpy==1.26.4                  # embedding array ops
openai==1.35.0                 # GPT-OSS-20B generator + judge
tiktoken==0.7.0                # 4000-token context truncation
pyyaml==6.0.1                  # config loading
tqdm==4.66.4                   # progress bars
pandas==2.2.2                  # results aggregation / CSV export
datasets==2.20.0               # HuggingFace dataset loading
```

### requirements-dev.txt
```
pytest==8.2.0
pytest-timeout==2.3.1   # enforce short timeouts in unit tests
pytest-cov==5.0.0
black==24.4.2
ruff==0.4.7
mypy==1.10.0
ipykernel==6.29.4
```

### environment.yaml
```yaml
name: rag-chunking-bench
channels: [conda-forge, pytorch]
dependencies:
  - python=3.10
  - pip
  - pip:
    - -r requirements.txt
```

---

## 6. Entrypoints (CLI)

| Script | Purpose | Key Args |
|---|---|---|
| `chunk.py` | Run one chunker on one dataset, save chunks + timing | `--method`, `--dataset`, `--merged`, `--config`, `--output_dir` |
| `index.py` | Build FAISS index from saved chunks | `--chunks_dir`, `--output_dir`, `--config` |
| `evaluate.py` | Experiment 1: Accuracy@5, Recall@10 | `--method`, `--dataset`, `--index_dir`, `--config` |
| `generate_and_judge.py` | Experiment 2: RAG generation + LLM-judge scoring | `--method`, `--dataset`, `--index_dir`, `--config` |
| `run_full_bench.py` | **Master** — all methods × all datasets, both experiments | `--config`, `--methods`, `--datasets`, `--skip_generation`, `--resume` |
| `report.py` | Aggregate results → Tables 1–4 (CSV, JSON, LaTeX) | `--results_dir`, `--output_dir`, `--format` |

The `run_full_bench.py` master script is the primary reproduction entrypoint. It chains: `chunk → index → evaluate → generate_and_judge` for every (method, dataset) combination, with the 48h timeout respected per pair and `--resume` supporting interrupted runs.

---

## 7. Docker Spec

```dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential git curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt punkt_tab
RUN python -m spacy download en_core_web_sm

COPY . .

# Volumes for data and results (mount at runtime)
VOLUME ["/workspace/data", "/workspace/results"]

ENV OPENAI_API_KEY=""
ENV OPENAI_API_BASE=""

CMD ["python", "run_full_bench.py", "--config", "configs/config.yaml"]
```

> For GPU: switch base to `nvidia/cuda:12.1.0-runtime-ubuntu22.04` and replace `faiss-cpu` with `faiss-gpu`.

---

## 8. Risk Assessment

| ID | Severity | Category | Description | Mitigation |
|---|---|---|---|---|
| RISK-01 | 🔴 High | Dataset Access | GutenQA, LiteraryQA, NovelQA are non-HuggingFace datasets requiring manual download | `download_datasets.py` with per-dataset loaders; smoke-test mode on SQuAD only |
| RISK-02 | 🔴 High | LLM API Access | GPT-OSS-20B is a non-public model identifier; users may lack access | Abstract behind OpenAI-compatible interface; README warns exact reproduction requires access; any compatible model substitutable via config |
| RISK-03 | 🟡 Medium | HAC Threshold | `similarity_threshold=0.85` is assumed from Qu et al. 2025 defaults (SIR conf: 0.88) | Configurable + `# ASSUMED` comment; cross-reference instruction in README |
| RISK-04 | 🟡 Medium | Max-Min alpha | `alpha=0.5` assumed from Kiss et al. 2025 (SIR conf: 0.83) | Same as RISK-03 |
| RISK-05 | 🟡 Medium | spaCy S-marker | Reproducing the memory error behavior depends on document size vs available RAM | Log document character length; `--force_spacy_limit` flag to cap memory |
| RISK-06 | 🟡 Medium | PIRB vs dataset queries | Ambiguous whether PIRB queries or dataset-native queries are used (SIR conf: 0.70) | Implement EQ5/EQ6 independently of PIRB tooling; both query sources supported via config |
| RISK-07 | 🟢 Low | Hardware | H100 × 20 nodes vs typical user setup; 48h timeout fires differently | Resumable pipeline; timeout configurable; timing differences expected and documented |
| RISK-08 | 🟢 Low | Experimental chunkers | GraphSeg, LumberChunker, DenseX expected to fail or be extremely slow | Placed in `chunkers/experimental/`; stub classes with NotImplementedError + paper citation |

---

## Stage 3 Checklist

- [x] Framework selection with reasoning
- [x] Complete module hierarchy (39 files, all SIR components covered)
- [x] Data flow for all 3 pipeline paths (chunking, retrieval eval, generation eval)
- [x] Config schema with all SIR hyperparameters + confidence annotations
- [x] Dependencies manifest (requirements.txt, requirements-dev.txt, environment.yaml)
- [x] All 6 entrypoints defined with CLI schemas
- [x] Docker spec included
- [x] Risk assessment (8 risks: 2 High, 4 Medium, 2 Low)
- [x] architecture_plan.json written to registry
- [x] architecture_plan_summary.md (this file)

---

**→ Ready for Stage 4: Code Generator**
