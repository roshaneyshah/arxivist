"""
data/dataset_loader.py
======================
Dataset loading utilities for the RAG Chunking Benchmark.

Paper: "Chunking Methods on Retrieval-Augmented Generation" (arXiv:2606.00881)
Appendix A: Dataset selection criteria and statistics.

Supports all 11 dataset configurations:
  - GutenQA, GutenQA merged
  - LiteraryQA, NovelQA
  - Natural Questions (NQ)
  - PoQuAD, PoQuAD merged
  - Qasper
  - SQuAD
  - TriviaQA, TriviaQA merged

Returns standardized (document, query, answer_span) records for use
by the evaluation pipeline.

RISK-01 NOTE: GutenQA, LiteraryQA, NovelQA are not on HuggingFace Hub.
These require manual download. See data/download_datasets.py for instructions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Query record schema
# ---------------------------------------------------------------------------
# Each query record is a dict with keys:
#   query_id      (str)  — unique identifier
#   query         (str)  — question text
#   answer_span   (str)  — extractive answer text (may be empty string)
#   relevant_doc_id (str) — document id containing the answer
#   ground_truth  (str)  — reference answer for LLM-judge evaluation


class DatasetLoader:
    """
    Standardized loader for all paper benchmark datasets.

    Paper Appendix A: "Dataset selection for chunking evaluation requires
    careful balancing of: (1) sufficiently long documents, (2) availability
    of answer-grounded annotations, (3) diversity of domains and structures."

    Args:
        data_dir: Root directory containing dataset subdirectories.
        dataset_name: One of the 11 dataset configuration names.
    """

    SUPPORTED = {
        # HuggingFace-available datasets
        "squad", "triviaqa", "triviaqa_merged",
        "poquad", "poquad_merged",
        "nq",
        "qasper",
        # Requires manual download (RISK-01)
        "gutenqa", "gutenqa_merged",
        "literaryqa",
        "novelqa",
    }

    # Datasets available on HuggingFace Hub (auto-downloadable)
    HF_AVAILABLE = {"squad", "triviaqa", "triviaqa_merged", "nq", "qasper"}

    def __init__(self, data_dir: str, dataset_name: str) -> None:
        if dataset_name not in self.SUPPORTED:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Supported: {sorted(self.SUPPORTED)}"
            )
        self._data_dir = Path(data_dir)
        self._dataset_name = dataset_name
        self._is_merged = dataset_name.endswith("_merged")
        self._base_name = dataset_name.replace("_merged", "")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_documents(self) -> List[str]:
        """
        Load raw document texts for this dataset configuration.

        For merged variants: concatenate all documents into a single string.

        Returns:
            List of raw document text strings.
        """
        docs = self._load_documents_base()
        if self._is_merged:
            merged_text = "\n\n".join(docs)
            return [merged_text]
        return docs

    def load_queries(self) -> List[Dict]:
        """
        Load query records for this dataset.

        Returns:
            List of query dicts with keys:
                query_id, query, answer_span, relevant_doc_id, ground_truth
        """
        return self._load_queries_base()

    def create_merged_variant(self, documents: List[str]) -> str:
        """
        Concatenate documents into a single merged document (stress-test).

        Paper Appendix A: merged variants create "stress-test conditions."

        Args:
            documents: List of document strings.

        Returns:
            Single concatenated document string.
        """
        return "\n\n".join(documents)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _load_documents_base(self) -> List[str]:
        dispatch = {
            "squad": self._load_squad_docs,
            "triviaqa": self._load_triviaqa_docs,
            "nq": self._load_nq_docs,
            "qasper": self._load_qasper_docs,
            "poquad": self._load_poquad_docs,
            "gutenqa": self._load_gutenqa_docs,
            "literaryqa": self._load_literaryqa_docs,
            "novelqa": self._load_novelqa_docs,
        }
        loader = dispatch.get(self._base_name)
        if loader is None:
            raise NotImplementedError(f"Document loader not found for {self._base_name}")
        return loader()

    def _load_queries_base(self) -> List[Dict]:
        dispatch = {
            "squad": self._load_squad_queries,
            "triviaqa": self._load_triviaqa_queries,
            "nq": self._load_nq_queries,
            "qasper": self._load_qasper_queries,
            "poquad": self._load_poquad_queries,
            "gutenqa": self._load_gutenqa_queries,
            "literaryqa": self._load_literaryqa_queries,
            "novelqa": self._load_novelqa_queries,
        }
        loader = dispatch.get(self._base_name)
        if loader is None:
            raise NotImplementedError(f"Query loader not found for {self._base_name}")
        return loader()

    # ------------------------------------------------------------------
    # HuggingFace-based loaders
    # ------------------------------------------------------------------

    def _load_squad_docs(self) -> List[str]:
        """
        SQuAD: 100 documents (paper Appendix A: avg 29,971 chars).
        Loaded from HuggingFace datasets.
        """
        from datasets import load_dataset
        ds = load_dataset("rajpurkar/squad", split="validation")
        seen = {}
        for ex in ds:
            ctx = ex["context"]
            if ctx not in seen:
                seen[ctx] = True
        docs = list(seen.keys())
        # Paper uses 100 documents
        return docs[:100]

    def _load_squad_queries(self) -> List[Dict]:
        from datasets import load_dataset
        ds = load_dataset("rajpurkar/squad", split="validation")
        records = []
        for i, ex in enumerate(ds):
            answers = ex["answers"]["text"]
            answer_span = answers[0] if answers else ""
            records.append({
                "query_id": f"squad_{i}",
                "query": ex["question"],
                "answer_span": answer_span,
                "relevant_doc_id": ex["context"][:50],
                "ground_truth": answer_span,
            })
        return records

    def _load_triviaqa_docs(self) -> List[str]:
        """
        TriviaQA: 1,000 documents (paper Appendix A: avg 14,239 chars).
        """
        from datasets import load_dataset
        ds = load_dataset("trivia_qa", "rc.wikipedia", split="validation")
        docs = []
        seen_ids = set()
        for ex in ds:
            if len(docs) >= 1000:
                break
            # Each example has an entity_pages field with wiki passages
            if "entity_pages" in ex and ex["entity_pages"]["wiki_context"]:
                for ctx in ex["entity_pages"]["wiki_context"]:
                    doc_id = ctx[:50]
                    if doc_id not in seen_ids and ctx.strip():
                        seen_ids.add(doc_id)
                        docs.append(ctx)
                        if len(docs) >= 1000:
                            break
        return docs

    def _load_triviaqa_queries(self) -> List[Dict]:
        from datasets import load_dataset
        ds = load_dataset("trivia_qa", "rc.wikipedia", split="validation")
        records = []
        for i, ex in enumerate(ds):
            if i >= 1000:
                break
            answer = ex["answer"]["value"] if "answer" in ex else ""
            records.append({
                "query_id": f"triviaqa_{i}",
                "query": ex["question"],
                "answer_span": answer,
                "relevant_doc_id": ex.get("question_id", str(i)),
                "ground_truth": answer,
            })
        return records

    def _load_nq_docs(self) -> List[str]:
        """
        Natural Questions: 300 documents (paper Appendix A: avg 45,634 chars).
        """
        from datasets import load_dataset
        ds = load_dataset("google-research-datasets/natural_questions",
                          "default", split="validation", trust_remote_code=True)
        docs = []
        for ex in ds:
            if len(docs) >= 300:
                break
            # NQ has HTML document text
            doc_tokens = ex.get("document", {}).get("tokens", {})
            if doc_tokens and "token" in doc_tokens:
                text = " ".join(doc_tokens["token"])
                if len(text) > 100:
                    docs.append(text)
        return docs[:300]

    def _load_nq_queries(self) -> List[Dict]:
        from datasets import load_dataset
        ds = load_dataset("google-research-datasets/natural_questions",
                          "default", split="validation", trust_remote_code=True)
        records = []
        for i, ex in enumerate(ds):
            if i >= 300:
                break
            annotations = ex.get("annotations", {})
            short_answers = annotations.get("short_answers", [])
            answer_span = ""
            if short_answers and short_answers[0].get("text"):
                answer_span = short_answers[0]["text"][0]
            records.append({
                "query_id": f"nq_{i}",
                "query": ex["question"]["text"],
                "answer_span": answer_span,
                "relevant_doc_id": str(i),
                "ground_truth": answer_span,
            })
        return records

    def _load_qasper_docs(self) -> List[str]:
        """
        Qasper: 416 scientific papers (paper Appendix A: avg 1,014 chars per section).
        """
        from datasets import load_dataset
        ds = load_dataset("allenai/qasper", split="validation")
        docs = []
        for ex in ds:
            # Concatenate all full-text sections
            sections = []
            for title, paragraphs in zip(
                ex.get("full_text", {}).get("section_name", []),
                ex.get("full_text", {}).get("paragraphs", []),
            ):
                if paragraphs:
                    sections.append(f"# {title}\n" + "\n".join(paragraphs))
            if sections:
                docs.append("\n\n".join(sections))
        return docs

    def _load_qasper_queries(self) -> List[Dict]:
        from datasets import load_dataset
        ds = load_dataset("allenai/qasper", split="validation")
        records = []
        idx = 0
        for ex in ds:
            for qas in ex.get("qas", []):
                question = qas.get("question", "")
                answers = qas.get("answers", [])
                answer_span = ""
                if answers and answers[0].get("answer", {}).get("extractive_spans"):
                    answer_span = answers[0]["answer"]["extractive_spans"][0]
                records.append({
                    "query_id": f"qasper_{idx}",
                    "query": question,
                    "answer_span": answer_span,
                    "relevant_doc_id": ex.get("id", str(idx)),
                    "ground_truth": answer_span,
                })
                idx += 1
        return records

    def _load_poquad_docs(self) -> List[str]:
        """
        PoQuAD: Polish QA dataset, 1,449 documents (paper Appendix A: avg 922 chars).
        RISK-01: May need manual download from original PoQuAD release.
        """
        path = self._data_dir / "poquad" / "poquad.json"
        if not path.exists():
            raise FileNotFoundError(
                f"PoQuAD data not found at {path}. "
                "Download from: https://github.com/PoQuAD/PoQuAD "
                "and place in data/poquad/"
            )
        with open(path) as f:
            data = json.load(f)
        docs = []
        for item in data.get("data", []):
            for para in item.get("paragraphs", []):
                ctx = para.get("context", "")
                if ctx.strip():
                    docs.append(ctx)
        return docs[:1449]

    def _load_poquad_queries(self) -> List[Dict]:
        path = self._data_dir / "poquad" / "poquad.json"
        if not path.exists():
            raise FileNotFoundError(f"PoQuAD not found at {path}")
        with open(path) as f:
            data = json.load(f)
        records = []
        idx = 0
        for item in data.get("data", []):
            for para in item.get("paragraphs", []):
                for qa in para.get("qas", []):
                    answers = qa.get("answers", [])
                    answer_span = answers[0]["text"] if answers else ""
                    records.append({
                        "query_id": f"poquad_{idx}",
                        "query": qa.get("question", ""),
                        "answer_span": answer_span,
                        "relevant_doc_id": para.get("context", "")[:50],
                        "ground_truth": answer_span,
                    })
                    idx += 1
        return records

    # ------------------------------------------------------------------
    # Manual-download loaders (RISK-01)
    # ------------------------------------------------------------------

    def _load_gutenqa_docs(self) -> List[str]:
        """
        GutenQA: 36,917 documents from Project Gutenberg books.
        RISK-01: Not on HuggingFace. Download from Duarte et al. 2024 (EMNLP).
        Paper Appendix A: avg document length 1,814 chars.
        """
        path = self._data_dir / "gutenqa" / "documents.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"GutenQA documents not found at {path}. "
                "Download from: https://github.com/avduarte333/LumberChunker "
                "(GutenQA is released alongside LumberChunker)."
            )
        docs = []
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                text = obj.get("text", "") or obj.get("content", "")
                if text.strip():
                    docs.append(text)
        return docs

    def _load_gutenqa_queries(self) -> List[Dict]:
        path = self._data_dir / "gutenqa" / "queries.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"GutenQA queries not found at {path}")
        records = []
        with open(path) as f:
            for i, line in enumerate(f):
                obj = json.loads(line)
                records.append({
                    "query_id": f"gutenqa_{i}",
                    "query": obj.get("question", obj.get("query", "")),
                    "answer_span": obj.get("answer", ""),
                    "relevant_doc_id": obj.get("doc_id", str(i)),
                    "ground_truth": obj.get("answer", ""),
                })
        return records

    def _load_literaryqa_docs(self) -> List[str]:
        """
        LiteraryQA: 138 documents, avg 411,471 chars.
        RISK-01: From Bonomo et al. 2025 (EMNLP). Manual download required.
        """
        path = self._data_dir / "literaryqa" / "documents.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"LiteraryQA not found at {path}. "
                "Download from: https://github.com/Babelscape/LiteraryQA "
                "(Bonomo et al. 2025, EMNLP)."
            )
        docs = []
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                docs.append(obj.get("text", ""))
        return docs

    def _load_literaryqa_queries(self) -> List[Dict]:
        path = self._data_dir / "literaryqa" / "queries.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"LiteraryQA queries not found at {path}")
        records = []
        with open(path) as f:
            for i, line in enumerate(f):
                obj = json.loads(line)
                records.append({
                    "query_id": f"literaryqa_{i}",
                    "query": obj.get("question", ""),
                    "answer_span": obj.get("answer", ""),
                    "relevant_doc_id": obj.get("doc_id", str(i)),
                    "ground_truth": obj.get("answer", ""),
                })
        return records

    def _load_novelqa_docs(self) -> List[str]:
        """
        NovelQA: 60 novels, avg 1,007,786 chars (up to 6.8M chars per novel).
        RISK-01: From Wang et al. 2025 (arXiv:2403.12766). Manual download.
        """
        path = self._data_dir / "novelqa" / "documents.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"NovelQA not found at {path}. "
                "Download from: https://github.com/NovelQA/novelqa "
                "(Wang et al. 2025, arXiv:2403.12766)."
            )
        docs = []
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                docs.append(obj.get("text", ""))
        return docs

    def _load_novelqa_queries(self) -> List[Dict]:
        path = self._data_dir / "novelqa" / "queries.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"NovelQA queries not found at {path}")
        records = []
        with open(path) as f:
            for i, line in enumerate(f):
                obj = json.loads(line)
                records.append({
                    "query_id": f"novelqa_{i}",
                    "query": obj.get("question", ""),
                    "answer_span": obj.get("answer", ""),
                    "relevant_doc_id": obj.get("doc_id", str(i)),
                    "ground_truth": obj.get("answer", ""),
                })
        return records

    def __repr__(self) -> str:
        return f"DatasetLoader(dataset={self._dataset_name}, dir={self._data_dir})"
