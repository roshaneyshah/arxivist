#!/usr/bin/env python3
"""
run_full_bench.py
=================
Master entrypoint for the RAG Chunking Benchmark.

Paper: "Chunking Methods on Retrieval-Augmented Generation" (arXiv:2606.00881)

Runs all stages (chunk → index → Experiment 1 → Experiment 2) for every
enabled (method, dataset) combination, respecting the 48-hour timeout per
pair and outputting Tables 1-4 equivalent results.

Usage:
    python run_full_bench.py --config configs/config.yaml
    python run_full_bench.py --config configs/config.yaml --methods fixed_size recursive_semantic
    python run_full_bench.py --config configs/config.yaml --datasets squad triviaqa --skip_generation
    python run_full_bench.py --config configs/config.yaml --resume

Environment:
    OPENAI_API_BASE — base URL for GPT-OSS-20B API
    OPENAI_API_KEY  — API key for generation and judge
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure package is importable when run from repo root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_chunking_bench.chunkers import ChunkerRegistry
from rag_chunking_bench.data.dataset_loader import DatasetLoader
from rag_chunking_bench.embedding.embedder import ChunkEmbedder
from rag_chunking_bench.embedding.reranker import ChunkReranker
from rag_chunking_bench.evaluation.metrics import RetrievalMetrics
from rag_chunking_bench.evaluation.reporter import (
    ResultReporter, FAILURE_T, FAILURE_S
)
from rag_chunking_bench.generation.generator import RAGGenerator
from rag_chunking_bench.generation.judge import LLMJudge
from rag_chunking_bench.pipeline.chunking_pipeline import (
    ChunkingPipeline, ChunkingTimeoutError, ChunkingMemoryError
)
from rag_chunking_bench.pipeline.eval_pipeline import EvalPipeline
from rag_chunking_bench.retrieval.index import FAISSChunkIndex
from rag_chunking_bench.retrieval.retriever import RAGRetriever
from rag_chunking_bench.utils.config import BenchConfig, set_seed
from rag_chunking_bench.utils.timing import Timer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG Chunking Benchmark — arXiv:2606.00881"
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to YAML config (default: configs/config.yaml)"
    )
    parser.add_argument(
        "--methods", nargs="*", default=None,
        help="Chunker methods to run (default: all enabled in config)"
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Datasets to evaluate (default: all enabled in config)"
    )
    parser.add_argument(
        "--skip_generation", action="store_true",
        help="Skip Experiment 2 (generation + judge scoring)"
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Override output directory from config"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip already-completed (method, dataset) pairs"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Resume state helpers
# ---------------------------------------------------------------------------

def _state_path(output_dir: str) -> Path:
    return Path(output_dir) / "run_state.json"


def _load_state(output_dir: str) -> dict:
    p = _state_path(output_dir)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {"completed": []}


def _save_state(output_dir: str, state: dict) -> None:
    with open(_state_path(output_dir), "w") as f:
        json.dump(state, f, indent=2)


def _pair_key(method: str, dataset: str) -> str:
    return f"{method}::{dataset}"


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Load config
    cfg = BenchConfig.from_yaml(args.config)
    set_seed(cfg.pipeline.seed)

    output_dir = args.output_dir or cfg.pipeline.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Resolve method and dataset lists
    methods = args.methods or cfg.chunkers.enabled
    datasets = args.datasets or cfg.datasets.enabled

    logger.info(f"Methods:  {methods}")
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Output:   {output_dir}")

    # Resume state
    state = _load_state(output_dir) if args.resume or cfg.pipeline.resume else {"completed": []}

    # Shared components
    embedder = ChunkEmbedder(
        model_name=cfg.embedding.model,
        device=cfg.embedding.device,
        batch_size=cfg.embedding.batch_size,
        normalize=cfg.embedding.normalize,
        embedding_dim=cfg.embedding.embedding_dim,
    )
    reranker = ChunkReranker(
        model_name=cfg.reranker.model,
        device=cfg.reranker.device,
    )
    reporter = ResultReporter(output_dir=output_dir)
    metrics = RetrievalMetrics()

    # Generator and judge (Experiment 2)
    generator = judge = None
    if not args.skip_generation and cfg.generation.api_key:
        generator = RAGGenerator(
            model=cfg.generation.model,
            api_base=cfg.generation.api_base,
            api_key=cfg.generation.api_key,
            max_context_tokens=cfg.generation.max_context_tokens,
            temperature=cfg.generation.temperature,
        )
        judge = LLMJudge(
            model=cfg.judge.model,
            api_base=cfg.judge.api_base,
            api_key=cfg.judge.api_key,
            scale_min=cfg.judge.scale_min,
            scale_max=cfg.judge.scale_max,
        )
        logger.info("Experiment 2 (generation + judge) enabled.")
    else:
        logger.info(
            "Experiment 2 disabled "
            "(--skip_generation or no API key configured)."
        )

    # -----------------------------------------------------------------------
    # Main loop: for each (method, dataset) pair
    # -----------------------------------------------------------------------
    total = len(methods) * len(datasets)
    pair_idx = 0

    for method_name in methods:
        # Get chunker config for this method
        method_cfg = getattr(cfg.chunkers, method_name, None)
        chunker_config = vars(method_cfg) if method_cfg else {}

        for dataset_name in datasets:
            pair_idx += 1
            key = _pair_key(method_name, dataset_name)

            logger.info(
                f"\n{'='*60}\n"
                f"[{pair_idx}/{total}] {method_name} × {dataset_name}\n"
                f"{'='*60}"
            )

            # Resume skip
            if key in state["completed"]:
                logger.info(f"  → SKIPPED (already completed)")
                continue

           # ----------------------------------------------------------------
            # 1. Load dataset
            # ----------------------------------------------------------------
            try:
                logger.info(f"  [DEBUG] Attempting to load dataset '{dataset_name}'...")
                loader = DatasetLoader(cfg.datasets.data_dir, dataset_name)
                documents = loader.load_documents()
                queries = loader.load_queries()
                logger.info(
                    f"  Dataset loaded: {len(documents)} docs, {len(queries)} queries"
                )
            except Exception as e:
                logger.error(f"  DATASET FATAL ERROR: {e}", exc_info=True)
                reporter.record_failure(method_name, dataset_name, FAILURE_T)
                continue

            # ----------------------------------------------------------------
            # 2. Chunking + indexing (with 48h timeout)
            # ----------------------------------------------------------------
            try:
                chunker = ChunkerRegistry.get(method_name, chunker_config)
            except (KeyError, NotImplementedError) as e:
                logger.error(f"  CHUNKER ERROR: {e}")
                reporter.record_failure(method_name, dataset_name, FAILURE_T)
                continue

            index_dir = str(Path(output_dir) / "indices" / method_name / dataset_name)
            chunk_pipeline = ChunkingPipeline(
                chunker=chunker,
                embedder=embedder,
                timeout_hours=cfg.pipeline.timeout_hours,
                index_dir=index_dir,
            )

            try:
                with Timer() as total_timer:
                    faiss_index, all_chunks_nested, chunk_elapsed = chunk_pipeline.run(documents)

                flat_chunks = [c for nested in all_chunks_nested for c in nested]
                reporter.record_timing(method_name, dataset_name, chunk_elapsed)
                logger.info(
                    f"  Chunks: {len(flat_chunks)}, "
                    f"chunking time: {Timer().elapsed_human() if False else _fmt_s(chunk_elapsed)}"
                )

            except ChunkingTimeoutError as e:
                logger.warning(f"  T-MARKER: {e}")
                reporter.record_failure(method_name, dataset_name, FAILURE_T)
                reporter.record_timing(method_name, dataset_name, FAILURE_T)
                state["completed"].append(key)
                _save_state(output_dir, state)
                continue

            except ChunkingMemoryError as e:
                logger.warning(f"  S-MARKER: {e}")
                reporter.record_failure(method_name, dataset_name, FAILURE_S)
                reporter.record_timing(method_name, dataset_name, FAILURE_S)
                state["completed"].append(key)
                _save_state(output_dir, state)
                continue

            except Exception as e:
                logger.error(f"  UNEXPECTED ERROR during chunking: {e}", exc_info=True)
                reporter.record_failure(method_name, dataset_name, FAILURE_T)
                state["completed"].append(key)
                _save_state(output_dir, state)
                continue

            # ----------------------------------------------------------------
            # 3. Evaluation (Experiments 1 and 2)
            # ----------------------------------------------------------------
            retriever = RAGRetriever(
                index=faiss_index,
                embedder=embedder,
                reranker=reranker,
                top_k_index=cfg.retrieval.top_k_index,
            )

            eval_pipeline = EvalPipeline(
                retriever=retriever,
                metrics=metrics,
                reporter=reporter,
                all_chunks=flat_chunks,
                generator=generator,
                judge=judge,
                top_k_acc=cfg.retrieval.top_k_accuracy,
                top_k_rec=cfg.retrieval.top_k_recall,
                top_k_gen=cfg.retrieval.top_k_generation,
            )

            # Experiment 1: Evidence Retrieval
            try:
                eval_pipeline.run_retrieval_eval(queries, method_name, dataset_name)
            except Exception as e:
                logger.error(f"  EXP1 ERROR: {e}", exc_info=True)

            # Experiment 2: End-to-End RAG
            if generator is not None and judge is not None:
                try:
                    eval_pipeline.run_generation_eval(queries, method_name, dataset_name)
                except Exception as e:
                    logger.error(f"  EXP2 ERROR: {e}", exc_info=True)

            # Mark pair complete and persist state
            state["completed"].append(key)
            _save_state(output_dir, state)

    # -----------------------------------------------------------------------
    # Export final results (Tables 1-4 equivalent)
    # -----------------------------------------------------------------------
    logger.info("\n" + "="*60)
    logger.info("BENCHMARK COMPLETE — exporting results")
    logger.info("="*60)

    reporter.export_json()
    reporter.export_csv()
    reporter.print_summary_table("accuracy_at_5")
    reporter.print_summary_table("recall_at_10")

    logger.info(f"\nAll results written to: {output_dir}")
    logger.info(
        "Compare with paper Tables 1-4 (arXiv:2606.00881).\n"
        "NOTE: Results may differ due to:\n"
        "  - Different generator model (substitute for GPT-OSS-20B)\n"
        "  - Hardware differences affecting timeout behavior\n"
        "  - ASSUMED hyperparameters (HAC threshold, Max-min alpha)\n"
        "See architecture_plan_summary.md for full risk assessment."
    )


def _fmt_s(seconds: float) -> str:
    if seconds < 1:
        return "<1s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.2f}m"
    return f"{seconds/3600:.2f}h"


if __name__ == "__main__":
    main()
