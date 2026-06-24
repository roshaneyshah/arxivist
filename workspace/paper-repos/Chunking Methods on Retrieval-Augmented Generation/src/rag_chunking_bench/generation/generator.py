"""
generation/generator.py
=======================
LLM generation module for Experiment 2 (end-to-end RAG answer generation).

Paper: "Chunking Methods on Retrieval-Augmented Generation" (arXiv:2606.00881)
Section 3.2: "GPT-OSS-20B model was employed as the generative model. For each
query, the top-5 retrieved chunks were provided as contextual input with a
restriction to 4,000 tokens."
"""

from __future__ import annotations

from typing import List, Optional

from rag_chunking_bench.utils.text_utils import count_tokens, truncate_to_tokens


_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question based solely on the "
    "provided context. If the context does not contain enough information, "
    "say so briefly."
)

_USER_TEMPLATE = (
    "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
)


class RAGGenerator:
    """
    LLM generator for RAG Experiment 2.

    Builds a prompt from the query and top-5 retrieved chunks, respecting the
    4,000-token context limit (explicitly stated in paper Section 3.2), then
    calls the OpenAI-compatible API.

    Args:
        model: Model identifier (default: gpt-oss-20b as used in paper).
        api_base: OpenAI-compatible API base URL.
        api_key: API key string.
        max_context_tokens: Hard limit on context tokens (4000, paper Section 3.2).
        temperature: Sampling temperature.
            ASSUMED default 1.0 — not stated in paper. SIR confidence: 0.55.
    """

    def __init__(
        self,
        model: str = "gpt-oss-20b",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        max_context_tokens: int = 4000,  # SIR conf 0.99 — explicitly stated
        temperature: float = 1.0,        # ASSUMED: SIR conf 0.55
    ) -> None:
        self._model = model
        self._api_base = api_base
        self._api_key = api_key
        self._max_context_tokens = max_context_tokens
        self._temperature = temperature

    def _get_client(self):
        """Lazy-init OpenAI client."""
        import openai
        kwargs = {}
        if self._api_base:
            kwargs["base_url"] = self._api_base
        if self._api_key:
            kwargs["api_key"] = self._api_key
        return openai.OpenAI(**kwargs)

    def _truncate_context(self, chunks: List[str]) -> List[str]:
        """
        Truncate chunks to fit within max_context_tokens.

        Paper Section 3.2: "restriction to 4,000 tokens."
        Adds chunks greedily until token limit is reached.

        Args:
            chunks: Top-k retrieved chunk strings.

        Returns:
            Subset of chunks fitting within token budget.
        """
        kept: List[str] = []
        used_tokens = 0
        for chunk in chunks:
            t = count_tokens(chunk)
            if used_tokens + t > self._max_context_tokens:
                break
            kept.append(chunk)
            used_tokens += t
        return kept

    def _build_prompt(self, query: str, context_chunks: List[str]) -> str:
        """
        Build the generation prompt from query and context chunks.

        Args:
            query: Query string.
            context_chunks: List of retrieved chunk strings (already truncated).

        Returns:
            Formatted prompt string.
        """
        context = "\n\n".join(
            f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks)
        )
        return _USER_TEMPLATE.format(context=context, query=query)

    def generate(self, query: str, context_chunks: List[str]) -> str:
        """
        Generate an answer for the given query using retrieved context.

        Paper Experiment 2: top-5 chunks, 4000-token context limit.

        Args:
            query: Query string.
            context_chunks: Top-k retrieved chunk strings (pre-reranked).

        Returns:
            Generated answer string.
        """
        truncated = self._truncate_context(context_chunks)
        prompt = self._build_prompt(query, truncated)

        client = self._get_client()
        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self._temperature,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()

    def __repr__(self) -> str:
        return f"RAGGenerator(model={self._model}, max_ctx={self._max_context_tokens})"
