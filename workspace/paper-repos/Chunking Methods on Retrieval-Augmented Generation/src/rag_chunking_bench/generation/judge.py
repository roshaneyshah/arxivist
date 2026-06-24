"""
generation/judge.py
===================
LLM-as-a-Judge module for Experiment 2 evaluation.

Paper: "Chunking Methods on Retrieval-Augmented Generation" (arXiv:2606.00881)
Section 3.2: "generated responses were evaluated against ground-truth answers
using an LLM-as-a-judge approach, with GPT-OSS-20B serving as the judge model.
The evaluation was conducted using a five-point Likert scale, where scores
ranged from 1 (poor answer quality and relevance) to 5 (highly accurate and
relevant answer)."

Reference: Zheng et al. 2023 (MT-Bench / LLM-as-a-Judge), NeurIPS 2023.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional


_JUDGE_SYSTEM = (
    "You are an expert evaluator assessing the quality and accuracy of "
    "AI-generated answers. You will be given a question, a reference answer, "
    "and a candidate answer. Score the candidate answer on a 5-point Likert scale:\n"
    "1 = Poor: incorrect, irrelevant, or missing key information\n"
    "2 = Below average: partially correct but with major errors or omissions\n"
    "3 = Average: partially correct with some useful information\n"
    "4 = Good: mostly correct and relevant with minor issues\n"
    "5 = Excellent: accurate, complete, and highly relevant\n\n"
    "Respond with ONLY a single integer (1, 2, 3, 4, or 5). No explanation."
)

_JUDGE_USER = (
    "Question: {query}\n\n"
    "Reference Answer: {ground_truth}\n\n"
    "Candidate Answer: {answer}\n\n"
    "Score (1-5):"
)


class LLMJudge:
    """
    LLM-as-a-Judge for RAG answer quality evaluation.

    Uses GPT-OSS-20B (same model as generator) to score generated answers
    on a 5-point Likert scale against ground-truth answers.

    Paper reference: Section 3.2; Table 4.

    Args:
        model: Judge model (default: gpt-oss-20b, as stated in paper).
        api_base: OpenAI-compatible API base URL.
        api_key: API key string.
        scale_min: Minimum scale value (1, as stated in paper).
        scale_max: Maximum scale value (5, as stated in paper).
    """

    def __init__(
        self,
        model: str = "gpt-oss-20b",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        scale_min: int = 1,   # SIR conf 0.99 — explicitly stated
        scale_max: int = 5,   # SIR conf 0.99 — explicitly stated
    ) -> None:
        self._model = model
        self._api_base = api_base
        self._api_key = api_key
        self._scale_min = scale_min
        self._scale_max = scale_max

    def _get_client(self):
        import openai
        kwargs = {}
        if self._api_base:
            kwargs["base_url"] = self._api_base
        if self._api_key:
            kwargs["api_key"] = self._api_key
        return openai.OpenAI(**kwargs)

    def _parse_score(self, response_text: str) -> int:
        """
        Extract integer score from judge response.

        Falls back to scale_min on parse failure.

        Args:
            response_text: Raw judge model output string.

        Returns:
            Integer in [scale_min, scale_max].
        """
        # Find first integer in response
        matches = re.findall(r"\b([1-5])\b", response_text.strip())
        if matches:
            score = int(matches[0])
            return max(self._scale_min, min(self._scale_max, score))
        return self._scale_min  # fallback on parse failure

    def score(self, query: str, answer: str, ground_truth: str) -> int:
        """
        Score a single (query, answer, ground_truth) triple.

        Paper Experiment 2: 5-point Likert scale, GPT-OSS-20B judge.

        Args:
            query: Original question string.
            answer: LLM-generated answer to evaluate.
            ground_truth: Reference answer string.

        Returns:
            Integer score in {1, 2, 3, 4, 5}.
        """
        prompt = _JUDGE_USER.format(
            query=query, ground_truth=ground_truth, answer=answer
        )
        client = self._get_client()
        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,  # deterministic judge output
            max_tokens=5,
        )
        raw = response.choices[0].message.content.strip()
        return self._parse_score(raw)

    def score_batch(self, records: List[Dict]) -> List[int]:
        """
        Score a batch of (query, answer, ground_truth) records.

        Args:
            records: List of dicts with keys 'query', 'answer', 'ground_truth'.

        Returns:
            List of integer scores.
        """
        return [
            self.score(r["query"], r["answer"], r["ground_truth"])
            for r in records
        ]

    def __repr__(self) -> str:
        return f"LLMJudge(model={self._model}, scale={self._scale_min}-{self._scale_max})"
