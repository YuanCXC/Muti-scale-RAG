from __future__ import annotations

import json
import re
from itertools import cycle
from threading import Lock
from typing import Any

from openai import OpenAI

from ..models import EvidenceUnit


class SemanticEvaluator:
    protocol_version = "v4.2-complete-evidence-chain"
    metric_names = (
        "accuracy",
        "faithfulness",
        "answer_relevance",
        "context_relevance",
    )

    def __init__(self, model: str, api_keys: list[str], base_url: str) -> None:
        self.model = model
        self.clients = [
            OpenAI(api_key=api_key, base_url=base_url, timeout=60.0, max_retries=0)
            for api_key in api_keys
        ]
        self.client_cycle = cycle(enumerate(self.clients))
        self.client_lock = Lock()
        self.request_counts = [0 for _ in self.clients]

    @staticmethod
    def is_refusal(answer: str) -> bool:
        text = re.sub(r"\s+", " ", str(answer or "").strip().lower())
        patterns = (
            "cannot be determined",
            "can't be determined",
            "can not be determined",
            "insufficient evidence",
            "insufficient information",
            "not enough information",
            "unable to determine",
            "not provided in the evidence",
        )
        return not text or any(pattern in text for pattern in patterns)

    def evaluate(
        self,
        question: str,
        reference_answer: str,
        generated_answer: str,
        evidence: list[EvidenceUnit],
    ) -> dict[str, Any]:
        context = "\n".join(
            f"[{unit.title} | sentence {unit.sent_id}] {unit.text}" for unit in evidence
        )
        prompt = f"""You are a strict evaluator of retrieval-augmented question answering.
Score every metric from 0 to 1.

Question: {question}
Reference answer: {reference_answer}
Generated answer: {generated_answer}

Retrieved evidence:
{context}

Metrics:
1. accuracy: agreement with the reference answer, including valid aliases and equivalent wording.
2. faithfulness: whether the generated answer is supported by and does not contradict the evidence.
3. answer_relevance: whether the answer directly and concisely addresses the question.
4. context_relevance: whether the retrieved evidence contains the complete set of facts needed
   to answer the question, without being dominated by irrelevant material.

Rules:
- Judge the generated answer, not its writing style.
- A refusal is incorrect when the reference answer is specific.
- Give context_relevance 1.0 only when the complete evidence chain is present.
- When one useful fact is present but another required relation or supporting fact is missing,
  context_relevance must be below 1.0; use about 0.5 for a materially incomplete chain.
- Use context_relevance 0.0 when the evidence contains no fact useful for the answer.
- Return only one JSON object with exactly these numeric keys:
  accuracy, faithfulness, answer_relevance, context_relevance.
"""
        with self.client_lock:
            client_index, client = next(self.client_cycle)
            self.request_counts[client_index] += 1
        response = client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            max_tokens=256,
            extra_body={"thinking": {"type": "disabled"}},
            messages=[{"role": "user", "content": prompt}],
        )
        content = (response.choices[0].message.content or "").strip()
        parsed = self._parse_json_object(content)
        scores: dict[str, Any] = {}
        for metric in self.metric_names:
            value = float(parsed[metric])
            scores[metric] = max(0.0, min(1.0, value))

        refusal = self.is_refusal(generated_answer)
        if refusal:
            scores["accuracy"] = 0.0
            scores["answer_relevance"] = min(scores["answer_relevance"], 0.25)

        usage = response.usage
        scores.update(
            {
                "is_refusal": refusal,
                "raw_evaluation": content,
                "prompt_tokens": int(usage.prompt_tokens or 0) if usage else 0,
                "completion_tokens": int(usage.completion_tokens or 0) if usage else 0,
                "total_tokens": int(usage.total_tokens or 0) if usage else 0,
            }
        )
        return scores

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        cleaned = text.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError("Semantic evaluator did not return a JSON object")
        parsed = json.loads(match.group())
        missing = [name for name in SemanticEvaluator.metric_names if name not in parsed]
        if missing:
            raise ValueError(f"Semantic evaluator omitted metrics: {', '.join(missing)}")
        return parsed
