from __future__ import annotations

from itertools import cycle
from threading import Lock
from typing import Any

from openai import OpenAI

from ..models import EvidenceUnit


class AnswerGenerator:
    def __init__(
        self,
        model: str,
        api_keys: list[str],
        base_url: str,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.clients = [
            OpenAI(api_key=api_key, base_url=base_url, timeout=60.0, max_retries=0)
            for api_key in api_keys
        ]
        self.client_cycle = cycle(enumerate(self.clients))
        self.client_lock = Lock()
        self.request_counts = [0 for _ in self.clients]

    def generate(self, question: str, evidence: list[EvidenceUnit]) -> str:
        return str(self.generate_with_metadata(question, evidence)["answer"])

    def generate_with_metadata(
        self,
        question: str,
        evidence: list[EvidenceUnit],
    ) -> dict[str, Any]:
        context = "\n".join(
            f"[{unit.title} | sentence {unit.sent_id}] {unit.text}" for unit in evidence
        )
        with self.client_lock:
            client_index, client = next(self.client_cycle)
            self.request_counts[client_index] += 1
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body={"thinking": {"type": "disabled"}},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer the question using only the supplied evidence. "
                        "Perform any necessary comparison, temporal reasoning, or relation "
                        "linking across the evidence. Do not refuse merely because the answer "
                        "is not stated in one sentence. Say that it cannot be determined only "
                        "when a required fact is genuinely absent. Return only the short answer "
                        "without explanation."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Evidence:\n{context}\n\nQuestion: {question}\nAnswer:",
                },
            ],
        )
        usage = response.usage
        return {
            "answer": (response.choices[0].message.content or "").strip(),
            "prompt_tokens": int(usage.prompt_tokens or 0) if usage else 0,
            "completion_tokens": int(usage.completion_tokens or 0) if usage else 0,
            "total_tokens": int(usage.total_tokens or 0) if usage else 0,
        }
