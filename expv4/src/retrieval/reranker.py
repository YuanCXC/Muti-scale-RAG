from __future__ import annotations

import httpx

from ..models import EvidenceUnit


class APIReranker:
    def __init__(self, model: str, api_key: str, base_url: str) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=60.0)

    def rerank(
        self,
        query: str,
        candidates: list[EvidenceUnit],
        top_k: int,
    ) -> list[EvidenceUnit]:
        if not candidates:
            return []
        response = self.client.post(
            f"{self.base_url}/rerank",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "query": query,
                "documents": [unit.text for unit in candidates],
                "top_n": min(top_k, len(candidates)),
            },
        )
        response.raise_for_status()
        ranked: list[EvidenceUnit] = []
        for result in response.json().get("results", []):
            unit = candidates[int(result["index"])]
            unit.score = float(result.get("relevance_score", unit.score))
            unit.metadata["rerank_score"] = unit.score
            ranked.append(unit)
        return ranked or candidates[:top_k]
