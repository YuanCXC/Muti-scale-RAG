from __future__ import annotations

import time
from typing import Sequence

import numpy as np
from openai import OpenAI, RateLimitError


class EmbeddingClient:
    def __init__(
        self,
        model: str,
        mode: str,
        api_key: str,
        base_url: str,
        max_rate_limit_retries: int = 8,
    ) -> None:
        self.model = model
        self.mode = mode
        self._local_model = None
        self._client = None
        self.max_rate_limit_retries = max_rate_limit_retries

        if mode == "local":
            from sentence_transformers import SentenceTransformer

            self._local_model = SentenceTransformer(model)
        else:
            self._client = OpenAI(api_key=api_key, base_url=base_url)

    def embed(self, texts: str | Sequence[str]) -> np.ndarray:
        batch = [texts] if isinstance(texts, str) else list(texts)
        if not batch:
            return np.empty((0, 0), dtype=np.float32)

        if self._local_model is not None:
            vectors = self._local_model.encode(
                batch,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        else:
            response = self._embed_api(batch)
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors = np.asarray([item.embedding for item in ordered], dtype=np.float32)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-12)
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def _embed_api(self, batch: list[str]):
        for attempt in range(self.max_rate_limit_retries + 1):
            try:
                return self._client.embeddings.create(model=self.model, input=batch)
            except RateLimitError as exc:
                if attempt >= self.max_rate_limit_retries:
                    raise
                retry_after = None
                if exc.response is not None:
                    retry_after = exc.response.headers.get("retry-after")
                delay = (
                    float(retry_after)
                    if retry_after
                    else min(60.0, 2.0 ** (attempt + 1))
                )
                time.sleep(max(1.0, delay))
        raise RuntimeError("Embedding request exhausted rate-limit retries")
