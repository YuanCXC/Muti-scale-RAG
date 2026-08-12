from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class ExperimentConfig:
    validation_path: str = "data/hotpotqa/validation-00000-of-00001.parquet"
    triplet_documents_path: str = "data/hotpotqa/valid_title_sentence.json"
    v4_data_dir: str = "data/v4"
    v4_work_data_dir: str = "data/hotpotqa/v4"
    results_dir: str = "expv4/results"
    bridge_index_path: str = "data/v4/bridge_index_semantic_v4.pkl"

    embedding_model: str = "BAAI/bge-m3"
    embedding_mode: str = "api"
    embedding_batch_size: int = 64
    embedding_workers: int = 2
    vector_dimension: int = 1024

    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    vector_top_k: int = 20
    keyword_top_k: int = 20
    rerank_top_k: int = 7

    structural_window: int = 2
    bridge_target_window: int = 1
    max_structural_probes: int = 8
    max_bridge_seeds: int = 4
    max_bridge_links_per_seed: int = 3
    max_bridge_hops: int = 2
    bridge_max_document_frequency: int = 300
    anchor_sigma: float = 2.0

    gain_relevance_weight: float = 0.45
    gain_marginal_weight: float = 0.30
    gain_prior_weight: float = 0.25
    gain_cost_weight: float = 0.10
    structural_gain_threshold: float = 0.42
    bridge_gain_threshold: float = 0.42
    second_hop_gain_threshold: float = 0.44

    context_budget: int = 3600
    max_context_units: int = 12
    support_min_facts: int = 2
    support_max_facts: int = 5
    support_margin: float = 0.01

    generation_model: str = "deepseek-chat"
    generation_temperature: float = 0.0
    generation_max_tokens: int = 128

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_database: str = "neo4j"
    bridge_annotation_model: str = "deepseek-chat"
    bridge_annotation_workers: int = 4
    bridge_annotation_max_tokens: int = 2400

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ExperimentConfig":
        values: dict[str, Any] = {}
        if path:
            with Path(path).open("r", encoding="utf-8") as handle:
                values = json.load(handle)
        allowed = {item.name for item in fields(cls)}
        config = cls(**{key: value for key, value in values.items() if key in allowed})

        config.embedding_model = os.getenv("EMBEDDING_MODEL", config.embedding_model)
        config.embedding_mode = os.getenv("EMBEDDING_MODE", config.embedding_mode)
        config.vector_dimension = int(os.getenv("VECTOR_DIM", config.vector_dimension))
        config.rerank_model = os.getenv("RERANK_MODEL", config.rerank_model)
        config.generation_model = os.getenv(
            "BIGMOD_API_MODEL",
            os.getenv("LLM_MODEL", config.generation_model),
        )
        config.neo4j_uri = os.getenv("NEO4J_URI", config.neo4j_uri)
        config.neo4j_user = os.getenv("NEO4J_USER", config.neo4j_user)
        config.neo4j_database = os.getenv("NEO4J_DATABASE", config.neo4j_database)
        config.bridge_annotation_model = os.getenv(
            "BRIDGE_ANNOTATION_MODEL",
            os.getenv(
                "BIGMOD_API_MODEL",
                os.getenv("LLM_MODEL", config.bridge_annotation_model),
            ),
        )
        return config

    def resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def validation_file(self) -> Path:
        return self.resolve(self.validation_path)

    @property
    def triplet_documents_file(self) -> Path:
        return self.resolve(self.triplet_documents_path)

    @property
    def data_dir(self) -> Path:
        return self.resolve(self.v4_data_dir)

    @property
    def work_data_dir(self) -> Path:
        return self.resolve(self.v4_work_data_dir)

    @property
    def sentence_file(self) -> Path:
        return self.work_data_dir / "official_sentences.jsonl"

    @property
    def sentence_index_dir(self) -> Path:
        return self.data_dir / "sentence_index"

    @property
    def bridge_index_file(self) -> Path:
        return self.resolve(self.bridge_index_path)

    @property
    def raw_bridge_index_file(self) -> Path:
        return self.work_data_dir / "bridge_index.pkl"

    @property
    def bridge_relation_dataset_file(self) -> Path:
        return self.data_dir / "bridge_relation_dataset_v4.jsonl"

    @property
    def bridge_edge_annotations_file(self) -> Path:
        return self.work_data_dir / "bridge_edge_semantic_annotations_v4.jsonl"

    @property
    def neo4j_bridge_paths_file(self) -> Path:
        return self.work_data_dir / "neo4j_bridge_paths_v4.jsonl"

    @property
    def neo4j_bridge_supervision_file(self) -> Path:
        return self.work_data_dir / "neo4j_bridge_supervision_v4.jsonl"

    @property
    def neo4j_bridge_annotations_file(self) -> Path:
        return self.work_data_dir / "neo4j_bridge_semantic_annotations_v4.jsonl"

    @property
    def bridge_supervision_file(self) -> Path:
        return self.work_data_dir / "bridge_supervision_v4.jsonl"

    @property
    def bridge_semantic_annotations_file(self) -> Path:
        return self.work_data_dir / "bridge_semantic_annotations_v4.jsonl"

    @property
    def bridge_clean_dataset_file(self) -> Path:
        return self.work_data_dir / "bridge_clean_dataset_v4.jsonl"

    @property
    def output_dir(self) -> Path:
        return self.resolve(self.results_dir)

    @property
    def embedding_api_key(self) -> str:
        return os.getenv("EMBED_API_KEY") or os.getenv("OPENAI_API_KEY", "")

    @property
    def embedding_base_url(self) -> str:
        return os.getenv("EMBED_BASE_URL") or os.getenv("OPENAI_API_BASE", "")

    @property
    def rerank_api_key(self) -> str:
        return os.getenv("RERANK_API_KEY") or os.getenv("OPENAI_API_KEY", "")

    @property
    def rerank_base_url(self) -> str:
        return os.getenv("RERANK_BASE_URL") or os.getenv("OPENAI_API_BASE", "")

    @property
    def generation_api_key(self) -> str:
        return self.generation_api_keys[0] if self.generation_api_keys else ""

    @property
    def generation_api_keys(self) -> list[str]:
        keys = [
            os.getenv("BIGMOD_API_KEY", ""),
            os.getenv("BIGMOD_API_KEY_2", ""),
            os.getenv("BIGMOD_API_KEY_3", ""),
            os.getenv("BIGMOD_API_KEY_4", ""),
            os.getenv("BIGMOD_API_KEY_5", ""),
        ]
        configured = [key for key in keys if key]
        if configured:
            return configured
        fallback = os.getenv("OPENAI_API_KEY", "")
        return [fallback] if fallback else []

    @property
    def generation_base_url(self) -> str:
        return os.getenv("BIGMOD_API_URL") or os.getenv("OPENAI_API_BASE", "")

    @property
    def neo4j_password(self) -> str:
        return os.getenv("NEO4J_PASSWORD", "")
