from __future__ import annotations

from copy import deepcopy
import time

import numpy as np

from .config import ExperimentConfig
from .generation.answer_generator import AnswerGenerator
from .indexing.bridge_index import load_bridge_index
from .indexing.sentence_index import SentenceIndex
from .models import EvidenceUnit, MethodResult, RecoveryDecision
from .recovery.gating import decide_recovery
from .recovery.operators import RecoveryOperators
from .recovery.probes import ProbeEngine
from .retrieval.embedding import EmbeddingClient
from .retrieval.initial_retriever import InitialRetriever
from .selection.evidence_selection import ContextSelector, SupportPredictor, deduplicate


class AdaptiveRecoveryPipeline:
    RECOVERY_STRATEGIES = {
        "none",
        "always_structural",
        "always_bridge",
        "always_both",
        "adaptive",
    }

    def __init__(
        self, config: ExperimentConfig, enable_generation: bool = True
    ) -> None:
        self.config = config
        self.sentence_index = SentenceIndex(config.sentence_index_dir)
        self.bridge_index = load_bridge_index(config.bridge_index_file)
        self.embedder = EmbeddingClient(
            config.embedding_model,
            config.embedding_mode,
            config.embedding_api_key,
            config.embedding_base_url,
        )
        self.retriever = InitialRetriever(config, self.sentence_index, self.embedder)
        self.probes = ProbeEngine(config, self.sentence_index, self.bridge_index)
        self.operators = RecoveryOperators(config, self.sentence_index)
        self.context_selector = ContextSelector(config, self.sentence_index)
        self.support_predictor = SupportPredictor(config, self.sentence_index)
        self.generator = None
        if enable_generation:
            self.generator = AnswerGenerator(
                config.generation_model,
                config.generation_api_keys,
                config.generation_base_url,
                config.generation_temperature,
                config.generation_max_tokens,
            )

    def retrieve(self, query: str) -> tuple[list[EvidenceUnit], np.ndarray, float]:
        started_at = time.perf_counter()
        initial, query_vector = self.retriever.retrieve(query)
        retrieval_time_ms = (time.perf_counter() - started_at) * 1000
        return initial, query_vector, retrieval_time_ms

    def run_variants(
        self,
        query: str,
        recovery_strategies: list[str],
        retrieved: tuple[list, np.ndarray, float] | None = None,
    ) -> dict[str, MethodResult]:
        if retrieved is None:
            retrieved = self.retrieve(query)
        return {
            strategy: self.run(query, strategy, retrieved)
            for strategy in recovery_strategies
        }

    def run(
        self,
        query: str,
        recovery_strategy: str = "adaptive",
        retrieved: tuple[list, np.ndarray, float] | None = None,
    ) -> MethodResult:
        if recovery_strategy not in self.RECOVERY_STRATEGIES:
            allowed = ", ".join(sorted(self.RECOVERY_STRATEGIES))
            raise ValueError(
                f"Unknown recovery strategy: {recovery_strategy}. Allowed: {allowed}"
            )

        if retrieved is None:
            initial, query_vector, retrieval_time_ms = self.retrieve(query)
        else:
            initial = deepcopy(retrieved[0])
            query_vector = retrieved[1]
            retrieval_time_ms = retrieved[2]
        recovery_started_at = time.perf_counter()

        structural_probes = []
        if recovery_strategy == "adaptive":
            structural_probes = self.probes.structural(initial, query_vector)
        structural_gain = max((probe.gain for probe in structural_probes), default=0.0)
        if recovery_strategy == "adaptive":
            structural_activated = (
                structural_gain > self.config.structural_gain_threshold
            )
        else:
            structural_activated = recovery_strategy in {
                "always_structural",
                "always_both",
            }
        structural = self.operators.structural(initial) if structural_activated else []

        bridge_probes = []
        if recovery_strategy in {"adaptive", "always_bridge", "always_both"}:
            bridge_seeds = deduplicate(initial + structural)
            bridge_probes = self.probes.bridge(
                bridge_seeds,
                initial + structural,
                query,
                query_vector,
                hop=1,
            )

        if recovery_strategy == "adaptive":
            decision = decide_recovery(structural_probes, bridge_probes, self.config)
        else:
            decision = RecoveryDecision(
                structural_activated=structural_activated,
                bridge_activated=recovery_strategy in {"always_bridge", "always_both"},
                structural_gain=structural_gain,
                bridge_gain=max((probe.gain for probe in bridge_probes), default=0.0),
            )

        if recovery_strategy == "adaptive":
            bridge_1_probes = [
                probe
                for probe in bridge_probes
                if probe.gain > self.config.bridge_gain_threshold
            ]
        else:
            bridge_1_probes = bridge_probes
        bridge_1 = (
            self.operators.bridge(bridge_1_probes, hop=1)
            if decision.bridge_activated
            else []
        )

        bridge_2 = []
        bridge_2_probes = []
        if (
            decision.bridge_activated
            and self.config.max_bridge_hops >= 2
            and bridge_1_probes
        ):
            hop_2_seeds = [probe.evidence for probe in bridge_1_probes]
            provisional = deduplicate(initial + structural + bridge_1)
            bridge_2_probes = self.probes.bridge(
                hop_2_seeds,
                provisional,
                query,
                query_vector,
                hop=2,
            )
            if recovery_strategy == "adaptive":
                bridge_2_probes = [
                    probe
                    for probe in bridge_2_probes
                    if probe.gain > self.config.second_hop_gain_threshold
                ]
            if bridge_2_probes:
                bridge_2 = self.operators.bridge(bridge_2_probes, hop=2)

        candidates = deduplicate(initial + structural + bridge_1 + bridge_2)
        context = self.context_selector.select(candidates, query_vector)
        supporting_facts = self.support_predictor.predict(context, query_vector)
        answer = self.generator.generate(query, context) if self.generator else ""

        bridge_chains = []
        for probe in bridge_1_probes + bridge_2_probes:
            if not probe.bridge_link:
                continue
            bridge_chains.append(
                {
                    "source_title": probe.bridge_link.source_title,
                    "source_sent_id": probe.bridge_link.source_sentence_id,
                    "bridge_entity": probe.bridge_link.bridge_entity,
                    "predicate": probe.bridge_link.predicate,
                    "target_title": probe.bridge_link.target_title,
                    "target_sent_id": probe.evidence.sent_id,
                    "gain": probe.gain,
                    "evidence_hop": probe.evidence.metadata.get("evidence_hop", 1),
                }
            )

        recovery_time_ms = (time.perf_counter() - recovery_started_at) * 1000
        return MethodResult(
            initial_evidence=initial,
            candidate_evidence=candidates,
            context_evidence=context,
            supporting_facts=supporting_facts,
            answer=answer,
            stats={
                "recovery_strategy": recovery_strategy,
                "structural_activated": decision.structural_activated,
                "bridge_activated": decision.bridge_activated,
                "second_bridge_hop": bool(bridge_2),
                "activation_pattern": decision.activation_pattern,
                "structural_gain": decision.structural_gain,
                "bridge_gain": decision.bridge_gain,
                "initial_units": len(initial),
                "structural_probe_count": len(structural_probes),
                "bridge_probe_count": len(bridge_probes),
                "second_hop_probe_count": len(bridge_2_probes),
                "structural_added_units": len(structural),
                "bridge_added_units": len(bridge_1) + len(bridge_2),
                "candidate_units": len(candidates),
                "candidate_tokens": sum(unit.token_count for unit in candidates),
                "selected_context_units": len(context),
                "selected_context_tokens": sum(unit.token_count for unit in context),
                "bridge_chains": bridge_chains,
                "retrieval_time_ms": retrieval_time_ms,
                "recovery_time_ms": recovery_time_ms,
                "time_ms": retrieval_time_ms + recovery_time_ms,
            },
        )
