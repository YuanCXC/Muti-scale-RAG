from __future__ import annotations

import numpy as np

from ..config import ExperimentConfig
from ..models import EvidenceUnit, ProbeCandidate, RecoveryDecision
from ..text_utils import cosine


class GainCalculator:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def score(
        self,
        operator: str,
        evidence: EvidenceUnit,
        candidate_vector: np.ndarray,
        query_vector: np.ndarray,
        base_vectors: list[np.ndarray],
        prior: float,
    ) -> ProbeCandidate:
        relevance = max(0.0, cosine(query_vector, candidate_vector))
        redundancy = max(
            (cosine(candidate_vector, vector) for vector in base_vectors), default=0.0
        )
        marginal = relevance * (1.0 - min(1.0, max(0.0, redundancy)))
        cost = evidence.token_count / max(self.config.context_budget, 1)
        gain = (
            self.config.gain_relevance_weight * relevance
            + self.config.gain_marginal_weight * marginal
            + self.config.gain_prior_weight * prior
            - self.config.gain_cost_weight * cost
        )
        return ProbeCandidate(
            operator=operator,
            evidence=evidence,
            relevance=relevance,
            marginal_semantic_gain=marginal,
            prior=prior,
            cost=cost,
            gain=gain,
        )


def decide_recovery(
    structural_probes: list[ProbeCandidate],
    bridge_probes: list[ProbeCandidate],
    config: ExperimentConfig,
) -> RecoveryDecision:
    structural_gain = max((probe.gain for probe in structural_probes), default=0.0)
    bridge_gain = max((probe.gain for probe in bridge_probes), default=0.0)
    return RecoveryDecision(
        structural_activated=structural_gain > config.structural_gain_threshold,
        bridge_activated=bridge_gain > config.bridge_gain_threshold,
        structural_gain=structural_gain,
        bridge_gain=bridge_gain,
    )
