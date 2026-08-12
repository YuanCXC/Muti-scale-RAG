from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SentenceRecord:
    vector_id: int
    title: str
    sent_id: int
    text: str


@dataclass
class EvidenceUnit:
    title: str
    sent_id: int
    text: str
    score: float
    source: str
    vector_id: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, int]:
        return self.title, self.sent_id

    @property
    def token_count(self) -> int:
        return max(1, len(self.text) // 4)


@dataclass(frozen=True)
class BridgeLink:
    source_title: str
    source_sentence_id: int
    bridge_entity: str
    target_title: str
    predicate: str
    entity_role: str
    match_confidence: float
    grounding_confidence: float
    entity_specificity: float
    bridge_prior: float


@dataclass
class ProbeCandidate:
    operator: str
    evidence: EvidenceUnit
    relevance: float
    marginal_semantic_gain: float
    prior: float
    cost: float
    gain: float
    bridge_link: BridgeLink | None = None


@dataclass(frozen=True)
class RecoveryDecision:
    structural_activated: bool
    bridge_activated: bool
    structural_gain: float
    bridge_gain: float

    @property
    def activation_pattern(self) -> str:
        if self.structural_activated and self.bridge_activated:
            return "structural_bridge"
        if self.structural_activated:
            return "structural_only"
        if self.bridge_activated:
            return "bridge_only"
        return "none"


@dataclass
class MethodResult:
    initial_evidence: list[EvidenceUnit]
    candidate_evidence: list[EvidenceUnit]
    context_evidence: list[EvidenceUnit]
    supporting_facts: list[tuple[str, int]]
    answer: str = ""
    stats: dict[str, Any] = field(default_factory=dict)
