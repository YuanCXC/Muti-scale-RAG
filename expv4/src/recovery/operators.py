from __future__ import annotations

from ..config import ExperimentConfig
from ..indexing.sentence_index import SentenceIndex
from ..models import EvidenceUnit, ProbeCandidate


class RecoveryOperators:
    def __init__(self, config: ExperimentConfig, sentence_index: SentenceIndex) -> None:
        self.config = config
        self.sentence_index = sentence_index

    def structural(self, initial: list[EvidenceUnit]) -> list[EvidenceUnit]:
        recovered: dict[tuple[str, int], EvidenceUnit] = {}
        for seed in initial:
            for record in self.sentence_index.title_sentences(seed.title):
                if abs(record.sent_id - seed.sent_id) > self.config.structural_window:
                    continue
                unit = EvidenceUnit(
                    title=record.title,
                    sent_id=record.sent_id,
                    text=record.text,
                    score=seed.score,
                    source="structural",
                    vector_id=record.vector_id,
                    metadata={"anchor_sent_id": seed.sent_id},
                )
                current = recovered.get(unit.key)
                if current is None or unit.score > current.score:
                    recovered[unit.key] = unit
        return list(recovered.values())

    def bridge(
        self,
        probes: list[ProbeCandidate],
        hop: int,
    ) -> list[EvidenceUnit]:
        recovered: dict[tuple[str, int], EvidenceUnit] = {}
        for probe in probes:
            center = probe.evidence
            for record in self.sentence_index.title_sentences(center.title):
                if (
                    abs(record.sent_id - center.sent_id)
                    > self.config.bridge_target_window
                ):
                    continue
                unit = EvidenceUnit(
                    title=record.title,
                    sent_id=record.sent_id,
                    text=record.text,
                    score=probe.gain
                    if record.sent_id != center.sent_id
                    else max(probe.gain, center.score),
                    source="bridge",
                    vector_id=record.vector_id,
                    metadata={
                        **center.metadata,
                        "target_center_sent_id": center.sent_id,
                        "evidence_hop": hop,
                        "probe_gain": probe.gain,
                    },
                )
                current = recovered.get(unit.key)
                if current is None or unit.score > current.score:
                    recovered[unit.key] = unit
        return list(recovered.values())
