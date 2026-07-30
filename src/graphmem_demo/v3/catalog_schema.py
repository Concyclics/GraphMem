from __future__ import annotations

from dataclasses import dataclass, field


GRAPHMEM_V3_SCHEMA = "graphmem_v3"


@dataclass
class EventFrameV3:
    """A role-neutral, cross-turn event aggregate used for coarse routing."""

    frame_id: str
    question_id: str
    label: str
    label_key: str
    participant_keys: list[str] = field(default_factory=list)
    status: str = "unknown"
    event_time: str | None = None
    observed_at: str | None = None
    session_ids: list[str] = field(default_factory=list)
    claim_ids: list[str] = field(default_factory=list)
    event_ids: list[str] = field(default_factory=list)
    source_turn_ids: list[str] = field(default_factory=list)
    semantic_type_keys: list[str] = field(default_factory=list)
    attributes: dict[str, list[str]] = field(default_factory=dict)
    confidence: float = 1.0
    retrieval_text: str = ""
    embedding: list[float] | None = None
    schema_version: str = GRAPHMEM_V3_SCHEMA

    @property
    def node_id(self) -> str:
        return self.frame_id


@dataclass
class OperandRecordV3:
    """Lossless typed operand projected from one atomic claim."""

    operand_id: str
    question_id: str
    subject_key: str
    predicate_key: str
    object_key: str
    object_text: str
    context_key: str = ""
    event_frame_id: str | None = None
    state_op: str = "none"
    polarity: str = "unknown"
    modality: str = "unknown"
    event_time: str | None = None
    observed_at: str | None = None
    recurrence_days: list[str] = field(default_factory=list)
    recurrence_count: int | None = None
    quantity: float | None = None
    unit: str = ""
    source_claim_ids: list[str] = field(default_factory=list)
    source_turn_ids: list[str] = field(default_factory=list)
    session_ids: list[str] = field(default_factory=list)
    event_type_keys: list[str] = field(default_factory=list)
    confidence: float = 1.0
    retrieval_text: str = ""
    embedding: list[float] | None = None
    object_embedding: list[float] | None = None
    schema_version: str = GRAPHMEM_V3_SCHEMA

    @property
    def node_id(self) -> str:
        return self.operand_id
