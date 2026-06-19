from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


def to_dict(value: Any) -> dict[str, Any]:
    return asdict(value)


@dataclass
class QuestionCase:
    question_id: str
    question_type: str
    question: str
    answer: Any
    question_date: str | None
    haystack_sessions: list[list[dict[str, Any]]]
    haystack_session_ids: list[str]
    haystack_dates: list[str | None]
    answer_session_ids: list[str]
    memory_cache_key: str | None = None


@dataclass
class LeafNode:
    node_id: str
    question_id: str
    session_id: str
    session_date: str | None
    turn_index: int
    raw_text: str
    user_text: str
    message_count: int
    retrieval_text: str = ""
    embedding: list[float] | None = None


@dataclass
class SummaryNode:
    node_id: str
    question_id: str
    session_id: str
    session_date: str | None
    level: int
    child_ids: list[str]
    leaf_ids: list[str]
    summary: str
    retrieval_text: str = ""
    anchor_terms: dict[str, list[str]] = field(default_factory=dict)
    summary_mode: str = "legacy_kway"
    summary_schema_version: str = "minimal_memory_v1"
    parsed_summary: dict[str, Any] | None = None
    raw_summary_text: str = ""
    truncated: bool = False
    parse_error: str | None = None
    source_level: int = 0
    embedding: list[float] | None = None


@dataclass
class GraphEdge:
    src: str
    dst: str
    score: float
    relation: Literal[
        "semantic_neighbor",
        "temporal_neighbor",
        "keyword_neighbor",
        "state_neighbor",
        "entity_neighbor",
        "time_neighbor",
        "event_neighbor",
        "update_neighbor",
    ]


@dataclass
class RetrievedContext:
    question_id: str
    variant: str
    summary_node_ids: list[str]
    leaf_node_ids: list[str]
    edge_count: int
    context_text: str
    answer_session_hit: bool
    retrieved_session_ids: list[str]
    latency_sec: float
    answer_session_all_hit: bool = False
    answer_session_recall: float = 0.0
    retrieved_answer_session_count: int = 0
    gold_answer_session_count: int = 0


@dataclass
class DeepSeekCallRecord:
    question_id: str
    variant: str
    stage: Literal[
        "build_summary_leaf",
        "build_summary_internal",
        "build_summary_raw_group",
        "build_summary_session_direct",
        "build_summary_session_merge",
        "answer_qa",
    ]
    call_id: str
    model: str
    thinking_mode: Literal["enabled", "disabled", "none"]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0
    reasoning_tokens: int = 0
    latency_sec: float = 0.0
    retry_count: int = 0
    error_status: str | None = None
    finish_reason: str | None = None
    max_tokens: int | None = None
    response_format: str | None = None


@dataclass
class CompressionRecord:
    question_id: str
    variant: str
    stage: str
    origin_tokens: int
    compressed_tokens: int
    latency_sec: float
    compressor: str
    chunk_count: int = 1
    error_status: str | None = None


@dataclass
class EmbeddingCallRecord:
    question_id: str
    variant: str
    item_count: int
    prompt_tokens: int
    total_tokens: int
    latency_sec: float
    model: str
    error_status: str | None = None


@dataclass
class QuestionStats:
    question_id: str
    variant: str
    session_count: int
    leaf_count: int
    summary_count: int
    edge_count: int
    build_prompt_tokens: int
    build_completion_tokens: int
    answer_prompt_tokens: int
    answer_completion_tokens: int
    reasoning_tokens: int
    total_deepseek_tokens: int
    deepseek_call_count: int
    build_latency_sec: float
    retrieval_latency_sec: float
    answer_latency_sec: float
    retrieved_answer_session_hit: bool
    retrieved_answer_session_all_hit: bool = False
    retrieved_answer_session_recall: float = 0.0
    retrieved_answer_session_count: int = 0
    gold_answer_session_count: int = 0
    wall_time_sec: float = 0.0
    summary_parse_error_count: int = 0
    summary_truncation_count: int = 0
    build_calls_per_session: float = 0.0
    ready_job_counts: list[dict[str, Any]] = field(default_factory=list)
    peak_inflight_deepseek: int = 0


@dataclass
class VariantStats:
    variant: str
    question_count: int
    session_count: int
    leaf_count: int
    summary_count: int
    edge_count: int
    build_prompt_tokens: int
    build_completion_tokens: int
    answer_prompt_tokens: int
    answer_completion_tokens: int
    reasoning_tokens: int
    total_deepseek_tokens: int
    deepseek_call_count: int
    avg_tokens_per_question: float
    avg_tokens_per_session: float
    retrieval_answer_session_hit_rate: float
    retrieval_answer_session_all_hit_rate: float = 0.0
    avg_retrieved_answer_session_recall: float = 0.0
    token_budget_avg_under_300k: bool = False
    build_latency_sec: float = 0.0
    retrieval_latency_sec: float = 0.0
    answer_latency_sec: float = 0.0
    wall_time_sec: float = 0.0
    summary_parse_error_count: int = 0
    summary_truncation_count: int = 0
    build_calls_per_session: float = 0.0
    peak_inflight_deepseek: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
