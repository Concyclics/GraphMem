"""Usage records the judge client emits.

Lifted verbatim from the retired ``graphmem_demo.models``; the judge is the
only surviving consumer.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


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
class DeepSeekCallRecord:
    question_id: str
    variant: str
    stage: str
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
    breakdown_inferred: bool = False
    excluded_from_budget: bool = False


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
