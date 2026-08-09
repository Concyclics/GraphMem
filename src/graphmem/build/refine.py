from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from ..config import CacheIdentity, GraphMemV5Config, config_hash
from ..domain import RelationType, canonical_json, stable_id
from ..storage import SQLiteGraphStore


@dataclass(frozen=True, slots=True)
class RefineCandidate:
    candidate_id: str
    kind: str
    left_id: str
    right_id: str
    left_context: str
    right_context: str
    allowed_relations: tuple[str, ...]
    score_margin: float
    cross_scene: bool
    cross_session: bool
    affects_portal: bool
    similarity: float | None = None
    gate_level: int = 0
    estimated_child_pairs: int = 0
    priority: float = 0.0


@dataclass(frozen=True, slots=True)
class RefineDecision:
    candidate_id: str
    decision: str
    confidence: float
    source: str


PROMPT_VERSION = "graphmem-v5-selective-refine-v2-compact"
SYSTEM_PROMPT = (
    "Resolve only the supplied graph candidates. Do not invent entities or endpoints. "
    "Return one compact JSON object per input line using keys i,d,c, where i is the supplied "
    "integer row index, d is an offered candidate/relation or NONE, and c is confidence. "
    "No markdown and no explanation."
)


class Qwen30BRefiner:
    """Value-gated single-model refiner with complete, cacheable call logging."""

    def __init__(self, store: SQLiteGraphStore, config: GraphMemV5Config,
                 dataset_hash: str, client: Any | None = None) -> None:
        self.store = store
        self.config = config
        self.dataset_hash = dataset_hash
        self.prompt_hash = hashlib.sha256(
            (PROMPT_VERSION + "\n" + SYSTEM_PROMPT).encode("utf-8")
        ).hexdigest()
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url=config.models.llm_base_url, api_key="local")
        self.client = client

    def eligible(self, candidate: RefineCandidate) -> bool:
        if candidate.similarity is not None:
            mode = self.config.edges.refine_mode
            ambiguous = (self.config.edges.low_threshold <= candidate.similarity
                         < self.config.edges.high_threshold)
            if mode == "none":
                return False
            if mode == "ambiguous_only":
                return ambiguous
            if mode == "high_value_only":
                return ambiguous and (
                    candidate.cross_session or candidate.estimated_child_pairs > 1)
            if mode == "all_bounded_candidates":
                return candidate.similarity >= self.config.edges.low_threshold
        return (
            candidate.score_margin < self.config.scenes.coreference_margin
            and (candidate.cross_scene or candidate.cross_session)
            and (candidate.affects_portal or candidate.cross_session)
        )

    def refine(self, memory_id: str, candidates: Sequence[RefineCandidate]) -> tuple[
        tuple[RefineDecision, ...], tuple[str, ...]
    ]:
        unique = {candidate.candidate_id: candidate for candidate in candidates if self.eligible(candidate)}
        ordered = sorted(unique.values(), key=lambda row: (-row.priority, row.candidate_id))
        turn_count = len(self.store.turns(memory_id))
        rate = self.config.edges.max_refine_calls_per_1000_turns
        max_batches = 0 if rate == 0 else max(1, math.ceil(turn_count * rate / 1000))
        decisions: list[RefineDecision] = []
        truncated: list[str] = []
        size = self.config.edges.refine_batch_size
        batches = [ordered[index:index + size] for index in range(0, len(ordered), size)]
        for batch_index, batch in enumerate(batches):
            if batch_index >= max_batches:
                truncated.extend(item.candidate_id for item in batch)
                continue
            decisions.extend(self._call(memory_id, batch))
        return tuple(decisions), tuple(truncated)

    def _call(self, memory_id: str, batch: Sequence[RefineCandidate]) -> list[RefineDecision]:
        endpoint_limit = self.config.models.refine_input_tokens_per_endpoint
        rows = []
        for item_index, item in enumerate(batch):
            rows.append({
                "i": item_index,
                "candidate_id": item.candidate_id,
                "kind": item.kind,
                "left_id": item.left_id,
                "right_id": item.right_id,
                "left": self._truncate(item.left_context, endpoint_limit),
                "right": self._truncate(item.right_context, endpoint_limit),
                "allowed": item.allowed_relations,
                "priority": round(item.priority, 6),
            })
        payload = "\n".join(canonical_json(row) for row in rows)
        identity = CacheIdentity(
            dataset_hash=self.dataset_hash,
            model_id=self.config.models.llm_model,
            prompt_hash=self.prompt_hash,
            schema_version=self.config.schema_version,
            config_hash=config_hash(self.config),
            stage="selective_refine:" + hashlib.sha256(payload.encode()).hexdigest(),
        )
        cache_key = identity.key()
        request = {
            "model": self.config.models.llm_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": payload},
            ],
            "temperature": 0,
            "max_tokens": self.config.models.bridge_refine_output_tokens
            if any(item.cross_session for item in batch)
            else self.config.models.refine_output_tokens,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        cached = self.store.cache_get(cache_key)
        started = time.perf_counter()
        retry_count = 0
        if cached:
            response_payload = cached["response"]
            prior = cached["usage"]
            cached_tokens = int(prior.get("cached_input_tokens", 0)) + int(
                prior.get("uncached_input_tokens", 0)
            )
            usage = {
                "cached_input_tokens": cached_tokens,
                "uncached_input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "total_tokens": cached_tokens,
            }
            is_cached = True
        else:
            response = self.client.chat.completions.create(**request)
            message = response.choices[0].message
            reasoning = getattr(message, "reasoning_content", None)
            if reasoning:
                raise RuntimeError("Qwen refine returned forbidden reasoning content")
            content = message.content or ""
            usage_object = getattr(response, "usage", None)
            usage = {
                "cached_input_tokens": 0,
                "uncached_input_tokens": int(getattr(usage_object, "prompt_tokens", 0) or 0),
                "output_tokens": int(getattr(usage_object, "completion_tokens", 0) or 0),
                "reasoning_tokens": 0,
                "total_tokens": int(getattr(usage_object, "total_tokens", 0) or 0),
            }
            response_payload = {"content": content, "model": getattr(response, "model", "")}
            self.store.cache_put(cache_key, "selective_refine", request, response_payload, usage, self.prompt_hash)
            is_cached = False
        latency = (time.perf_counter() - started) * 1000
        occurrence = int(self.store._read_one(
            "SELECT count(*) FROM llm_calls WHERE memory_id=? AND cache_key=?",
            (memory_id, cache_key),
        )[0])
        call_id = stable_id("llm-call", memory_id, cache_key, is_cached, occurrence)
        self.store.log_llm_call(
            call_id=call_id, memory_id=memory_id, stage="selective_refine",
            cache_key=cache_key, cached=is_cached, request=request, response=response_payload,
            usage=usage, latency_ms=latency, retry_count=retry_count,
            batch_size=len(batch), prompt_hash=self.prompt_hash,
        )
        allowed = {item.candidate_id: set(item.allowed_relations) | {"NONE"} for item in batch}
        aliases = {index: item.candidate_id for index, item in enumerate(batch)}
        parsed: dict[str, RefineDecision] = {}
        for line in str(response_payload.get("content", "")).splitlines():
            try:
                row = json.loads(line)
                candidate_id = (aliases[int(row["i"])] if "i" in row else str(row["candidate_id"]))
                decision = str(row.get("d", row.get("decision")))
                confidence = float(row.get("c", row.get("confidence", 0.0)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if candidate_id in allowed and decision in allowed[candidate_id] and 0 <= confidence <= 1:
                parsed[candidate_id] = RefineDecision(candidate_id, decision, confidence, "qwen30b")
        return [parsed.get(item.candidate_id, RefineDecision(
            item.candidate_id, "NONE", 0.0, "parse_fallback"
        )) for item in batch]

    @staticmethod
    def _truncate(text: str, token_limit: int) -> str:
        # Conservative whitespace approximation keeps endpoint payload bounded even
        # when a tokenizer service is unavailable during build preflight.
        words = text.split()
        return " ".join(words[: max(1, int(token_limit / 1.3))])
