from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import math
from typing import Any, Callable

from ..clients import rough_token_count
from ..models import DeepSeekCallRecord, QuestionCase
from .build import (
    build_hypergraph,
    build_turn_nodes,
    parse_session_extraction,
    session_extraction_messages,
    validate_hypergraph,
)
from .event_entities import (
    event_entity_candidate_payload,
    parse_event_entities,
)
from .reference_consolidation import (
    consolidation_candidate_payload,
    parse_event_identity_edges,
    parse_reference_edges,
    reference_consolidation_messages,
)
from .schema import V3Index


@dataclass
class V3BuildResult:
    index: V3Index
    records: list[DeepSeekCallRecord]
    diagnostics: list[dict[str, Any]]
    parse_error_count: int


_BUILD_BUDGET_RESERVE = 3_000


def _message_token_estimate(messages: list[dict[str, str]]) -> int:
    text = "\n".join(str(message.get("content") or "") for message in messages)
    if not text:
        return 0
    # Slightly more conservative than the normal report estimator.  The
    # reserve below absorbs provider chat framing/tokenizer differences.
    return max(rough_token_count(text), math.ceil(len(text.encode("utf-8")) / 3.2)) + 64


def _uniform_session_completion_cap(
    prompt_estimates: list[int],
    *,
    requested_max_tokens: int,
    build_budget_tokens: int,
) -> int:
    """Pre-allocate a hard per-session output cap within the build budget."""
    if not prompt_estimates:
        return requested_max_tokens
    completion_budget = (
        build_budget_tokens
        - _BUILD_BUDGET_RESERVE
        - sum(prompt_estimates)
    )
    return min(
        requested_max_tokens,
        max(256, completion_budget // len(prompt_estimates)),
    )


def _bounded_optional_max_tokens(
    *,
    spent_tokens: int,
    prompt_estimate: int,
    requested_max_tokens: int,
    build_budget_tokens: int,
    minimum: int,
) -> int:
    remaining = (
        build_budget_tokens
        - _BUILD_BUDGET_RESERVE
        - spent_tokens
        - prompt_estimate
    )
    if remaining < minimum:
        return 0
    return min(requested_max_tokens, remaining)


def build_index(
    *,
    case: QuestionCase,
    variant: str,
    chat: Callable[..., Any],
    embed: Callable[[list[Any], str, str], None],
    max_tokens: int,
    workers: int,
    build_budget_tokens: int = 300_000,
) -> V3BuildResult:
    turns = build_turn_nodes(case)
    grouped: dict[str, list[Any]] = {}
    for turn in turns:
        grouped.setdefault(turn.session_id, []).append(turn)
    session_dates = dict(zip(case.haystack_session_ids, case.haystack_dates))

    extraction_messages = {
        session_id: session_extraction_messages(
            session_id, session_dates.get(session_id), grouped[session_id]
        )
        for session_id in grouped
    }
    prompt_estimates = [
        _message_token_estimate(extraction_messages[session_id])
        for session_id in grouped
    ]
    extraction_max_tokens = _uniform_session_completion_cap(
        prompt_estimates,
        requested_max_tokens=max_tokens,
        build_budget_tokens=build_budget_tokens,
    )

    def extract(session_id: str) -> tuple[str, Any, Any, Any, Any, DeepSeekCallRecord]:
        session_turns = grouped[session_id]
        result = chat(
            stage="build_v3_session",
            messages=extraction_messages[session_id],
            max_tokens=extraction_max_tokens,
            json_mode=True,
        )
        claims, events, episodes, error = parse_session_extraction(
            result.text,
            question_id=case.question_id,
            session_id=session_id,
            session_date=session_dates.get(session_id),
            turns=session_turns,
        )
        return session_id, claims, events, episodes, error, result.record

    extracted = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(grouped)))) as pool:
        futures = {pool.submit(extract, session_id): session_id for session_id in grouped}
        for future in as_completed(futures):
            extracted.append(future.result())
    order = {session_id: index for index, session_id in enumerate(case.haystack_session_ids)}
    extracted.sort(key=lambda row: order[row[0]])

    retry_records: list[DeepSeekCallRecord] = []
    retry_diagnostics: list[dict[str, Any]] = []
    spent_tokens = sum(row[5].total_tokens for row in extracted)
    retry_indexes = [
        index for index, row in enumerate(extracted)
        if row[4] in {"empty_claims", "invalid_json"}
    ][:2]
    for extracted_index in retry_indexes:
        session_id, _claims, _events, _episodes, initial_error, _record = (
            extracted[extracted_index]
        )
        session_turns = grouped[session_id]
        messages = session_extraction_messages(
            session_id, session_dates.get(session_id), session_turns
        )
        messages[0] = {
            "role": "system",
            "content": messages[0]["content"] + (
                " FORMAT RETRY: emit claims/events as positional arrays exactly matching "
                "the supplied schema. Copy source turn IDs verbatim from the input rows; "
                "do not rename keys, wrap the graph, or use object-shaped claims."
            ),
        }
        retry_max_tokens = _bounded_optional_max_tokens(
            spent_tokens=spent_tokens,
            prompt_estimate=_message_token_estimate(messages),
            requested_max_tokens=max_tokens,
            build_budget_tokens=build_budget_tokens,
            minimum=256,
        )
        if retry_max_tokens == 0:
            retry_diagnostics.append({
                "question_id": case.question_id,
                "variant": variant,
                "stage": "v3_session_extraction_retry",
                "session_id": session_id,
                "initial_error": initial_error,
                "accepted": False,
                "skipped": True,
                "reason": "build_budget_guard",
                "spent_tokens": spent_tokens,
            })
            continue
        retry_result = chat(
            stage="build_v3_session_retry",
            messages=messages,
            max_tokens=retry_max_tokens,
            json_mode=True,
        )
        retry_records.append(retry_result.record)
        spent_tokens += retry_result.record.total_tokens
        retry_claims, retry_events, retry_episodes, retry_error = (
            parse_session_extraction(
                retry_result.text,
                question_id=case.question_id,
                session_id=session_id,
                session_date=session_dates.get(session_id),
                turns=session_turns,
            )
        )
        accepted = bool(retry_claims) and retry_error not in {
            "empty_claims", "invalid_json"
        }
        if accepted:
            extracted[extracted_index] = (
                session_id, retry_claims, retry_events, retry_episodes,
                retry_error, _record,
            )
        retry_diagnostics.append({
            "question_id": case.question_id,
            "variant": variant,
            "stage": "v3_session_extraction_retry",
            "session_id": session_id,
            "initial_error": initial_error,
            "retry_error": retry_error,
            "accepted": accepted,
            "claim_count": len(retry_claims),
            "event_count": len(retry_events),
            "prompt_tokens": retry_result.record.prompt_tokens,
            "completion_tokens": retry_result.record.completion_tokens,
            "total_tokens": retry_result.record.total_tokens,
        })

    claims = [claim for _session, rows, _events, _episodes, _error, _record in extracted for claim in rows]
    events = [event for _session, _claims, rows, _episodes, _error, _record in extracted for event in rows]
    proposals = {
        session_id: rows
        for session_id, _claims, _events, rows, _error, _record in extracted
    }
    for index, claim in enumerate(claims):
        claim.observation_order = index
    embed(turns, "retrieval_text")
    embed(claims, "retrieval_text")
    embed(events, "retrieval_text")
    index = build_hypergraph(
        question_id=case.question_id,
        session_ids=case.haystack_session_ids,
        session_dates=session_dates,
        turns=turns,
        claims=claims,
        events=events,
        episode_proposals=proposals,
    )
    reference_record: DeepSeekCallRecord | None = None
    reference_payload = consolidation_candidate_payload(turns, events) or {}
    entity_candidates = event_entity_candidate_payload(events, turns)
    if entity_candidates:
        reference_payload.update(entity_candidates)
    if not any(reference_payload.values()):
        reference_payload = None
    reference_edges = []
    event_identity_edges = []
    event_entities = []
    event_entity_edges = []
    if reference_payload is not None:
        reference_messages = reference_consolidation_messages(reference_payload)
        reference_max_tokens = _bounded_optional_max_tokens(
            spent_tokens=spent_tokens,
            prompt_estimate=_message_token_estimate(reference_messages),
            requested_max_tokens=min(2048, max_tokens),
            build_budget_tokens=build_budget_tokens,
            minimum=128,
        )
        if reference_max_tokens > 0:
            reference_result = chat(
                stage="build_v3_reference_consolidation",
                messages=reference_messages,
                max_tokens=reference_max_tokens,
                json_mode=True,
            )
            reference_record = reference_result.record
            spent_tokens += reference_result.record.total_tokens
            reference_edges = parse_reference_edges(
                reference_result.text,
                question_id=case.question_id,
                turns=turns,
            )
            event_identity_edges = parse_event_identity_edges(
                reference_result.text,
                question_id=case.question_id,
                events=events,
                candidate_pairs=reference_payload.get("event_pairs", []),
            )
            event_entities, event_entity_edges = parse_event_entities(
                reference_result.text,
                question_id=case.question_id,
                events=events,
                candidate_payload=reference_payload,
            )
            index.event_entities.extend(event_entities)
            index.hyperedges.extend([
                *reference_edges, *event_identity_edges, *event_entity_edges,
            ])
            embed(
                [*event_entities, *reference_edges, *event_identity_edges,
                 *event_entity_edges], "retrieval_text"
            )
            errors = validate_hypergraph(index)
            if errors:
                raise ValueError(f"V3 reference hypergraph validation failed: {errors[:8]}")
    embed(index.operands, "object_text", "object_embedding")
    diagnostics = [
        {
            "question_id": case.question_id,
            "variant": variant,
            "stage": "v3_session_extraction",
            "session_id": session_id,
            "turn_count": len(grouped[session_id]),
            "claim_count": len(session_claims),
            "event_count": len(session_events),
            "episode_proposal_count": len(session_episodes),
            "parse_error": error,
            "prompt_tokens": record.prompt_tokens,
            "completion_tokens": record.completion_tokens,
            "prompt_cache_hit_tokens": record.prompt_cache_hit_tokens,
            "prompt_cache_miss_tokens": record.prompt_cache_miss_tokens,
            "total_tokens": record.total_tokens,
        }
        for session_id, session_claims, session_events, session_episodes, error, record
        in extracted
    ]
    diagnostics.extend(retry_diagnostics)
    diagnostics.append(
        {
            "question_id": case.question_id,
            "variant": variant,
            "stage": "v3_hypergraph",
            "turn_count": len(index.turns),
            "claim_count": len(index.claims),
            "event_count": len(index.events),
            "event_entity_count": len(index.event_entities),
            "episode_count": len(index.episodes),
            "theme_count": len(index.themes),
            "hyperedge_count": len(index.hyperedges),
            "state_chain_count": len(index.state_chains),
            "reference_candidate_anchor_count": len(
                (reference_payload or {}).get("anchors", [])
            ),
            "reference_edge_count": len(reference_edges),
            "event_identity_candidate_count": len(
                (reference_payload or {}).get("event_pairs", [])
            ),
            "event_identity_edge_count": len(event_identity_edges),
            "event_entity_candidate_count": len(
                (reference_payload or {}).get("event_candidates", [])
            ),
            "event_entity_neighborhood_count": len(
                (reference_payload or {}).get("event_neighborhoods", [])
            ),
            "event_entity_edge_count": len(event_entity_edges),
            "fallback_session_count": sum(row[4] in {"empty_claims", "invalid_json"} for row in extracted),
            "salvaged_session_count": sum(row[4] == "partial_json_salvaged" for row in extracted),
            "session_completion_cap": extraction_max_tokens,
            "build_budget_reserve": _BUILD_BUDGET_RESERVE,
            "reference_budget_skipped": reference_payload is not None and reference_record is None,
        }
    )
    return V3BuildResult(
        index=index,
        records=[
            *[row[5] for row in extracted],
            *retry_records,
            *([reference_record] if reference_record is not None else []),
        ],
        diagnostics=diagnostics,
        parse_error_count=sum(row[4] == "invalid_json" for row in extracted),
    )
