from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from ..clients import rough_token_count
from ..models import DeepSeekCallRecord, QuestionCase
from .build import (
    add_routing_semantic_edges,
    build_index as assemble_index,
    build_inverted_indexes,
    build_turn_nodes,
    lossless_session_extraction,
    parse_session_extraction,
    session_extraction_messages,
    validate_index,
)
from .schema import (
    EvidenceGroup,
    GraphEdgeV36,
    RoleFrameNode,
    V36Index,
)


@dataclass
class V36BuildResult:
    index: V36Index
    records: list[DeepSeekCallRecord]
    diagnostics: list[dict[str, Any]]
    parse_error_count: int


_REPAIR_RESERVE = 15_000
_CONSOLIDATION_RESERVE = 5_000
_HARD_SAFETY_RESERVE = 5_000
_BUDGET_RESERVE = (
    _REPAIR_RESERVE + _CONSOLIDATION_RESERVE + _HARD_SAFETY_RESERVE
)


def _session_information_density(turns: list[Any]) -> tuple[int, int, int]:
    text = " ".join(str(getattr(turn, "text", "") or "") for turn in turns)
    terms = re.findall(r"[\w'-]+", text.casefold())
    return len(set(terms)), len(terms), len(text)


def _select_llm_sessions(
    grouped: dict[str, list[Any]], cap: int,
) -> set[str]:
    """Select question-independent, timeline-stratified high-density sessions."""
    ordered = list(grouped)
    if cap <= 0 or cap >= len(ordered):
        return set(ordered)
    selected: set[str] = set()
    total = len(ordered)
    for bucket in range(cap):
        start = bucket * total // cap
        end = (bucket + 1) * total // cap
        candidates = ordered[start:end]
        selected.add(max(
            candidates,
            key=lambda session_id: (
                _session_information_density(grouped[session_id]),
                -ordered.index(session_id),
            ),
        ))
    return selected


def _message_token_estimate(messages: list[dict[str, str]]) -> int:
    text = "\n".join(str(item.get("content") or "") for item in messages)
    return max(
        rough_token_count(text),
        math.ceil(len(text.encode("utf-8")) / 3.5),
    ) + 64


def _completion_caps(
    prompt_estimates: dict[str, int], *, requested: int, budget: int,
) -> dict[str, int]:
    if not prompt_estimates:
        return {}
    assumed_prompt = sum(math.ceil(value * 1.05) for value in prompt_estimates.values())
    output_budget = budget - _BUDGET_RESERVE - assumed_prompt
    if output_budget < 128 * len(prompt_estimates):
        raise RuntimeError("V3.6 prompts cannot fit the build token envelope")
    base = 128
    remaining = max(0, output_budget - base * len(prompt_estimates))
    weights = {session_id: math.sqrt(max(1, estimate)) for session_id, estimate in prompt_estimates.items()}
    total_weight = sum(weights.values()) or 1.0
    caps = {session_id: min(requested, base + math.floor(remaining * weights[session_id] / total_weight)) for session_id in prompt_estimates}
    spare = output_budget - sum(caps.values())
    for session_id in sorted(caps, key=lambda value: (-prompt_estimates[value], value)):
        if spare <= 0:
            break
        addition = min(spare, requested - caps[session_id])
        caps[session_id] += addition
        spare -= addition
    return caps


def _repair_budget_remaining(
    *, build_budget_tokens: int, spent: int, prompt_cost: int,
) -> int:
    """Budget available to repair after protecting later and hard reserves."""
    return (
        build_budget_tokens
        - _CONSOLIDATION_RESERVE
        - _HARD_SAFETY_RESERVE
        - spent
        - prompt_cost
    )


def _consolidation_budget_remaining(
    *, build_budget_tokens: int, spent: int, prompt_cost: int,
) -> int:
    """Final-stage budget retaining a provider-accounting safety margin."""
    return build_budget_tokens - _HARD_SAFETY_RESERVE - spent - prompt_cost


def _checkpoint_signature(
    messages: list[dict[str, str]], max_tokens: int,
) -> str:
    payload = {"messages": messages, "max_tokens": max_tokens}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _load_call_checkpoint(
    directory: Path | None, *, stage: str, key: str,
    messages: list[dict[str, str]], max_tokens: int,
) -> tuple[str, DeepSeekCallRecord] | None:
    if directory is None:
        return None
    marker = hashlib.sha256(f"{stage}:{key}".encode("utf-8")).hexdigest()[:24]
    path = directory / f"{stage}-{marker}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("signature") != _checkpoint_signature(messages, max_tokens):
            return None
        text = payload.get("text")
        record = payload.get("record")
        if not isinstance(text, str) or not isinstance(record, dict):
            return None
        return text, DeepSeekCallRecord(**record)
    except Exception:
        return None


def _save_call_checkpoint(
    directory: Path | None, *, stage: str, key: str,
    messages: list[dict[str, str]], max_tokens: int, text: str,
    record: DeepSeekCallRecord,
) -> None:
    if directory is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    marker = hashlib.sha256(f"{stage}:{key}".encode("utf-8")).hexdigest()[:24]
    path = directory / f"{stage}-{marker}.json"
    payload = {
        "signature": _checkpoint_signature(messages, max_tokens),
        "text": text, "record": asdict(record),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    temporary.replace(path)


def _repair_positions(extracted: list[tuple[Any, ...]]) -> list[int]:
    """Return every structurally invalid session; the budget loop is the cap."""
    return [
        position for position, row in enumerate(extracted)
        if row[4] in {"invalid_json", "empty_frames", "coverage_gap"}
    ]


def _identity_key(value: str) -> str:
    key = value.casefold().strip()
    return "" if key in {"", "none", "unknown", "fact", "event", "predicate", "said", "dialogue answer"} else key


_IDENTITY_PHRASE_STOP = {
    "a", "an", "from", "in", "made", "moving", "of", "on", "said", "the",
    "this", "to", "went",
}


def _grounded_identity_phrases(frame: RoleFrameNode) -> set[str]:
    phrases: set[str] = set()
    for value in (
        frame.entity_key, frame.predicate_key, frame.object_key,
        frame.context_key, frame.event_identity_key,
    ):
        tokens = [
            token for token in re.findall(r"[a-z0-9]+", value.casefold())
            if token not in _IDENTITY_PHRASE_STOP
        ]
        for width in range(2, min(4, len(tokens)) + 1):
            phrases.update(
                " ".join(tokens[start:start + width])
                for start in range(len(tokens) - width + 1)
            )
    return phrases


def _candidate_pairs(frames: list[RoleFrameNode], limit: int = 240) -> list[list[str]]:
    """Generate only identity-bearing cross-session candidates.

    A broad entity+predicate bucket creates hubs whenever extraction emits a
    generic predicate. Exact event identity, or an exact grounded proposition,
    is required before the bounded LLM validator sees a pair.
    """
    buckets: dict[str, list[RoleFrameNode]] = {}
    for frame in frames:
        event_key = _identity_key(frame.event_identity_key)
        if event_key:
            buckets.setdefault(f"event:{event_key}", []).append(frame)
        entity = _identity_key(frame.entity_key)
        predicate = _identity_key(frame.predicate_key)
        obj = _identity_key(frame.object_key)
        context = _identity_key(frame.context_key)
        if entity and predicate and obj:
            buckets.setdefault(f"proposition:{entity}|{predicate}|{obj}", []).append(frame)
        elif entity and predicate and context:
            buckets.setdefault(f"context:{entity}|{predicate}|{context}", []).append(frame)
        if frame.owner_key:
            for phrase in sorted(_grounded_identity_phrases(frame)):
                buckets.setdefault(
                    f"phrase:{frame.owner_key}|{phrase}", []
                ).append(frame)
    scored: dict[tuple[str, str], tuple[int, int, int]] = {}
    for bucket_key, rows in sorted(buckets.items()):
        is_phrase = bucket_key.startswith("phrase:")
        if is_phrase and not 2 <= len(rows) <= 6:
            continue
        phrase = bucket_key.rsplit("|", 1)[-1] if is_phrase else ""
        for left_index, left in enumerate(rows):
            for right in rows[left_index + 1:]:
                if left.session_ids == right.session_ids:
                    continue
                pair = tuple(sorted((left.frame_id, right.frame_id)))
                anchored = bool(_anchored_identity_phrase(left, right)) if is_phrase else False
                score = (
                    100 if not is_phrase else 50 if anchored else 10,
                    len(phrase.split()),
                    -len(rows),
                )
                if score > scored.get(pair, (-1, -1, -999)):
                    scored[pair] = score
    ordered = sorted(
        scored, key=lambda pair: (*scored[pair], pair), reverse=True
    )
    return [[*pair] for pair in ordered[:limit]]


def _identity_supported(left: RoleFrameNode, right: RoleFrameNode) -> bool:
    left_event = _identity_key(left.event_identity_key)
    right_event = _identity_key(right.event_identity_key)
    if left_event and left_event == right_event:
        return True
    left_signature = tuple(_identity_key(value) for value in (left.entity_key, left.predicate_key, left.object_key))
    right_signature = tuple(_identity_key(value) for value in (right.entity_key, right.predicate_key, right.object_key))
    if all(left_signature) and left_signature == right_signature:
        return True
    left_context = tuple(_identity_key(value) for value in (left.entity_key, left.predicate_key, left.context_key))
    right_context = tuple(_identity_key(value) for value in (right.entity_key, right.predicate_key, right.context_key))
    if all(left_context) and left_context == right_context:
        return True
    return (
        bool(left.owner_key)
        and left.owner_key == right.owner_key
        and bool(
            _grounded_identity_phrases(left)
            & _grounded_identity_phrases(right)
        )
    )


def _consolidation_messages(
    frames: list[RoleFrameNode], candidate_pairs: list[list[str]]
) -> list[dict[str, str]]:
    by_id = {frame.frame_id: frame for frame in frames}
    candidates = [
        {
            "left": pair[0],
            "right": pair[1],
            "left_text": by_id[pair[0]].retrieval_text,
            "right_text": by_id[pair[1]].retrieval_text,
            "left_sources": by_id[pair[0]].source_turn_ids,
            "right_sources": by_id[pair[1]].source_turn_ids,
        }
        for pair in candidate_pairs
    ]
    return [
        {
            "role": "system",
            "content": (
                "Consolidate only the supplied question-independent frame pairs. "
                "Return JSON {edges:[[left_id,right_id,relation,confidence]]}. "
                "relation is reference or same_event. Accept reference only when "
                "one mention genuinely refers to the other identity; accept "
                "same_event only when both describe the same real event, not merely "
                "the same topic or participants. Do not create IDs, facts, temporal "
                "order, participant links, semantic links or evaluation-specific rules. "
                "Confidence must be at least 0.8. JSON only."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"candidate_pairs": candidates},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]


def _anchored_identity_phrase(left: RoleFrameNode, right: RoleFrameNode) -> str:
    common = sorted(
        _grounded_identity_phrases(left) & _grounded_identity_phrases(right),
        key=lambda value: (-len(value.split()), value),
    )
    left_anchors = {_identity_key(left.entity_key), _identity_key(left.context_key)}
    right_anchors = {_identity_key(right.entity_key), _identity_key(right.context_key)}
    left_value = _identity_key(left.object_key)
    right_value = _identity_key(right.object_key)
    for phrase in common:
        left_is_anchor = any(phrase == anchor or phrase in anchor for anchor in left_anchors)
        right_is_anchor = any(phrase == anchor or phrase in anchor for anchor in right_anchors)
        if (left_is_anchor and phrase in right_value) or (right_is_anchor and phrase in left_value):
            return phrase
    return ""


def _apply_consolidation(
    index: V36Index,
    text: str,
    candidate_pairs: list[list[str]],
) -> tuple[int, str | None]:
    try:
        payload = json.loads(text)
    except Exception:
        return 0, "invalid_json"
    if not isinstance(payload, dict):
        return 0, "invalid_json"
    allowed_pairs = {tuple(sorted(pair)) for pair in candidate_pairs}
    frame_by_id = {frame.frame_id: frame for frame in index.frames}
    reference_group_start = len(index.evidence_groups)
    accepted = 0
    accepted_pairs: set[tuple[str, str]] = set()
    for raw in payload.get("edges", []):
        if not isinstance(raw, list) or len(raw) < 4:
            continue
        left, right, relation = str(raw[0]), str(raw[1]), str(raw[2])
        try:
            confidence = float(raw[3])
        except (TypeError, ValueError):
            continue
        pair = tuple(sorted((left, right)))
        if (
            pair not in allowed_pairs
            or relation not in {"reference", "same_event"}
            or confidence < 0.8
            or left not in frame_by_id
            or right not in frame_by_id
            or not _identity_supported(frame_by_id[left], frame_by_id[right])
        ):
            continue
        sources = list(dict.fromkeys([
            *frame_by_id[left].source_turn_ids,
            *frame_by_id[right].source_turn_ids,
        ]))
        index.edges.append(GraphEdgeV36(
            edge_id=f"{index.turns[0].question_id}:edge:{len(index.edges)}",
            question_id=index.turns[0].question_id,
            src=left,
            dst=right,
            relation=relation,  # type: ignore[arg-type]
            directed=True,
            confidence=confidence,
            provenance={
                "llm_rule": "bounded_identity_consolidation",
                "candidate_pair": list(pair),
                "source_turn_ids": sources,
            },
        ))
        index.evidence_groups.append(EvidenceGroup(
            group_id=(
                f"{index.turns[0].question_id}:group:"
                f"{len(index.evidence_groups)}"
            ),
            question_id=index.turns[0].question_id,
            group_kind="reference_chain",
            member_frame_ids=[left, right],
            source_turn_ids=sources,
            required_roles=["reference", "identity", "source"],
            completeness_mask={
                "reference": True, "identity": True, "source": bool(sources)
            },
            provenance_complete=bool(sources),
            confidence=confidence,
            retrieval_text=(
                f"{frame_by_id[left].retrieval_text} | "
                f"{frame_by_id[right].retrieval_text}"
            ),
            session_ids=list(dict.fromkeys([
                *frame_by_id[left].session_ids,
                *frame_by_id[right].session_ids,
            ])),
        ))
        accepted += 1
        accepted_pairs.add(pair)
    fallback_degree: Counter[str] = Counter()
    fallback_added = 0
    for raw_pair in candidate_pairs:
        pair = tuple(sorted(raw_pair))
        if pair in accepted_pairs or fallback_added >= 48:
            continue
        left, right = pair
        if fallback_degree[left] >= 3 or fallback_degree[right] >= 3:
            continue
        if left not in frame_by_id or right not in frame_by_id:
            continue
        phrase = _anchored_identity_phrase(frame_by_id[left], frame_by_id[right])
        if (
            not phrase
            or not frame_by_id[left].owner_key
            or frame_by_id[left].owner_key != frame_by_id[right].owner_key
        ):
            continue
        sources = list(dict.fromkeys([
            *frame_by_id[left].source_turn_ids,
            *frame_by_id[right].source_turn_ids,
        ]))
        confidence = 0.82
        index.edges.append(GraphEdgeV36(
            edge_id=f"{index.turns[0].question_id}:edge:{len(index.edges)}",
            question_id=index.turns[0].question_id, src=left, dst=right,
            relation="reference", directed=True, confidence=confidence,
            provenance={
                "local_rule": "bounded_anchored_identity_phrase",
                "identity_phrase": phrase, "candidate_pair": list(pair),
                "source_turn_ids": sources,
            },
        ))
        index.evidence_groups.append(EvidenceGroup(
            group_id=f"{index.turns[0].question_id}:group:{len(index.evidence_groups)}",
            question_id=index.turns[0].question_id, group_kind="reference_chain",
            member_frame_ids=[left, right], source_turn_ids=sources,
            required_roles=["reference", "identity", "source"],
            completeness_mask={
                "reference": True, "identity": True, "source": bool(sources),
            },
            provenance_complete=bool(sources), confidence=confidence,
            retrieval_text=(
                f"{frame_by_id[left].retrieval_text} | "
                f"{frame_by_id[right].retrieval_text}"
            ),
            session_ids=list(dict.fromkeys([
                *frame_by_id[left].session_ids, *frame_by_id[right].session_ids,
            ])),
        ))
        accepted += 1
        fallback_added += 1
        fallback_degree[left] += 1
        fallback_degree[right] += 1
    for group in index.evidence_groups[reference_group_start:]:
        if group.group_kind != "reference_chain":
            continue
        for frame_id in group.member_frame_ids:
            index.edges.append(GraphEdgeV36(
                edge_id=f"{index.turns[0].question_id}:edge:{len(index.edges)}",
                question_id=index.turns[0].question_id, src=group.group_id,
                dst=frame_id, relation="reference", directed=True,
                confidence=group.confidence,
                provenance={
                    "local_rule": "reference_group_member",
                    "group_id": group.group_id,
                    "source_turn_ids": group.source_turn_ids,
                }, role="member",
            ))
    build_inverted_indexes(index)
    return accepted, None


def build_index(
    *,
    case: QuestionCase,
    variant: str,
    chat: Callable[..., Any],
    embed: Callable[[list[Any], str, str], None],
    max_tokens: int,
    workers: int,
    build_budget_tokens: int = 300_000,
    checkpoint_dir: Path | None = None,
    llm_session_cap: int = 0,
) -> V36BuildResult:
    turns = build_turn_nodes(case)
    grouped: dict[str, list[Any]] = {}
    for turn in turns:
        grouped.setdefault(turn.session_id, []).append(turn)
    llm_sessions = _select_llm_sessions(grouped, llm_session_cap)
    session_dates = {
        session_id: session_turns[0].session_date
        for session_id, session_turns in grouped.items() if session_turns
    }
    messages = {
        session_id: session_extraction_messages(
            session_id, session_dates.get(session_id), session_turns
        )
        for session_id, session_turns in grouped.items()
    }
    estimates = {
        session_id: _message_token_estimate(value)
        for session_id, value in messages.items()
    }
    completion_caps = _completion_caps(
        estimates, requested=max_tokens, budget=build_budget_tokens
    )

    def extract(session_id: str) -> tuple[Any, ...]:
        if session_id not in llm_sessions:
            frames, card, coverage, error = lossless_session_extraction(
                question_id=case.question_id, session_id=session_id,
                turns=grouped[session_id],
            )
            return session_id, frames, card, coverage, error, None
        session_messages = [dict(item) for item in messages[session_id]]
        session_messages[0]["content"] += (
            f" Output JSON budget: at most {completion_caps[session_id]} tokens. "
            "Use compact strings; cover durable facts before optional detail."
        )
        stage = "build_v36_session"
        call_cap = completion_caps[session_id]
        cached = _load_call_checkpoint(
            checkpoint_dir, stage=stage, key=session_id,
            messages=session_messages, max_tokens=call_cap,
        )
        if cached is None:
            result = chat(
                stage=stage, messages=session_messages,
                max_tokens=call_cap, json_mode=True,
            )
            response_text, record = result.text, result.record
            _save_call_checkpoint(
                checkpoint_dir, stage=stage, key=session_id,
                messages=session_messages, max_tokens=call_cap,
                text=response_text, record=record,
            )
        else:
            response_text, record = cached
        frames, card, coverage, error = parse_session_extraction(
            response_text,
            question_id=case.question_id,
            session_id=session_id,
            session_date=session_dates.get(session_id),
            turns=grouped[session_id],
        )
        return session_id, frames, card, coverage, error, record

    extracted: list[tuple[Any, ...]] = []
    with ThreadPoolExecutor(
        max_workers=max(1, min(workers, len(grouped)))
    ) as pool:
        futures = {
            pool.submit(extract, session_id): session_id for session_id in grouped
        }
        for future in as_completed(futures):
            extracted.append(future.result())
    session_order = {
        session_id: order
        for order, session_id in enumerate(grouped)
    }
    extracted.sort(key=lambda row: session_order[row[0]])
    records = [row[5] for row in extracted if row[5] is not None]
    spent = sum(record.total_tokens for record in records)
    diagnostics: list[dict[str, Any]] = []

    # A repair is allowed only for structurally invalid extraction. It reuses the
    # original source and remains subject to the per-question build hard limit.
    for position in _repair_positions(extracted):
        row = extracted[position]
        session_id, frames, card, coverage, error, original_record = row
        if error not in {"invalid_json", "empty_frames", "coverage_gap"}:
            continue
        repair_messages = list(messages[session_id])
        repair_messages[0] = {
            "role": "system",
            "content": repair_messages[0]["content"] + (
                " FORMAT REPAIR: emit valid F objects, a compact routing "
                "card, and exactly one coverage row per supplied turn. Copy source "
                "turn IDs exactly. Frame sources are the sole provenance mapping; "
                "coverage rows contain only turn ID and class. Do not add commentary."
            ),
        }
        prompt_cost = _message_token_estimate(repair_messages)
        remaining = _repair_budget_remaining(
            build_budget_tokens=build_budget_tokens,
            spent=spent,
            prompt_cost=prompt_cost,
        )
        if remaining < 256:
            diagnostics.append({
                "stage": "v36_session_repair",
                "session_id": session_id,
                "accepted": False,
                "reason": "build_budget_guard",
            })
            continue
        stage = "build_v36_session_repair"
        call_cap = min(max_tokens, max(256, int(remaining * 0.65)))
        cached = _load_call_checkpoint(
            checkpoint_dir, stage=stage, key=session_id,
            messages=repair_messages, max_tokens=call_cap,
        )
        if cached is None:
            result = chat(
                stage=stage, messages=repair_messages,
                max_tokens=call_cap, json_mode=True,
            )
            response_text, repair_record = result.text, result.record
            _save_call_checkpoint(
                checkpoint_dir, stage=stage, key=session_id,
                messages=repair_messages, max_tokens=call_cap,
                text=response_text, record=repair_record,
            )
        else:
            response_text, repair_record = cached
        records.append(repair_record)
        spent += repair_record.total_tokens
        repaired = parse_session_extraction(
            response_text,
            question_id=case.question_id,
            session_id=session_id,
            session_date=session_dates.get(session_id),
            turns=grouped[session_id],
        )
        accepted = repaired[3] not in {"invalid_json", "empty_frames", "coverage_gap"}
        if accepted:
            extracted[position] = (
                session_id, repaired[0], repaired[1], repaired[2], repaired[3],
                original_record,
            )
        diagnostics.append({
            "stage": "v36_session_repair",
            "session_id": session_id,
            "accepted": accepted,
            "initial_error": error,
            "repair_error": repaired[3],
            "total_tokens": repair_record.total_tokens,
        })

    frames = [frame for row in extracted for frame in row[1]]
    cards = [row[2] for row in extracted]
    coverage = [item for row in extracted for item in row[3]]
    index = assemble_index(
        question_id=case.question_id,
        turns=turns,
        frames=frames,
        routing_cards=cards,
        coverage=coverage,
    )
    embed(index.turns, "retrieval_text")
    embed(index.frames, "retrieval_text")
    embed(index.routing_cards, "routing_text")
    embed(index.evidence_groups, "retrieval_text")
    add_routing_semantic_edges(index)

    candidate_pairs = _candidate_pairs(index.frames)
    consolidation_record: DeepSeekCallRecord | None = None
    consolidation_accepted = 0
    consolidation_error: str | None = None
    if candidate_pairs:
        consolidation_messages = _consolidation_messages(
            index.frames, candidate_pairs
        )
        prompt_cost = _message_token_estimate(consolidation_messages)
        remaining = _consolidation_budget_remaining(
            build_budget_tokens=build_budget_tokens,
            spent=spent,
            prompt_cost=prompt_cost,
        )
        if remaining >= 128:
            stage = "build_v36_identity_consolidation"
            call_cap = min(2048, remaining)
            cached = _load_call_checkpoint(
                checkpoint_dir, stage=stage, key=case.question_id,
                messages=consolidation_messages, max_tokens=call_cap,
            )
            if cached is None:
                result = chat(
                    stage=stage, messages=consolidation_messages,
                    max_tokens=call_cap, json_mode=True,
                )
                response_text, consolidation_record = result.text, result.record
                _save_call_checkpoint(
                    checkpoint_dir, stage=stage, key=case.question_id,
                    messages=consolidation_messages, max_tokens=call_cap,
                    text=response_text, record=consolidation_record,
                )
            else:
                response_text, consolidation_record = cached
            records.append(consolidation_record)
            spent += consolidation_record.total_tokens
            consolidation_accepted, consolidation_error = _apply_consolidation(
                index, response_text, candidate_pairs
            )
            if consolidation_accepted:
                new_groups = [
                    group for group in index.evidence_groups
                    if group.embedding is None
                ]
                embed(new_groups, "retrieval_text")
        else:
            consolidation_error = "build_budget_guard"

    errors = validate_index(index)
    if errors:
        raise ValueError(f"V3.6 index validation failed: {errors[:12]}")
    diagnostics.extend({
        "question_id": case.question_id,
        "variant": variant,
        "stage": "v36_session_extraction",
        "session_id": row[0],
        "turn_count": len(grouped[row[0]]),
        "frame_count": len(row[1]),
        "coverage_count": len(row[3]),
        "lossless_only_count": sum(
            item.coverage_class == "lossless_only" for item in row[3]
        ),
        "local_lossless_frame_count": sum(
            "lossless fallback" in frame.semantic_type_keys for frame in row[1]
        ),
        "coverage_protocol": "source_derived_v2",
        "parse_error": row[4],
        "extraction_mode": ("llm" if row[5] is not None else "lossless_deterministic"),
        "prompt_tokens": (row[5].prompt_tokens if row[5] is not None else 0),
        "completion_tokens": (row[5].completion_tokens if row[5] is not None else 0),
        "total_tokens": (row[5].total_tokens if row[5] is not None else 0),
        "requested_completion_cap": completion_caps.get(row[0], 0),
        "provider_output_cap_honored": (
            row[5] is None or row[5].completion_tokens <= completion_caps[row[0]]
        ),
        "finish_reason": (row[5].finish_reason if row[5] is not None else None),
    } for row in extracted)
    edge_relation_counts = Counter(edge.relation for edge in index.edges)
    node_degree: Counter[str] = Counter()
    for edge in index.edges:
        node_degree[edge.src] += 1
        node_degree[edge.dst] += 1
    diagnostics.append({
        "question_id": case.question_id,
        "variant": variant,
        "stage": "v36_index",
        "turn_count": len(index.turns),
        "frame_count": len(index.frames),
        "routing_card_count": len(index.routing_cards),
        "evidence_group_count": len(index.evidence_groups),
        "edge_count": len(index.edges),
        "state_chain_count": len(index.state_chains),
        "candidate_identity_pair_count": len(candidate_pairs),
        "accepted_identity_edge_count": consolidation_accepted,
        "consolidation_error": consolidation_error,
        "participant_edge_count": 0,
        "temporal_scope_edge_count": 0,
        "duplicate_projection_node_count": 0,
        "edge_relation_counts": dict(edge_relation_counts),
        "max_node_degree": max(node_degree.values(), default=0),
        "incomplete_evidence_group_count": sum(not all(group.completeness_mask.values()) for group in index.evidence_groups),
        "provenance_incomplete_group_count": sum(not group.provenance_complete for group in index.evidence_groups),
        "lossless_only_turn_count": sum(item.coverage_class == "lossless_only" for item in index.coverage),
        "durable_covered_turn_count": sum(item.coverage_class == "memory_frame" and bool(item.frame_ids) for item in index.coverage),
        "provider_output_cap_violation_count": sum(
            row[5] is not None and row[5].completion_tokens > completion_caps[row[0]]
            for row in extracted
        ),
        "llm_session_cap": llm_session_cap,
        "llm_selected_session_count": len(llm_sessions),
        "deterministic_lossless_session_count": len(grouped) - len(llm_sessions),
        "routing_card_max_chars": max((len(card.routing_text) for card in index.routing_cards), default=0),
        "build_total_tokens": spent,
        "session_completion_cap_min": min(completion_caps.values(), default=0),
        "session_completion_cap_max": max(completion_caps.values(), default=0),
        "session_completion_cap_total": sum(completion_caps.values()),
        "budget_reserve": _BUDGET_RESERVE,
        "repair_reserve": _REPAIR_RESERVE,
        "consolidation_reserve": _CONSOLIDATION_RESERVE,
        "hard_safety_reserve": _HARD_SAFETY_RESERVE,
    })
    return V36BuildResult(
        index=index,
        records=records,
        diagnostics=diagnostics,
        parse_error_count=sum(
            row[4] in {"invalid_json", "empty_frames", "coverage_gap"} for row in extracted
        ),
    )
