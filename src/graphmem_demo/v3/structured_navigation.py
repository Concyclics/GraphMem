from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable

from .query_planning import answer_slot_phrase
from .retrieval import build_query_frame


_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")
_SET_WIDE_INTENTS = frozenset(
    {"count", "list", "ordering", "preference_list", "recurrence"}
)
_TEMPORAL_INTENTS = frozenset(
    {"date", "planned_date", "duration", "earliest", "latest", "ordering", "recurrence"}
)
_STATE_INTENTS = frozenset({"latest", "state"})

_RELATIONS_BY_INTENT: dict[str, tuple[str, ...]] = {
    "lookup": (
        "supports",
        "source",
        "event_frame_member",
        "event_entity_member",
        "episode_member",
    ),
    "count": (
        "quantity_collection",
        "same_entity",
        "same_predicate",
        "supports",
        "source",
        "event_frame_member",
        "event_entity_member",
        "episode_member",
        "theme_member",
    ),
    "list": (
        "same_entity",
        "same_predicate",
        "supports",
        "source",
        "event_frame_member",
        "event_entity_member",
        "episode_member",
        "theme_member",
    ),
    "preference_list": (
        "same_entity",
        "same_predicate",
        "supports",
        "contradicts",
        "source",
        "episode_member",
        "theme_member",
    ),
    "recurrence": (
        "quantity_collection",
        "same_event",
        "temporal_scope",
        "supports",
        "source",
        "event_frame_member",
        "episode_member",
    ),
    "date": (
        "same_event",
        "temporal_scope",
        "before",
        "after",
        "supports",
        "source",
        "event_frame_member",
        "operand_projection",
        "episode_member",
    ),
    "planned_date": (
        "same_event",
        "temporal_scope",
        "supports",
        "source",
        "event_frame_member",
        "operand_projection",
    ),
    "duration": (
        "same_event",
        "temporal_scope",
        "before",
        "after",
        "supports",
        "source",
        "event_frame_member",
        "operand_projection",
        "episode_member",
    ),
    "earliest": (
        "same_event",
        "temporal_scope",
        "before",
        "after",
        "supports",
        "source",
        "event_frame_member",
        "episode_member",
    ),
    "ordering": (
        "same_event",
        "temporal_scope",
        "before",
        "after",
        "supports",
        "source",
        "event_frame_member",
        "event_entity_member",
        "episode_member",
    ),
    "latest": (
        "state_history",
        "supersedes",
        "contradicts",
        "same_entity",
        "same_predicate",
        "temporal_scope",
        "supports",
        "source",
        "episode_member",
    ),
    "state": (
        "state_history",
        "supersedes",
        "contradicts",
        "same_entity",
        "same_predicate",
        "supports",
        "source",
        "episode_member",
    ),
    "location": (
        "same_event",
        "event_entity_member",
        "event_frame_member",
        "temporal_scope",
        "supports",
        "source",
        "episode_member",
    ),
    "recommendation": (
        "same_entity",
        "same_predicate",
        "supports",
        "contradicts",
        "source",
        "episode_member",
        "theme_member",
    ),
    "counterfactual": (
        "supports",
        "contradicts",
        "same_event",
        "event_entity_member",
        "episode_member",
        "theme_member",
    ),
}


@dataclass(frozen=True)
class QueryIR:
    """Benchmark-neutral query algebra used by graph navigation and packing."""

    intent: str
    answer_form: str
    subjects: tuple[str, ...]
    content_terms: tuple[str, ...]
    temporal_terms: tuple[str, ...]
    explicit_dates: tuple[str, ...]
    answer_slot: str | None
    required_slots: tuple[str, ...]
    allowed_relations: tuple[str, ...]
    set_wide: bool
    temporal: bool
    state_sensitive: bool
    exact_binding: bool


def build_query_ir(question: str) -> QueryIR:
    frame = build_query_frame(question)
    intent = str(frame.requested_operation)
    state_marked = bool(re.search(
        r"\b(?:current|currently|now|previously|formerly|still|no longer|"
        r"finally|ultimately)\b|\bused to\b",
        question.casefold(),
    ))
    allowed_relations = list(_RELATIONS_BY_INTENT.get(
        intent, _RELATIONS_BY_INTENT["lookup"]
    ))
    required = ["target_binding", "source_provenance"]
    if intent in _SET_WIDE_INTENTS:
        required.extend(("collection_members", "collection_closure"))
    if intent in _TEMPORAL_INTENTS:
        required.extend(("event_identity", "temporal_anchor"))
    if intent == "duration":
        required.extend(("start_operand", "end_operand", "unit"))
    if intent in {"earliest", "latest", "ordering"}:
        required.append("competing_events")
    if intent in _STATE_INTENTS:
        required.extend(("state_history", "current_valid_state"))
    elif state_marked:
        required.extend(("state_history", "current_valid_state"))
        for relation in _RELATIONS_BY_INTENT["state"]:
            if relation not in allowed_relations:
                allowed_relations.append(relation)
    if intent in {"recommendation", "preference_list"}:
        required.extend(("positive_constraints", "negative_constraints"))
    if intent == "counterfactual":
        required.extend(("condition", "supported_consequence"))
    subjects = tuple(dict.fromkeys(frame.participant_terms))
    slot = answer_slot_phrase(question)
    return QueryIR(
        intent=intent,
        answer_form=str(frame.answer_form),
        subjects=subjects,
        content_terms=tuple(frame.content_terms),
        temporal_terms=tuple(frame.temporal_terms),
        explicit_dates=tuple(frame.explicit_dates),
        answer_slot=slot,
        required_slots=tuple(dict.fromkeys(required)),
        allowed_relations=tuple(allowed_relations),
        set_wide=intent in _SET_WIDE_INTENTS,
        temporal=intent in _TEMPORAL_INTENTS,
        state_sensitive=intent in _STATE_INTENTS or state_marked,
        exact_binding=bool(subjects or slot or frame.content_terms),
    )


def canonical_operation(ir: QueryIR, proposed: str) -> str:
    """Keep a fixed executable intent while retaining a bounded planner hint."""

    proposed_tokens = {
        token.casefold() for token in _WORD_RE.findall(proposed.replace("_", " "))
    }
    if not proposed_tokens or proposed.casefold() in {"unknown", "evidence_closure"}:
        return ir.intent
    # The fixed prefix is the only part consumed by routing code.  The suffix
    # remains useful for audits but cannot invent a new operator class.
    suffix = "_".join(sorted(proposed_tokens))[:48]
    return f"{ir.intent}:{suffix}" if suffix else ir.intent


def merged_relations(ir: QueryIR, proposed: Iterable[str]) -> tuple[str, ...]:
    valid = set(_RELATIONS_BY_INTENT["lookup"])
    for relations in _RELATIONS_BY_INTENT.values():
        valid.update(relations)
    return tuple(dict.fromkeys(
        [
            *ir.allowed_relations,
            *(str(value) for value in proposed if str(value) in valid),
        ]
    ))


def multiview_frontier_ids(
    retrieval_trace: dict[str, Any],
    ir: QueryIR,
) -> list[tuple[str, str, float]]:
    """Return a bounded union of already-computed retrieval channels.

    The base retriever intentionally retains a wider audit trace than its final
    prompt.  Reusing that trace avoids another embedding call and lets graph
    navigation recover evidence that was lost during the first packing pass.
    """

    channels = retrieval_trace.get("channels") or {}
    quotas = {
        "exact": 64 if ir.exact_binding else 32,
        "bm25": 40,
        "dense": 28,
        "dense_view_1": 24,
        "dense_view_2": 24,
    }
    rows: list[tuple[str, str, float]] = []
    seen: set[str] = set()
    for channel, limit in quotas.items():
        for rank, raw_id in enumerate(channels.get(channel) or []):
            if rank >= limit:
                break
            node_id = str(raw_id)
            if not node_id or node_id in seen:
                continue
            seen.add(node_id)
            rows.append((node_id, f"multiview_{channel}", 1.0 / (rank + 1)))
    for rank, row in enumerate(retrieval_trace.get("rrf_top") or []):
        if rank >= 48:
            break
        node_id = str(row.get("node_id") or "")
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        rows.append((
            node_id,
            "multiview_rrf",
            float(row.get("score") or 0.0),
        ))
    return rows

def materialize_multiview_rows(
    graph_store: Any,
    question_id: str,
    retrieval_trace: dict[str, Any],
    ir: QueryIR,
) -> list[dict[str, Any]]:
    """Resolve saved channel node IDs into navigation-ledger rows."""

    nodes = graph_store.nodes_for(question_id)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node_id, source, score in multiview_frontier_ids(retrieval_trace, ir):
        suffix = node_id.split(":", 1)[1] if ":" in node_id else node_id
        if suffix in seen:
            continue
        raw = nodes.get(suffix)
        if raw is None:
            continue
        canonical_id = str(
            raw.get("node_id")
            or raw.get("operand_id")
            or raw.get("frame_id")
            or ""
        )
        canonical_suffix = (
            canonical_id.split(":", 1)[1]
            if ":" in canonical_id
            else canonical_id
        )
        source_turn_ids = []
        for value in raw.get("source_turn_ids") or []:
            value = str(value)
            value_suffix = value.split(":", 1)[1] if ":" in value else value
            source_turn_ids.append(f"{question_id}:{value_suffix}")
        rows.append({
            "node_id": f"{question_id}:{canonical_suffix}",
            "node_type": str(raw.get("node_type") or "unknown"),
            "selection_source": source,
            "score": round(float(score), 6),
            "text": str(raw.get("retrieval_text") or raw.get("text") or ""),
            "source_turn_ids": source_turn_ids,
            "session_id": raw.get("session_id"),
            "session_date": raw.get("session_date") or raw.get("observed_at"),
            "relation_path": [],
        })
        seen.add(suffix)
    return rows


def _operator_family_compatible(intent: str, operation: str) -> bool:
    """Match algebra families, never benchmark topics or named entities."""
    tokens = set(re.findall(r"[a-z]+", operation.casefold()))
    families = {
        "count": {"count", "total", "sum", "quantity", "money", "revenue", "average", "ratio", "percent", "cardinality", "scalar"},
        "duration": {"duration", "difference", "interval", "elapsed", "week", "day", "month", "year"},
        "date": {"date", "time", "event", "onset"},
        "planned_date": {"date", "time", "event", "plan"},
        "earliest": {"earliest", "before", "after", "order"},
        "ordering": {"earliest", "before", "after", "order", "collection"},
        "latest": {"latest", "state", "frequency", "relation", "choice"},
        "state": {"latest", "state", "frequency", "relation", "choice"},
        "recurrence": {"frequency", "recurrence", "weekly", "count", "state"},
        "location": {"location", "relation", "event"},
        "list": {"collection", "ordered", "entity", "action"},
    }
    allowed = families.get(intent)
    if not allowed:
        return False
    return bool(tokens & allowed)


def _unsafe_scope_operator(query_ir: QueryIR, operation: str) -> bool:
    """Reject locally plausible operators that answer a different scalar slot."""

    normalized = operation.casefold()
    # These operators describe the size of a state or collection, but a count
    # question can instead ask for a delta, occurrence count, or scoped total.
    # Without a complete closure proof they are proposals for the wrong slot
    # surprisingly often.
    if query_ir.intent == "count" and any(
        marker in normalized
        for marker in (
            "cardinality_state",
            "distinct_action_entity_collection",
            "event_occurrence_count",
        )
    ):
        return True
    return False


def certified_trace_hint(
    retrieval_trace: dict[str, Any], query_ir: QueryIR | None = None
) -> dict[str, Any] | None:
    """Keep only mechanically certified generic operator outputs."""

    certificate = retrieval_trace.get("closure_certificate") or {}
    payload: dict[str, Any] = {}
    if (
        certificate.get("complete")
        and certificate.get("provenance_complete")
        and not certificate.get("missing_requirements")
        and isinstance(retrieval_trace.get("operator_result"), dict)
        and (
            query_ir is None
            or _operator_family_compatible(
                query_ir.intent,
                str(retrieval_trace["operator_result"].get("operation") or ""),
            )
        )
        and (
            query_ir is None
            or not _unsafe_scope_operator(
                query_ir,
                str(retrieval_trace["operator_result"].get("operation") or ""),
            )
        )
    ):
        payload["operator_result"] = retrieval_trace["operator_result"]
        payload["operator_operand_node_ids"] = list(
            certificate.get("operand_node_ids") or []
        )[:24]
    operator_result = retrieval_trace.get("operator_result")
    if (
        query_ir is not None
        and isinstance(operator_result, dict)
        and certificate.get("provenance_complete")
        and _operator_family_compatible(
            query_ir.intent, str(operator_result.get("operation") or "")
        )
        and not _unsafe_scope_operator(
            query_ir, str(operator_result.get("operation") or "")
        )
        and "operator_result" not in payload
    ):
        payload["operator_result_proposal"] = operator_result
    for key in (
        "duration_hint",
        "relative_time_hint",
        "relative_age_hint",
        "latest_state_hint",
        "location_at_time_hint",
        "before_after_relation_hint",
    ):
        value = retrieval_trace.get(key)
        if isinstance(value, dict) and value.get("complete"):
            payload[key] = value
    catalog = retrieval_trace.get("catalog_operator_hint")
    if (
        query_ir is not None
        and isinstance(catalog, dict)
        and catalog.get("complete")
        and catalog.get("packed_provenance_complete") is True
        and _operator_family_compatible(
            query_ir.intent, str(catalog.get("operation") or "")
        )
        and not _unsafe_scope_operator(
            query_ir, str(catalog.get("operation") or "")
        )
        and "operator_result" not in payload
        and "operator_result_proposal" not in payload
    ):
        payload["catalog_operator_proposal"] = catalog
    if (
        isinstance(catalog, dict)
        and catalog.get("operation") == "exact_entity_mismatch"
        and catalog.get("global_lossless_scan_complete")
    ):
        payload["catalog_operator_hint"] = catalog
    return payload or None



def authoritative_trace_answer(retrieval_trace: dict[str, Any]) -> str | None:
    """Render only globally certified exact-entity absence conclusions."""
    catalog = retrieval_trace.get("catalog_operator_hint")
    if (
        isinstance(catalog, dict)
        and catalog.get("complete")
        and catalog.get("operation") == "exact_entity_mismatch"
        and catalog.get("global_lossless_scan_complete")
        and catalog.get("contrast_proof_complete")
    ):
        target = str(catalog.get("required_entity") or "the requested entity").strip()
        if target:
            return f"The memory did not mention {target}; it only mentioned a different entity."
    return None


def structured_navigation_summary(ir: QueryIR) -> dict[str, Any]:
    return {
        "intent": ir.intent,
        "answer_form": ir.answer_form,
        "subjects": list(ir.subjects),
        "content_terms": list(ir.content_terms),
        "temporal_terms": list(ir.temporal_terms),
        "explicit_dates": list(ir.explicit_dates),
        "answer_slot": ir.answer_slot,
        "required_slots": list(ir.required_slots),
        "allowed_relations": list(ir.allowed_relations),
        "set_wide": ir.set_wide,
        "temporal": ir.temporal,
        "state_sensitive": ir.state_sensitive,
        "exact_binding": ir.exact_binding,
    }
