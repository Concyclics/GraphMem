from __future__ import annotations

import heapq
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from typing import Any, Iterable

from ..clients import cosine_similarity, rough_token_count
from ..models import QuestionCase, RetrievedContext
from .build import canonical_key
from .binding_hints import missing_count_target_hint, missing_possessive_anchor_hint
from .answer_packing import contract_evidence_to_ids, fit_answer_payload
from .answer_hints import (
    before_after_relation_hint, calendar_window_hint, date_scope_hint,
    structured_section_hint,
)
from .catalog import ensure_catalog
from .catalog_schema import EventFrameV3, OperandRecordV3
from .operators import duration_hint
from .ordinal_event import ordinal_event_hint
from .ordinal_operator import ordinal_list_hint
from .catalog_operators import catalog_operator_hint
from .compact_packing import pack_context
from .coarse_projection import project_reached_episodes
from .contrast_relation import contrast_alternative_hint
from .dense import dense_rank_many
from .scope import score_scope_posteriors as _score_scope_posteriors
from .scope_expansion import lossless_event_turn_candidates, total_scope_candidates
from .relative_entity import resolve_relative_entity
from .recommendation_resources import (
    recommendation_resource_turn_ids, recommendation_scope_session_ids,
    resource_evidence_text,
)
from .relation_focus import relation_focus_turn_ids
from .media_relation import media_attribute_hint
from .location_state import location_at_time_hint
from .event_interval import event_lifecycle_duration_hint
from .movement_collection import movement_location_collection_hint
from .planned_event_count import planned_event_identity_count
from .quantified_relation import all_subjects_relation_hint
from .query_planning import answer_slot_phrase, query_views as planned_query_views
from .weekday_operator import weekday_scope_hint
from .query_focus import (
    focused_evidence_capsule,
    infer_answer_slot,
    should_use_focused_capsule,
)
from .state_temporal_operators import (
    latest_state_hint,
    relative_age_hint,
    relative_time_hint,
)
from .temporal_normalize import resolve_evidence_time
from .schema import (
    ClaimNode,
    ClosureCertificate,
    EpisodeNode,
    EventNode,
    EventEntityNode,
    HyperEdge,
    QueryFrame,
    ThemeNode,
    TurnNode,
    V3Index,
)


V3_RETRIEVAL_VERSION = "graphmem_v3_generic_closure_20260727do"

_AUTHORITATIVE_CATALOG_OPERATIONS = frozenset({
    "aggregate_revenue_total",
    "arrival_clock_time",
    "catalog_duration",
    "combined_named_duration",
    "dimensional_quantity_total",
    "distinct_action_entity_collection",
    "earliest_named_alternative",
    "exact_entity_mismatch",
    "event_onset_from_lossless_evidence",
    "frequency_state_comparison",
    "latest_cardinality_state",
    "money_amount",
    "money_difference",
    "named_scalar_average",
    "ordered_event_collection",
    "ordinal_list_item",
    "partitioned_scalar_total",
    "per_item_amount",
    "ratio_percent",
    "scalar_attribute_state",
    "scalar_comparison",
    "total_money",
    "weekly_recurrence_count",
})

_WORD_RE = re.compile(r"[\w'-]+", re.UNICODE)
_QUERY_FUNCTION_WORDS = {
    "a", "about", "am", "can", "could", "after", "all", "an", "and", "any", "are", "as", "at",
    "be", "before", "between", "by", "did", "do", "does", "earliest", "for",
    "from", "had", "has", "have", "how", "i", "in", "is", "latest", "many",
    "me", "most", "my", "of", "on", "or", "our", "please", "recent",
    "recently", "currently", "should", "some", "the", "their", "them", "they", "to", "was", "we", "were",
    "suggest", "tell", "that", "this", "these", "those", "current",
    "distinct", "different", "type", "types", "order", "past", "last",
    "second", "seconds", "minute", "minutes", "hour", "hours",
    "day", "days", "week", "weeks", "month", "months", "year", "years",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "what", "when", "would", "where", "which", "who", "why", "with", "you", "your",
}
_MONTHS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}
_MONTH_NUMBERS = {name: index for index, name in enumerate((
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
), start=1)}


def _token_key(value: str) -> str:
    lowered = value.casefold().strip("\x27\"")
    if lowered.endswith("'s"):
        lowered = lowered[:-2]
    if len(lowered) > 5 and lowered.endswith("ing"):
        lowered = lowered[:-3]
    elif len(lowered) > 4 and lowered.endswith("ed"):
        lowered = lowered[:-2]
    if len(lowered) > 4 and lowered.endswith("s") and not lowered.endswith("ss"):
        return lowered[:-1]
    return lowered


def _tokens(text: str) -> list[str]:
    normalized = text.replace("_", " ").replace("-", " ")
    return [_token_key(value) for value in _WORD_RE.findall(normalized)]


def _authoritative_catalog_hint(hint: Any) -> dict[str, Any] | None:
    """Return only operator results whose closure is mechanically auditable."""

    if not isinstance(hint, dict) or not hint.get("complete"):
        return None
    if hint.get("operation") not in _AUTHORITATIVE_CATALOG_OPERATIONS:
        return None
    if hint.get("packed_provenance_complete") is not True:
        return None
    return hint


def _globally_closed_certificate(certificate: Any) -> bool:
    """Require a complete, untruncated graph scope before answer short-circuiting.

    Packed provenance only proves that the sources for the selected operands are
    present. It does not prove that the selected operands exhaust the memory
    scope. Aggregate and collection operators therefore cannot be authoritative
    when graph expansion reports a truncated or otherwise incomplete closure.
    """

    return bool(
        isinstance(certificate, dict)
        and certificate.get("complete") is True
        and certificate.get("truncated") is False
        and certificate.get("provenance_complete") is True
        and not certificate.get("missing_requirements")
    )


def authoritative_catalog_answer(retrieval_trace: dict[str, Any]) -> str | None:
    """Render a mechanically complete operator proof without another model vote."""

    hint = _authoritative_catalog_hint(
        retrieval_trace.get("catalog_operator_hint")
    )
    if hint is not None and not _globally_closed_certificate(
        retrieval_trace.get("closure_certificate")
    ):
        hint = None
    if hint is None:
        local_duration = retrieval_trace.get("duration_hint")
        if (
            isinstance(local_duration, dict)
            and local_duration.get("operation") == "duration_since_bound_event"
            and local_duration.get("provenance_complete") is True
        ):
            hint = local_duration
    if hint is None:
        return None
    value = hint.get("value")
    if value is None:
        values = hint.get("values")
        if not isinstance(values, list) or not values:
            return None
        value = ", ".join(str(item) for item in values)
    if isinstance(value, bool):
        rendered = "yes" if value else "no"
    else:
        rendered = str(value).strip()
    if not rendered:
        return None
    unit = str(hint.get("unit") or "").strip()
    if unit and unit.casefold() not in rendered.casefold():
        rendered = f"{rendered} {unit}"
    return rendered


def _natural_dates(question: str) -> list[str]:
    values: list[str] = []
    month_names = "|".join(_MONTH_NUMBERS)
    patterns = (
        rf"\b({month_names})\s+(\d{{1,2}})(?:st|nd|rd|th)?[,]?\s+((?:19|20)\d{{2}})\b",
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_names})[,]?\s+((?:19|20)\d{{2}})\b",
    )
    for pattern_index, pattern in enumerate(patterns):
        for match in re.finditer(pattern, question.casefold()):
            if pattern_index == 0:
                month_name, raw_day, raw_year = match.groups()
            else:
                raw_day, month_name, raw_year = match.groups()
            try:
                value = datetime(
                    int(raw_year), _MONTH_NUMBERS[month_name], int(raw_day)
                ).date().isoformat()
            except ValueError:
                continue
            if value not in values:
                values.append(value)
    for match in re.finditer(
        rf"\b({month_names})[,]?\s+((?:19|20)\d{{2}})\b",
        question.casefold(),
    ):
        month_name, raw_year = match.groups()
        value = f"{int(raw_year):04d}-{_MONTH_NUMBERS[month_name]:02d}"
        if value not in values:
            values.append(value)
    return values


def _is_recommendation_request(lowered: str) -> bool:
    """Distinguish requests for new advice from lookups about prior advice."""

    return bool(
        re.search(
            r"\b(?:can|could|would|will) you\s+"
            r"(?:recommend|suggest|advise|help (?:me|us) (?:choose|find))\b|"
            r"\b(?:please|kindly)\s+(?:recommend|suggest|advise)\b|"
            r"\bdo you have\s+(?:any\s+|some\s+|helpful\s+)?"
            r"(?:advice|recommendations?|suggestions?|tips|resources?)\b|"
            r"\bany\s+(?:advice|recommendations?|suggestions?|tips|resources?)\b|"
            r"^(?:what|which)\b.{0,80}\bwould be\s+(?:an?\s+)?good\b|"
            r"^should\s+(?:i|we|they|he|she)\b|"
            r"\bdo you think\b.{0,100}\b(?:a good idea|should|worth)\b|"
            r"^(?:what|which)\b.{0,80}\bshould\s+\w+\s+"
            r"(?:do|use|try|choose|buy|read|visit|take|learn|serve|make|prepare|cook|wear)\b|"
            r"^(?:where|how)\b.{0,60}\b(?:can|could|should)\b.{0,60}\blearn\b",
            lowered,
        )
    )


def _is_relational_order_request(lowered: str) -> bool:
    """Recognize questions whose answer is the event/state relative to an anchor."""

    if not re.search(r"\b(?:before|after)\b", lowered):
        return False
    return bool(
        re.match(r"^what did\b.{0,60}\bdo (?:before|after)\b", lowered)
        or re.match(
            r"^(?:which|what)\b.{0,80}\bdid i\b"
            r".{0,60}\b(?:before|after)\b",
            lowered,
        )
        or re.match(
            r"^(?:which|what|where)\b.{0,70}\b(?:was|were)\b"
            r".{0,60}\b(?:before|after)\b",
            lowered,
        )
        or re.match(
            r"^(?:which|what)\b.{0,60}\b(?:happened|occurred|came)\b"
            r".{0,40}\b(?:before|after)\b",
            lowered,
        )
    )


def build_query_frame(question: str) -> QueryFrame:
    raw_tokens = _tokens(question)
    content = list(dict.fromkeys(
        token for token in raw_tokens
        if len(token) > 1 and token not in _QUERY_FUNCTION_WORDS
    ))
    lowered = " ".join(raw_tokens)
    dates = list(dict.fromkeys([
        *_natural_dates(question),
        *re.findall(r"\b(?:19|20)\d{2}(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?\b", question),
    ]))
    temporal = list(dict.fromkeys(
        token for token in raw_tokens
        if (
            token in _MONTHS
            or token in {"before", "after", "during", "between"}
            or re.fullmatch(r"(?:19|20)\d{2}", token)
        )
    ))
    if re.search(
        r"\bif\b.{0,100}\b(?:hadn't|had not|would not|wouldn't|without)\b|"
        r"\bwould\b.{0,80}\bif\b",
        lowered,
    ):
        operation, answer_form = "counterfactual", "state"
    elif re.search(
        r"\bwhat (?:percentage|percent)\b|\bwhat percentage of\b|"
        r"\bpercentage (?:is|was|are|were)\b",
        lowered,
    ):
        operation, answer_form = "count", "number"
    elif re.search(r"\bin (?:the )?order\b|\border(?:ing)? of\b|\bfirst to last\b|"
        r"\bfrom earliest to latest\b|\bchronological(?:ly)?\b", lowered):
        operation, answer_form = "ordering", "list"
    elif _is_recommendation_request(lowered):
        operation, answer_form = "recommendation", "recommendation"
    elif re.search(
        r"\bhow many\s+(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b"
        r".{0,40}\b(?:per|each|typical|usually|weekly|monthly|yearly)\b",
        lowered,
    ):
        operation, answer_form = "count", "number"
    elif (
        re.search(r"\bhow many\s+(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b", lowered)
        and re.search(
            r"\b(?:take|took|spend|spent|pass|passed|between|before|after|"
            r"ago|since|elapsed|apart|from)\b",
            lowered,
        )
    ):
        operation, answer_form = "duration", "duration"
    elif re.match(r"^when\b", lowered) and re.search(r"\bplan(?:n)?\b", lowered):
        operation, answer_form = "planned_date", "date"
    elif re.search(r"\bwhich\s+(?:day|week|month|year|date)\b", lowered):
        operation, answer_form = "date", "date"
    elif re.search(r"\bhow old\b", lowered):
        operation, answer_form = "lookup", "number"
    elif (
        re.search(r"\bwhere\b", lowered)
        and re.search(r"\b(?:latest|most recent|currently|current|now)\b", lowered)
    ):
        operation, answer_form = "latest", "entity"
    elif re.search(r"^where\b|\bwhere (?:was|were|is|are|did|does)\b", lowered):
        operation, answer_form = "location", "list"
    elif re.search(r"\bwhat do\b.{0,80}\blike\b", lowered):
        operation, answer_form = "preference_list", "list"
    elif re.search(
        r"\b(?:finally|ultimately)\s+(?:decid(?:e|ed)|chos(?:e|en)|named?)\b|"
        r"\b(?:ended up|settled on)\b|\bfinal(?:ly)?\s+(?:choice|name|decision)\b",
        lowered,
    ):
        operation, answer_form = "latest", "state"
    elif re.search(r"\b(?:more|less)\s+frequent(?:ly)?\b", lowered):
        operation, answer_form = "latest", "state"
    elif re.search(r"\bhow often\b|\bfrequency\b", lowered):
        operation, answer_form = "recurrence", "frequency"
    elif re.search(
        r"\bwhat (?:is|was|were|are)\s+(?:the\s+)?"
        r"(?:total|combined|aggregate|sum)\b|"
        r"\b(?:total|combined|aggregate)\s+(?:amount|number|count|weight|value)\b|"
        r"\b(?:in total|altogether)\b",
        lowered,
    ):
        operation, answer_form = "count", "number"
    elif re.search(r"\bhow (?:many|much)\b", lowered):
        operation, answer_form = "count", "number"
    elif re.search(r"\b(latest|most recent|most recently|recently|currently|current|now)\b", lowered):
        operation, answer_form = "latest", "state"
    elif _is_relational_order_request(lowered):
        operation = "ordering"
        answer_form = "entity" if re.search(r"\b(?:which|what|where)\b", lowered) else "list"
    elif re.search(r"\b(earliest|first)\b", lowered):
        operation = "earliest"
        answer_form = "entity" if re.search(r"\bwhich\b", lowered) else "date"
    elif re.search(r"\bhow long\b|\bduration\b|\bdifference\b", lowered):
        operation, answer_form = "duration", "duration"
    elif re.match(r"^when\b|^what (?:date|year)\b", lowered):
        operation, answer_form = "date", "date"
    elif re.search(r"\bwhich\b|\blist\b|\bwhat (?:are|were)\b", lowered):
        operation, answer_form = "list", "list"
    elif re.search(r"\bstate\b|\bstatus\b", lowered):
        operation, answer_form = "state", "state"
    else:
        operation, answer_form = "lookup", "span"
    original_words = _WORD_RE.findall(question)
    participants: list[str] = []
    for word in original_words:
        key = _token_key(word)
        if (
            word[:1].isupper()
            and key not in _QUERY_FUNCTION_WORDS
            and key not in _MONTHS
            and key not in participants
        ):
            participants.append(key)
    hypotheses = ["direct_evidence", "episode_context", "theme_context"]
    if operation in {"latest", "earliest", "state"}:
        hypotheses.append("state_history")
    if operation in {"date", "planned_date", "duration", "latest", "earliest", "ordering", "recurrence"}:
        hypotheses.append("temporal_scope")
    elif temporal:
        hypotheses.append("temporal_scope")
    if operation in {"count", "list", "location", "preference_list", "recurrence"}:
        hypotheses.append("collection_scope")
    if operation == "recommendation":
        hypotheses.append("preference_constraints")
    return QueryFrame(
        raw_question=question,
        content_terms=content,
        participant_terms=participants,
        temporal_terms=temporal,
        explicit_dates=dates,
        requested_operation=operation,  # type: ignore[arg-type]
        answer_form=answer_form,  # type: ignore[arg-type]
        hypotheses=hypotheses,
    )


def _node_text(node: Any) -> str:
    return str(getattr(node, "retrieval_text", "") or "")


def _node_vector(node: Any) -> list[float] | None:
    return getattr(node, "embedding", None)


def _bm25_rank(query_terms: list[str], rows: list[tuple[str, str]]) -> list[str]:
    if not query_terms or not rows:
        return []
    docs = [Counter(_tokens(text)) for _node_id, text in rows]
    lengths = [sum(doc.values()) for doc in docs]
    average = sum(lengths) / max(1, len(lengths))
    frequencies: Counter[str] = Counter()
    for doc in docs:
        frequencies.update(doc.keys())
    scored: list[tuple[float, str]] = []
    for (node_id, _text), doc, length in zip(rows, docs, lengths):
        score = 0.0
        for term in query_terms:
            tf = doc.get(term, 0)
            if not tf:
                continue
            df = frequencies[term]
            idf = math.log(1.0 + (len(rows) - df + 0.5) / (df + 0.5))
            score += idf * (tf * 2.2) / (
                tf + 1.2 * (1.0 - 0.75 + 0.75 * length / max(1.0, average))
            )
        if score > 0:
            scored.append((score, node_id))
    return [node_id for _score, node_id in sorted(scored, reverse=True)]


def _dense_rank(query_vector: list[float], nodes: list[Any]) -> list[str]:
    return [
        node.node_id if hasattr(node, "node_id") else node.edge_id
        for _score, node in sorted(
            (
                (cosine_similarity(query_vector, _node_vector(node)), node)
                for node in nodes
            ),
            key=lambda item: (item[0], getattr(item[1], "node_id", getattr(item[1], "edge_id", ""))),
            reverse=True,
        )
        if _score > 0
    ]


def _exact_rank(frame: QueryFrame, nodes: list[Any]) -> list[str]:
    terms = set(frame.content_terms + frame.participant_terms + frame.temporal_terms)
    scored: list[tuple[float, str]] = []
    for node in nodes:
        node_id = getattr(node, "node_id", getattr(node, "edge_id", ""))
        values = set(_tokens(_node_text(node)))
        overlap = len(terms & values)
        if overlap:
            specificity = overlap / max(1, min(len(terms), len(values)))
            scored.append((overlap + specificity, node_id))
    return [node_id for _score, node_id in sorted(scored, reverse=True)]


def _rrf(channels: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for channel in channels:
        for rank, node_id in enumerate(channel):
            scores[node_id] += 1.0 / (k + rank + 1)
    return dict(scores)


def _query_overlap(frame: QueryFrame, text: str) -> float:
    query = set(frame.content_terms + frame.participant_terms + frame.temporal_terms)
    if not query:
        return 0.0
    return len(query & set(_tokens(text))) / len(query)


def _node_type(node: Any) -> str:
    if isinstance(node, TurnNode):
        return "turn"
    if isinstance(node, ClaimNode):
        return "claim"
    if isinstance(node, EventNode):
        return "event"
    if isinstance(node, EventEntityNode):
        return "event_entity"
    if isinstance(node, EpisodeNode):
        return "episode"
    if isinstance(node, ThemeNode):
        return "theme"
    if isinstance(node, EventFrameV3):
        return "event_frame"
    if isinstance(node, OperandRecordV3):
        return "operand"
    return "hyperedge"


def _annotate_packed_provenance(
    hint: dict[str, Any], kept_ids: set[str]
) -> dict[str, Any]:
    """Keep graph completeness distinct from bounded prompt coverage."""
    result = dict(hint)
    source_ids = set(result.get("source_turn_ids", []))
    operand_ids = set(result.get("operand_ids", []))
    result["packed_source_coverage"] = (
        len(source_ids & kept_ids) / max(1, len(source_ids))
    )
    result["packed_operand_coverage"] = (
        len(operand_ids & kept_ids) / max(1, len(operand_ids))
        if operand_ids else None
    )
    result["packed_provenance_complete"] = bool(
        source_ids and source_ids.issubset(kept_ids)
    ) or bool(operand_ids and operand_ids.issubset(kept_ids))
    return result


def _node_session_id(node: Any) -> str | None:
    value = getattr(node, "session_id", None)
    if value:
        return str(value)
    sessions = getattr(node, "session_ids", None) or []
    return str(sessions[0]) if sessions else None


def _relation_prior(frame: QueryFrame, relation: str) -> float:
    weights = {
        "supports": 1.0, "refers_to": 1.0, "same_event": 0.9,
        "event_entity_member": 1.0, "episode_member": 0.78,
        "theme_member": 0.45, "participant": 0.30, "semantic_cluster": 0.38,
        "contradiction": 0.65, "state_history": 0.72,
        "temporal_scope": 0.70, "quantity_collection": 0.72,
        "event_frame_member": 0.82, "operand_projection": 0.92,
    }
    operation = frame.requested_operation
    if operation in {"latest", "earliest", "state"}:
        weights.update(state_history=1.0, contradiction=0.95, temporal_scope=0.92, quantity_collection=0.30)
    elif operation in {"date", "planned_date", "duration", "ordering"}:
        weights.update(temporal_scope=1.0, state_history=0.78, contradiction=0.50, quantity_collection=0.35)
    elif operation in {"count", "list", "location", "preference_list", "recurrence"}:
        weights.update(quantity_collection=1.0, supports=0.95, temporal_scope=0.48, state_history=0.55)
    elif operation == "counterfactual":
        weights.update(supports=1.0, same_event=0.95, contradiction=0.85, temporal_scope=0.55)
    elif operation == "recommendation":
        weights.update(supports=1.0, contradiction=0.85, participant=0.58, semantic_cluster=0.62)
    return weights.get(relation, 0.25)


def _legacy_scope_posteriors(
    frame: QueryFrame,
    nodes: dict[str, Any],
    channels: dict[str, list[str]],
    rrf_scores: dict[str, float],
) -> list[dict[str, Any]]:
    query_terms = set(frame.content_terms)
    grouped: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    for node_id, node in nodes.items():
        session_id = _node_session_id(node)
        if session_id:
            grouped[session_id].append((node_id, node))
    channel_top = {name: set(values[:80]) for name, values in channels.items()}
    max_rrf = max(rrf_scores.values(), default=1.0)
    rows: list[dict[str, Any]] = []
    for session_id, values in grouped.items():
        covered: set[str] = set()
        best_overlap = 0.0
        relevant_count = 0
        best_rrf = 0.0
        node_ids = {node_id for node_id, _node in values}
        for node_id, node in values:
            overlap_terms = query_terms & set(_tokens(_node_text(node)))
            if overlap_terms:
                covered.update(overlap_terms)
                relevant_count += 1
            best_overlap = max(best_overlap, _query_overlap(frame, _node_text(node)))
            best_rrf = max(best_rrf, rrf_scores.get(node_id, 0.0))
        channel_hits = sum(bool(node_ids & ids) for ids in channel_top.values())
        coverage = len(covered) / max(1, len(query_terms))
        density = min(1.0, math.log1p(relevant_count) / math.log(8.0))
        posterior = (
            0.46 * coverage + 0.20 * best_overlap
            + 0.14 * (channel_hits / max(1, len(channel_top)))
            + 0.12 * (best_rrf / max_rrf) + 0.08 * density
        )
        rows.append({
            "session_id": session_id, "posterior": round(posterior, 6),
            "query_coverage": round(coverage, 6), "covered_terms": sorted(covered),
            "channel_hits": channel_hits, "relevant_node_count": relevant_count,
        })
    rows.sort(key=lambda row: (
        row["posterior"], row["query_coverage"], row["channel_hits"], row["session_id"]
    ), reverse=True)
    return rows


def _scope_posteriors(
    frame: QueryFrame,
    nodes: dict[str, Any],
    channels: dict[str, list[str]],
    rrf_scores: dict[str, float],
) -> list[dict[str, Any]]:
    return _score_scope_posteriors(
        frame, nodes, channels, rrf_scores,
        tokenize=_tokens, node_text=_node_text, query_overlap=_query_overlap,
    )


def _temporal_compatibility(frame: QueryFrame, node: Any) -> float:
    if not (frame.temporal_terms or frame.explicit_dates):
        return 0.0
    text = _node_text(node).casefold()
    terms = frame.temporal_terms + frame.explicit_dates
    return 1.0 if any(term.casefold() in text for term in terms) else 0.0


def _expand(
    frame: QueryFrame,
    nodes: dict[str, Any],
    edges: dict[str, HyperEdge],
    rrf_scores: dict[str, float],
    seed_ids: list[str],
    *,
    max_depth: int = 2,
    max_nodes: int = 96,
) -> tuple[dict[str, float], list[dict[str, Any]], list[str]]:
    incident: dict[str, list[str]] = defaultdict(list)
    for edge in edges.values():
        for incidence in edge.incidences:
            incident[incidence.node_id].append(edge.edge_id)
    selected: dict[str, float] = {}
    trace: list[dict[str, Any]] = []
    visited_edges: set[str] = set()
    queue: list[tuple[float, int, str, str | None]] = []
    max_rrf = max(rrf_scores.values(), default=1.0)
    for node_id in seed_ids:
        base = rrf_scores.get(node_id, 0.0) / max_rrf
        heapq.heappush(queue, (-base, 0, node_id, None))
    while queue and len(selected) < max_nodes:
        negative, depth, item_id, via_edge = heapq.heappop(queue)
        score = -negative
        if item_id in selected and selected[item_id] >= score:
            continue
        selected[item_id] = score
        if via_edge:
            via = edges.get(via_edge)
            trace.append({
                "node_id": item_id,
                "via_hyperedge": via_edge,
                "depth": depth,
                "score": round(score, 6),
                "node_type": _node_type(nodes[item_id]),
                "relation": via.relation if via else None,
                "relation_prior": round(_relation_prior(frame, via.relation), 6) if via else 0.0,
                "hyperedge_size": len(via.incidences) if via else 0,
            })
        if depth >= max_depth:
            continue
        candidate_edges: list[str] = []
        if item_id in edges:
            candidate_edges.append(item_id)
        candidate_edges.extend(incident.get(item_id, []))
        for edge_id in candidate_edges:
            edge = edges[edge_id]
            visited_edges.add(edge_id)
            fanout_scale = min(1.0, math.sqrt(8.0 / max(8, len(edge.incidences))))
            edge_gain = fanout_scale * (
                0.28 * _query_overlap(frame, edge.retrieval_text)
                + 0.10 * _temporal_compatibility(frame, edge)
                + 0.08 * edge.confidence
                + 0.16 * _relation_prior(frame, edge.relation)
            )
            ranked_incidences = sorted(
                edge.incidences,
                key=lambda value: (
                    _query_overlap(frame, _node_text(nodes.get(value.node_id))),
                    rrf_scores.get(value.node_id, 0.0),
                    value.node_id,
                ),
                reverse=True,
            )[:12]
            for incidence in ranked_incidences:
                neighbor_id = incidence.node_id
                if neighbor_id not in nodes:
                    continue
                neighbor = nodes[neighbor_id]
                base = rrf_scores.get(neighbor_id, 0.0) / max_rrf
                specificity = 0.08 if _node_type(neighbor) in {"turn", "claim", "event"} else 0.03
                gain = (
                    0.40 * score
                    + 0.34 * _query_overlap(frame, _node_text(neighbor))
                    + 0.12 * _temporal_compatibility(frame, neighbor)
                    + 0.10 * base
                    + specificity
                    + edge_gain
                )
                heapq.heappush(queue, (-gain, depth + 1, neighbor_id, edge_id))
    return selected, trace, sorted(visited_edges)


def _date_key(value: str | None) -> tuple[int, str]:
    if not value:
        return (0, "")
    normalized = value.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y/%m", "%Y"):
        try:
            return (1, datetime.strptime(normalized, fmt).isoformat())
        except ValueError:
            pass
    return (1, normalized)


def _relevant_claims(
    frame: QueryFrame, claims: list[ClaimNode], scores: dict[str, float]
) -> list[ClaimNode]:
    ranked = sorted(
        claims,
        key=lambda item: (
            _query_overlap(frame, item.retrieval_text),
            scores.get(item.node_id, 0.0),
            item.confidence,
        ),
        reverse=True,
    )
    positive = [item for item in ranked if _query_overlap(frame, item.retrieval_text) > 0]
    return positive or ranked[:4]


def _closure_and_operator(
    frame: QueryFrame,
    index: V3Index,
    selected_ids: set[str],
    visited_edge_ids: list[str],
    scores: dict[str, float],
    truncated: bool,
) -> tuple[ClosureCertificate, dict[str, Any] | None]:
    claims = _relevant_claims(
        frame, [item for item in index.claims if item.node_id in selected_ids], scores
    )
    operands: list[str] = []
    missing: list[str] = []
    complete = True
    relevant_edges: list[HyperEdge] = []
    if frame.requested_operation == "recurrence":
        occurrence_claims = [
            item for item in claims
            if item.polarity != "negative"
            and (item.event_time or item.observed_at)
        ]
        operands = [item.node_id for item in occurrence_claims[:12]]
        if len(operands) < 2:
            complete = False
            missing.append("recurrence_occurrences")
    elif frame.requested_operation in {"count", "list"}:
        relevant_edges = [
            edge for edge in index.hyperedges
            if edge.relation == "quantity_collection"
            and (
                _query_overlap(frame, edge.retrieval_text) > 0
                or any(incidence.node_id in {item.node_id for item in claims} for incidence in edge.incidences)
            )
        ]
        if relevant_edges:
            operands = list(dict.fromkeys(
                incidence.node_id
                for edge in relevant_edges
                for incidence in edge.incidences
                if incidence.role == "operand"
            ))
            missing_operands = [value for value in operands if value not in selected_ids]
            if missing_operands:
                complete = False
                missing.append("collection_operands")
        else:
            complete = False
            missing.append("collection_scope")
    elif frame.requested_operation in {"latest", "earliest", "state"}:
        chain_candidates = [
            chain for chain in index.state_chains
            if any(value in selected_ids for value in chain.history_claim_ids)
        ]
        if chain_candidates:
            operands = list(dict.fromkeys(
                value for chain in chain_candidates for value in chain.history_claim_ids
            ))
            if any(value not in selected_ids for value in operands):
                complete = False
                missing.append("state_history")
        else:
            operands = [item.node_id for item in claims]
            if not operands:
                complete = False
                missing.append("state_evidence")
    elif frame.requested_operation in {"date", "planned_date", "duration", "ordering"}:
        temporal_edges = [
            edge for edge in index.hyperedges
            if edge.relation == "temporal_scope"
            and any(incidence.node_id in selected_ids for incidence in edge.incidences)
        ]
        operands = list(dict.fromkeys(
            incidence.node_id for edge in temporal_edges for incidence in edge.incidences
        ))
        if not operands:
            complete = False
            missing.append("temporal_scope")
    else:
        operands = [item.node_id for item in claims[:8]]
        if not operands:
            complete = False
            missing.append("direct_evidence")
    if truncated and frame.requested_operation in {"count", "list", "duration", "recurrence"}:
        complete = False
        missing.append("untruncated_scope")
    contradiction_ids = list(dict.fromkeys(
        incidence.node_id
        for edge in index.hyperedges
        if edge.relation == "contradiction"
        and any(incidence.node_id in selected_ids for incidence in edge.incidences)
        for incidence in edge.incidences
    ))
    certificate = ClosureCertificate(
        requested_operation=frame.requested_operation,
        complete=complete,
        visited_hyperedge_ids=visited_edge_ids,
        operand_node_ids=operands,
        contradiction_node_ids=contradiction_ids,
        missing_requirements=list(dict.fromkeys(missing)),
        truncated=truncated,
        provenance_complete=all(
            item.source_turn_ids for item in claims
        ),
        scope_description=(
            "local hypergraph closure over retrieved evidence; no global memory scan"
        ),
    )
    if not certificate.complete:
        return certificate, None
    by_id = {item.node_id: item for item in index.claims}
    operand_claims = [by_id[value] for value in operands if value in by_id]
    if frame.requested_operation == "recurrence":
        ordered = sorted(
            operand_claims,
            key=lambda item: (_date_key(item.event_time or item.observed_at), item.observation_order),
        )
        return certificate, {
            "operation": "recurrence",
            "occurrences": [
                {
                    "subject": item.subject,
                    "predicate": item.predicate,
                    "value": item.object,
                    "time": item.event_time or item.observed_at,
                    "source_turn_ids": item.source_turn_ids,
                }
                for item in ordered
            ],
        }
    if frame.requested_operation in {"count", "list"}:
        distinct: dict[str, ClaimNode] = {}
        for claim in operand_claims:
            if claim.polarity == "negative" or claim.state_op in {"remove", "retract", "cancel"}:
                distinct.pop(claim.object_key, None)
            else:
                distinct[claim.object_key] = claim
        values = [item.object for item in distinct.values()]
        result: dict[str, Any] = {"operation": "distinct", "values": values}
        if frame.requested_operation == "count":
            result["count"] = len(values)
        return certificate, result
    if frame.requested_operation in {"latest", "earliest", "state"} and operand_claims:
        reverse = frame.requested_operation != "earliest"
        ordered = sorted(
            operand_claims,
            key=lambda item: (_date_key(item.event_time or item.observed_at), item.observation_order),
            reverse=reverse,
        )
        selected = ordered[0]
        return certificate, {
            "operation": frame.requested_operation,
            "subject": selected.subject,
            "predicate": selected.predicate,
            "value": selected.object,
            "time": selected.event_time or selected.observed_at,
            "polarity": selected.polarity,
            "modality": selected.modality,
        }
    return certificate, None


def _render_block(kind: str, node: Any) -> str:
    if isinstance(node, TurnNode):
        return (
            f"[TURN {node.node_id} | session={node.session_id} | "
            f"date={node.session_date or 'unknown'} | speaker={node.speaker}]\n{node.text}"
        )
    if isinstance(node, ClaimNode):
        return (
            f"[CLAIM {node.node_id} | time={node.event_time or node.observed_at or 'unknown'} | "
            f"modality={node.modality} | polarity={node.polarity} | sources={','.join(node.source_turn_ids)}]\n"
            f"{node.subject} | {node.predicate} | {node.object}"
        )
    if isinstance(node, EventNode):
        return (
            f"[EVENT {node.node_id} | time={node.event_time or 'unknown'} | "
            f"status={node.status} | sources={','.join(node.source_turn_ids)}]\n{node.label}"
        )
    if isinstance(node, EpisodeNode):
        return (
            f"[EPISODE {node.node_id} | session={node.session_id} | "
            f"turns={','.join(node.turn_ids)}]\n{node.retrieval_text}"
        )
    if isinstance(node, ThemeNode):
        return (
            f"[THEME {node.node_id} | episodes={','.join(node.episode_ids)}]\n"
            f"{node.retrieval_text}"
        )
    if isinstance(node, OperandRecordV3):
        return (
            f"[OPERAND {node.node_id} | event_time={node.event_time or 'unknown'} | "
            f"observed_at={node.observed_at or 'unknown'} | modality={node.modality} | "
            f"polarity={node.polarity} | sources={','.join(node.source_turn_ids)}]\n"
            f"{node.subject_key} | {node.predicate_key} | {node.object_text}"
        )
    if isinstance(node, EventFrameV3):
        return (
            f"[EVENT_FRAME {node.node_id} | event_time={node.event_time or 'unknown'} | "
            f"observed_at={node.observed_at or 'unknown'} | sources={','.join(node.source_turn_ids)}]\n"
            f"{node.label}"
        )
    return f"[{kind.upper()}]\n{_node_text(node)}"


_SCALAR_RE = re.compile(
    r"(?<!\w)(?:"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?|"
    r"\d{1,3}:\d{2}(?::\d{2})?|"
    r"[$€£]\s?\d+(?:[.,]\d+)?|"
    r"\d+(?:[.,]\d+)?\s?(?:%|km|mi|miles?|minutes?|hours?|days?|weeks?|"
    r"months?|years?|kg|lb|lbs|degrees?|items?)"
    r")(?!\w)",
    re.IGNORECASE,
)


def _compact_relevant_text(frame: QueryFrame, text: str, limit: int = 280) -> str:
    segments = [
        value.strip()
        for value in re.split(r"(?<=[.!?])\s+|\n+", text)
        if value.strip()
    ]
    if not segments:
        return text[:limit]
    ranked = sorted(
        enumerate(segments),
        key=lambda item: (
            _query_overlap(frame, item[1]),
            len(set(frame.content_terms) & set(_tokens(item[1]))),
            -item[0],
        ),
        reverse=True,
    )
    chosen = sorted(index for index, _value in ranked[:2])
    return " ".join(segments[index] for index in chosen)[:limit]


def _evidence_time(node: Any) -> str | None:
    if isinstance(node, TurnNode):
        return node.session_date
    if isinstance(node, ClaimNode):
        return node.event_time or node.observed_at
    if isinstance(node, EventNode):
        return node.event_time
    if isinstance(node, EpisodeNode):
        return node.time_end or node.time_start or node.session_date
    if isinstance(node, ThemeNode):
        return node.time_end or node.time_start
    if isinstance(node, OperandRecordV3):
        return node.event_time or node.observed_at
    if isinstance(node, EventFrameV3):
        return node.event_time or node.observed_at
    return None


def _temporal_evidence_ledger(
    frame: QueryFrame,
    kept: list[tuple[str, Any, float, str]],
) -> list[dict[str, Any]]:
    """Create a compact chronology from packed evidence only."""
    candidates: list[tuple[tuple[int, str], float, dict[str, Any]]] = []
    for kind, node, score, _source in kept:
        observed = _evidence_time(node)
        text = _node_text(node)
        overlap = _query_overlap(frame, text)
        if not observed or overlap <= 0:
            continue
        compact = _compact_relevant_text(frame, text)
        scalars = list(dict.fromkeys(match.group(0) for match in _SCALAR_RE.finditer(compact)))
        if not scalars and frame.requested_operation not in {
            "latest", "earliest", "state", "date", "duration", "ordering"
        }:
            continue
        candidates.append(
            (
                _date_key(observed),
                overlap * 10.0 + score + (8.0 if scalars else 0.0),
                {
                    "node_id": node.node_id,
                    "node_type": kind,
                    "observed_at": observed,
                    "exact_value_spans": scalars[:6],
                    "evidence": compact,
                },
            )
        )
    candidates.sort(key=lambda item: (item[1], item[0]), reverse=True)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...], str]] = set()
    for _date, _score, row in candidates:
        key = (
            row["observed_at"],
            tuple(row["exact_value_spans"]),
            canonical_key(row["evidence"]),
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
        if len(rows) >= 6:
            break
    rows.sort(key=lambda row: _date_key(row["observed_at"]), reverse=True)
    if (
        frame.requested_operation not in {"date", "planned_date", "duration", "ordering", "latest", "earliest", "state"}
        and len({value for row in rows for value in row["exact_value_spans"]}) < 2
    ):
        return []
    return rows


def _newest_scalar_hint(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates: list[tuple[str, str, dict[str, Any]]] = []
    distinct: set[str] = set()
    for row in rows:
        match = re.search(r"\b((?:19|20)\d{2})[/.-](\d{1,2})[/.-](\d{1,2})\b", row["observed_at"])
        if not match:
            continue
        date_key = f"{int(match.group(1)):04d}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
        for value in row["exact_value_spans"]:
            distinct.add(value)
            candidates.append((date_key, value, row))
    if len(distinct) < 2 or not candidates:
        return None
    date_key, value, row = max(candidates, key=lambda item: (item[0], item[1]))
    supporters = [
        item[2]["node_id"] for item in candidates
        if item[0] == date_key and item[1] == value
    ]
    return {
        "operation": "newest_scalar_from_local_evidence",
        "value": value,
        "observed_at": row["observed_at"],
        "supporting_node_ids": list(dict.fromkeys(supporters)),
        "candidate_values": sorted(distinct),
    }


def _planned_event_hint(
    frame: QueryFrame,
    kept: list[tuple[str, Any, float, str]],
) -> dict[str, Any] | None:
    if frame.requested_operation != "planned_date":
        return None
    turns = {
        node.node_id: node for kind, node, _score, _source in kept if kind == "turn"
    }
    candidates: list[tuple[float, dict[str, Any]]] = []
    for kind, node, score, _source in kept:
        if isinstance(node, TurnNode):
            plan_match = re.search(
                r"\b(?:plan(?:ning|ned)?|thinking about|intend(?:ing)?|going to)\b",
                node.text,
                flags=re.IGNORECASE,
            )
            time_match = re.search(
                r"\b(?:next\s+(?:day|week|month|season|year|weekend)|"
                r"tomorrow|in\s+\d+\s+(?:days?|weeks?|months?|years?))\b",
                node.text,
                flags=re.IGNORECASE,
            )
            overlap = _query_overlap(frame, _node_text(node))
            if plan_match and time_match and overlap > 0:
                candidates.append((
                    overlap * 10.0 + score,
                    {
                        "operation": "planned_event_from_local_turn",
                        "event_time": time_match.group(0),
                        "anchor_date": node.session_date,
                        "node_id": node.node_id,
                        "source_turn_ids": [node.node_id],
                        "evidence": _compact_relevant_text(frame, node.text, 320),
                    },
                ))
            continue
        is_planned = (
            isinstance(node, ClaimNode) and node.modality == "planned"
        ) or (
            isinstance(node, EventNode) and node.status == "planned"
        )
        event_time = getattr(node, "event_time", None)
        if not is_planned or not event_time:
            continue
        overlap = _query_overlap(frame, _node_text(node))
        if overlap <= 0:
            continue
        source_ids = list(getattr(node, "source_turn_ids", []))
        source_turns = [turns[value] for value in source_ids if value in turns]
        anchor = next((item.session_date for item in source_turns if item.session_date), None)
        if anchor is None and isinstance(node, ClaimNode):
            anchor = node.observed_at
        candidates.append((
            overlap * 10.0 + score,
            {
                "operation": "planned_event_from_local_evidence",
                "event_time": event_time,
                "anchor_date": anchor,
                "node_id": node.node_id,
                "source_turn_ids": source_ids,
                "evidence": _compact_relevant_text(frame, _node_text(node), 320),
            },
        ))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]["node_id"]))[1]


def _recommendation_constraints(
    frame: QueryFrame,
    kept: list[tuple[str, Any, float, str]],
    scope_posteriors: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if frame.requested_operation != "recommendation":
        return []
    scope_rows = list(scope_posteriors or [])
    allowed_sessions = set(
        recommendation_scope_session_ids(frame, scope_rows)
    )

    raw_terms = set(_tokens(frame.raw_question))
    self_query = bool(raw_terms & {"i", "me", "my", "mine", "we", "our", "ours"})

    def speaker_compatible(node: Any) -> bool:
        if isinstance(node, TurnNode):
            speaker_key = canonical_key(node.speaker_key)
            if self_query:
                return node.transport_role == "user" or speaker_key in {"participant 1", "user"}
            if frame.participant_terms:
                return bool(set(frame.participant_terms) & set(_tokens(node.speaker_key)))
            return True
        subject_key = canonical_key(getattr(node, "subject_key", ""))
        if self_query and subject_key:
            return subject_key in {"participant 1", "user"}
        if frame.participant_terms and subject_key:
            return bool(set(frame.participant_terms) & set(_tokens(subject_key)))
        return True

    ownership_pattern = (
        r"\b(?:already|currently|current|have|has|having|own|owns|owned|"
        r"use|uses|using|bought|purchased|got|compatible|setup|equipment)\b"
    )
    relation_pattern = r"\b(?:consider|like|love|need|plan|prefer|replace|seek|want)\w*\b"

    def constraint_strength(node: Any) -> int:
        text = _node_text(node).casefold()
        if re.search(ownership_pattern, text):
            return 3
        if isinstance(node, ClaimNode) and node.kind in {"preference", "state"}:
            return 2
        if re.search(relation_pattern, text):
            return 1
        return 0

    ranked = sorted(
        kept,
        key=lambda item: (
            _node_session_id(item[1]) in allowed_sessions if allowed_sessions else True,
            item[3] == "recommendation_resource_provenance",
            speaker_compatible(item[1]),
            constraint_strength(item[1]),
            _query_overlap(frame, _node_text(item[1])),
            item[2],
        ),
        reverse=True,
    )
    rows: list[dict[str, Any]] = []
    target_terms = set(frame.content_terms) - {
        "recommend", "resource", "learn", "more", "where", "some",
        "online", "good", "help", "find", "think", "rearrang", "tips", "weekend",
    }
    seen: set[str] = set()
    for kind, node, _score, source in ranked:
        session_id = _node_session_id(node)
        if allowed_sessions and session_id not in allowed_sessions:
            continue
        if not speaker_compatible(node):
            continue
        relation_text = _node_text(node).casefold()
        if isinstance(node, ClaimNode):
            if not (
                node.kind in {"preference", "state"}
                or node.modality == "planned"
                or re.search(relation_pattern, node.predicate, flags=re.IGNORECASE)
                or re.search(ownership_pattern, relation_text, flags=re.IGNORECASE)
            ):
                continue
        elif isinstance(node, OperandRecordV3):
            if not (
                node.modality == "planned"
                or re.search(relation_pattern, node.predicate_key, flags=re.IGNORECASE)
                or re.search(ownership_pattern, relation_text, flags=re.IGNORECASE)
            ):
                continue
        elif isinstance(node, TurnNode):
            if not (
                re.search(relation_pattern, relation_text, flags=re.IGNORECASE)
                or re.search(ownership_pattern, relation_text, flags=re.IGNORECASE)
            ):
                continue
        else:
            continue
        text = (
            resource_evidence_text(_node_text(node))
            if source == "recommendation_resource_provenance"
            else _compact_relevant_text(frame, _node_text(node), 240)
        )
        normalized = canonical_key(text)
        overlap = _query_overlap(frame, text)
        if (
            not normalized or normalized in seen
            or (
                overlap < 0.12
                and source != "recommendation_resource_provenance"
            )
            or (
                target_terms
                and len(target_terms & set(_tokens(text))) < 1
                and source != "recommendation_resource_provenance"
            )
        ):
            continue
        seen.add(normalized)
        rows.append({
            "node_id": node.node_id,
            "node_type": kind,
            "constraint": text,
            "polarity": getattr(node, "polarity", "unknown"),
            "modality": getattr(node, "modality", "unknown"),
            "session_id": session_id,
            "selection_source": source,
        })
        if len(rows) >= 5:
            break
    return rows


def _pack(
    ordered: list[tuple[str, Any, float, str]],
    budget: int,
) -> tuple[list[tuple[str, Any, float, str]], str, list[dict[str, Any]]]:
    kept: list[tuple[str, Any, float, str]] = []
    blocks: list[str] = []
    decisions: list[dict[str, Any]] = []
    used = 0
    for kind, node, score, source in ordered:
        block = _render_block(kind, node)
        cost = rough_token_count(block)
        if used + cost > budget:
            decisions.append({
                "node_id": getattr(node, "node_id", ""),
                "decision": "drop_budget",
                "rough_tokens": cost,
                "source": source,
            })
            continue
        kept.append((kind, node, score, source))
        blocks.append(block)
        used += cost
        decisions.append({
            "node_id": getattr(node, "node_id", ""),
            "decision": "keep",
            "rough_tokens": cost,
            "source": source,
        })
    return kept, "\n\n".join(blocks), decisions


def retrieve(
    *,
    case: QuestionCase,
    variant: str,
    index: V3Index,
    query_vector: list[float],
    query_vectors: list[list[float]] | None = None,
    token_budget: int = 7200,
) -> RetrievedContext:
    started = time.perf_counter()
    ensure_catalog(index)
    frame = build_query_frame(case.question)
    dense_vectors = query_vectors or [query_vector]
    primary_dense_scores: dict[str, float] = {}

    def query_similarity(node: Any) -> float:
        # Auxiliary views are routing-only; the original question remains the
        # sole semantic authority for ranking and deterministic operators.
        node_id = getattr(node, "node_id", getattr(node, "edge_id", ""))
        if node_id in primary_dense_scores:
            return primary_dense_scores[node_id]
        return cosine_similarity(query_vector, _node_vector(node))

    def object_query_similarity(node: Any) -> float:
        return cosine_similarity(
            query_vector, getattr(node, "object_embedding", None)
        )
    planned_views = planned_query_views(frame)
    slot_phrase = answer_slot_phrase(frame.raw_question)
    slot_vector = None
    if slot_phrase and slot_phrase in planned_views:
        slot_index = planned_views.index(slot_phrase)
        if slot_index < len(dense_vectors):
            slot_vector = dense_vectors[slot_index]

    def temporal_semantic_similarity(node: Any) -> float:
        primary = query_similarity(node)
        if slot_vector is None:
            return primary
        return 0.55 * primary + 0.45 * cosine_similarity(
            slot_vector, _node_vector(node)
        )

    def answer_slot_similarity(node: Any) -> float:
        primary = query_similarity(node)
        if slot_vector is None:
            return primary
        return 0.40 * primary + 0.60 * cosine_similarity(
            slot_vector, _node_vector(node)
        )

    def target_semantic_similarity(node: Any) -> float:
        if node is None:
            return 0.0
        vector = slot_vector or query_vector
        return max(
            cosine_similarity(vector, _node_vector(node)),
            cosine_similarity(vector, getattr(node, "object_embedding", None)),
        )

    retrieval_claims = [
        item for item in index.claims
        if not (item.predicate_key == "said" and item.confidence <= 0.4)
    ]
    node_list: list[Any] = [
        *index.turns, *retrieval_claims, *index.events, *index.event_entities,
        *index.episodes, *index.themes,
        *index.event_frames, *index.operands,
    ]
    node_by_id = {node.node_id: node for node in node_list}
    edge_by_id = {edge.edge_id: edge for edge in index.hyperedges}
    relation_scores = {
        relation: _relation_prior(frame, relation)
        for relation in sorted({edge.relation for edge in index.hyperedges})
    }
    relation_total = sum(relation_scores.values()) or 1.0
    relation_posteriors = {
        relation: round(score / relation_total, 6)
        for relation, score in relation_scores.items()
    }
    candidates: list[Any] = [*node_list, *index.hyperedges]
    text_rows = [
        (getattr(node, "node_id", getattr(node, "edge_id", "")), _node_text(node))
        for node in candidates
    ]
    dense_rankings, primary_dense_scores = dense_rank_many(dense_vectors, candidates)
    channels = {"dense": dense_rankings[0] if dense_rankings else []}
    for view_index, ranking in enumerate(dense_rankings[1:], start=1):
        channels[f"dense_view_{view_index}"] = ranking
    channels["bm25"] = _bm25_rank(
        frame.content_terms + frame.participant_terms + frame.temporal_terms,
        text_rows,
    )
    channels["exact"] = _exact_rank(frame, candidates)
    primary_channels = {
        name: channels[name] for name in ("dense", "bm25", "exact")
    }
    rrf_scores = _rrf(list(primary_channels.values()))
    scope_posteriors = _scope_posteriors(
        frame, node_by_id, primary_channels, rrf_scores
    )
    scope_by_session = {
        row["session_id"]: row["posterior"] for row in scope_posteriors
    }
    turns_by_session: dict[str, list[TurnNode]] = defaultdict(list)
    for turn in index.turns:
        turns_by_session[turn.session_id].append(turn)
    turn_session_by_id = {turn.node_id: turn.session_id for turn in index.turns}
    relation_focus_ids = relation_focus_turn_ids(frame, index.turns)
    relation_focus_sessions = list(dict.fromkeys(
        turn_session_by_id[node_id]
        for node_id in relation_focus_ids if node_id in turn_session_by_id
    ))
    # Fine-grained operand/event-frame hits must feed back into coarse routing
    # before local turn retrieval and graph expansion. This closes the loop
    # between coarse-to-fine and fine-to-coarse retrieval.
    catalog_route_scores: defaultdict[str, float] = defaultdict(float)
    for ranking in channels.values():
        catalog_rank = 0
        for node_id in ranking[:80]:
            node = node_by_id.get(node_id)
            if node is None or _node_type(node) not in {"operand", "event_frame"}:
                continue
            catalog_rank += 1
            contribution = 1.0 / (20.0 + catalog_rank)
            for source_id in getattr(node, "source_turn_ids", []):
                session_id = turn_session_by_id.get(source_id)
                if session_id:
                    catalog_route_scores[session_id] += contribution
    catalog_route_limit = (
        8 if frame.requested_operation in {"count", "list", "ordering", "location", "recurrence"}
        else 5
    )
    catalog_route_sessions = [
        session_id for session_id, _score in sorted(
            catalog_route_scores.items(), key=lambda item: (item[1], item[0]), reverse=True,
        )[:catalog_route_limit]
    ]
    local_scope_rows = list(scope_posteriors[:10])
    local_scope_ids = {str(row["session_id"]) for row in local_scope_rows}
    for session_id in list(dict.fromkeys([*relation_focus_sessions, *catalog_route_sessions])):
        if session_id not in local_scope_ids:
            local_scope_rows.append({"session_id": session_id})
            local_scope_ids.add(session_id)
    recommendation_scope_ids = recommendation_scope_session_ids(
        frame, local_scope_rows,
    )
    recommendation_resource_ids = recommendation_resource_turn_ids(
        frame,
        index.turns,
        recommendation_scope_ids,
        semantic_similarity=query_similarity,
    )
    scope_local_turn_ids: list[str] = []
    scope_local_turn_sources: dict[str, str] = {}
    local_query_terms = frame.content_terms + frame.participant_terms + frame.temporal_terms
    for scope_rank, scope_row in enumerate(local_scope_rows):
        local_turns = turns_by_session.get(scope_row["session_id"], [])
        local_rows = [(turn.node_id, _node_text(turn)) for turn in local_turns]
        local_channels = [
            (_dense_rank(query_vector, local_turns), 4),
            (_bm25_rank(local_query_terms, local_rows), 1),
            (_exact_rank(frame, local_turns), 1),
        ]
        local_by_id = {turn.node_id: turn for turn in local_turns}
        local_selected: list[str] = []
        for ranking, limit in local_channels:
            for rank, node_id in enumerate(ranking[:limit]):
                if node_id not in local_selected:
                    local_selected.append(node_id)
                source = (
                    "scope_local_turn_primary" if rank == 0
                    else "scope_local_turn_secondary"
                )
                if (
                    source == "scope_local_turn_primary"
                    or node_id not in scope_local_turn_sources
                ):
                    scope_local_turn_sources[node_id] = source
        # Dialogue meaning frequently spans an adjacency pair: a question or
        # reference in one turn and its value in the immediately following
        # turn.  Close only around the strongest local anchors in the top
        # scopes, preserving coarse-to-fine locality and a bounded pack.
        if (scope_rank < 4 or scope_row["session_id"] in set(catalog_route_sessions[:4])) and local_selected:
            ordered_local = sorted(local_turns, key=lambda item: item.turn_index)
            position_by_id = {
                item.node_id: position for position, item in enumerate(ordered_local)
            }
            for anchor_id in local_selected[:2]:
                anchor_position = position_by_id.get(anchor_id)
                if anchor_position is None:
                    continue
                for neighbor_position in (anchor_position - 1, anchor_position + 1):
                    if not (0 <= neighbor_position < len(ordered_local)):
                        continue
                    neighbor_id = ordered_local[neighbor_position].node_id
                    if neighbor_id not in local_selected:
                        local_selected.append(neighbor_id)
                        scope_local_turn_sources[neighbor_id] = (
                            "scope_local_turn_adjacent"
                        )
        scope_local_turn_ids.extend(
            node_id for node_id in local_selected if node_id in local_by_id
        )
    for node_id in relation_focus_ids:
        if node_id not in scope_local_turn_ids:
            scope_local_turn_ids.append(node_id)
        scope_local_turn_sources[node_id] = "relation_focus"
    max_scope = max(scope_by_session.values(), default=1.0)
    scope_claim_active = max(
        (row.get("claim_joint_match", 0.0) for row in scope_posteriors),
        default=0.0,
    ) >= 0.25
    scope_lexical_active = bool(
        scope_posteriors
        and scope_posteriors[0].get("query_coverage", 0.0) >= 0.80
        and (len(scope_posteriors) == 1
             or scope_posteriors[0]["posterior"] - scope_posteriors[1]["posterior"] >= 0.08)
        and scope_posteriors[0].get("channel_hits", 0) >= 2
    )
    scope_active = scope_claim_active or scope_lexical_active
    scoped_rrf_scores: dict[str, float] = {}
    for node_id, score in rrf_scores.items():
        node = node_by_id.get(node_id)
        session_id = _node_session_id(node) if node is not None else None
        if session_id and scope_active:
            posterior = scope_by_session.get(session_id, 0.0)
            if frame.requested_operation in {"count", "list", "location", "preference_list", "recurrence"}:
                multiplier = 0.86 + 0.44 * posterior / max_scope
            else:
                multiplier = 0.72 + 0.78 * posterior / max_scope
        else:
            multiplier = 1.0
        scoped_rrf_scores[node_id] = score * multiplier
    quota = {"turn": 10, "claim": 12, "event": 8, "event_entity": 4, "episode": 8, "theme": 5, "event_frame": 6, "operand": 12, "hyperedge": 8}
    seeds: list[str] = []
    for node_id, _score in sorted(scoped_rrf_scores.items(), key=lambda item: item[1], reverse=True):
        node = node_by_id.get(node_id) or edge_by_id.get(node_id)
        if node is None:
            continue
        kind = _node_type(node)
        if quota.get(kind, 0) <= 0:
            continue
        quota[kind] -= 1
        seeds.append(node_id)

    relation_channel_top_ids = {
        node_id
        for ranking in primary_channels.values()
        for node_id in ranking[:120]
    }
    reference_relation_seeds = sorted(
        (
            edge.edge_id for edge in index.hyperedges
            if edge.relation == "refers_to"
            and edge.edge_id in relation_channel_top_ids
        ),
        key=lambda node_id: scoped_rrf_scores.get(node_id, 0.0),
        reverse=True,
    )[:2]
    event_identity_relation_seeds = sorted(
        (
            edge.edge_id for edge in index.hyperedges
            if edge.relation == "same_event"
            and edge.provenance.get("generator")
            == "bounded_event_identity_consolidation"
            and edge.edge_id in relation_channel_top_ids
        ),
        key=lambda node_id: scoped_rrf_scores.get(node_id, 0.0),
        reverse=True,
    )[:2]
    event_entity_relation_seeds = sorted(
        (
            edge.edge_id for edge in index.hyperedges
            if edge.relation == "event_entity_member"
            and edge.provenance.get("generator")
            == "bounded_event_entity_consolidation"
            and (
                edge.edge_id in relation_channel_top_ids
                or edge.provenance.get("event_entity_id") in relation_channel_top_ids
            )
        ),
        key=lambda node_id: scoped_rrf_scores.get(node_id, 0.0),
        reverse=True,
    )[:2]
    relation_route_seed_ids = list(dict.fromkeys([
        *reference_relation_seeds, *event_identity_relation_seeds,
        *event_entity_relation_seeds,
    ]))
    relation_seed_score = max(scoped_rrf_scores.values(), default=1.0) * 1.05
    for node_id in relation_route_seed_ids:
        scoped_rrf_scores[node_id] = max(
            scoped_rrf_scores.get(node_id, 0.0), relation_seed_score
        )
    seeds.extend(
        node_id for node_id in relation_route_seed_ids if node_id not in set(seeds)
    )
    seed_id_set = set(seeds)
    seeds.extend(
        node_id for node_id in scope_local_turn_ids if node_id not in seed_id_set
    )
    expanded_scores, expansion_trace, visited_edges = _expand(
        frame, node_by_id, edge_by_id, scoped_rrf_scores, seeds, max_nodes=160
    )
    reference_rows: list[tuple[float, dict[str, Any]]] = []
    for edge_id in visited_edges:
        edge = edge_by_id.get(edge_id)
        if edge is None or edge.relation != "refers_to":
            continue
        antecedent = next(
            (item.node_id for item in edge.incidences if item.role == "antecedent"), ""
        )
        anaphor = next(
            (item.node_id for item in edge.incidences if item.role == "anaphor"), ""
        )
        if antecedent not in node_by_id or anaphor not in node_by_id:
            continue
        score = (
            0.45 * max(expanded_scores.get(antecedent, 0.0), expanded_scores.get(anaphor, 0.0))
            + 0.35 * _query_overlap(frame, edge.retrieval_text)
            + 0.20 * edge.confidence
        )
        reference_rows.append((score, {
            "edge_id": edge_id,
            "antecedent_node_id": antecedent,
            "anaphor_node_id": anaphor,
            "antecedent_text": _node_text(node_by_id[antecedent]),
            "anaphor_text": _node_text(node_by_id[anaphor]),
            "confidence": edge.confidence,
            "resolved_value": edge.provenance.get("resolved_value", ""),
        }))
    reference_rows.sort(key=lambda row: (row[0], row[1]["edge_id"]), reverse=True)
    reference_chain_hints = [row for _score, row in reference_rows[:4]]
    reference_chain_source_ids = {
        value
        for row in reference_chain_hints
        for value in (row["antecedent_node_id"], row["anaphor_node_id"])
    }

    event_identity_rows: list[tuple[float, dict[str, Any]]] = []
    for edge_id in visited_edges:
        edge = edge_by_id.get(edge_id)
        if (
            edge is None
            or edge.relation != "same_event"
            or edge.provenance.get("generator")
            != "bounded_event_identity_consolidation"
        ):
            continue
        earlier_id = next((
            item.node_id for item in edge.incidences
            if item.role.startswith("earlier_")
        ), "")
        later_id = next((
            item.node_id for item in edge.incidences
            if item.role.startswith("later_")
        ), "")
        earlier = node_by_id.get(earlier_id)
        later = node_by_id.get(later_id)
        if not isinstance(earlier, EventNode) or not isinstance(later, EventNode):
            continue
        score = (
            0.45 * max(
                expanded_scores.get(earlier_id, 0.0),
                expanded_scores.get(later_id, 0.0),
            )
            + 0.35 * _query_overlap(frame, edge.retrieval_text)
            + 0.20 * edge.confidence
        )
        event_identity_rows.append((score, {
            "edge_id": edge_id,
            "earlier_event_id": earlier_id,
            "later_event_id": later_id,
            "earlier_status": earlier.status,
            "later_status": later.status,
            "earlier_time": earlier.event_time,
            "later_time": later.event_time,
            "earlier_text": earlier.label,
            "later_text": later.label,
            "source_turn_ids": list(dict.fromkeys([
                *earlier.source_turn_ids, *later.source_turn_ids
            ])),
            "confidence": edge.confidence,
            "identity_basis": edge.provenance.get("identity_basis", ""),
        }))
    event_identity_rows.sort(
        key=lambda row: (row[0], row[1]["edge_id"]), reverse=True
    )
    event_identity_chain_hints = [row for _score, row in event_identity_rows[:2]]
    event_identity_source_ids = {
        value
        for row in event_identity_chain_hints
        for value in (
            row["earlier_event_id"], row["later_event_id"],
            *row["source_turn_ids"],
        )
    }
    event_entity_rows: list[tuple[float, dict[str, Any]]] = []
    for edge_id in visited_edges:
        edge = edge_by_id.get(edge_id)
        if (
            edge is None
            or edge.relation != "event_entity_member"
            or edge.provenance.get("generator")
            != "bounded_event_entity_consolidation"
        ):
            continue
        entity_id = next((
            incidence.node_id for incidence in edge.incidences
            if incidence.role == "identity"
        ), "")
        entity = node_by_id.get(entity_id)
        if not isinstance(entity, EventEntityNode):
            continue
        members = [
            node_by_id.get(value) for value in entity.member_event_ids
            if isinstance(node_by_id.get(value), EventNode)
        ]
        score = (
            0.45 * max(
                [expanded_scores.get(entity_id, 0.0)]
                + [expanded_scores.get(value, 0.0)
                   for value in entity.member_event_ids]
            )
            + 0.35 * _query_overlap(frame, entity.retrieval_text)
            + 0.20 * entity.confidence
        )
        member_rows = []
        for event in members:
            source_turn = next((
                node_by_id.get(value) for value in event.source_turn_ids
                if isinstance(node_by_id.get(value), TurnNode)
            ), None)
            observed_at = source_turn.session_date if source_turn else None
            resolved, resolution_basis = resolve_evidence_time(
                event.event_time, observed_at
            )
            resolved_value = None
            if resolved is not None and resolution_basis != "observed_fallback":
                raw_time = str(event.event_time or "").casefold()
                if "month" in raw_time:
                    resolved_value = resolved.strftime("%B %Y")
                elif "year" in raw_time:
                    resolved_value = resolved.strftime("%Y")
                else:
                    resolved_value = resolved.date().isoformat()
            member_rows.append({
                "event_id": event.node_id,
                "label": event.label,
                "status": event.status,
                "event_time": event.event_time,
                "observed_at": observed_at,
                "resolved_time": resolved_value,
                "resolution_basis": resolution_basis,
                "source_turn_ids": event.source_turn_ids,
            })
        current_row = next((
            row for row in member_rows
            if row["event_id"] == entity.current_event_id
        ), member_rows[-1] if member_rows else {})
        event_entity_rows.append((score, {
            "edge_id": edge_id,
            "event_entity_id": entity_id,
            "canonical_label": entity.canonical_label,
            "identity_anchors": entity.anchor_terms,
            "member_event_ids": entity.member_event_ids,
            "lifecycle_status": entity.lifecycle_status,
            "current_event_id": entity.current_event_id,
            "time_start": entity.time_start,
            "time_end": entity.time_end,
            "current_resolved_time": current_row.get("resolved_time"),
            "current_resolution_basis": current_row.get("resolution_basis"),
            "members": member_rows,
            "source_turn_ids": entity.source_turn_ids,
            "confidence": entity.confidence,
        }))
    event_entity_rows.sort(
        key=lambda row: (row[0], row[1]["edge_id"]), reverse=True
    )
    event_entity_hints = [row for _score, row in event_entity_rows[:2]]
    event_entity_source_ids = {
        value for row in event_entity_hints
        for value in (
            row["event_entity_id"], *row["member_event_ids"],
            *row["source_turn_ids"],
        )
    }
    # A coarse node reached at the graph-depth boundary must still be allowed
    # to reveal a bounded amount of its own evidence. This is a provenance
    # projection rather than an extra semantic graph hop.
    coarse_fine_projection_ids, coarse_fine_projection_trace = (
        project_reached_episodes(
            nodes=node_by_id,
            expanded_scores=expanded_scores,
            primary_similarity=query_similarity,
            slot_similarity=answer_slot_similarity,
            query_overlap=lambda node: _query_overlap(frame, _node_text(node)),
        )
    )
    scope_local_seed_set = set(scope_local_turn_ids)
    projections_by_seed: dict[str, set[str]] = defaultdict(set)
    for edge in index.hyperedges:
        if edge.relation not in {"operand_projection", "event_frame_member"}:
            continue
        edge_seeds = [
            incidence.node_id for incidence in edge.incidences
            if incidence.node_id in scope_local_seed_set
        ]
        if not edge_seeds:
            continue
        projected = [
            incidence.node_id for incidence in edge.incidences
            if _node_type(node_by_id.get(incidence.node_id)) in {"operand", "event_frame"}
        ]
        for seed_id in edge_seeds:
            projections_by_seed[seed_id].update(projected)
    projection_rows_by_session: dict[str, list[tuple[float, str, str, str]]] = defaultdict(list)
    fallback_rows: list[tuple[float, str]] = []
    for seed_id in scope_local_turn_ids:
        projected_ids = projections_by_seed.get(seed_id, set())
        raw_overlap = _query_overlap(frame, _node_text(node_by_id[seed_id]))
        projected_overlap = max((
            _query_overlap(frame, _node_text(node_by_id[node_id]))
            for node_id in projected_ids
        ), default=0.0)
        session_id = _node_session_id(node_by_id[seed_id]) or ""
        if not projected_ids or raw_overlap > projected_overlap + 0.05:
            fallback_rows.append((
                raw_overlap - projected_overlap
                + 0.20 * scope_by_session.get(session_id, 0.0),
                seed_id,
            ))
        inherited = scope_local_turn_sources[seed_id]
        source = (
            "scope_local_projection_primary"
            if inherited == "scope_local_turn_primary"
            else "scope_local_projection_secondary"
        )
        for node_id in sorted(projected_ids):
            score = (
                0.65 * max(0.0, query_similarity(node_by_id[node_id]))
                + 0.35 * _query_overlap(frame, _node_text(node_by_id[node_id]))
            )
            projection_rows_by_session[session_id].append((score, node_id, source, seed_id))
    scope_local_projection_ids: set[str] = set()
    scope_local_projection_sources: dict[str, str] = {}
    for scope_rank, scope_row in enumerate(scope_posteriors[:10]):
        rows = projection_rows_by_session.get(scope_row["session_id"], [])
        best_by_seed: dict[str, tuple[float, str, str]] = {}
        for score, node_id, source, seed_id in rows:
            old = best_by_seed.get(seed_id)
            if old is None or (score, node_id, source) > old:
                best_by_seed[seed_id] = (score, node_id, source)
        seed_rows = sorted(best_by_seed.values(), reverse=True)
        if scope_rank < 4:
            primary_rows = [
                row for row in seed_rows if row[2] == "scope_local_projection_primary"
            ]
            secondary_rows = [
                row for row in seed_rows if row[2] == "scope_local_projection_secondary"
            ]
            ranked = [*primary_rows, *secondary_rows[:1]]
        else:
            ranked = seed_rows[:1]
        seen_projection_ids: set[str] = set()
        for _score, node_id, source in ranked:
            if node_id in seen_projection_ids:
                continue
            seen_projection_ids.add(node_id)
            scope_local_projection_ids.add(node_id)
            scope_local_projection_sources[node_id] = source
    for _score, seed_id in sorted(fallback_rows, reverse=True)[:4]:
        scope_local_turn_sources[seed_id] = "scope_local_turn_unprojected"
    seed_set = set(seeds)
    graph_rescues = {
        node_id for node_id in expanded_scores
        if node_id not in seed_set and node_id in node_by_id
        and _query_overlap(frame, _node_text(node_by_id[node_id])) > 0
    }

    combined_scores = {
        node_id: scoped_rrf_scores.get(node_id, 0.0) + expanded_scores.get(node_id, 0.0)
        for node_id in node_by_id
    }
    turn_ids = {item.node_id for item in index.turns}
    turn_channel_lists = {
        name: [node_id for node_id in ranking if node_id in turn_ids][:40]
        for name, ranking in channels.items()
    }
    turn_channel_sets = {name: set(values) for name, values in turn_channel_lists.items()}
    direct_candidates = set().union(*turn_channel_sets.values())
    eligible_direct = {
        node_id for node_id in direct_candidates
        if (
            _query_overlap(frame, _node_text(node_by_id[node_id])) >= 0.25
            or sum(node_id in values for values in turn_channel_sets.values()) >= 2
        )
    }
    uncovered_terms = set(frame.content_terms)
    direct_ordered: list[str] = []
    direct_sessions: Counter[str] = Counter()
    direct_limit = (
        18 if frame.requested_operation in {"count", "list", "location", "preference_list", "ordering", "recurrence"}
        else 12 if frame.requested_operation in {"latest", "earliest", "duration"}
        else 8
    )
    while eligible_direct and len(direct_ordered) < direct_limit:
        node_id = max(
            eligible_direct,
            key=lambda value: (
                len(uncovered_terms & set(_tokens(_node_text(node_by_id[value])))),
                -direct_sessions[_node_session_id(node_by_id[value]) or ""],
                sum(value in values for values in turn_channel_sets.values()),
                _query_overlap(frame, _node_text(node_by_id[value])),
                combined_scores.get(value, 0.0),
                value,
            ),
        )
        direct_ordered.append(node_id)
        direct_sessions[_node_session_id(node_by_id[node_id]) or ""] += 1
        uncovered_terms -= set(_tokens(_node_text(node_by_id[node_id])))
        eligible_direct.remove(node_id)
    direct_protected = set(direct_ordered)
    rescue_protected = set(sorted(
        graph_rescues, key=lambda node_id: combined_scores.get(node_id, 0.0), reverse=True
    )[:6])
    catalog_limit = (
        24 if frame.requested_operation in {"count", "list", "location", "preference_list", "ordering", "recurrence"}
        else 16 if frame.requested_operation in {"latest", "earliest", "duration", "state"}
        else 10
    )
    top_scope_sessions = (
        {row["session_id"] for row in scope_posteriors[:10]}
        | set(catalog_route_sessions)
    )
    total_scope_sessions, total_scope_operand_ids = total_scope_candidates(
        frame, index.operands, index.turns, tokenize=_tokens,
        similarity=query_similarity,
    )
    lossless_event_candidate_ids = lossless_event_turn_candidates(
        frame, index.turns, tokenize=_tokens, similarity=query_similarity,
    )
    top_scope_sessions |= set(total_scope_sessions) | {
        turn_session_by_id[node_id] for node_id in lossless_event_candidate_ids
        if node_id in turn_session_by_id
    }
    local_evidence_turn_ids = (
        direct_protected | set(scope_local_turn_ids) | set(lossless_event_candidate_ids)
    )

    def in_local_catalog_scope(node: Any) -> bool:
        source_turn_ids = set(getattr(node, "source_turn_ids", []))
        return bool(
            getattr(node, "node_id", "") in expanded_scores
            or source_turn_ids & local_evidence_turn_ids
            or any(
                turn_session_by_id.get(source_id) in top_scope_sessions
                for source_id in source_turn_ids
            )
        )

    local_operands = [
        item for item in index.operands if in_local_catalog_scope(item)
    ]
    local_event_frames = [
        item for item in index.event_frames if in_local_catalog_scope(item)
    ]
    catalog_node_ids = [
        node_id for node_id, node in node_by_id.items()
        if _node_type(node) in {"event_frame", "operand"}
        and in_local_catalog_scope(node)
    ]
    catalog_protected = set(sorted(
        catalog_node_ids,
        key=lambda node_id: combined_scores.get(node_id, 0.0), reverse=True,
    )[:catalog_limit])
    # Independent dense beam within the routed scope prevents lexical/RRF
    # dominance without turning the operator into a global benchmark scan.
    catalog_dense_protected = set(sorted(
        catalog_node_ids,
        key=lambda node_id: query_similarity(node_by_id[node_id]),
        reverse=True,
    )[:min(8, catalog_limit)])
    local_turns_for_operator = [
        turn for turn in index.turns
        if turn.session_id in top_scope_sessions
        or turn.node_id in local_evidence_turn_ids
    ]
    prepack_relative_time_hint = relative_time_hint(
        frame,
        [
            *[("operand", item, query_similarity(item), "date_index") for item in index.operands],
            *[("event_frame", item, query_similarity(item), "date_index") for item in index.event_frames],
            *[("turn", item, query_similarity(item), "date_index") for item in index.turns],
        ],
        case.question_date, tokenize=_tokens, node_text=_node_text,
        evidence_time=_evidence_time,
        semantic_similarity=temporal_semantic_similarity,
    )
    prepack_location_hint = location_at_time_hint(frame, index.operands)
    prepack_event_lifecycle_hint = event_lifecycle_duration_hint(
        frame, index.turns
    )
    prepack_contrast_alternative_hint = contrast_alternative_hint(
        frame, index.turns
    )
    prepack_weekday_hint = weekday_scope_hint(
        frame,
        [
            *[("operand", item, query_similarity(item), "local_catalog") for item in local_operands],
            *[("event_frame", item, query_similarity(item), "local_catalog") for item in local_event_frames],
            *[("turn", item, query_similarity(item), "local_scope") for item in local_turns_for_operator],
        ],
        case.question_date,
        tokenize=_tokens,
        node_text=_node_text,
        evidence_time=_evidence_time,
    )
    prepack_before_after_hint = before_after_relation_hint(
        frame,
        [
            *[("event_frame", item, query_similarity(item), "local_catalog") for item in local_event_frames],
            *[("turn", item, query_similarity(item), "local_scope") for item in local_turns_for_operator],
        ],
        query_overlap=_query_overlap,
    )
    # Ordinal questions require a complete event chain. Restricting the
    # operator to initially routed sessions can shift every later ordinal.
    # Typed subject/relation/event filtering keeps this closure deterministic.
    global_catalog_hint = catalog_operator_hint(
        frame, index.operands,
        event_frames=index.event_frames,
        query_overlap=_query_overlap,
        semantic_similarity=lambda item: query_similarity(item),
        object_semantic_similarity=lambda item: object_query_similarity(item),
        target_semantic_similarity=target_semantic_similarity,
        turns=index.turns,
    )
    if (
        not isinstance(global_catalog_hint, dict)
        or global_catalog_hint.get("operation") not in {
            "aggregate_revenue_total", "dimensional_quantity_total",
            "money_amount", "money_difference", "total_money",
            "frequency_state_comparison", "scalar_attribute_state_ambiguous",
            "combined_named_duration",
            "combined_named_duration_incomplete", "named_alternative_incomplete",
            "earliest_named_alternative",
            "ordered_event_collection", "relation_slot_location",
            "before_after_operand_relation",
            "distinct_action_entity_collection",
            "named_scalar_average",
            "dialogue_answer_span", "dialogue_followup_plan",
            "event_onset_from_lossless_evidence", "scalar_comparison",
            "arrival_clock_time", "arrival_clock_ambiguous",
            "weekly_recurrence_count",
        }
    ):
        global_catalog_hint = None
    prepack_catalog_hint = movement_location_collection_hint(
        frame, index.operands, index.event_frames,
    ) or planned_event_identity_count(
        frame, index.event_frames,
    ) or all_subjects_relation_hint(
        frame, index.operands,
    ) or ordinal_event_hint(
        frame, index.operands, index.turns,
    ) or ordinal_list_hint(
        frame, local_turns_for_operator, query_overlap=_query_overlap,
    ) or global_catalog_hint or catalog_operator_hint(
        frame,
        local_operands,
        event_frames=local_event_frames,
        query_overlap=_query_overlap,
        semantic_similarity=lambda item: query_similarity(item),
        object_semantic_similarity=lambda item: object_query_similarity(item),
        target_semantic_similarity=target_semantic_similarity,
        turns=local_turns_for_operator,
    )
    if (
        isinstance(prepack_catalog_hint, dict)
        and prepack_catalog_hint.get("operation") == "event_occurrence_count"
        and lossless_event_candidate_ids
        and len(lossless_event_candidate_ids) < 12
    ):
        represented = set(prepack_catalog_hint.get("source_turn_ids", [])) | set(
            prepack_catalog_hint.get("missing_source_turn_ids", [])
        )
        if represented == set(lossless_event_candidate_ids):
            prepack_catalog_hint = dict(prepack_catalog_hint)
            prepack_catalog_hint.update({
                "value": len(lossless_event_candidate_ids),
                "candidate_value": len(lossless_event_candidate_ids),
                "complete": True,
                "completion_basis": "bounded_lossless_event_turn_index",
                "source_turn_ids": list(lossless_event_candidate_ids),
                "missing_source_turn_ids": [],
            })
    catalog_operator_sources = set(
        prepack_catalog_hint.get("source_turn_ids", [])
        if isinstance(prepack_catalog_hint, dict) else []
    ) & turn_ids
    temporal_operator_ids = set(
        prepack_weekday_hint.get("supporting_node_ids", [])
        if isinstance(prepack_weekday_hint, dict) else []
    ) | set(
        prepack_relative_time_hint.get("candidate_node_ids", [])
        if isinstance(prepack_relative_time_hint, dict) else []
    )
    temporal_operator_sources = {
        source_id
        for node_id in temporal_operator_ids
        for source_id in getattr(node_by_id.get(node_id), "source_turn_ids", [])
        if source_id in turn_ids
    }
    location_operator_ids = set(
        prepack_location_hint.get("operand_ids", [])
        if isinstance(prepack_location_hint, dict) else []
    )
    location_operator_sources = set(
        prepack_location_hint.get("source_turn_ids", [])
        if isinstance(prepack_location_hint, dict) else []
    )
    lifecycle_operator_sources = set(
        prepack_event_lifecycle_hint.get("source_turn_ids", [])
        if isinstance(prepack_event_lifecycle_hint, dict) else []
    )
    contrast_operator_sources = set(
        prepack_contrast_alternative_hint.get("source_turn_ids", [])
        if isinstance(prepack_contrast_alternative_hint, dict) else []
    )
    relation_operator_ids = set()
    relation_operator_sources = set()
    if isinstance(prepack_before_after_hint, dict):
        for key in ("anchor_event", "nearest_qualifying_event"):
            row = prepack_before_after_hint.get(key) or {}
            node_id = str(row.get("node_id") or "")
            if node_id:
                relation_operator_ids.add(node_id)
        relation_operator_sources.update(prepack_before_after_hint.get("source_turn_ids", []))
    protected = (
        direct_protected | rescue_protected | catalog_protected | catalog_dense_protected
        | set(scope_local_turn_ids) | scope_local_projection_ids
        | set(coarse_fine_projection_ids)
        | catalog_operator_sources | temporal_operator_ids | temporal_operator_sources
        | location_operator_ids | location_operator_sources
        | lifecycle_operator_sources
        | contrast_operator_sources
        | relation_operator_ids | relation_operator_sources
        | reference_chain_source_ids
        | event_identity_source_ids
        | event_entity_source_ids
        | set(recommendation_resource_ids)
        | set(total_scope_operand_ids) | set(lossless_event_candidate_ids)
    )
    type_caps = {"theme": 5, "episode": 8, "event_entity": 2, "event_frame": 8, "operand": 24, "claim": 18, "event": 10, "turn": 28}
    selected: list[tuple[str, Any, float, str]] = []
    selected_ids: set[str] = set()

    def add(node_id: str, source: str) -> None:
        if node_id in selected_ids or node_id not in node_by_id:
            return
        node = node_by_id[node_id]
        kind = _node_type(node)
        if type_caps.get(kind, 0) <= 0 and node_id not in protected:
            return
        type_caps[kind] = type_caps.get(kind, 0) - 1
        selected_ids.add(node_id)
        selected.append((kind, node, combined_scores.get(node_id, 0.0), source))

    for node_id in sorted(direct_protected, key=lambda value: combined_scores.get(value, 0.0), reverse=True):
        add(node_id, "protected_direct")
    for node_id in sorted(relation_operator_ids | relation_operator_sources):
        if node_id in selected_ids:
            for position, item in enumerate(selected):
                if getattr(item[1], "node_id", "") == node_id:
                    selected[position] = (
                        item[0], item[1], item[2], "relation_operator_provenance"
                    )
                    break
        else:
            add(node_id, "relation_operator_provenance")
    for node_id in sorted(reference_chain_source_ids):
        add(node_id, "reference_chain_provenance")
    for node_id in sorted(event_identity_source_ids):
        add(node_id, "event_identity_provenance")
    for node_id in sorted(event_entity_source_ids):
        add(node_id, "event_entity_provenance")
    for node_id in sorted(catalog_protected, key=lambda value: combined_scores.get(value, 0.0), reverse=True):
        add(node_id, "protected_catalog")
    for node_id in sorted(catalog_dense_protected, key=lambda value: query_similarity(node_by_id[value]), reverse=True):
        add(node_id, "protected_catalog_dense")
    for node_id in sorted(catalog_operator_sources):
        add(node_id, "catalog_operator_provenance")
    for node_id in total_scope_operand_ids:
        add(node_id, "scope_total_index")
    for node_id in lossless_event_candidate_ids:
        add(node_id, "scope_lossless_event")
    for node_id in sorted(temporal_operator_ids):
        add(node_id, "temporal_operator_provenance")
    for node_id in sorted(temporal_operator_sources):
        add(node_id, "temporal_operator_provenance")
    for node_id in sorted(location_operator_ids | location_operator_sources):
        add(node_id, "location_operator_provenance")
    for node_id in sorted(lifecycle_operator_sources):
        add(node_id, "lifecycle_operator_provenance")
    for node_id in sorted(contrast_operator_sources):
        add(node_id, "contrast_operator_provenance")
    for node_id in recommendation_resource_ids:
        if node_id in selected_ids:
            for position, item in enumerate(selected):
                if getattr(item[1], "node_id", "") == node_id:
                    selected[position] = (
                        item[0], item[1], item[2],
                        "recommendation_resource_provenance",
                    )
                    break
        else:
            add(node_id, "recommendation_resource_provenance")
    for node_id in scope_local_turn_ids:
        add(node_id, scope_local_turn_sources[node_id])
    for node_id in sorted(scope_local_projection_ids):
        add(node_id, scope_local_projection_sources[node_id])
    for node_id in coarse_fine_projection_ids:
        add(node_id, "coarse_fine_projection")
    for node_id in sorted(rescue_protected, key=lambda value: combined_scores.get(value, 0.0), reverse=True):
        add(node_id, "protected_graph_rescue")
    for node_id, _score in sorted(combined_scores.items(), key=lambda item: item[1], reverse=True):
        add(node_id, "fused_multilevel")
    # A selected coarse/typed node must retain its direct lossless evidence.
    # Promote only a bounded set of the strongest query-bound parents so this
    # remains a fine projection, not an unbounded extra graph expansion.
    provenance_parents = sorted(
        [
            item for item in selected
            if item[0] in {"claim", "event", "event_frame", "operand"}
            and getattr(item[1], "source_turn_ids", [])
        ],
        key=lambda item: (
            _query_overlap(frame, _node_text(item[1])),
            query_similarity(item[1]),
            item[2],
        ),
        reverse=True,
    )[:8]
    focused_source_ids = {
        source_id
        for _kind, node, _score, _source in provenance_parents
        for source_id in getattr(node, "source_turn_ids", [])[:3]
        if source_id in turn_ids
    }
    for source_id in sorted(focused_source_ids):
        protected.add(source_id)
        if source_id not in selected_ids:
            add(source_id, "focused_provenance_expansion")
            continue
        for position, item in enumerate(selected):
            if getattr(item[1], "node_id", "") == source_id:
                if item[3] == "recommendation_resource_provenance":
                    break
                selected[position] = (
                    item[0], item[1], item[2], "focused_provenance_expansion"
                )
                break
    catalog_members = [
        member
        for kind, node, _score, _source in selected
        if kind in {"operand", "event_frame"}
        for member in getattr(node, "source_turn_ids", [])
    ]
    for member in catalog_members:
        add(member, "catalog_provenance")
    claim_sources = [
        source
        for kind, node, _score, _source in selected
        if kind == "claim"
        for source in node.source_turn_ids
    ]
    event_sources = [
        source
        for kind, node, _score, _source in selected
        if kind == "event"
        for source in node.source_turn_ids
    ]
    frame_sources = [
        source
        for kind, node, _score, _source in selected
        if kind in {"event_frame", "operand"}
        for source in node.source_turn_ids
    ]
    for source in claim_sources + event_sources + frame_sources:
        add(source, "provenance_expansion")
    selected.sort(
        key=lambda item: (
            item[3] in {
                "protected_direct", "protected_catalog",
                "catalog_operator_provenance", "protected_graph_rescue",
            },
            item[0] in {"turn", "claim", "event", "event_frame", "operand"},
            item[2],
        ),
        reverse=True,
    )
    kept, context, initial_pack_trace = pack_context(frame, selected, token_budget)
    # Close retained typed nodes over their lossless sources to a bounded fixed
    # point. A single repack is insufficient because a parent can enter only
    # after another source promotion changes the budget ordering.
    final_pack_trace = initial_pack_trace
    provenance_closure_trace: list[dict[str, Any]] = []
    for _closure_pass in range(8):
        packed_typed_parents = sorted(
            [
                item for item in kept
                if item[0] in {"claim", "event", "operand"}
                and getattr(item[1], "source_turn_ids", [])
            ],
            key=lambda item: (
                _query_overlap(frame, _node_text(item[1])),
                query_similarity(item[1]),
                item[2],
            ),
            reverse=True,
        )[:12]
        packed_frame_parents = [
            item for item in kept
            if item[0] == "event_frame"
            and getattr(item[1], "source_turn_ids", [])
        ]
        packed_source_ids = {
            source_id
            for _kind, node, _score, _source in [
                *packed_frame_parents, *packed_typed_parents,
            ]
            for source_id in getattr(node, "source_turn_ids", [])[:3]
            if source_id in turn_ids
        }
        promoted = False
        for source_id in sorted(packed_source_ids):
            protected.add(source_id)
            if source_id not in selected_ids:
                add(source_id, "focused_provenance_expansion")
                promoted = True
                continue
            for position, item in enumerate(selected):
                if getattr(item[1], "node_id", "") != source_id:
                    continue
                if item[3] in {
                    "recommendation_resource_provenance",
                    "focused_provenance_expansion",
                }:
                    break
                selected[position] = (
                    item[0], item[1], item[2], "focused_provenance_expansion"
                )
                promoted = True
                break
        provenance_closure_trace.append({
            "pass": _closure_pass + 1,
            "packed_parent_ids": [
                getattr(item[1], "node_id", "")
                for item in [*packed_frame_parents, *packed_typed_parents]
            ],
            "required_source_ids": sorted(packed_source_ids),
            "promoted": promoted,
        })
        if not promoted:
            break
        kept, context, final_pack_trace = pack_context(
            frame, selected, token_budget
        )
    pack_trace = [
        {**row, "pack_pass": "initial"} for row in initial_pack_trace
    ] + [
        {**row, "pack_pass": "source_complete"} for row in final_pack_trace
    ]
    kept_ids = {node.node_id for _kind, node, _score, _source in kept}
    catalog_hint = prepack_catalog_hint or catalog_operator_hint(
        frame, [node for kind, node, _score, _source in kept if kind == "operand"],
        event_frames=[
            node for kind, node, _score, _source in kept if kind == "event_frame"
        ],
        query_overlap=_query_overlap,
        semantic_similarity=lambda item: query_similarity(item),
        target_semantic_similarity=target_semantic_similarity,
        turns=[node for kind, node, _score, _source in kept if kind == "turn"],
    )
    if isinstance(catalog_hint, dict) and catalog_hint.get("complete"):
        catalog_hint = _annotate_packed_provenance(catalog_hint, kept_ids)
    temporal_ledger = _temporal_evidence_ledger(frame, kept)
    local_binding_hint = missing_possessive_anchor_hint(
        frame, kept, node_text=_node_text, tokenize=_tokens,
    ) or missing_count_target_hint(
        frame, kept, node_text=_node_text, tokenize=_tokens,
    )
    explicit_date_hint = date_scope_hint(
        frame, kept, node_text=_node_text, node_session_id=_node_session_id,
        query_overlap=_query_overlap, compact_text=_compact_relevant_text,
    )
    local_calendar_window_hint = calendar_window_hint(
        frame, kept, query_overlap=_query_overlap,
    )
    local_before_after_hint = prepack_before_after_hint or before_after_relation_hint(
        frame, kept, query_overlap=_query_overlap,
    )
    if (
        isinstance(catalog_hint, dict)
        and catalog_hint.get("operation") == "before_after_operand_relation"
    ):
        local_before_after_hint = catalog_hint
    local_structured_section_hint = structured_section_hint(
        frame, kept, query_overlap=_query_overlap,
    )
    recommendation_constraints = _recommendation_constraints(
        frame, kept, scope_posteriors
    )
    newest_scalar_hint = _newest_scalar_hint(temporal_ledger)
    local_duration_hint = duration_hint(
        frame, kept, tokenize=_tokens, node_text=_node_text,
        evidence_time=_evidence_time, query_overlap=_query_overlap,
        question_date=case.question_date,
    )
    local_latest_state_hint = latest_state_hint(
        frame, kept, tokenize=_tokens, node_text=_node_text,
        evidence_time=_evidence_time,
    )
    local_relative_time_hint = relative_time_hint(
        frame, kept, case.question_date, tokenize=_tokens, node_text=_node_text,
        evidence_time=_evidence_time,
        semantic_similarity=temporal_semantic_similarity,
    )
    if local_relative_time_hint is None:
        local_relative_time_hint = prepack_weekday_hint or weekday_scope_hint(
            frame, kept, case.question_date, tokenize=_tokens, node_text=_node_text,
            evidence_time=_evidence_time,
        )
    relative_entity_binding = resolve_relative_entity(
        frame, local_relative_time_hint, node_by_id, index.operands,
        similarity=query_similarity,
    )
    if (
        relative_entity_binding is not None
        and set(relative_entity_binding.get("source_turn_ids", [])).issubset(kept_ids)
        and (
            not isinstance(local_relative_time_hint, dict)
            or local_relative_time_hint.get("operation") != "relative_time_scope_from_local_evidence"
            or int(local_relative_time_hint.get("matched_query_term_count", 0)) > 0
            or int(local_relative_time_hint.get("action_alignment", 0)) > 0
        )
    ):
        local_relative_time_hint = dict(local_relative_time_hint or {})
        local_relative_time_hint["entity_binding"] = relative_entity_binding
        local_relative_time_hint["resolved_value"] = relative_entity_binding["value"]
        local_relative_time_hint["complete"] = True
    local_relative_age_hint = relative_age_hint(
        frame, kept, case.question_date, tokenize=_tokens, node_text=_node_text,
    )
    planned_event_hint = _planned_event_hint(frame, kept)
    local_media_attribute_hint = media_attribute_hint(
        frame, [node for kind, node, _score, _source in kept if kind == "turn"]
    )
    local_contrast_alternative_hint = (
        prepack_contrast_alternative_hint
        or contrast_alternative_hint(
            frame, [
                node for kind, node, _score, _source in kept
                if kind == "turn"
            ],
        )
    )
    truncated = len(kept) < len(selected)
    certificate, operator = _closure_and_operator(
        frame, index, kept_ids, visited_edges, combined_scores, truncated
    )
    ledger: list[dict[str, Any]] = []
    for kind, node, score, source in kept:
        row = {
            "node_id": node.node_id,
            "node_type": kind,
            "score": round(score, 6),
            "selection_source": source,
            "text": _node_text(node),
        }
        if isinstance(node, (ClaimNode, EventNode, EventFrameV3, OperandRecordV3)):
            row["source_turn_ids"] = list(node.source_turn_ids)
            row["time"] = node.event_time
        if isinstance(node, EventEntityNode):
            row["source_turn_ids"] = list(node.source_turn_ids)
            row["time"] = node.time_end
            row["member_event_ids"] = list(node.member_event_ids)
            row["identity_anchors"] = list(node.anchor_terms)
        if isinstance(node, OperandRecordV3):
            row["source_claim_ids"] = list(node.source_claim_ids)
            row["recurrence_days"] = list(node.recurrence_days)
        if isinstance(node, EventFrameV3):
            row["claim_ids"] = list(node.claim_ids)
            row["event_ids"] = list(node.event_ids)
        ledger.append(row)
    if catalog_hint is not None:
        ledger.append({"catalog_operator_hint": catalog_hint})
    ledger.append({"closure_certificate": asdict(certificate)})
    if operator is not None:
        ledger.append({"operator_result": operator})

    kept_turns = [node for kind, node, _score, _source in kept if kind == "turn"]
    kept_claims = [node for kind, node, _score, _source in kept if kind == "claim"]
    kept_events = [node for kind, node, _score, _source in kept if kind == "event"]
    kept_entities = [node for kind, node, _score, _source in kept if kind == "event_entity"]
    kept_episodes = [node for kind, node, _score, _source in kept if kind == "episode"]
    kept_themes = [node for kind, node, _score, _source in kept if kind == "theme"]
    kept_frames = [node for kind, node, _score, _source in kept if kind == "event_frame"]
    kept_operands = [node for kind, node, _score, _source in kept if kind == "operand"]
    session_ids = list(dict.fromkeys(
        node.session_id
        for node in [*kept_turns, *kept_claims, *kept_events, *kept_episodes]
    ))
    return RetrievedContext(
        question_id=case.question_id,
        variant=variant,
        summary_node_ids=[node.node_id for node in [*kept_themes, *kept_entities, *kept_frames, *kept_episodes]],
        leaf_node_ids=[node.node_id for node in kept_turns],
        edge_count=len(visited_edges),
        context_text=context,
        answer_session_hit=False,
        retrieved_session_ids=session_ids,
        latency_sec=time.perf_counter() - started,
        routing_card_ids=[node.node_id for node in [*kept_themes, *kept_entities, *kept_frames]],
        fact_node_ids=[node.node_id for node in [*kept_claims, *kept_events, *kept_operands]],
        evidence_leaf_ids=[node.node_id for node in kept_turns],
        evidence_ledger=ledger,
        query_kind=frame.requested_operation,
        packed_rough_tokens=rough_token_count(context),
        schema_version="graphmem_v3",
        retrieval_trace={
            "relation_route_seed_ids": relation_route_seed_ids,
            "scope_active": scope_active,
            "retrieval_version": V3_RETRIEVAL_VERSION,
            "query_frame": asdict(frame),
            "scope_posteriors": scope_posteriors[:12],
            "relation_posteriors": relation_posteriors,
            "primary_scope_hint": (
                scope_posteriors[0] if scope_posteriors
                and scope_active
                and frame.requested_operation not in {"count", "list", "location", "preference_list", "recurrence"}
                and scope_posteriors[0]["query_coverage"] >= 0.80
                and (len(scope_posteriors) == 1
                     or scope_posteriors[0]["posterior"] - scope_posteriors[1]["posterior"] >= 0.06)
                else None
            ),
            "channels": {name: values[:50] for name, values in channels.items()},
            "rrf_top": [
                {"node_id": node_id, "score": score}
                for node_id, score in sorted(
                    rrf_scores.items(), key=lambda item: item[1], reverse=True
                )[:80]
            ],
            "seed_ids": seeds,
            "visited_hyperedge_ids": visited_edges,
            "expansion_steps": expansion_trace,
            "catalog_protected_ids": sorted(catalog_protected),
            "catalog_dense_protected_ids": sorted(catalog_dense_protected),
            "catalog_route_sessions": catalog_route_sessions,
            "relation_focus_ids": relation_focus_ids,
            "local_scope_session_ids": [str(row["session_id"]) for row in local_scope_rows],
            "scope_local_turn_ids": scope_local_turn_ids,
            "recommendation_scope_ids": recommendation_scope_ids,
            "recommendation_resource_ids": recommendation_resource_ids,
            "scope_local_projection_ids": sorted(scope_local_projection_ids),
            "coarse_fine_projection_ids": coarse_fine_projection_ids,
            "coarse_fine_projection_trace": coarse_fine_projection_trace,
            "reference_chain_hints": reference_chain_hints,
            "event_identity_chain_hints": event_identity_chain_hints,
            "event_entity_hints": event_entity_hints,
            "graph_rescue_ids": sorted(graph_rescues),
            "graph_rescue_kept_ids": sorted(graph_rescues & kept_ids),
            "pack_decisions": pack_trace,
            "provenance_closure_trace": provenance_closure_trace,
            "closure_certificate": asdict(certificate),
            "operator_result": operator,
            "catalog_operator_hint": catalog_hint,
            "prepack_weekday_hint": prepack_weekday_hint,
            "location_at_time_hint": prepack_location_hint,
            "event_lifecycle_duration_hint": prepack_event_lifecycle_hint,
            "temporal_evidence_ledger": temporal_ledger,
            "binding_hint": local_binding_hint,
            "explicit_date_hint": explicit_date_hint,
            "calendar_window_hint": local_calendar_window_hint,
            "before_after_relation_hint": local_before_after_hint,
            "structured_section_hint": local_structured_section_hint,
            "newest_scalar_hint": newest_scalar_hint,
            "duration_hint": local_duration_hint,
            "latest_state_hint": local_latest_state_hint,
            "relative_time_hint": local_relative_time_hint,
            "relative_age_hint": local_relative_age_hint,
            "planned_event_hint": planned_event_hint,
            "media_attribute_hint": local_media_attribute_hint,
            "contrast_alternative_hint": local_contrast_alternative_hint,
            "recommendation_constraints": recommendation_constraints,
            "selected_node_types": dict(Counter(kind for kind, _node, _score, _source in kept)),
        },
    )


def answer_messages(
    case: QuestionCase, retrieval: RetrievedContext, *, max_prompt_tokens: int = 8300
) -> list[dict[str, str]]:
    closure = retrieval.retrieval_trace.get("closure_certificate") or {}
    operator = retrieval.retrieval_trace.get("operator_result")
    catalog_hint = retrieval.retrieval_trace.get("catalog_operator_hint")
    catalog_answer_hint = _authoritative_catalog_hint(catalog_hint)
    catalog_candidate_hint = (
        catalog_hint
        if (
            isinstance(catalog_hint, dict)
            and catalog_hint.get("complete")
            and catalog_answer_hint is None
        )
        else None
    )
    query_frame = retrieval.retrieval_trace.get("query_frame") or {}
    temporal_ledger = retrieval.retrieval_trace.get("temporal_evidence_ledger") or []
    local_binding_hint = retrieval.retrieval_trace.get("binding_hint")
    explicit_date_hint = retrieval.retrieval_trace.get("explicit_date_hint")
    local_calendar_window_hint = retrieval.retrieval_trace.get(
        "calendar_window_hint"
    )
    local_before_after_hint = retrieval.retrieval_trace.get(
        "before_after_relation_hint"
    )
    local_structured_section_hint = retrieval.retrieval_trace.get("structured_section_hint")
    recommendation_constraints = (
        retrieval.retrieval_trace.get("recommendation_constraints") or []
    )
    newest_scalar_hint = retrieval.retrieval_trace.get("newest_scalar_hint")
    local_duration_hint = retrieval.retrieval_trace.get("duration_hint")
    local_latest_state_hint = retrieval.retrieval_trace.get("latest_state_hint")
    local_relative_time_hint = retrieval.retrieval_trace.get("relative_time_hint")
    local_relative_age_hint = retrieval.retrieval_trace.get("relative_age_hint")
    planned_event_hint = retrieval.retrieval_trace.get("planned_event_hint")
    local_media_attribute_hint = retrieval.retrieval_trace.get("media_attribute_hint")
    local_location_at_time_hint = retrieval.retrieval_trace.get(
        "location_at_time_hint"
    )
    local_event_lifecycle_hint = retrieval.retrieval_trace.get(
        "event_lifecycle_duration_hint"
    )
    local_contrast_alternative_hint = retrieval.retrieval_trace.get(
        "contrast_alternative_hint"
    )
    scope_posteriors = retrieval.retrieval_trace.get("scope_posteriors") or []
    if not retrieval.retrieval_trace.get("scope_active", False):
        scope_posteriors = []
    primary_scope_hint = retrieval.retrieval_trace.get("primary_scope_hint")
    answer_form = query_frame.get("answer_form", "span")
    operation = query_frame.get("requested_operation", "lookup")
    catalog_operation = (catalog_answer_hint or {}).get("operation")
    catalog_trace_operation = (catalog_hint or {}).get("operation") if isinstance(catalog_hint, dict) else None
    if local_before_after_hint and local_before_after_hint.get("complete"):
        answer_contract = (
            "Use before_after_relation_hint as the locally verified temporal binding. "
            "The anchor_event is excluded from the answer. Return the requested semantic slot "
            "from nearest_qualifying_event, grounded in its cited lossless source turns."
        )
    elif local_calendar_window_hint and local_calendar_window_hint.get("complete"):
        answer_contract = (
            "Use only calendar_window_hint.evidence from the computed ordinal weekend window. "
            "Return the requested semantic slot from those dated turns; do not substitute a "
            "similar event from another weekend in the same month."
        )
    elif local_media_attribute_hint and local_media_attribute_hint.get("complete"):
        answer_contract = (
            "Answer the requested visual attribute concisely from media_attribute_hint.value "
            "and media_attribute_hint.evidence. The artifact type, date, occurrence, caption, "
            "and bounded same-speaker follow-up are jointly bound; preserve exact descriptive "
            "details and do not substitute another media item from the same date."
        )
    elif (
        local_location_at_time_hint
        and local_location_at_time_hint.get("complete")
    ):
        answer_contract = (
            "Return location_at_time_hint.value as the location on the requested date. "
            "It is derived from the latest typed movement before that date and its next "
            "return boundary; do not substitute a later retrospectively reported trip."
        )
    elif (
        local_contrast_alternative_hint
        and local_contrast_alternative_hint.get("complete")
    ):
        if local_contrast_alternative_hint.get("relation_kind") == "causal_displacement":
            answer_contract = (
                "Answer the why-question using contrast_alternative_hint.value as the "
                "displacing activity. Phrase it concisely as the reason the rejected "
                "activity was deferred; do not invent an additional motive."
            )
        else:
            answer_contract = (
                "Return contrast_alternative_hint.value as the planned alternative. "
                "It is bound through a local instead-of/rather-than dialogue adjacency; "
                "do not replace it with a newer unrelated plan."
            )
    elif (
        local_event_lifecycle_hint
        and local_event_lifecycle_hint.get("complete")
        and catalog_operation != "catalog_duration"
    ):
        answer_contract = (
            "Return event_lifecycle_duration_hint.value and unit as an approximate "
            "duration. Its subject-bound start and end evidence supersede unrelated "
            "dated mentions; preserve the approximation when either endpoint is a "
            "relative week or month."
        )
    elif local_binding_hint and local_binding_hint.get("complete"):
        answer_contract = (
            "State that memory is insufficient for the exact requested relationship or count target. "
            "Do not answer from a sibling possessor or related event."
        )
    elif local_structured_section_hint and local_structured_section_hint.get("complete"):
        answer_contract = (
            "Return structured_section_hint.value exactly. It was extracted from the "
            "requested ordinal artifact and named section in a lossless source turn."
        )
    elif explicit_date_hint and explicit_date_hint.get("complete") and operation == "date":
        answer_contract = (
            "Return explicit_date_hint.value exactly as the date bound to the locally "
            "relevant session; do not substitute the observation timestamp."
        )
    elif (
        isinstance(local_relative_age_hint, dict)
        and local_relative_age_hint.get("complete")
    ):
        answer_contract = (
            "Return relative_age_hint.value exactly as the age or elapsed amount bound to "
            "the requested event in lossless local evidence; do not substitute a date or "
            "a sibling event."
        )
    elif (
        isinstance(local_relative_time_hint, dict)
        and local_relative_time_hint.get("complete")
        and local_relative_time_hint.get("resolved_value")
    ):
        answer_contract = (
            "Return relative_time_hint.resolved_value exactly. The dated event was bound through "
            "an explicit event-frame-to-operand/source relation whose lossless source is packed."
        )
    elif catalog_trace_operation == "scalar_attribute_state_ambiguous":
        answer_contract = (
            "The catalog found temporally ordered competing scalar values for the requested "
            "attribute but intentionally did not force one. Inspect each cited lossless source. "
            "Use the newer value when its sentence grammatically treats that number as the value "
            "of the requested existing attribute, even if that reference is nested inside a plan "
            "or comparison. If the number is only a desired target, threshold, or estimate, retain "
            "the asserted state instead. Do not default to the older value merely because its "
            "typed predicate is more specific."
        )
    elif catalog_operation == "named_alternative_incomplete":
        answer_contract = (
            "State that memory is insufficient because at least one explicitly named "
            "alternative lacks evidence for the compared relationship. Do not infer its "
            "date from unrelated facts about that entity."
        )
    elif catalog_operation == "combined_named_duration":
        answer_contract = (
            "Return catalog_hint.value and catalog_hint.unit exactly as the sum of the "
            "independently entity-bound completed durations in catalog_hint.proofs. Every "
            "explicitly named entity has its own lossless source; ignore plans and estimates."
        )
    elif catalog_operation == "combined_named_duration_incomplete":
        answer_contract = (
            "State that memory is insufficient because at least one explicitly named entity "
            "does not have one unambiguous asserted completion duration. Do not use an estimate "
            "or a duration belonging to a different entity."
        )
    elif catalog_operation == "frequency_state_comparison":
        answer_contract = (
            "Return catalog_hint.value as yes or no. It compares the latest and previous "
            "weekly rates for the same bound activity using typed dated evidence."
        )
    elif catalog_operation == "earliest_named_alternative":
        answer_contract = (
            "Return catalog_hint.value exactly as the earlier of the two alternatives "
            "explicitly named in the question."
        )
    elif catalog_operation == "final_choice_state":
        answer_contract = (
            "Return catalog_hint.value exactly as the accepted or adopted final state. "
            "Do not replace it with an earlier brainstormed proposal."
        )
    elif catalog_operation == "ordered_event_collection":
        answer_contract = (
            "Return every catalog_hint.values entry in its supplied chronological order, "
            "using the event value rather than a bare date."
        )
    elif catalog_operation == "ordered_action_entity_candidates":
        answer_contract = (
            "The catalog performed a global lossless scan for completed instances of the "
            "requested action. Inspect every cited candidate in date order, bind the requested "
            "target entity from that event or its same-turn context, exclude plans and current-"
            "day uncompleted choices, deduplicate retrospective repeats of the same event/entity, "
            "and return the complete chronological entity list."
        )
    elif catalog_operation == "all_subjects_relation":
        answer_contract = (
            "Return catalog_hint.value exactly. The ALL/both quantifier is true only because "
            "each named subject has independent positive typed provenance in catalog_hint.proofs."
        )
    elif catalog_operation == "ordinal_list_item":
        answer_contract = (
            "Return the complete catalog hint value exactly as the requested ordinal list item. "
            "The position is resolved from the lossless ordered-list source turn."
        )
    elif (
        catalog_operation == "ordinal_event_attribute"
        and isinstance(catalog_hint, dict)
        and catalog_hint.get("complete")
    ):
        answer_contract = (
            "Return catalog_hint.value exactly as the attribute or date of the requested ordinal "
            "completed event. The event sequence is ordered by grounded observation time, and the "
            "attribute is bound inside that event session; do not substitute a habitual attribute "
            "or a different event occurrence."
        )
    elif catalog_operation == "ordinal_event_attribute":
        answer_contract = (
            "The ordinal event occurrence and session are complete, but its requested attribute was "
            "not preserved in the typed operand. Extract that attribute only from catalog_hint.evidence "
            "and its cited same-session dialogue window. Follow contrast and follow-up answers: distinguish "
            "what is usual from what happened this time. Do not use another occurrence."
        )
    elif catalog_operation == "exact_entity_mismatch":
        answer_contract = (
            "State that the memory is insufficient for the exact requested multiword entity. "
            "Do not answer from evidence that matches only a strict substring or sibling entity."
        )
    elif (
        catalog_operation == "movement_location_collection"
        and isinstance(catalog_hint, dict)
        and catalog_hint.get("complete")
    ):
        answer_contract = (
            "Return every catalog_hint.values entry exactly as the locations where "
            "the requested subject was physically present in the requested time "
            "window. Exclude plans, invitations, bookings, other subjects, media "
            "locations, and places that were only mentioned."
        )
    elif (
        catalog_operation == "planned_event_identity_count"
        and isinstance(catalog_hint, dict)
        and catalog_hint.get("complete")
    ):
        answer_contract = (
            "Return catalog_hint.value exactly as the number of distinct planned "
            "events. Repeated mentions with the same participants, action, and "
            "resolved target interval are one plan identity; do not count completed "
            "events or repeated conversations as additional plans."
        )
    elif (
        catalog_operation == "event_occurrence_count"
        and isinstance(catalog_hint, dict)
        and catalog_hint.get("complete")
    ):
        answer_contract = (
            "Return the complete catalog hint value exactly as the number of event occurrences. "
            "It is an occurrence count, not a distinct-entity count; use its groups as the audit breakdown."
        )
    elif (
        catalog_operation == "distinct_action_entity_collection"
        and isinstance(catalog_hint, dict)
        and catalog_hint.get("complete")
    ):
        answer_contract = (
            "Return catalog_hint.value exactly as the number of distinct target entities. "
            "The global typed closure already unions every action requested by the question, "
            "deduplicates repeated mentions of one entity, and excludes relation-only mentions."
        )
    elif catalog_operation == "distinct_action_entity_collection":
        answer_contract = (
            "catalog_hint.value is only a typed lower bound because lossless user turns in "
            "uncovered_source_turn_ids matched the requested action and target type but lacked "
            "an operand projection. Inspect those cited turns, recover any distinct target "
            "entities, deduplicate them with catalog_hint.items, and then return the full count."
        )
    elif catalog_operation == "event_occurrence_count":
        answer_contract = (
            "Treat catalog_hint.value_lower_bound as projected occurrences only. Inspect every "
            "lossless turn in missing_source_turn_ids, count each distinct completed matching event, "
            "and use candidate_value only when each cited turn independently supports one event. "
            "Do not count plans or recommendations."
        )
    elif catalog_operation in {
        "per_item_amount", "money_amount", "money_difference", "total_money",
        "aggregate_revenue_total",
        "dimensional_quantity_total", "ratio_percent",
        "partitioned_scalar_total",
    }:
        answer_contract = (
            "Return the complete catalog hint value and unit directly. Its operands already encode the "
            "requested arithmetic; do not recount distinct entities or substitute an intermediate total."
        )
    elif catalog_operation == "latest_relation_state":
        answer_contract = (
            "Return the complete catalog hint value as the newest valid relation state, then inspect every cited lossless source turn for relation arguments omitted by the projection. For a location, return the full containment chain and resolve pronouns such as it or there to the nearest explicit containing place in that turn. Do not replace it "
            "with an older, more lexically similar relation."
        )
    elif catalog_operation == "relation_slot_location":
        answer_contract = (
            "Return catalog_hint.value exactly as the location attached to the requested "
            "relation. The hint follows one predicate-matched operand through its event-frame "
            "and lossless source; do not borrow a place from another event involving the same "
            "person or nearby dialogue."
        )
    elif catalog_operation == "dialogue_followup_plan":
        answer_contract = (
            "Return catalog_hint.value as the target speaker's plan. It is bound through a "
            "same-session object-bearing commitment and the recipient's adjacent response; "
            "do not substitute a later topically similar plan."
        )
    elif catalog_operation == "event_onset_from_lossless_evidence":
        answer_contract = (
            "Return catalog_hint.value exactly as the event onset. It is computed from a "
            "subject- and activity-bound lossless relative expression or present-perfect "
            "duration anchored to that dialogue session date; preserve approximation wording."
        )
    elif catalog_operation == "dialogue_answer_span":
        answer_contract = (
            "Return catalog_hint.value exactly. It is an explicitly named value from the "
            "next answer turn after a semantically matching historical question, with the "
            "requested speaker bound before extraction. Do not replace it with an older "
            "topically related value."
        )
    elif catalog_operation == "scalar_comparison":
        answer_contract = (
            "Return catalog_hint.value and unit exactly. The difference is computed from "
            "two separately entity-bound scalar mentions in cited lossless turns; do not "
            "substitute a timestamp, ordinal, or another person's value."
        )
    elif catalog_operation == "catalog_duration":
        answer_contract = (
            "Return catalog_hint.value and catalog_hint.unit from the complete catalog hint; use "
            "inclusive_days only when the question explicitly asks for inclusive day counting."
        )
    elif catalog_operation == "scalar_snapshot":
        answer_contract = (
            "Return the complete catalog hint value and unit as the explicit quantity snapshot. "
            "Do not recount mentions or replace a scoped historical snapshot with a later total."
        )
    elif catalog_operation == "weekly_recurrence_count":
        answer_contract = (
            "Return the complete catalog hint value as scheduled occurrences per requested period, using "
            "recurrence_days as the audit breakdown."
        )
    elif catalog_operation == "arrival_clock_time":
        answer_contract = (
            "Return catalog_hint.value exactly as the arrival clock. It is computed from a "
            "relation-bound departure time plus travel duration in the cited lossless turns."
        )
    elif catalog_operation == "arrival_clock_ambiguous":
        answer_contract = (
            "State that the arrival time is not uniquely supported; do not borrow a business "
            "hour or a clock from another travel path."
        )
    elif catalog_operation == "scalar_attribute_state":
        answer_contract = (
            "Return catalog_hint.value exactly as the newest valid scalar state of the requested "
            "attribute. The operator binds numbers from operand objects rather than timestamps and "
            "selects the attribute family before applying chronology."
        )
    elif re.search(
        r"\b(?:might|likely|probably|possibly|could)\b|\bbased on\b|\bwould\b.{0,40}\benjoy\b",
        case.question.casefold(),
    ):
        answer_contract = (
            "The question explicitly asks for a best-supported inference. Combine the supplied evidence "
            "with ordinary semantic implications, and return the most strongly supported concise conclusion. "
            "Do not require the conclusion to appear verbatim; abstain only when the evidence leaves the "
            "leading alternatives genuinely tied or unsupported."
        )
    elif operation == "earliest" and operator is not None:
        answer_contract = (
            "Use the complete local operator result to choose only among the alternatives "
            "named in the question. Return the named alternative whose matching event is earliest."
        )
    elif operation == "counterfactual":
        answer_contract = (
            "Answer the stated counterfactual, not the present-day fact. Identify explicit causal "
            "evidence connecting the removed condition to the outcome. If the evidence says that "
            "condition motivated or caused the outcome, removing it should negate or materially "
            "reduce the outcome unless independent evidence supports the same result."
        )
    elif (
        operation == "ordering"
        and re.search(r"\b(?:before|after)\b", case.question.casefold())
    ):
        answer_contract = (
            "Treat the event named after before or after as the anchor. Return only the requested "
            "semantic slot from the nearest qualifying event on that side of the anchor, using "
            "event/session dates and lossless source turns; do not return the anchor itself."
        )
    elif operation == "ordering":
        answer_contract = (
            "Return exactly the requested completed or started events in chronological order. "
            "Exclude events that are only planned, recommended, hypothetical, or outside the requested "
            "time window. Prefer a lossless raw turn when its event clause is missing from its projections, "
            "and map every date back to the event name."
        )
    elif operation == "latest" and re.search(r"\bwhere\b", case.question.casefold()):
        answer_contract = (
            "Return the place in the newest valid evidence about the requested object or state. "
            "A newer explicit storage or location update supersedes an older one; do not select an "
            "older place merely because its wording overlaps more strongly."
        )
    elif (
        operation == "location"
        and local_relative_time_hint
        and local_relative_time_hint.get("within_tolerance")
    ):
        answer_contract = (
            "Return the location stated in relative_time_hint.evidence. That fine-grained "
            "dated fact defines the requested relative-time scope; do not substitute a "
            "nearby sibling event from another evidence block."
        )
    elif operation == "location":
        answer_contract = (
            "Return the distinct place names that answer where. Do not substitute dates, companions, "
            "activities, or generic occasions for locations."
        )
    elif operation == "preference_list":
        answer_contract = (
            "Return only the distinct objects or qualities that evidence explicitly says the subject "
            "likes, loves, or enjoys. Do not substitute activities they merely did, places they visited, "
            "or things inferred from attendance unless the preference itself is stated."
        )
    elif operation == "planned_date":
        answer_contract = (
            "Answer when the explicitly planned future event is scheduled. When planned_event_hint "
            "is present, use its event_time and anchor_date unless its cited evidence is negated or cancelled. Resolve relative time against "
            "that evidence block session date. Do not replace the plan with a later completed event of "
            "the same kind, and do not infer cancellation unless cancellation is explicit."
        )
    elif (
        operation == "duration"
        and local_duration_hint
        and local_duration_hint.get("operation")
        == "duration_since_consecutive_event_sequence"
    ):
        answer_contract = (
            "Return duration_hint.value and duration_hint.unit exactly. The operator first "
            "resolved the requested consecutive event sequence, then measured from the "
            "sequence endpoint to the question date using the requested calendar unit."
        )
    elif (
        operation == "duration"
        and local_duration_hint
        and local_duration_hint.get("operation") == "explicit_relative_duration"
    ):
        answer_contract = (
            "Return duration_hint.value and duration_hint.unit exactly. The hint resolves "
            "explicit state-duration and relative-event expressions from locally bound "
            "lossless evidence; do not replace it with calendar dates from sibling facts."
        )
    elif (
        operation == "duration"
        and local_relative_age_hint
        and local_relative_age_hint.get("operation")
        == "relative_age_from_evidence_expression"
    ):
        answer_contract = (
            "Return relative_age_hint.value exactly as the relative elapsed expression bound "
            "to the requested event in a lossless source. Do not require a second dated endpoint."
        )
    elif operation == "duration":
        answer_contract = (
            "When duration_hint is present and its two endpoints match the two events in the question, "
            "return duration_hint.value and duration_hint.unit; inclusive_days is also acceptable only when the question requests "
            "inclusive day counting. If either exact named endpoint is absent, state that the "
            "memory is insufficient and do not substitute a sibling entity. A completed event described as today or just completed in a dated "
            "memory inherits that memory date unless contrary evidence supplies another date."
        )
    elif operation == "date":
        answer_contract = (
            "Resolve relative time expressions against the evidence block session date. If evidence says "
            "last week, two weeks ago, or a few weeks ago, preserve that range or compute the anchored "
            "range; do not abstain merely because an exact day is unavailable."
        )
    elif operation == "lookup" and re.search(
        r"\b(?:or not|did|was|were|has|have)\b", case.question.casefold()
    ):
        answer_contract = (
            "For an event-specific yes-or-no question, bind every modifier, especially relative time, "
            "to the same event. If relative_time_hint is within tolerance, evaluate the requested relation "
            "only in that dated scope and never transfer a person or attribute from a similar event at "
            "another date. Answer yes only when the scoped evidence asserts the relation; otherwise state "
            "that the relation is not present in the scoped event evidence."
        )
    elif operation == "lookup":
        answer_contract = (
            "Return the explicit referent requested by the question. If the evidence does not give a "
            "proper name but uniquely identifies the referent with a descriptive noun phrase, return "
            "that description instead of abstaining solely because a name is absent. Preserve the "
            "attributes that distinguish the referent from alternatives."
        )
    elif answer_form == "recommendation":
        answer_contract = (
            "Match the requested response form: when asked for resources, courses, places, or tools, name concrete resources, resource categories, places, or tools rather than returning learning tips alone. For a resource request, the first sentence must explicitly recommend resources tailored to the most specific positive target and depth constraints in memory; do not begin with project or practice advice. The entity, location, or scenario named in the current question is the mandatory target. Honor the most specific positive tool, product, skill level, feature depth, and format constraints in the supplied memory; do not dilute them into a generic recommendation. "
            "Answer for that target even when memory only contains a different older plan; never replace "
            "the requested target or refuse solely because its plan is absent from memory. Use general "
            "knowledge for the recommendation and remembered preferences only to personalize it. State the "
            "relevant preference or constraint, then give concrete categories and named examples. "
            "For advice, every recommendation_constraints item marked recommendation_resource_provenance is an existing resource: explicitly reuse each one and never recommend acquiring it again. Explicitly incorporate each locally relevant planned or current object change "
            "and each stated style or format constraint; ignore unrelated facts that merely share a room, place, or date."
        )
    elif answer_form == "frequency":
        answer_contract = (
            "Return a concrete recurrence interval, never a vague adverb such as regularly. "
            "Prefer an explicit schedule; otherwise bind same-subject, same-activity dated "
            "occurrences and report the best-supported approximate cadence with units."
        )
    elif answer_form == "number" and re.search(
        r"\bhow many\s+(?:different\s+)?types?\s+of\b",
        case.question.casefold(),
    ):
        answer_contract = (
            "Count distinct subtype names that occur in completed, asserted use of the requested target. "
            "A component explicitly mixed into, served in, incorporated into, or used to prepare a completed "
            "item counts as used. Exclude merely suggested, provided, planned, possible, or desired components. "
            "Normalize repeated names, give an auditable list, and end with Total = N."
        )
    elif answer_form == "number" and re.search(
        r"\b(?:per|each|typical)\s+(?:day|week|month|year)\b|\bin a typical week\b",
        case.question.casefold(),
    ):
        answer_contract = (
            "Count scheduled occurrences in the requested period, not only distinct activity types. "
            "A recurring activity assigned to multiple days contributes one occurrence per listed day. "
            "Deduplicate repeated mentions of the same schedule, then show the per-activity frequency "
            "and end with Total = N."
        )
    elif answer_form == "number":
        answer_contract = (
            "Build a set of distinct entities for which evidence explicitly states the subject-predicate "
            "relation requested by the question. Normalize aliases and enforce the requested entity type, "
            "time, polarity, and modality. Do not infer the predicate from association, participation, "
            "ownership, a role or team, completion, or merely working on an entity. Include both historical "
            "and current matches only when the question asks for both. For pending-action questions, each "
            "distinct action-object-state pair is an operand; an original item and its replacement remain "
            "distinct when evidence assigns them different pending actions. Otherwise count each entity once. "
            "If no evidence matches the exact requested entity type, state that memory is "
            "insufficient and do not turn absence into Total = 0. Give a short auditable "
            "breakdown, and end with Total = N only when at least one exact operand exists."
        )
    elif answer_form == "list":
        answer_contract = (
            "Name every requested entity or event and attach its relevant order, date, or attribute. "
            "For ordering questions, return an explicit first-to-last sequence using the event names; "
            "never answer with bare dates or values that are not mapped back to their events."
        )
    else:
        answer_contract = (
            "Return the direct value, entity, list, date, duration, or state requested."
        )
    catalog_complete = bool(
        isinstance(catalog_hint, dict) and catalog_hint.get("complete")
    )
    if not catalog_complete:
        lowered_question = case.question.casefold()
        if re.search(r"^what\s+did\s+.+?\s+[a-z]+\b", lowered_question):
            answer_contract += (
                " Bind the grammatical subject, relation, and direct-object slot independently. "
                "Return only the object of the requested relation for that subject; do not substitute "
                "a nearby subject, habitual object, or another relation from the same session."
            )
        if re.search(r"^why\s+did\b", lowered_question):
            answer_contract += (
                " Return the explicitly stated cause or motivation, not merely the delayed activity, "
                "a consequence, or a general personality inference."
            )
        if re.search(r"\bfavou?rite\b", lowered_question):
            answer_contract += (
                " Require preference evidence for the exact named subject. Resolve this, that, or the "
                "series through adjacent turns in the same dialogue before declaring the name absent. "
                "If explicit favorite wording is absent but exactly one current candidate is described "
                "with strong positive affect and no competing candidate is supported, return it as the "
                "best-supported likely favorite rather than abstaining."
            )
        if query_frame.get("explicit_dates"):
            answer_contract += (
                " Treat explicit_dates as a hard temporal scope and prefer evidence whose session or "
                "event date matches it exactly over a more recent or more frequent sibling fact."
            )
    answer_slot = infer_answer_slot(case.question)
    relation_evidence_ids: set[str] = set()
    if local_before_after_hint and local_before_after_hint.get("complete"):
        relation_evidence_ids.update(local_before_after_hint.get("source_turn_ids", []))
        for key in ("anchor_event", "nearest_qualifying_event"):
            node_id = str((local_before_after_hint.get(key) or {}).get("node_id") or "")
            if node_id:
                relation_evidence_ids.add(node_id)
    relation_evidence = contract_evidence_to_ids(
        retrieval.context_text, relation_evidence_ids
    )
    focused_evidence = relation_evidence or (
        focused_evidence_capsule(case.question, retrieval.context_text)
        if should_use_focused_capsule(
            answer_form=answer_form, operation=operation
        )
        else ""
    )
    focused_only = False
    answer_evidence = relation_evidence or retrieval.context_text
    system = (
        "Answer using only the supplied memory evidence. Named speakers are equal "
        "participants; do not prefer evidence based on user/assistant transport roles. "
        "Use structured coarse claims for routing and raw turns for exact grounding. "
        "A reference_chain_hint is a build-time discourse link; use it only when both cited raw "
        "turns verify the antecedent and anaphoric statement. "
        "When a verified reference_chain_hint matches the question's discourse reference, "
        "answer from its antecedent and do not replace it with a nearby entity of the same type. "
        "The cited raw turns remain the authority. "
        "When a verified reference_chain_hint has a non-empty resolved_value matching the requested "
        "reference, answer with that resolved value rather than repeating the anaphor phrase. "
        "Catalog hints are deterministic projections of packed evidence: use a complete hint directly; "
        "Only catalog_operator_hint is authoritative. catalog_candidate_hint is a routed semantic "
        "candidate and must be verified against its cited lossless turns before use. "
        "Treat a multiword entity phrase as atomic: evidence for a strict substring or sibling entity "
        "does not establish the full requested entity. "
        "use an incomplete hint only as a candidate list and verify it against cited evidence. "
        "A scope posterior is routing confidence, not evidence. When primary_scope_hint is present, "
        "prefer evidence from that session; include lower-ranked sessions only when they provide an "
        "explicitly requested operand, update, or contradiction. "
        "Evidence blocks are relevance-ranked, not chronological. Resolve relative times such as "
        "today, yesterday, last week, two weeks ago, a few weeks ago, or just got back against that evidence block session date; a completed "
        "event stated in a dated turn inherits that date unless a different event date is explicit. "
        "When relative_time_hint is present and within_tolerance is true, use its selected evidence scope "
        "instead of a more lexically salient event from another date, and evaluate the requested relation "
        "only inside that selected dated scope. When relative_age_hint is present, use its anchored "
        "event_date and computed value directly unless the cited evidence is about a different event. "
        "When latest_state_hint is present, prefer its dated evidence for current-state questions unless contradicted by a newer exact match. "
        "When newest_scalar_hint is present and the question gives no explicit historical cutoff, "
        "treat it as the locally computed newest candidate and prefer it unless its cited evidence "
        "is only planned, possible, negated, or about a different attribute. If the same attribute "
        "has different dated values, compare dates and use the newest valid asserted state, "
        "while preserving completed events over later plans. For a recommendation or general "
        "explanation, you may use general knowledge, but personalize it from memory and never "
        "invent user history. Use an operator result only when its closure certificate is "
        "complete. An incomplete closure certificate only disables the local operator; it does not mean "
        "the packed evidence is insufficient. Never abstain solely because closure is incomplete or the "
        "pack is truncated. The query-focused evidence capsule is a contraction of "
        "the same packed subgraph, not a separate source: inspect it first and use "
        "extended evidence only to fill a missing operand, update, or contradiction. "
        "When an event_entity_hint specifically matches the requested event, take dates and status "
        "only from that entity's member events or cited source turns. Evidence for a similar or "
        "sibling event outside member_event_ids must not answer the question. A cluster is identity "
        "evidence, not permission to transfer an unrelated attribute between members. "
        "Answer the requested semantic slot; evidence about the same event but a "
        "For a date question, when the matching event_entity_hint has current_resolved_time, return "
        "that value exactly rather than its relative event_time phrase. "
        "different attribute is not an answer. Return only a concise answer, without reasoning."
    )
    payload = {
        "question": case.question,
        "question_date": case.question_date,
        "answer_form": answer_form,
        "answer_contract": answer_contract,
        "closure_certificate": closure,
        "operator_result": operator,
        "catalog_operator_hint": catalog_answer_hint,
        "catalog_candidate_hint": catalog_candidate_hint,
        "temporal_evidence_ledger_newest_first": temporal_ledger,
        "binding_hint": local_binding_hint,
        "explicit_date_hint": explicit_date_hint,
        "calendar_window_hint": local_calendar_window_hint,
        "before_after_relation_hint": local_before_after_hint,
        "reference_chain_hints": retrieval.retrieval_trace.get(
            "reference_chain_hints", []
        ),
        "event_identity_chain_hints": retrieval.retrieval_trace.get(
            "event_identity_chain_hints", []
        ),
        "event_entity_hints": retrieval.retrieval_trace.get(
            "event_entity_hints", []
        ),
        "structured_section_hint": local_structured_section_hint,
        "newest_scalar_hint": newest_scalar_hint,
        "duration_hint": local_duration_hint,
        "planned_event_hint": planned_event_hint,
        "media_attribute_hint": local_media_attribute_hint,
        "location_at_time_hint": local_location_at_time_hint,
        "event_lifecycle_duration_hint": local_event_lifecycle_hint,
        "contrast_alternative_hint": local_contrast_alternative_hint,
        "latest_state_hint": local_latest_state_hint,
        "relative_time_hint": local_relative_time_hint,
        "relative_age_hint": local_relative_age_hint,
        "scope_posteriors": scope_posteriors[:4],
        "primary_scope_hint": primary_scope_hint,
        "recommendation_constraints": recommendation_constraints,
        "evidence": answer_evidence,
        "answer_slot": {
            "kind": answer_slot.kind,
            "instruction": answer_slot.instruction,
        },
        "query_focused_evidence": "" if focused_only else focused_evidence,
    }
    messages, prepack = fit_answer_payload(
        system, payload, max_prompt_tokens=max_prompt_tokens
    )
    retrieval.retrieval_trace["answer_prepack"] = prepack
    retrieval.retrieval_trace["catalog_hint_authoritative"] = bool(
        catalog_answer_hint
    )
    retrieval.retrieval_trace["answer_focused_only"] = focused_only
    retrieval.retrieval_trace["answer_evidence_text"] = answer_evidence
    retrieval.retrieval_trace["answer_evidence_chars"] = len(answer_evidence)
    retrieval.retrieval_trace["answer_evidence_block_ids"] = [
        match.group(1)
        for match in re.finditer(
            r"^\[(?:TURN|CLAIM|EVENT|EVENT_ENTITY|EVENT_FRAME|OPERAND|EPISODE|THEME)\s+([^\s|\]]+)",
            answer_evidence,
            flags=re.MULTILINE | re.IGNORECASE,
        )
    ]
    if not prepack["fit_pass"]:
        raise ValueError(f"V3 answer prepack failed: {prepack}")
    return messages
