from __future__ import annotations

import json
import os
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any, Iterable

from ..clients import cosine_similarity, rough_token_count
from ..models import QuestionCase, RetrievedContext
from ..v3.build import canonical_key
from ..v3.action_semantics import action_families
from .source_spans import (
    binding_tokens, build_source_span_closure, fuzzy_term_overlap, query_binding_terms,
)
from .dialogue_topology import infer_dialogue_topology, is_memory_source
from .operators import (
    evaluate_operators, query_bound_collection_ledger, counterfactual_dependency_hint, record_time_source_hint, temporal_order_source_hint,
    open_temporal_sequence_from_sources_hint,
    temporal_source_pair_hint, relative_time_from_sources_hint,
    source_bound_date_lookup_hint, transaction_sum_from_sources_hint,
    exact_entity_absence_hint,
    named_individual_event_members_hint, repeated_event_total_from_sources_hint,
    age_arithmetic_from_sources_hint, advance_booking_recency_from_sources_hint,
    current_role_duration_from_sources_hint, weekly_schedule_days_from_sources_hint,
    family_relation_total_from_sources_hint, linked_event_date_from_sources_hint,
    latest_category_start_from_sources_hint, scoped_completed_event_members_hint,
    preference_constraints_from_sources_hint, dialogue_attribute_match_hint,
    currency_extreme_entity_from_sources_hint,
    presupposed_event_absence_hint,
    dialogue_final_choice_from_sources_hint,
    completed_item_metric_total_from_sources_hint,
    scoped_completed_duration_total_from_sources_hint,
    relative_value_multiplier_from_sources_hint,
    relative_duration_at_event_from_sources_hint,
    prior_candidate_count_from_sources_hint,
    completed_carrier_sequence_from_sources_hint,
    event_endpoint_difference_from_sources_hint,
    travel_arrival_time_from_sources_hint,
    completed_work_subtype_total_from_sources_hint,
    incomplete_terminal_event_hint,
    state_change_members_from_sources_hint, provenance_acquisition_members_hint,
    explicit_cuisine_categories_hint, subset_percentage_from_sources_hint,
    excluded_collection_members_hint, paired_metric_total_from_sources_hint,
    binary_savings_from_sources_hint, temporal_predecessor_entity_hint,
    labeled_scalar_difference_from_sources_hint,
    repeated_activity_duration_total_hint, relative_anchor_source_hint,
    dated_event_count_from_sources_hint, named_event_attendance_count_hint,
    same_unit_state_difference_hint,
    maintenance_entity_count_hint, category_acquisition_members_hint,
    pending_operation_target_pairs_hint,
    latest_scalar_state_from_sources_hint,
    threshold_progress_remaining_hint, latest_approx_scalar_state_hint,
    latest_labeled_currency_state_hint, latest_weekly_schedule_time_hint,
    same_unit_acquisition_total_hint,
)
from .schema import (
    CompletenessCertificate,
    EvidenceGroup,
    QueryIR,
    RoleFrameNode,
    RoutingCard,
    TurnNodeV36,
    V36Index,
)


V36_RETRIEVAL_VERSION = "graphmem_v36_atomic_question_time_closure_20260730"


def _navigation_repair_enabled() -> bool:
    return os.environ.get("GRAPHMEM_V36_NAVIGATION_REPAIR", "").casefold() in {
        "1", "true", "yes", "on",
    }


def _atomic_group_repair_enabled() -> bool:
    return os.environ.get("GRAPHMEM_V36_ATOMIC_GROUP_REPAIR", "").casefold() in {
        "1", "true", "yes", "on",
    }


def _dialogue_closure_enabled() -> bool:
    return os.environ.get("GRAPHMEM_V36_DIALOGUE_CLOSURE", "").casefold() in {
        "1", "true", "yes", "on",
    }

_WORD_RE = re.compile(r"[\w'-]+", re.UNICODE)
_FUNCTION_WORDS = {
    "a", "about", "after", "again", "all", "am", "an", "and", "any", "are", "as",
    "at", "be", "before", "between", "by", "can", "checking", "could", "current",
    "currently", "did", "do", "does", "earliest", "for", "from", "had",
    "has", "have", "he", "her", "hers", "him", "his", "how", "i", "in",
    "is", "it", "its", "latest", "many", "me", "most", "my", "of", "on",
    "or", "our", "ours", "please", "previous", "recent", "recently", "she", "should",
    "some", "remind", "that", "the", "their", "theirs", "them", "they", "this",
    "those", "to", "was", "we", "were", "what", "when", "where", "which",
    "who", "why", "wondering", "if", "will", "with", "would", "you", "your", "yours",
    "i'm", "im", "i've", "ive", "i'll", "ill", "getting", "excited",
    "tip", "tips", "look", "looking",
}
_DATE_RE = re.compile(
    r"\b(?:19|20)\d{2}(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?\b|"
    r"\b(?:today|yesterday|tomorrow|last|past|next|ago|before|after)\b",
    re.IGNORECASE,
)
_NEGATIVE_RE = re.compile(
    r"\b(?:not|never|no longer|dislike|hate|avoid|without|cancelled|canceled)\b",
    re.IGNORECASE,
)
_STATE_RE = re.compile(
    r"\b(?:current|currently|latest|now|still|status|state|changed|updated|"
    r"switch(?:ed|ing)?|became|become|no longer|highest|lowest|best|record)\b",
    re.IGNORECASE,
)
_LIST_RE = re.compile(
    r"\b(?:list|all|each|different|types?|items?|things?)\b|"
    r"\b(?:what|which)\s+(?!(?:was|is|has|does)\b)[\w'-]+s\b",
    re.IGNORECASE,
)
_COUNT_RE = re.compile(
    r"\bhow many\b|\bcount\b|\bnumber of\b|\btotal\b", re.IGNORECASE
)
_DURATION_RE = re.compile(
    r"\bhow long\b|\bduration\b|\btime between\b|\bhow many "
    r"(?:days|weeks|months|years|hours|minutes)\b|"
    r"\bhow much time\b|\b(?:total|combined)\s+time\b|"
    r"\btime\s+(?:it\s+)?takes?\b",
    re.IGNORECASE,
)
_RECORD_STATE_RE = re.compile(
    r"\b(?:personal\s+best|all-time\s+best|record\s+(?:time|score|distance|weight|speed))\b",
    re.IGNORECASE,
)
_RECOMMENDATION_RE = re.compile(
    r"\b(?:recommend(?:ed|ing|ation|ations)?|suggest(?:ed|ing|ion|ions)?)\b|"
    r"\b(?:advice|tips?)\b|"
    r"\bwhat\s+to\s+look\s+for\b|"
    r"\bhelp(?:\s+\w+){0,3}\s+(?:choose|pick|decide)\b|"
    r"\bdo you think\b(?:.{0,80}\b(?:might|could|should)\b)?|"
    r"\bwhich\b.+\bshould\s+(?:i|we)\b|"
    r"\bany\s+(?:ideas?|suggestions?)\b|"
    r"\b(?:would|is)\s+it\s+(?:be\s+)?a\s+good\s+idea\b|"
    r"\bwhat\s+should\s+(?:i|we)\b|"
    r"\bwhat\s+(?:[A-Za-z'-]+\s+){1,4}should\s+(?:i|we)\s+(?:use|choose|pick|try|make|do)\b|"
    r"\bwhat\s+(?:could|can)\s+(?:i|we)\s+(?:serve|make|do|try|watch|read)\b",
    re.IGNORECASE,
)
_TEMPORAL_ORDER_RE = re.compile(
    r"\b(?:who|what|which)\b.{0,80}\b"
    r"(?:first|earlier|later|earliest|latest)\b.{0,160}\bor\b",
    re.IGNORECASE,
)
_PREFERENCE_RE = re.compile(
    r"\b(?:prefer|preference|favorite|favourite|like|dislike|love|hate|enjoy)\b",
    re.IGNORECASE,
)
_TEMPORAL_SEQUENCE_RE = re.compile(
    r"\b(?:order|sequence|chronological order)\s+(?:of|for)\b|"
    r"\bfrom\s+(?:earliest|oldest)\s+to\s+(?:latest|newest)\b",
    re.IGNORECASE,
)
_TEMPORAL_LATEST_RE = re.compile(
    r"\b(?:most recently|started using most recently|latest one|newest one)\b",
    re.IGNORECASE,
)
_TEMPORAL_AFTER_FIRST_RE = re.compile(
    r"\bfirst\b.{0,120}\bafter\b|\bafter\b.{0,120}\bfirst\b",
    re.IGNORECASE,
)
_AVERAGE_RE = re.compile(r"\b(?:average|mean)\b", re.IGNORECASE)
_SUM_RE = re.compile(
    r"\bhow much\b(?!\s+time\b).{0,80}\b(?:spend|spent|pay|paid|cost)\b|"
    r"\b(?:total money|total amount|total cost)\b", re.IGNORECASE,
)
_DIFFERENCE_RE = re.compile(
    r"\bhow much (?:more|less)\b|\b(?:difference|compared to)\b",
    re.IGNORECASE,
)
_DIALOGUE_RE = re.compile(
    r"\b(?:tell|told|say|said|ask|asked|answer|answered|reply|replied|"
    r"recommended|suggested|advice)\b",
    re.IGNORECASE,
)
_REFERENCE_RE = re.compile(
    r"\b(?:its|those|these|former|latter|same|another)\b|"
    r"\b(?:that|this)\s+(?!(?:is|was|would|could|should|has|have|did|does|can|will)\b)[\w'-]+",
    re.IGNORECASE,
)
_ORDINAL_RE = re.compile(
    r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|"
    r"tenth|\d+(?:st|nd|rd|th))\b",
    re.IGNORECASE,
)


def _tokens(text: str) -> list[str]:
    return [
        token.casefold().strip("'\"")
        for token in _WORD_RE.findall(text.replace("_", " "))
        if token.casefold().strip("'\"") not in _FUNCTION_WORDS
    ]


def _content_tokens(text: str) -> list[str]:
    return list(dict.fromkeys(
        token for token in _tokens(text)
        if len(token) > 1 and token not in _FUNCTION_WORDS
    ))


def _comparison_targets(question: str) -> list[str]:
    """Extract the two explicitly contrasted event descriptions, if present."""
    if not _TEMPORAL_ORDER_RE.search(question):
        return []
    body = question.strip().rstrip("?.!")
    if ":" in body:
        body = body.split(":", 1)[1]
    elif "," in body:
        body = body.split(",", 1)[1]
    parts = re.split(r"\s+\bor\b\s+", body, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return []
    targets = [" ".join(_content_tokens(part)) for part in parts]
    return targets if all(targets) else []



def _quoted_targets(question: str) -> list[str]:
    values = re.findall(r"['\"]([^'\"]{3,})['\"]", question)
    return [
        " ".join(_content_tokens(value)) for value in values
        if _content_tokens(value)
    ]


def _sum_operand_targets(question: str) -> list[str]:
    """Extract an explicit coordinated object list for a sum query."""
    match = re.search(
        r"\b(?:cost\s+of|spent\s+on|spend\s+on|paid\s+for|pay\s+for)\s+"
        r"(.+?)(?:\s+I\s+(?:got|bought|purchased|ordered)|"
        r"\s+since\b|\s+during\b|\?|$)",
        question, re.IGNORECASE,
    )
    if match is None:
        return []
    body = match.group(1).strip(" ,")
    if "," in body:
        parts = [part.strip(" ,") for part in body.split(",") if part.strip(" ,")]
        parts = [re.sub(r"^(?:and|or)\s+", "", part, flags=re.IGNORECASE) for part in parts]
    else:
        parts = [part.strip() for part in re.split(r"\s+and\s+", body, maxsplit=1, flags=re.IGNORECASE)]
    targets = [target for part in parts if (target := " ".join(_content_tokens(part)))]
    return targets if len(targets) >= 2 else []


def _difference_targets(question: str) -> list[str]:
    match = re.search(
        r"\b(?:in|for)\s+(.+?)\s+compared\s+to\s+(.+?)(?:\?|$)",
        question, re.IGNORECASE,
    )
    if match is None:
        return []
    return [
        " ".join(_content_tokens(match.group(1))),
        " ".join(_content_tokens(match.group(2))),
    ]


def _relative_event_targets(question: str) -> list[str]:
    """Bind the requested event and an explicit before/after boundary.

    This is intentionally limited to date questions.  A bare relative marker is
    not enough: both sides must contain durable content terms, so ordinary uses
    such as "after work" do not manufacture a temporal comparison.
    """
    match = re.search(
        r"^\s*(?:when|what\s+date)\s+"
        r"(?:(?:did|does|was|were|has|have)\s+)?"
        r"(.+?)\s+\b(?:after|before)\b\s+(.+?)(?:[?.!]|$)",
        question, re.IGNORECASE,
    )
    if match is None:
        return []
    targets = [" ".join(_content_tokens(value)) for value in match.groups()]
    return targets if all(targets) else []


def _normalized_owner_candidate(value: str) -> str:
    candidate = canonical_key(
        re.sub(r"(?:['’]s)$", "", value.strip(), flags=re.IGNORECASE)
    )
    return "" if candidate in {
        "a", "all", "an", "any", "both", "either", "neither", "some", "the",
    } else candidate


def build_query_ir(question: str) -> QueryIR:
    content = _content_tokens(question)
    lowered = question.casefold()
    comparison_targets = _comparison_targets(question)
    relative_event_targets = _relative_event_targets(question)
    if not comparison_targets and relative_event_targets:
        comparison_targets = relative_event_targets
    sequence_targets = (
        _quoted_targets(question) if _TEMPORAL_SEQUENCE_RE.search(question)
        else []
    )
    operand_targets: list[str] = []
    aggregation_op = "none"
    temporal = list(dict.fromkeys(
        match.group(0).casefold() for match in _DATE_RE.finditer(question)
    ))
    frequency_count = bool(re.search(
        r"\bhow many\s+(?:days|times|classes|sessions)\s+"
        r"(?:a|per|each)\s+(?:week|month|year)\b",
        lowered,
    ))
    counterfactual = bool(re.search(
        r"\bif\b.{0,160}\b(?:hadn\x27t|didn\x27t|weren\x27t|wasn\x27t|had not|did not|were not|was not|never|without)\b",
        lowered,
    ))
    relative_age_difference = bool(re.search(
        r"\bhow many\s+(?:days|weeks|months|years)\s+(?:older|younger)\b",
        lowered,
    ))
    geographic_state_lookup = bool(re.search(
        r"\b(?:u\.?s\.?|united states|american)\s+state\b|"
        r"\bwhat\s+state\s+(?:did|does|is|was|were|has|have)\b.{0,80}"
        r"\b(?:visit|travel|live|reside|move|stay|meet|go|went)\b",
        lowered,
    ))
    if len(sequence_targets) >= 2:
        comparison_targets = sequence_targets
        value_type = "temporal_order"
        roles = ["events", "times", "source"]
    elif _TEMPORAL_SEQUENCE_RE.search(question):
        value_type = "temporal_order"
        roles = ["events", "times", "source"]
    elif _TEMPORAL_LATEST_RE.search(question):
        value_type = "temporal_order"
        roles = ["events", "times", "source"]
    elif _TEMPORAL_AFTER_FIRST_RE.search(question) or relative_event_targets:
        value_type = "temporal_order"
        roles = ["event_a", "event_b", "time_a", "time_b", "identity", "source"]
    elif comparison_targets:
        value_type = "temporal_order"
        roles = ["event_a", "event_b", "time_a", "time_b", "source"]
    elif _AVERAGE_RE.search(question):
        value_type = "aggregate"
        aggregation_op = "average"
        roles = ["quantity", "source"]
    elif _DIFFERENCE_RE.search(question):
        value_type = "aggregate"
        aggregation_op = "difference"
        operand_targets = _difference_targets(question)
        roles = ["quantity", "source"]
    elif relative_age_difference:
        value_type = "aggregate"
        aggregation_op = "difference"
        roles = ["quantity", "source"]
    elif _SUM_RE.search(question):
        value_type = "aggregate"
        aggregation_op = "sum"
        operand_targets = _sum_operand_targets(question)
        roles = ["quantity", "source"]
        if operand_targets:
            roles.insert(0, "components")
    elif frequency_count:
        value_type = "count"
        roles = ["scope", "members", "source"]
    elif _DURATION_RE.search(question):
        value_type = "duration"
        if re.search(
            r"\b(?:between|before|after|since|until|from\b.+\bto)\b|"
            r"\bhow long had\b.+\bwhen\b",
            lowered,
        ):
            roles = ["event_a", "event_b", "time_a", "time_b", "source"]
        else:
            roles = ["duration", "source"]
            if re.search(r"\b(?:and|combined|altogether|in total)\b", lowered):
                roles.insert(1, "components")
    elif _COUNT_RE.search(question):
        value_type = "count"
        roles = ["scope", "members", "source"]
    elif counterfactual:
        value_type = "boolean"
        roles = ["condition", "effect", "source"]
    elif _PREFERENCE_RE.search(question):
        value_type = "preference"
        roles = ["owner", "preference", "polarity", "context", "source"]
        if re.match(r"\s*what\b", lowered) and re.search(
            r"\b(?:like|love|prefer|enjoy)\b", lowered
        ):
            roles.insert(1, "members")
    elif _ORDINAL_RE.search(question) and _DIALOGUE_RE.search(question):
        value_type = "span"
        roles = ["scope", "ordered_items", "source"]
    elif _RECOMMENDATION_RE.search(question):
        value_type = "recommendation"
        roles = ["current_state", "context", "source"]
    elif geographic_state_lookup:
        # State is also a geographic answer type. Do not compile a question
        # such as "What state did X visit?" into lifecycle-chain navigation.
        value_type = "location"
        roles = ["event", "location", "source"]
    elif _STATE_RE.search(question) or _RECORD_STATE_RE.search(question):
        value_type = "state"
        roles = ["current_state", "time", "source"]
        if re.search(r"\b(?:changed|updated|switch(?:ed|ing)?|became|become|no longer)\b", lowered):
            roles.insert(0, "previous_state")
    elif temporal and re.match(r"\s*(?:which|who)\b", lowered):
        value_type = "entity"
        roles = ["entity", "event", "time", "source"]
    elif _LIST_RE.search(question):
        value_type = "list"
        roles = ["scope", "members", "source"]
    elif re.search(r"\bwhen\b|\bwhat date\b", lowered):
        value_type = "date"
        roles = ["event", "time", "source"]
    elif re.search(r"\bwho\b|\bwhich (?:person|place|organization)\b", lowered):
        value_type = "entity"
        roles = ["entity", "relation", "source"]
    elif re.match(r"\s*where\b", lowered) or re.search(
        r"\bwhere\s+(?:did|does|is|are|was|were|has|have|can|could|will|would)\b",
        lowered,
    ):
        value_type = "location"
        roles = (
            ["scope", "members", "event", "source"]
            if re.search(r"\bwhere\s+(?:has|have)\b", lowered)
            else ["event", "location", "source"]
        )
    elif re.match(r"\s*(?:did|does|is|are|was|were|has|have|can)\b", lowered):
        value_type = "boolean"
        roles = ["fact", "polarity", "source"]
    else:
        value_type = "span"
        roles = ["fact", "source"]
    if _DIALOGUE_RE.search(question):
        roles = list(dict.fromkeys([
            "prompt_turn", "reply_turn", "reply_content", "source", *roles
        ]))
    if _REFERENCE_RE.search(question):
        roles = list(dict.fromkeys([
            "reference", "identity", "source", *roles
        ]))
    relation = " ".join(content)
    # Keep case sensitivity on the captured name.  The former IGNORECASE
    # pattern accepted function words (notably "both") as people and retained
    # possessive suffixes such as ``Caroline's`` in the canonical owner key.
    owner_match = re.search(
        r"(?i:\b(?:did|does|is|was|has|have|can|could|would|will|should|"
        r"might|may)\s+)"
        r"([A-Z][\w'’-]+)\b",
        question,
    )
    target_owner = _normalized_owner_candidate(
        owner_match.group(1) if owner_match else ""
    )
    if not target_owner:
        relative_owner = re.search(
            r"(?i:\b(?:that|which|for)\s+)"
            r"([A-Z][\w'’-]+)\s+"
            r"(?i:might|may|would|could|can|will|should|to)\b",
            question,
        )
        if relative_owner:
            target_owner = _normalized_owner_candidate(relative_owner.group(1))
    if not target_owner:
        possessive_owner = re.search(
            r"\b([A-Z][\w'-]+?)(?:['’]s)\b", question,
        )
        if possessive_owner:
            target_owner = _normalized_owner_candidate(possessive_owner.group(1))
    return QueryIR(
        raw_question=question,
        target_entities=content,
        target_relation=relation,
        target_owner=target_owner,
        requested_value_type=value_type,  # type: ignore[arg-type]
        temporal_constraints=temporal,
        state_constraints=(
            ["latest_valid_state"] if value_type == "state" else []
        ),
        collection_constraints=(
            ["distinct", "complete_scope"]
            if value_type in {"count", "list"} or "members" in roles else []
        ),
        polarity="negative" if _NEGATIVE_RE.search(question) else "unknown",
        required_roles=roles,
        comparison_targets=comparison_targets,
        aggregation_op=aggregation_op,  # type: ignore[arg-type]
        operand_targets=operand_targets,
    )

def query_views(ir: QueryIR) -> list[str]:
    views = [ir.raw_question]
    compact = " | ".join(filter(None, (
        "entities " + " ".join(ir.target_entities),
        "relation " + ir.target_relation,
        "value " + ir.requested_value_type,
        "time " + " ".join(ir.temporal_constraints),
    )))
    if compact and compact != ir.raw_question:
        views.append(compact)
    target_terms, relation_terms = query_binding_terms(ir)
    if target_terms:
        views.append("target " + " ".join(sorted(target_terms)))
    relation_view = [*sorted(action_families(ir.raw_question)), *sorted(relation_terms)]
    if relation_view:
        views.append("relation " + " ".join(dict.fromkeys(relation_view)))
    return list(dict.fromkeys(views))


def _node_id(node: Any) -> str:
    return str(getattr(node, "node_id"))


def _node_text(node: Any) -> str:
    return str(
        getattr(node, "retrieval_text", None)
        or getattr(node, "routing_text", None)
        or getattr(node, "text", "")
    )


def _node_sessions(node: Any) -> list[str]:
    if hasattr(node, "session_ids"):
        return list(getattr(node, "session_ids"))
    session_id = getattr(node, "session_id", "")
    return [session_id] if session_id else []


def _dense_rank(
    nodes: Iterable[Any], query_vectors: list[list[float]], limit: int
) -> list[tuple[str, float]]:
    scored: list[tuple[str, float]] = []
    for node in nodes:
        embedding = getattr(node, "embedding", None)
        if embedding is None:
            continue
        score = max(
            cosine_similarity(vector, embedding) for vector in query_vectors
        )
        scored.append((_node_id(node), score))
    scored.sort(key=lambda row: (-row[1], row[0]))
    return scored[:limit]


def _bm25_rank(
    nodes: Iterable[Any], query: str, limit: int
) -> list[tuple[str, float]]:
    rows = list(nodes)
    if not rows:
        return []
    documents = [Counter(_tokens(_node_text(node))) for node in rows]
    query_terms = _content_tokens(query)
    document_frequency = Counter(
        term for document in documents for term in document
    )
    average_length = (
        sum(sum(document.values()) for document in documents) / len(documents)
    ) or 1.0
    scored: list[tuple[str, float]] = []
    for node, document in zip(rows, documents):
        length = sum(document.values()) or 1
        score = 0.0
        for term in query_terms:
            frequency = document.get(term, 0)
            if not frequency:
                continue
            inverse = math.log(
                1.0 + (len(rows) - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            score += inverse * (
                frequency * 2.2
                / (frequency + 1.2 * (0.25 + 0.75 * length / average_length))
            )
        if score:
            scored.append((_node_id(node), score))
    scored.sort(key=lambda row: (-row[1], row[0]))
    return scored[:limit]


def _exact_rank(nodes: Iterable[Any], ir: QueryIR) -> list[tuple[str, float]]:
    query_terms = set(ir.target_entities)
    rows = []
    for node in nodes:
        text_terms = set(_content_tokens(_node_text(node)))
        overlap = query_terms & text_terms
        if overlap:
            rows.append((_node_id(node), len(overlap) / max(1, len(query_terms))))
    rows.sort(key=lambda row: (-row[1], row[0]))
    return rows


def _rrf(
    rankings: dict[str, list[tuple[str, float]]], *, rrf_k: int = 60
) -> tuple[list[tuple[str, float]], dict[str, dict[str, int]]]:
    scores: dict[str, float] = defaultdict(float)
    traces: dict[str, dict[str, int]] = defaultdict(dict)
    for channel, ranking in rankings.items():
        for rank, (node_id, _score) in enumerate(ranking, start=1):
            scores[node_id] += 1.0 / (rrf_k + rank)
            traces[node_id][channel] = rank
    return (
        sorted(scores.items(), key=lambda row: (-row[1], row[0])),
        dict(traces),
    )


def _structured_keys(ir: QueryIR) -> set[str]:
    terms = ir.target_entities
    keys = set(terms)
    for width in range(2, min(5, len(terms)) + 1):
        keys.update(" ".join(terms[start:start + width]) for start in range(len(terms) - width + 1))
    keys.update(ir.temporal_constraints)
    if ir.target_owner:
        keys.add(ir.target_owner)
    if ir.polarity != "unknown":
        keys.add(ir.polarity)
    return {canonical_key(key) or key for key in keys if key}


def _structured_rank(index: V36Index, ir: QueryIR, *, allowed_ids: set[str] | None = None) -> list[tuple[str, float]]:
    scores: Counter[str] = Counter()
    query_keys = _structured_keys(ir)
    for values in index.inverted_indexes.values():
        for key in query_keys:
            for node_id in values.get(key, []):
                if allowed_ids is None or node_id in allowed_ids:
                    scores[node_id] += 1
    return sorted(((node_id, float(score)) for node_id, score in scores.items()), key=lambda row: (-row[1], row[0]))


def _project_ranking_to_cards(
    ranking: list[tuple[str, float]],
    node_session: dict[str, str],
    card_by_session: dict[str, str],
    limit: int = 24,
) -> list[tuple[str, float]]:
    projected: dict[str, float] = {}
    for node_id, score in ranking:
        card_id = card_by_session.get(node_session.get(node_id, ""))
        if card_id:
            projected[card_id] = max(projected.get(card_id, float("-inf")), score)
    return sorted(projected.items(), key=lambda row: (-row[1], row[0]))[:limit]


def _card_ranking(
    index: V36Index, ir: QueryIR, query_vectors: list[list[float]]
) -> tuple[list[tuple[str, float]], dict[str, dict[str, int]]]:
    cards = index.routing_cards
    node_session = {node.node_id: session for node in [*index.turns, *index.frames] for session in _node_sessions(node)}
    card_by_session = {card.session_id: card.card_id for card in cards}
    structured_scores: Counter[str] = Counter()
    for node_id, score in _structured_rank(index, ir):
        card_id = card_by_session.get(node_session.get(node_id, ""))
        if card_id:
            structured_scores[card_id] += score
    rankings = {
        "dense": _dense_rank(cards, query_vectors, 24),
        "bm25": _bm25_rank(cards, ir.raw_question, 24),
        "exact": _exact_rank(cards, ir)[:24],
        "structured": sorted(((card_id, float(score)) for card_id, score in structured_scores.items()), key=lambda row: (-row[1], row[0]))[:24],
        "lossless_dense": _project_ranking_to_cards(_dense_rank(index.turns, query_vectors, 80), node_session, card_by_session),
        "lossless_bm25": _project_ranking_to_cards(_bm25_rank(index.turns, ir.raw_question, 80), node_session, card_by_session),
        "lossless_exact": _project_ranking_to_cards(_exact_rank(index.turns, ir)[:80], node_session, card_by_session),
    }
    return _rrf(rankings)


def _adaptive_cards(
    ranked: list[tuple[str, float]], ir: QueryIR,
    channels: dict[str, dict[str, int]],
) -> list[str]:
    if not ranked:
        return []
    ambiguous = (
        len(ranked) > 1
        and ranked[0][1] - ranked[1][1] < 0.08 / 60
    )
    wide = (
        ir.requested_value_type in {
            "count", "list", "location", "duration", "state", "preference",
            "recommendation", "temporal_order", "aggregate", "span", "entity",
            "date",
        }
        or len(ir.temporal_constraints) > 1
    )
    limit = min(8 if (ambiguous or wide) else 4, len(ranked))
    fused_position = {node_id: position for position, (node_id, _score) in enumerate(ranked)}

    def replace_weakest(selected: list[str], candidate: str) -> None:
        if candidate in selected or not selected:
            return
        pool = selected[2:] or selected

        def strength(node_id: str) -> tuple[int, float, int]:
            ranks = channels.get(node_id, {})
            supported = sum(rank <= 12 for rank in ranks.values())
            reciprocal = sum(1.0 / max(1, rank) for rank in ranks.values())
            return (
                supported, reciprocal, -fused_position.get(node_id, 10**9)
            )

        victim = min(pool, key=strength)
        selected[selected.index(victim)] = candidate
    direct_semantic = (
        ir.requested_value_type in {"span", "date"}
        or (ir.requested_value_type == "entity" and bool(ir.temporal_constraints))
    )
    state_multiversion = (
        ir.requested_value_type == "state"
        and ("previous_state" in ir.required_roles or bool(_RECORD_STATE_RE.search(ir.raw_question)))
    )
    if direct_semantic:
        selected = [node_id for node_id, _score in ranked[:min(4, limit)]]
        # Semantic paraphrases and misspellings need actual rank depth rather
        # than a round-robin that spends all slots at rank one.
        for channel in ("dense", "lossless_dense", "bm25", "lossless_bm25"):
            for rank in range(1, 5):
                candidate = next((
                    node_id for node_id, row in channels.items()
                    if row.get(channel) == rank
                ), None)
                if candidate and candidate not in selected:
                    selected.append(candidate)
                    if len(selected) >= limit:
                        break
            if len(selected) >= limit:
                break
    elif state_multiversion:
        selected = [node_id for node_id, _score in ranked[:min(4, limit)]]
        for rank in range(1, 5):
            for channel in (
                "lossless_exact", "lossless_bm25", "lossless_dense",
                "structured", "exact", "bm25", "dense",
            ):
                candidate = next((
                    node_id for node_id, row in channels.items()
                    if row.get(channel) == rank
                ), None)
                if candidate and candidate not in selected:
                    selected.append(candidate)
                    if len(selected) >= limit:
                        break
            if len(selected) >= limit:
                break
    else:
        selected = [node_id for node_id, _score in ranked[:limit]]
        dense_rows = sorted(
            (ranks["dense"], node_id, ranks)
            for node_id, ranks in channels.items() if "dense" in ranks
        )
        if dense_rows:
            dense_rank, dense_id, dense_channels = dense_rows[0]
            if (
                dense_rank == 1
                and any(channel != "dense" for channel in dense_channels)
                and dense_id not in selected and selected
            ):
                selected[-1] = dense_id
        consensus = sorted(
            (
                max(ranks["lossless_dense"], ranks["lossless_bm25"]),
                ranks["lossless_dense"] + ranks["lossless_bm25"], node_id,
            )
            for node_id, ranks in channels.items()
            if ranks.get("lossless_dense", 10**9) <= 3
            and ranks.get("lossless_bm25", 10**9) <= 3
        )
        if consensus and selected:
            candidate = consensus[0][2]
            if ir.requested_value_type in {"count", "list"}:
                replace_weakest(selected, candidate)
            elif candidate not in selected:
                selected[-1] = candidate
        exact_rescue = sorted(
            (
                min(ranks.get("lossless_bm25", 10**9), ranks.get("lossless_dense", 10**9)),
                ranks.get("lossless_exact", 10**9), node_id,
            )
            for node_id, ranks in channels.items()
            if ranks.get("lossless_exact", 10**9) <= 2
            and min(ranks.get("lossless_bm25", 10**9), ranks.get("lossless_dense", 10**9)) <= 12
        )
        if exact_rescue and selected:
            candidate = exact_rescue[0][2]
            if ir.requested_value_type in {"count", "list"}:
                replace_weakest(selected, candidate)
            elif candidate not in selected:
                selected[-1] = candidate
    for node_id, _score in ranked:
        if len(selected) >= limit:
            break
        if node_id not in selected:
            selected.append(node_id)
    return list(dict.fromkeys(selected))


def _activity_query_keys(ir: QueryIR) -> set[str]:
    keys: set[str] = set()
    for word in _content_tokens(ir.target_relation):
        term = word
        if len(word) > 5 and word.endswith("ing"):
            term = word[:-3]
            if len(term) > 2 and term[-1] == term[-2]:
                term = term[:-1]
        elif len(word) > 4 and word.endswith("ed"):
            term = word[:-2]
        if len(term) >= 4:
            keys.add(term)
    return keys


def _collection_scope_cards(
    index: V36Index, ir: QueryIR, selected_card_ids: list[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    if "members" not in ir.required_roles:
        return selected_card_ids, []
    activity_keys = _activity_query_keys(ir)
    if not activity_keys:
        return selected_card_ids, []
    group_by_id = {group.group_id: group for group in index.evidence_groups}
    frame_by_id = {frame.frame_id: frame for frame in index.frames}
    card_by_session = {card.session_id: card.card_id for card in index.routing_cards}
    candidates: dict[str, tuple[str, EvidenceGroup]] = {}
    for edge in index.edges:
        if (
            edge.relation != "collection_member"
            or edge.provenance.get("local_rule")
            != "bounded_cross_session_activity_collection"
        ):
            continue
        activity_key = str(edge.provenance.get("activity_key") or "")
        group = group_by_id.get(edge.src)
        if activity_key not in activity_keys or group is None:
            continue
        owners = {
            frame_by_id[frame_id].owner_key
            for frame_id in group.member_frame_ids if frame_id in frame_by_id
        }
        if ir.target_owner and owners != {ir.target_owner}:
            continue
        if not group.provenance_complete or not all(group.completeness_mask.values()):
            continue
        candidates[group.group_id] = (activity_key, group)
    if not candidates:
        return selected_card_ids, []
    activity_key, group = max(
        candidates.values(),
        key=lambda item: (len(set(item[1].session_ids)), item[1].confidence, item[1].group_id),
    )
    group_cards = [
        card_by_session[session_id]
        for session_id in group.session_ids if session_id in card_by_session
    ]
    merged = list(dict.fromkeys([
        *selected_card_ids[:2], *group_cards, *selected_card_ids[2:],
    ]))[:8]
    return merged, [{
        "group_id": group.group_id, "activity_key": activity_key,
        "card_ids": group_cards, "reason": "complete_collection_scope",
    }]


def _semantic_card_extension(index: V36Index, selected_card_ids: list[str]) -> tuple[str | None, dict[str, Any] | None]:
    selected = set(selected_card_ids)
    candidates = []
    for edge in index.edges:
        if edge.relation != "semantic_neighbor":
            continue
        if edge.src in selected and edge.dst not in selected:
            candidates.append((edge.confidence, edge.dst, edge))
        elif edge.dst in selected and edge.src not in selected:
            candidates.append((edge.confidence, edge.src, edge))
    if not candidates:
        return None, None
    score, card_id, edge = max(candidates, key=lambda row: (row[0], row[1]))
    return card_id, {"edge_id": edge.edge_id, "relation": edge.relation, "card_id": card_id, "confidence": score, "protected": False, "effect": "add_one_routing_card_only"}


def _per_session_rank(
    nodes: list[Any], ranker: Any, *, top_per_session: int = 2,
) -> list[tuple[str, float]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for node in nodes:
        for session_id in _node_sessions(node):
            grouped[session_id].append(node)
    rows: dict[str, float] = {}
    for session_nodes in grouped.values():
        for node_id, score in ranker(session_nodes)[:top_per_session]:
            rows[node_id] = max(rows.get(node_id, float("-inf")), score)
    return sorted(rows.items(), key=lambda row: (-row[1], row[0]))


def _morph_relation_rank(nodes: list[Any], ir: QueryIR) -> list[tuple[str, float]]:
    if ir.requested_value_type != "location" or "members" in ir.required_roles:
        return []
    ignored = {"ago", "year", "years", "month", "months", "week", "weeks", "day", "days"}
    query_terms = {
        _certificate_term(term) for term in _content_tokens(ir.target_relation)
        if term not in ignored and term != ir.target_owner
    }
    rows: list[tuple[str, float]] = []
    for node in nodes:
        if not isinstance(node, (RoleFrameNode, EvidenceGroup)):
            continue
        text_terms = {
            _certificate_term(term)
            for term in _content_tokens(getattr(node, "retrieval_text", ""))
        }
        overlap = len(query_terms & text_terms)
        if overlap:
            rows.append((_node_id(node), overlap / max(1, len(query_terms))))
    return sorted(rows, key=lambda row: (-row[1], row[0]))


def _fine_rankings(
    index: V36Index,
    nodes: list[Any],
    ir: QueryIR,
    query_vectors: list[list[float]],
) -> tuple[list[tuple[str, float]], dict[str, dict[str, int]]]:
    allowed_ids = {_node_id(node) for node in nodes}
    return _rrf({
        "dense": _dense_rank(nodes, query_vectors, 80),
        "bm25": _bm25_rank(nodes, ir.raw_question, 80),
        "exact": _exact_rank(nodes, ir)[:80],
        "structured": _structured_rank(index, ir, allowed_ids=allowed_ids)[:80],
        "role_relation": _morph_relation_rank(nodes, ir)[:80],
        "dense_session": _per_session_rank(nodes, lambda values: _dense_rank(values, query_vectors, len(values))),
        "bm25_session": _per_session_rank(nodes, lambda values: _bm25_rank(values, ir.raw_question, len(values))),
        "exact_session": _per_session_rank(nodes, lambda values: _exact_rank(values, ir)),
    })


def _fine_evidence_lists(
    fine_ranked: list[tuple[str, float]],
    fine_channels: dict[str, dict[str, int]],
    frame_by_id: dict[str, RoleFrameNode],
    group_by_id: dict[str, EvidenceGroup],
    routed_sessions: set[str],
    ir: QueryIR,
) -> tuple[list[tuple[RoleFrameNode, float]], list[tuple[EvidenceGroup, float]]]:
    memory_ids = set(frame_by_id) | set(group_by_id)
    protected: set[str] = set()
    for channel in ("dense", "bm25", "exact", "structured", "role_relation", "dense_session", "bm25_session", "exact_session"):
        rows = sorted(
            (ranks[channel], node_id) for node_id, ranks in fine_channels.items()
            if channel in ranks and node_id in memory_ids
        )
        # The role-relation channel is only enabled for scalar relation
        # completion and is intentionally much narrower than the broad
        # lexical/dense channels. Preserve a few more of its candidates so a
        # complete cross-session reference group is not lost merely because
        # its two member statements use different surface forms.
        channel_limit = 4 if channel == "role_relation" else 2
        protected.update(node_id for _rank, node_id in rows[:channel_limit])
    # Up to two best memory units per routed session prevent an otherwise
    # useful coarse card from contributing only its route text, while keeping
    # the protection bounded independently of any application domain.
    local_channels = ("dense_session", "bm25_session", "exact_session")
    for session_id in sorted(routed_sessions):
        candidates = []
        for node_id, ranks in fine_channels.items():
            node = frame_by_id.get(node_id) or group_by_id.get(node_id)
            if node is None or session_id not in node.session_ids:
                continue
            local_rank = min((ranks[channel] for channel in local_channels if channel in ranks), default=10**9)
            global_rank = min(ranks.values(), default=10**9)
            candidates.append((local_rank, global_rank, node_id))
        if candidates:
            protected.update(
                node_id for _local, _global, node_id
                in sorted(candidates)[:2]
            )
    # A temporal comparison is only answerable when both named alternatives
    # survive fine retrieval. Protect at most two best matching memory units
    # per side; this is role binding, not topic-specific retrieval.
    for target in ir.comparison_targets[:2]:
        target_terms = set(_content_tokens(target))
        matches: list[tuple[float, int, str]] = []
        for node_id in memory_ids:
            node = frame_by_id.get(node_id) or group_by_id.get(node_id)
            if node is None or not set(node.session_ids) <= routed_sessions:
                continue
            node_terms = set(_content_tokens(_node_text(node)))
            overlap = len(target_terms & node_terms)
            if overlap:
                matches.append((
                    overlap / max(1, len(target_terms)),
                    -min(fine_channels.get(node_id, {}).values(), default=10**9),
                    node_id,
                ))
        protected.update(
            node_id for _overlap, _rank, node_id
            in sorted(matches, reverse=True)[:2]
        )
    order = {node_id: rank for rank, (node_id, _score) in enumerate(fine_ranked)}
    score = dict(fine_ranked)
    candidate_ids = set(fine_channels) | {node_id for node_id, _score in fine_ranked}
    frames = [frame for frame in frame_by_id.values() if frame.frame_id in candidate_ids]
    frames.sort(key=lambda frame: (frame.frame_id not in protected, order.get(frame.frame_id, 10**9), frame.frame_id))
    groups = [
        group for group in group_by_id.values()
        if group.group_id in candidate_ids
        and _group_query_compatible(ir, group, frame_by_id)
    ]
    def group_order(group: EvidenceGroup) -> tuple[Any, ...]:
        protected_group = (
            group.group_id in protected
            or bool(set(group.member_frame_ids) & protected)
        )
        role_relation_rank = min(
            [
                fine_channels.get(node_id, {}).get("role_relation", 10**9)
                for node_id in [group.group_id, *group.member_frame_ids]
            ],
            default=10**9,
        )
        fused_rank = order.get(
            group.group_id,
            min(
                (order.get(frame_id, 10**9) for frame_id in group.member_frame_ids),
                default=10**9,
            ),
        )
        return (not protected_group, role_relation_rank, fused_rank, group.group_id)

    groups.sort(key=group_order)
    return (
        [(frame, score.get(frame.frame_id, 0.0)) for frame in frames[:40]],
        [(group, score.get(group.group_id, max((score.get(frame_id, 0.0) for frame_id in group.member_frame_ids), default=0.0))) for group in groups[:24]],
    )


def _fine_scope_nodes(index: V36Index, routed_sessions: set[str]) -> list[Any]:
    return [
        *[frame for frame in index.frames if set(frame.session_ids) <= routed_sessions],
        *[group for group in index.evidence_groups if set(group.session_ids) <= routed_sessions],
        *[turn for turn in index.turns if turn.session_id in routed_sessions],
    ]


def _gap_card_id(
    card_ranked: list[tuple[str, float]],
    card_channels: dict[str, dict[str, int]],
    selected_ids: list[str],
    missing_roles: list[str],
) -> str:
    selected = set(selected_ids)
    candidates = [node_id for node_id, _score in card_ranked if node_id not in selected]
    if not candidates:
        return ""
    missing = set(missing_roles)
    if missing & {"relation", "reply_content", "members", "operations"}:
        channel_order = ("bm25", "structured", "dense", "exact")
    elif missing & {"entity", "owner", "identity", "reference"}:
        channel_order = ("structured", "exact", "bm25", "dense")
    elif missing & {"time", "time_a", "time_b", "event", "event_a", "event_b"}:
        channel_order = ("exact", "bm25", "dense", "structured")
    else:
        return candidates[0]
    fallback = len(card_ranked) + 100
    return min(
        candidates,
        key=lambda node_id: tuple(card_channels.get(node_id, {}).get(channel, fallback) for channel in channel_order),
    )


_GROUP_BINDING_STOP = {
    "different", "type", "types", "kind", "kinds", "all", "recent",
    "recently", "used", "use", "using", "have", "had", "did", "does",
    "many", "much", "total", "number", "list", "which", "what",
}


def _group_query_compatible(
    ir: QueryIR, group: EvidenceGroup,
    frame_by_id: dict[str, RoleFrameNode],
) -> bool:
    """Require structural groups to bind to the query relation before use.

    Group kind supplies an evidence role, but never proves that it is the role
    requested by this query.  This lexical/canonical guard is deliberately
    domain independent and prevents an unrelated collection or state chain
    from making a completeness certificate true.
    """
    if group.group_kind not in {
        "collection", "state_transition", "temporal_pair", "reference_chain",
    }:
        return True
    query_terms = {
        _certificate_term(term) for term in _content_tokens(ir.target_relation)
        if _certificate_term(term) not in _GROUP_BINDING_STOP
        and _certificate_term(term) != ir.target_owner
    }
    if not query_terms:
        return False
    group_text = " ".join([
        group.retrieval_text,
        *(
            " ".join((
                frame_by_id[frame_id].owner_key,
                frame_by_id[frame_id].entity_key,
                frame_by_id[frame_id].predicate_key,
                frame_by_id[frame_id].object_key,
                frame_by_id[frame_id].context_key,
            ))
            for frame_id in group.member_frame_ids if frame_id in frame_by_id
        ),
    ])
    group_terms = {
        _certificate_term(term) for term in _content_tokens(group_text)
    }
    overlap = query_terms & group_terms
    required = 1 if len(query_terms) <= 3 else 2
    return len(overlap) >= required


def _roles_for_frame(frame: RoleFrameNode) -> set[str]:
    roles = {"fact", "source", "relation"}
    if frame.context_key:
        roles.add("context")
    if frame.state_op in {"set", "add", "increment", "complete"} or frame.lifecycle_status in {"ongoing", "completed"}:
        roles.add("current_state")
    if frame.entity_key or frame.object_key:
        roles.add("entity")
    if frame.frame_kind == "event" or (
        frame.predicate_key not in {"", "said", "predicate"}
        and bool(frame.temporal.event_time or frame.temporal.start)
    ):
        roles.add("event")
    location_keys = {frame.predicate_key, *frame.semantic_type_keys}
    if any(
        marker in location_keys
        for marker in {"location", "origin", "destination", "place"}
    ):
        roles.add("location")
    if frame.frame_kind == "preference":
        roles.update({"owner", "preference", "polarity", "context"})
    if frame.frame_kind == "quantity":
        roles.add("quantity")
        if frame.quantity.unit.casefold() in {"second", "seconds", "minute", "minutes", "hour", "hours", "day", "days", "week", "weeks", "month", "months", "year", "years"}:
            roles.add("duration")
    if frame.temporal.event_time or frame.temporal.start:
        roles.add("time")
    if frame.polarity != "unknown":
        roles.add("polarity")
    return roles


def _certificate_term(word: str) -> str:
    value = word.casefold()
    if len(value) > 5 and value.endswith("ing"):
        value = value[:-3]
        if len(value) > 2 and value[-1] == value[-2]:
            value = value[:-1]
    elif len(value) > 4 and value.endswith("ed"):
        value = value[:-2]
        if value.endswith("v"):
            value += "e"
    elif len(value) > 4 and value.endswith("s"):
        value = value[:-1]
    return value


def _location_evidence_coherent(
    ir: QueryIR, frames: list[RoleFrameNode], groups: list[EvidenceGroup],
) -> bool:
    ignored = {"ago", "year", "years", "month", "months", "week", "weeks", "day", "days"}
    relation_terms = {
        _certificate_term(term) for term in _content_tokens(ir.target_relation)
        if term not in ignored and term != ir.target_owner
    }
    frame_by_id = {frame.frame_id: frame for frame in frames}

    def owner_matches(rows: list[RoleFrameNode]) -> bool:
        owners = {row.owner_key for row in rows if row.owner_key}
        return not ir.target_owner or owners == {ir.target_owner}

    def relation_matches(text: str) -> bool:
        text_terms = {_certificate_term(term) for term in _content_tokens(text)}
        return not relation_terms or bool(relation_terms & text_terms)

    for frame in frames:
        if (
            "location" in _roles_for_frame(frame)
            and owner_matches([frame])
            and relation_matches(frame.retrieval_text)
        ):
            return True
    for group in groups:
        if group.group_kind != "reference_chain":
            continue
        rows = [
            frame_by_id[frame_id] for frame_id in group.member_frame_ids
            if frame_id in frame_by_id
        ]
        if (
            len(rows) == len(group.member_frame_ids)
            and any("location" in _roles_for_frame(row) for row in rows)
            and owner_matches(rows)
            and relation_matches(group.retrieval_text)
            and group.provenance_complete
        ):
            return True
    return False


def _certificate(
    ir: QueryIR,
    selected_frames: list[RoleFrameNode],
    selected_groups: list[EvidenceGroup],
    *,
    routed_sessions: set[str],
    excluded: list[str],
    expansion_rounds: int,
) -> CompletenessCertificate:
    present: set[str] = set()
    query_content = set(ir.target_entities)
    for frame in selected_frames:
        present.update(_roles_for_frame(frame))
    duration_component_count = sum("duration" in _roles_for_frame(frame) for frame in selected_frames)
    if duration_component_count >= 2:
        present.add("components")
    frame_by_id = {frame.frame_id: frame for frame in selected_frames}
    bound_groups = [
        group for group in selected_groups
        if _group_query_compatible(ir, group, frame_by_id)
    ]
    for group in bound_groups:
        present.update(
            role for role, available in group.completeness_mask.items()
            if available
        )
        if group.group_kind == "dialogue_pair":
            present.update({"prompt_turn", "reply_turn", "reply_content"})
            if (
                "reference" in ir.required_roles
                and len(query_content & set(_content_tokens(group.retrieval_text))) >= 2
            ):
                present.update({"reference", "identity"})
        elif group.group_kind == "reference_chain":
            present.update({"reference", "identity"})
    query_terms = ({ir.target_owner} if ir.target_owner else set(ir.target_entities)) - {""}
    selected_text = " ".join([
        *[frame.retrieval_text for frame in selected_frames],
        *[group.retrieval_text for group in bound_groups],
    ])
    selected_tokens = set(_content_tokens(selected_text))
    comparison_support: list[bool] = []
    if ir.comparison_targets:
        for target in ir.comparison_targets:
            terms = set(_content_tokens(target))
            required = 1 if len(terms) <= 1 else 2
            comparison_support.append(
                len(terms & selected_tokens) >= min(required, len(terms))
            )
        entity_match = (
            len(comparison_support) == len(ir.comparison_targets)
            and all(comparison_support)
        )
        # Mentions establish candidate identity, not occurrence or time. Only
        # the action-bound temporal operator may certify endpoint roles.
        for role in {
            "event_a", "event_b", "time_a", "time_b", "events", "times",
        }:
            present.discard(role)
    else:
        entity_match = not query_terms or bool(query_terms & selected_tokens)
    relation_terms = set(_content_tokens(ir.target_relation)) - query_terms
    relation_match = not relation_terms or bool(relation_terms & selected_tokens)
    provenance = all(
        frame.source_turn_ids for frame in selected_frames
    ) and all(group.provenance_complete for group in selected_groups)
    scope_match = (
        all(not set(frame.session_ids) or set(frame.session_ids) <= routed_sessions for frame in selected_frames)
        and all(not set(group.session_ids) or set(group.session_ids) <= routed_sessions for group in selected_groups)
    )
    missing = [role for role in ir.required_roles if role not in present]
    if (
        ir.requested_value_type == "location"
        and "members" not in ir.required_roles
        and not _location_evidence_coherent(ir, selected_frames, selected_groups)
        and "location" not in missing
    ):
        missing.append("location")
    if not entity_match and "entity" not in missing:
        missing.append("entity")
    if not relation_match and "relation" not in missing:
        missing.append("relation")
    if not scope_match and "routing_scope" not in missing:
        missing.append("routing_scope")
    if not provenance and "source" not in missing:
        missing.append("source")
    complete = not missing
    return CompletenessCertificate(
        entity_match=entity_match,
        relation_match=relation_match,
        scope_match=scope_match,
        provenance_complete=provenance,
        present_roles=sorted(present),
        missing_roles=missing,
        excluded_near_matches=excluded,
        complete=complete,
        expansion_rounds=expansion_rounds,
    )


def _relations_for_missing(missing: set[str]) -> set[str]:
    allowed = {"source"}
    if _navigation_repair_enabled() and missing & {"fact", "event", "condition", "effect"}:
        allowed.add("dialogue_pair")
    if _navigation_repair_enabled() and missing & {"condition", "effect"}:
        allowed.update({"reference", "same_event"})
    if missing & {"prompt_turn", "reply_turn", "reply_content"}:
        allowed.update({"dialogue_pair", "next_turn"})
    if missing & {"reference", "identity", "location"}:
        allowed.update({"reference", "same_event"})
    if missing & {"previous_state", "current_state"}:
        allowed.add("state_transition")
    if missing & {"scope", "members", "operations", "components"}:
        allowed.add("collection_member")
    if missing & {"event_a", "event_b", "time_a", "time_b", "time"}:
        allowed.update({"temporal_endpoint", "same_event"})
    if missing & {"positive", "negative", "polarity"}:
        allowed.add("contrast")
    return allowed


def _typed_expand(
    index: V36Index,
    seed_ids: set[str],
    missing_roles: set[str],
    *,
    max_depth: int = 2,
) -> tuple[set[str], list[dict[str, Any]]]:
    allowed = _relations_for_missing(missing_roles)
    adjacency: dict[str, list[Any]] = defaultdict(list)
    for edge in index.edges:
        if edge.relation not in allowed:
            continue
        adjacency[edge.src].append(edge)
        # Reliable structural relations can be inspected from either member, but
        # the stored edge remains directed and routing_contains is never reversed.
        if edge.relation != "source" or _navigation_repair_enabled():
            # Experimental inverse provenance projection: a lossless turn hit may
            # enter its source-bound frame before following typed relations.
            adjacency[edge.dst].append(edge)
    reached = set(seed_ids)
    frontier = set(seed_ids)
    trace: list[dict[str, Any]] = []
    for depth in range(max_depth):
        following: set[str] = set()
        for node_id in sorted(frontier):
            for edge in sorted(
                adjacency.get(node_id, []),
                key=lambda value: (-value.confidence, value.edge_id),
            ):
                other = edge.dst if edge.src == node_id else edge.src
                if other in reached:
                    continue
                reached.add(other)
                following.add(other)
                trace.append({
                    "edge_id": edge.edge_id,
                    "relation": edge.relation,
                    "from": node_id,
                    "to": other,
                    "depth": depth + 1,
                    "confidence": edge.confidence,
                })
        frontier = following
        if not frontier:
            break
    return reached, trace


def _format_card(card: RoutingCard) -> str:
    return f"[ROUTE {card.card_id} session={card.session_id}]\n{card.routing_text}"


def _format_group(
    group: EvidenceGroup,
    frame_by_id: dict[str, RoleFrameNode],
    turn_by_id: dict[str, TurnNodeV36],
) -> str:
    lines = [
        f"[EVIDENCE_GROUP {group.group_id} kind={group.group_kind} "
        f"complete={all(group.completeness_mask.values())}]"
    ]
    for frame_id in group.member_frame_ids:
        frame = frame_by_id[frame_id]
        time_value = (
            frame.temporal.event_time or frame.temporal.start
            or frame.temporal.observed_at or "unknown"
        )
        lines.append(
            f"FRAME {frame.frame_id}: kind={frame.frame_kind}; "
            f"owner={frame.owner_key}; entity={frame.entity_key}; "
            f"predicate={frame.predicate_key}; object={frame.object_key}; "
            f"polarity={frame.polarity}; modality={frame.modality}; "
            f"status={frame.lifecycle_status}; op={frame.state_op}; "
            f"time={time_value}"
        )
    for source in group.source_turn_ids:
        turn = turn_by_id.get(source)
        if turn is not None:
            lines.append(
                f"SOURCE {source} ({turn.session_date or 'unknown'}, "
                f"{turn.speaker}): {turn.text}"
            )
    return "\n".join(lines)


def _focused_turn_text(ir: QueryIR, turn: TurnNodeV36, max_chars: int = 1800) -> str:
    ordinal = re.search(
        r"\b(\d+)(?:st|nd|rd|th)\b|"
        r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b",
        ir.raw_question, re.IGNORECASE,
    )
    if ordinal:
        ordinal_words = {
            "first": "1", "second": "2", "third": "3", "fourth": "4",
            "fifth": "5", "sixth": "6", "seventh": "7", "eighth": "8",
            "ninth": "9", "tenth": "10",
        }
        number = ordinal.group(1) or ordinal_words[ordinal.group(2).casefold()]
        lines = turn.text.splitlines()
        for index, line in enumerate(lines):
            if re.match(rf"\s*(?:[*#-]+\s*)?{number}(?:[.)]|\s*[-:])", line):
                return "\n".join(lines[max(0, index - 1):index + 2])[:max_chars]
    anchor = re.search(r"\bafter\s+(.+?)(?:\?|\Z)", ir.raw_question, re.IGNORECASE)
    if anchor:
        needle = anchor.group(1).strip().casefold()
        position = turn.text.casefold().find(needle)
        if position >= 0:
            return turn.text[max(0, position - 200):position + len(needle) + 500][:max_chars]
    segments = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+|\n+", turn.text) if segment.strip()]
    if not segments:
        return turn.text[:max_chars]
    terms = set(ir.target_entities)
    scored = []
    for index, segment in enumerate(segments):
        tokens = set(_content_tokens(segment))
        overlap = len(tokens & terms)
        phrase_bonus = int(bool(ir.target_relation) and ir.target_relation in " ".join(_content_tokens(segment)))
        scored.append((overlap + phrase_bonus, index))
    anchors = [index for score, index in sorted(scored, key=lambda row: (-row[0], row[1])) if score > 0][:4]
    if not anchors:
        return turn.text[:max_chars]
    selected: set[int] = set(anchors)
    for index in anchors:
        if segments[index].lstrip().startswith("|"):
            start = index
            while start > 0 and segments[start - 1].lstrip().startswith("|"):
                start -= 1
            end = index
            while end + 1 < len(segments) and segments[end + 1].lstrip().startswith("|"):
                end += 1
            # A selected table row is meaningless without its column header.
            # Preserve the header and separator plus the matched row; this is
            # generic matrix evidence packing, not a domain-specific parser.
            selected.update({start, min(start + 1, end), index})
    text = "\n".join(segments[index] for index in sorted(selected))
    return text[:max_chars]


def _legacy_ranked_source_turns(
    index: V36Index, ir: QueryIR, query_vectors: list[list[float]],
    selected_cards: list[RoutingCard], fine_ranked: list[tuple[str, float]],
) -> list[tuple[TurnNodeV36, float]]:
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    fine_scores = dict(fine_ranked)
    chosen: list[str] = []
    global_chosen: list[str] = []
    per_card_lists: list[list[str]] = []
    global_fused, global_channels = _rrf({
        "exact": _exact_rank(index.turns, ir),
        "bm25": _bm25_rank(index.turns, ir.raw_question, 80),
        "dense": _dense_rank(index.turns, query_vectors, 80),
    }, rrf_k=10)
    global_limit = (
        8 if ir.requested_value_type in {
            "count", "list", "aggregate", "duration", "recommendation",
            "preference", "temporal_order",
        } else 4
    )
    for node_id, _score in global_fused:
        ranks = global_channels.get(node_id, {})
        high_confidence = (
            len(ranks) >= 2
            or ranks.get("exact", 10**9) <= 3
            or ranks.get("bm25", 10**9) <= 3
            or (
                ir.requested_value_type in {"span", "date"}
                and ranks.get("dense", 10**9) <= 3
            )
        )
        if (
            high_confidence
            and node_id in turn_by_id
            and node_id not in global_chosen
        ):
            global_chosen.append(node_id)
        if len(global_chosen) >= global_limit:
            break
    coverage_first = ir.requested_value_type in {
        "count", "list", "aggregate", "duration", "recommendation",
        "temporal_order",
    }
    for card in selected_cards:
        session_turns = [turn for turn in index.turns if turn.session_id == card.session_id]
        fine_per_card = [
            node_id
            for node_id, _score in fine_ranked
            if node_id in turn_by_id
            and turn_by_id[node_id].session_id == card.session_id
        ][:6]
        fused, _channels = _rrf({
            "exact": _exact_rank(session_turns, ir),
            "bm25": _bm25_rank(session_turns, ir.raw_question, len(session_turns)),
            "dense": _dense_rank(session_turns, query_vectors, len(session_turns)),
        }, rrf_k=10)
        # Structured questions need broad lossless coverage. Direct state,
        # entity and ordinal questions preserve the fine role-frame/source
        # projection so lexical neighbors cannot displace the exact proposition.
        fused_ids = [node_id for node_id, _score in fused]
        if (
            coverage_first
            and re.search(r"\b(?:i|me|my)\b", ir.raw_question, re.IGNORECASE)
        ):
            # First-person collection questions are assertions about the
            # questioner. Preserve a user-owned source from each routed region
            # before an adjacent assistant echo or explanation.
            user_ids = [
                node_id for node_id in fused_ids
                if turn_by_id[node_id].transport_role == "user"
            ]
            query_terms = set(_content_tokens(ir.raw_question))
            entity_phrases = [
                entity.casefold().strip()
                for entity in card.canonical_entities
                if len(_content_tokens(entity)) >= 1
                and not re.fullmatch(
                    r"(?:participant|speaker|questioner|assistant|user)[ _-]*\d*",
                    entity.casefold().strip(),
                )
            ]
            user_ids.sort(key=lambda node_id: (
                -4 * len(
                    query_terms
                    & set(_content_tokens(turn_by_id[node_id].text))
                )
                -3 * sum(
                    phrase in turn_by_id[node_id].text.casefold()
                    for phrase in entity_phrases
                ),
                fused_ids.index(node_id),
            ))
            fused_ids = [*user_ids, *[
                node_id for node_id in fused_ids if node_id not in user_ids
            ]]
        primary = fused_ids[:3] if coverage_first else fine_per_card[:3]
        fallback = (
            fine_per_card if coverage_first
            else [node_id for node_id, _score in fused]
        )
        per_card = list(primary)
        for node_id in fallback:
            if len(per_card) >= 3:
                break
            if node_id not in per_card:
                per_card.append(node_id)
        per_card_lists.append(per_card)
    if (
        ir.requested_value_type == "span"
        and (
            _ORDINAL_RE.search(ir.raw_question)
            or _DIALOGUE_RE.search(ir.raw_question)
            or re.search(
                r"\b(?:previous|initial|initially|first|next|immediately after)\b",
                ir.raw_question, re.IGNORECASE,
            )
        )
    ):
        # Ordinal/previous-dialogue questions need the local turn sequence, not
        # isolated semantically similar replies. Keep chronological adjacency
        # for the two strongest routed regions; the packer still enforces budget.
        for card in selected_cards[:2]:
            for turn in sorted(
                (item for item in index.turns if item.session_id == card.session_id),
                key=lambda item: item.turn_index,
            ):
                if turn.node_id not in chosen:
                    chosen.append(turn.node_id)
    if coverage_first:
        # First cover every routed region once, then interleave global
        # high-confidence evidence before taking second and third local turns.
        # This preserves coarse-to-fine coverage without starving an exact leaf
        # whose card narrowly missed the routing cap.
        for per_card in per_card_lists:
            if per_card and per_card[0] not in chosen:
                chosen.append(per_card[0])
        for node_id in global_chosen:
            if node_id not in chosen:
                chosen.append(node_id)
        for offset in range(1, 3):
            for per_card in per_card_lists:
                if offset < len(per_card) and per_card[offset] not in chosen:
                    chosen.append(per_card[offset])
    else:
        for node_id in global_chosen:
            if node_id not in chosen:
                chosen.append(node_id)
        # Direct lookup follows card rank and local proposition order.
        for per_card in per_card_lists:
            for node_id in per_card:
                if node_id not in chosen:
                    chosen.append(node_id)
    for node_id, _score in fine_ranked:
        if node_id in turn_by_id and node_id not in chosen:
            chosen.append(node_id)
        if len(chosen) >= 24:
            break
    return [
        (turn_by_id[node_id], fine_scores.get(node_id, 0.0))
        for node_id in chosen if node_id in turn_by_id
    ]


def _ranked_source_turns(
    index: V36Index, ir: QueryIR, query_vectors: list[list[float]],
    selected_cards: list[RoutingCard], fine_ranked: list[tuple[str, float]],
) -> list[tuple[TurnNodeV36, float]]:
    """Rank lossless turns globally, then expand only local dialogue neighbors.

    Routing cards constrain and boost the search region, but they do not force a
    fixed per-card quota.  This prevents weak cards from displacing a decisive
    source turn while still allowing a strong exact/BM25/dense hit outside the
    initial coarse scope to rescue routing.
    """
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    frame_by_id = {frame.frame_id: frame for frame in index.frames}
    group_by_id = {group.group_id: group for group in index.evidence_groups}
    routed_rank = {
        card.session_id: rank for rank, card in enumerate(selected_cards)
    }
    global_fused, channels = _rrf({
        "exact": _exact_rank(index.turns, ir),
        "bm25": _bm25_rank(index.turns, ir.raw_question, 120),
        "dense": _dense_rank(index.turns, query_vectors, 120),
    }, rrf_k=10)
    fused_score = dict(global_fused)

    # Project high-ranked frames/groups back to their lossless sources.  The
    # projection is bounded and only affects ranking; source turns remain the
    # evidence ultimately shown to the answer model.
    source_projection: dict[str, float] = defaultdict(float)
    for position, (node_id, score) in enumerate(fine_ranked[:80]):
        node = frame_by_id.get(node_id) or group_by_id.get(node_id)
        sources = list(getattr(node, "source_turn_ids", []) or [])
        if node_id in turn_by_id:
            sources.append(node_id)
        boost = max(0.25, 3.0 - position / 32.0) + 20.0 * score
        for source_id in sources:
            source_projection[source_id] = max(
                source_projection[source_id], boost
            )

    query_terms = {
        _certificate_term(token) for token in _content_tokens(ir.raw_question)
    }
    relation_terms = {
        _certificate_term(token) for token in _content_tokens(ir.target_relation)
    }
    first_person = bool(re.search(
        r"\b(?:i|me|my|mine)\b", ir.raw_question, re.IGNORECASE
    ))
    dialogue_lookup = bool(
        _DIALOGUE_RE.search(ir.raw_question)
        or _ORDINAL_RE.search(ir.raw_question)
        or re.search(
            r"\b(?:previous chat|earlier chat|you provided|you gave|you created)\b",
            ir.raw_question, re.IGNORECASE,
        )
    )
    temporal_lookup = ir.requested_value_type in {
        "date", "duration", "temporal_order", "state",
    }
    structured_lookup = ir.requested_value_type in {
        "count", "list", "aggregate", "duration", "temporal_order",
        "state", "preference", "recommendation",
    }

    candidate_ids = [node_id for node_id, _score in global_fused]
    candidate_ids.extend(
        source_id for source_id in source_projection
        if source_id not in fused_score
    )
    scored: list[tuple[float, TurnNodeV36]] = []
    for node_id in dict.fromkeys(candidate_ids):
        turn = turn_by_id.get(node_id)
        if turn is None:
            continue
        ranks = channels.get(node_id, {})
        routed = turn.session_id in routed_rank
        if not routed and not (
            len(ranks) >= 2
            or ranks.get("exact", 10**9) <= 4
            or ranks.get("bm25", 10**9) <= 4
            or source_projection.get(node_id, 0.0) >= 2.0
        ):
            continue
        turn_terms = {
            _certificate_term(token)
            for token in _content_tokens(
                f"{turn.speaker} {turn.listener} {turn.text}"
            )
        }
        overlap = len(query_terms & turn_terms)
        relation_overlap = len(relation_terms & turn_terms)
        score = 100.0 * fused_score.get(node_id, 0.0)
        score += source_projection.get(node_id, 0.0)
        score += 1.4 * overlap + 0.5 * relation_overlap
        score += 0.8 * sum(rank <= 8 for rank in ranks.values())
        if routed:
            score += max(0.25, 2.0 - 0.25 * routed_rank[turn.session_id])
        else:
            score -= 1.5
        if ir.target_owner:
            if turn.speaker_key == ir.target_owner:
                score += 5.0
            elif canonical_key(turn.listener) == ir.target_owner:
                score += 0.5
            elif ir.target_owner in turn_terms:
                score += 2.0
            else:
                score -= 1.0
        elif first_person and turn.transport_role == "user":
            score += 1.5
        if dialogue_lookup and turn.transport_role == "assistant":
            score += 1.0
        if temporal_lookup and (
            turn.session_date
            or re.search(
                r"\b(?:today|yesterday|tomorrow|ago|last|next|since|before|after|"
                r"january|february|march|april|may|june|july|august|september|"
                r"october|november|december|19\d{2}|20\d{2})\b",
                turn.text, re.IGNORECASE,
            )
        ):
            score += 0.75
        scored.append((score, turn))
    scored.sort(key=lambda row: (-row[0], row[1].session_id, row[1].turn_index))

    # Temporal comparisons need one independently matched anchor per named side.
    protected_ids: list[str] = []
    for target in ir.comparison_targets[:2]:
        target_terms = {
            _certificate_term(token) for token in _content_tokens(target)
        }
        matches = [
            (len(target_terms & {
                _certificate_term(token)
                for token in _content_tokens(turn.text)
            }), score, turn.node_id)
            for score, turn in scored
        ]
        matches = [row for row in matches if row[0] > 0]
        if matches:
            protected_ids.append(max(matches)[2])

    anchor_limit = 12 if structured_lookup else 8
    per_session_limit = 4 if structured_lookup else 3
    anchors: list[TurnNodeV36] = []
    per_session: Counter[str] = Counter()
    by_scored_id = {turn.node_id: (score, turn) for score, turn in scored}
    for node_id in protected_ids:
        row = by_scored_id.get(node_id)
        if row and node_id not in {turn.node_id for turn in anchors}:
            anchors.append(row[1])
            per_session[row[1].session_id] += 1
    for score, turn in scored:
        if len(anchors) >= anchor_limit:
            break
        if turn.node_id in {item.node_id for item in anchors}:
            continue
        if per_session[turn.session_id] >= per_session_limit:
            continue
        anchors.append(turn)
        per_session[turn.session_id] += 1

    session_turns: dict[str, list[TurnNodeV36]] = defaultdict(list)
    for turn in index.turns:
        session_turns[turn.session_id].append(turn)
    for rows in session_turns.values():
        rows.sort(key=lambda item: item.turn_index)
    position = {
        turn.node_id: index
        for rows in session_turns.values()
        for index, turn in enumerate(rows)
    }
    expand_neighbors = (
        dialogue_lookup
        or ir.requested_value_type in {
            "entity", "location", "date", "span", "preference",
            "recommendation", "state", "temporal_order",
        }
    )
    chosen: list[tuple[TurnNodeV36, float]] = []
    chosen_ids: set[str] = set()
    for anchor in anchors:
        anchor_score = by_scored_id.get(anchor.node_id, (0.0, anchor))[0]
        if anchor.node_id not in chosen_ids:
            chosen.append((anchor, anchor_score))
            chosen_ids.add(anchor.node_id)
        if not expand_neighbors:
            continue
        rows = session_turns[anchor.session_id]
        center = position[anchor.node_id]
        # Preserve the prompt/reply pair and one continuation turn.  This is
        # bounded local expansion, never a participant/theme hub traversal.
        offsets = (-1, 1, 2) if dialogue_lookup else (-1, 1)
        for offset in offsets:
            neighbor_index = center + offset
            if not 0 <= neighbor_index < len(rows):
                continue
            neighbor = rows[neighbor_index]
            if neighbor.node_id in chosen_ids:
                continue
            chosen.append((neighbor, anchor_score - 0.25 * abs(offset)))
            chosen_ids.add(neighbor.node_id)
            if len(chosen) >= 24:
                break
        if len(chosen) >= 24:
            break
    # Fuse the proposition-focused rank with the previous routed-coverage rank.
    # This is a generic rank ensemble: it preserves strong global lossless hits
    # while restoring one-per-region coverage for collections and multi-session facts.
    legacy_ranked = _legacy_ranked_source_turns(
        index, ir, query_vectors, selected_cards, fine_ranked,
    )
    combined = list(chosen[:8])
    remaining_new = list(chosen[8:])
    remaining_legacy = [
        row for row in legacy_ranked
        if row[0].node_id not in {item[0].node_id for item in combined}
    ]
    while len(combined) < 24 and (remaining_legacy or remaining_new):
        if remaining_legacy:
            row = remaining_legacy.pop(0)
            if row[0].node_id not in {item[0].node_id for item in combined}:
                combined.append(row)
        if len(combined) >= 24:
            break
        if remaining_new:
            row = remaining_new.pop(0)
            if row[0].node_id not in {item[0].node_id for item in combined}:
                combined.append(row)
    return combined[:24]


_AGGREGATE_RANK_STOP = {
    "average", "mean", "total", "amount", "much", "more", "less",
    "combined", "altogether", "difference", "compared", "what", "how",
    "did", "does", "do", "is", "are", "was", "were", "the", "a", "an",
    "of", "on", "in", "for", "to", "my", "me", "i",
    "since", "start", "year", "years", "have", "has",
}


def _priority_frame_ids(
    ir: QueryIR,
    ranked_frames: list[tuple[RoleFrameNode, float]],
    turn_by_id: dict[str, TurnNodeV36],
) -> list[str]:
    """Protect query-bound operands and state versions from pack eviction."""
    if ir.requested_value_type not in {"aggregate", "duration", "state", "count", "list"}:
        return []
    query_terms = {
        (
            _certificate_term(token)
            if ir.requested_value_type in {"aggregate", "count", "list"}
            else token
        )
        for token in _content_tokens(ir.raw_question.replace("-", " "))
        if token not in _AGGREGATE_RANK_STOP
        and (
            ir.requested_value_type not in {"count", "list"}
            or _certificate_term(token) not in _GROUP_BINDING_STOP
        )
    }
    time_units = {
        "second", "seconds", "minute", "minutes", "hour", "hours",
        "day", "days", "week", "weeks", "month", "months", "year", "years",
    }
    rows: list[tuple[int, float, str, str]] = []
    first_person = bool(re.search(r"\b(?:i|me|my)\b", ir.raw_question, re.IGNORECASE))
    for frame, fused_score in ranked_frames:
        if (
            frame.lifecycle_status in {"planned", "proposed", "cancelled"}
            or frame.polarity == "negative"
            or not frame.source_turn_ids
        ):
            continue
        if ir.requested_value_type in {"aggregate", "duration"} and frame.quantity.value is None:
            continue
        if (
            ir.requested_value_type == "duration"
            and frame.quantity.unit.casefold() not in time_units
        ):
            continue
        if ir.requested_value_type == "state" and frame.frame_kind not in {"state", "quantity", "fact"}:
            continue
        source_text = " ".join(
            turn_by_id[source].text
            for source in frame.source_turn_ids
            if source in turn_by_id
        )
        evidence_text = " ".join((
            frame.owner_key, frame.entity_key, frame.predicate_key,
            frame.object_key, frame.context_key, source_text,
        ))
        evidence_terms = {
            (_certificate_term(token) if ir.requested_value_type in {"aggregate", "count", "list"} else token)
            for token in _content_tokens(evidence_text.replace("-", " "))
        }
        structured_terms = {
            (_certificate_term(token) if ir.requested_value_type in {"aggregate", "count", "list"} else token)
            for token in _content_tokens(" ".join((
                frame.entity_key, frame.predicate_key, frame.object_key,
                frame.context_key,
            )))
        }
        structured_overlap = len(query_terms & structured_terms)
        if ir.requested_value_type == "state" and structured_overlap == 0:
            continue
        overlap = len(query_terms & evidence_terms) + structured_overlap
        if ir.target_owner:
            if frame.owner_key != ir.target_owner:
                continue
            overlap += 2
        elif first_person and any(
            turn_by_id[source].transport_role == "user"
            for source in frame.source_turn_ids if source in turn_by_id
        ):
            overlap += 2
        operand_bonus = max((
            len({_certificate_term(token) for token in _content_tokens(target)} & evidence_terms)
            for target in ir.operand_targets
        ), default=0)
        rows.append((
            overlap + operand_bonus, fused_score, frame.frame_id,
            frame.quantity.unit.casefold().rstrip("s"),
        ))
    if not rows:
        return []
    rows.sort(key=lambda row: (-row[0], -row[1], row[2]))
    best = rows[0][0]
    threshold = max(
        1, best - (2 if ir.aggregation_op == "sum" else 1)
    )
    candidates = [row for row in rows if row[0] >= threshold]
    frame_by_id = {frame.frame_id: frame for frame, _score in ranked_frames}
    if ir.requested_value_type in {"count", "list"}:
        eligible = [row for row in rows if row[0] >= 2]
        best_by_session: dict[str, tuple[int, float, str, str]] = {}
        for row in eligible:
            sessions = frame_by_id[row[2]].session_ids or [""]
            for session_id in sessions:
                prior = best_by_session.get(session_id)
                if prior is None or row[:2] > prior[:2]:
                    best_by_session[session_id] = row
        candidates = sorted(
            set(best_by_session.values()),
            key=lambda row: (-row[0], -row[1], row[2]),
        )
        candidates.extend(
            row for row in eligible
            if row[0] >= max(2, best - 2) and row not in candidates
        )
    if ir.aggregation_op == "sum":
        asks_money = bool(re.search(
            r"\b(?:money|spend|spent|pay|paid|cost|expense|expenses|"
            r"usd|dollars?|euros?|pounds?|yen)\b|[$€£¥]",
            ir.raw_question, re.IGNORECASE,
        ))
        candidates = [
            row for row in rows
            if row[0] >= 1
            and frame_by_id[row[2]].lifecycle_status == "completed"
            and (
                not asks_money
                or bool(re.search(
                    r"(?:[$€£¥]|usd|eur|gbp|jpy|dollar|euro|pound|yen)",
                    frame_by_id[row[2]].quantity.unit, re.IGNORECASE,
                ))
            )
        ]
    if ir.requested_value_type == "duration":
        completed = [
            row for row in rows
            if frame_by_id[row[2]].lifecycle_status == "completed"
        ]
        if completed:
            completed_best = max(row[0] for row in completed)
            protected_completed = [
                row for row in completed
                if row[0] >= max(1, completed_best - 2)
            ]
            candidates = list(dict.fromkeys([
                *protected_completed, *candidates,
            ]))
    if ir.requested_value_type == "duration":
        quoted_targets = [
            match[0] or match[1]
            for match in re.findall(r"'([^']+)'|\"([^\"]+)\"", ir.raw_question)
            if (match[0] or match[1]).strip()
        ]
        if quoted_targets:
            by_id = {frame.frame_id: frame for frame, _score in ranked_frames}
            relation_terms = set(_content_tokens(ir.raw_question)) - set().union(*(
                set(_content_tokens(target)) for target in quoted_targets
            ))
            for target in quoted_targets:
                target_terms = set(_content_tokens(target))
                ranked_target: list[tuple[int, float, str]] = []
                for frame, score in ranked_frames:
                    if (
                        frame.quantity.value is None
                        or frame.lifecycle_status != "completed"
                    ):
                        continue
                    source_text = " ".join(
                        turn_by_id[source].text for source in frame.source_turn_ids
                        if source in turn_by_id
                    )
                    terms = set(_content_tokens(
                        f"{frame.retrieval_text} {source_text}"
                    ))
                    target_overlap = len(target_terms & terms)
                    relation_overlap = len(relation_terms & terms)
                    if target_overlap:
                        ranked_target.append((
                            target_overlap * 4 + relation_overlap,
                            score, frame.frame_id,
                        ))
                if ranked_target:
                    frame_id = max(ranked_target)[2]
                    row = next(
                        (row for row in rows if row[2] == frame_id),
                        None,
                    )
                    if row is None:
                        frame = by_id[frame_id]
                        row = (
                            max(ranked_target)[0],
                            max(ranked_target)[1],
                            frame_id,
                            frame.quantity.unit.casefold().rstrip("s"),
                        )
                    if row not in candidates:
                        candidates.insert(0, row)
    if ir.aggregation_op == "average":
        unit_counts: dict[str, int] = defaultdict(int)
        for row in candidates:
            unit_counts[row[3]] += 1
        if unit_counts:
            dominant = max(unit_counts, key=lambda unit: unit_counts[unit])
            candidates = [row for row in candidates if row[3] == dominant]
    return [row[2] for row in candidates[:8]]


def _pack(
    *,
    cards: list[RoutingCard],
    ranked_groups: list[tuple[EvidenceGroup, float]],
    ranked_frames: list[tuple[RoleFrameNode, float]],
    ranked_turns: list[tuple[TurnNodeV36, float]] | None = None,
    priority_frame_ids: list[str] | None = None,
    ir: QueryIR | None = None,
    turn_by_id: dict[str, TurnNodeV36],
    token_budget: int,
) -> tuple[
    str, list[EvidenceGroup], list[RoleFrameNode], list[str],
    list[dict[str, Any]],
]:
    frame_by_id = {
        frame.frame_id: frame
        for frame, _score in ranked_frames
    }
    # Groups reached structurally may contain frames outside the ranked list.
    for group, _score in ranked_groups:
        for frame_id in group.member_frame_ids:
            if frame_id not in frame_by_id:
                continue
    parts: list[str] = []
    used_tokens = 0
    selected_groups: list[EvidenceGroup] = []
    selected_frames: list[RoleFrameNode] = []
    source_ids: list[str] = []
    ledger: list[dict[str, Any]] = []
    for card in cards[:8]:
        block = _format_card(card)
        cost = rough_token_count(block)
        if used_tokens + cost > token_budget:
            break
        parts.append(block)
        used_tokens += cost
    for turn, score in (ranked_turns or []):
        if turn.node_id in source_ids:
            continue
        focused_limit = (
            1100 if ir is not None and ir.requested_value_type in {
                "count", "list", "aggregate", "duration", "recommendation"
            } else 2200 if ir is not None and ir.requested_value_type == "span"
            else 1800
        )
        focused = (
            _focused_turn_text(ir, turn, max_chars=focused_limit)
            if ir is not None else turn.text[:focused_limit]
        )
        block = (
            f"[SOURCE_EVIDENCE {turn.node_id}]\n"
            f"date={turn.session_date or 'unknown'}; speaker={turn.speaker}\n{focused}"
        )
        cost = rough_token_count(block)
        if used_tokens + cost > token_budget:
            continue
        parts.append(block)
        used_tokens += cost
        source_ids.append(turn.node_id)
        ledger.append({
            "source_turn_id": turn.node_id, "score": score,
            "focused_lossless": True, "provenance_complete": True,
        })
        source_limit = (
            12 if ir is not None and ir.requested_value_type in {
                "count", "list", "aggregate", "duration", "recommendation"
            } else 12 if ir is not None and ir.requested_value_type == "span"
            else 8
        )
        if sum(1 for row in ledger if row.get("focused_lossless")) >= source_limit:
            break
    score_by_frame = {frame.frame_id: score for frame, score in ranked_frames}
    for frame_id in priority_frame_ids or []:
        frame = frame_by_id.get(frame_id)
        if frame is None or frame in selected_frames:
            continue
        sources = [
            turn_by_id[source] for source in frame.source_turn_ids
            if source in turn_by_id
        ]
        source_text = "\n".join(
            f"SOURCE {turn.node_id} ({turn.session_date or 'unknown'}, "
            f"{turn.speaker}): {turn.text}"
            for turn in sources
        )
        block = f"[FRAME {frame.frame_id}]\n{frame.retrieval_text}\n{source_text}"
        cost = rough_token_count(block)
        if used_tokens + cost > token_budget:
            continue
        parts.append(block)
        used_tokens += cost
        selected_frames.append(frame)
        source_ids.extend(
            source for source in frame.source_turn_ids if source not in source_ids
        )
        ledger.append({
            "frame_id": frame.frame_id,
            "frame_kind": frame.frame_kind,
            "source_turn_ids": frame.source_turn_ids,
            "score": score_by_frame.get(frame.frame_id, 0.0),
            "priority_role": "query_operand",
            "provenance_complete": True,
        })
    all_frame_by_id = dict(frame_by_id)
    for group, _score in ranked_groups:
        if not group.provenance_complete or not all(group.completeness_mask.values()):
            continue
        if any(frame_id not in all_frame_by_id for frame_id in group.member_frame_ids):
            continue
        block = _format_group(group, all_frame_by_id, turn_by_id)
        cost = rough_token_count(block)
        if used_tokens + cost > token_budget:
            continue
        parts.append(block)
        used_tokens += cost
        selected_groups.append(group)
        member_frames = [
            all_frame_by_id[frame_id] for frame_id in group.member_frame_ids
        ]
        selected_frames.extend(
            frame for frame in member_frames
            if frame.frame_id not in {
                item.frame_id for item in selected_frames
            }
        )
        source_ids.extend(
            source for source in group.source_turn_ids
            if source not in source_ids
        )
        ledger.append({
            "group_id": group.group_id,
            "group_kind": group.group_kind,
            "frame_ids": group.member_frame_ids,
            "source_turn_ids": group.source_turn_ids,
            "required_roles": group.required_roles,
            "completeness_mask": group.completeness_mask,
            "provenance_complete": group.provenance_complete,
        })
        if len(selected_groups) >= 12:
            break
    selected_ids = {frame.frame_id for frame in selected_frames}
    atomic_frame_ids = {
        frame_id for group in selected_groups
        if group.group_kind != "single_fact"
        for frame_id in group.member_frame_ids
    }
    for frame, score in ranked_frames:
        if frame.frame_id in selected_ids:
            continue
        if frame.frame_id in atomic_frame_ids:
            continue
        sources = [
            turn_by_id[source] for source in frame.source_turn_ids
            if source in turn_by_id
        ]
        source_text = "\n".join(
            f"SOURCE {turn.node_id} ({turn.session_date or 'unknown'}, "
            f"{turn.speaker}): {turn.text}"
            for turn in sources
        )
        block = (
            f"[FRAME {frame.frame_id}]\n{frame.retrieval_text}\n{source_text}"
        )
        cost = rough_token_count(block)
        if used_tokens + cost > token_budget:
            continue
        parts.append(block)
        used_tokens += cost
        selected_frames.append(frame)
        selected_ids.add(frame.frame_id)
        source_ids.extend(
            source for source in frame.source_turn_ids if source not in source_ids
        )
        ledger.append({
            "frame_id": frame.frame_id,
            "frame_kind": frame.frame_kind,
            "source_turn_ids": frame.source_turn_ids,
            "score": score,
            "provenance_complete": bool(frame.source_turn_ids),
        })
        if len(selected_frames) >= 16:
            break
    return (
        "\n\n".join(parts), selected_groups, selected_frames, source_ids, ledger
    )



def _compact_frame_block(frame: RoleFrameNode) -> str:
    temporal = (
        frame.temporal.event_time or frame.temporal.start
        or frame.temporal.observed_at or frame.temporal.end or "unknown"
    )
    quantity = ""
    if frame.quantity.value is not None:
        quantity = f"; quantity={frame.quantity.value} {frame.quantity.unit}".rstrip()
    return (
        f"[BOUND_FRAME {frame.frame_id} sources={','.join(frame.source_turn_ids)}]\n"
        f"kind={frame.frame_kind}; owner={frame.owner_key}; entity={frame.entity_key}; "
        f"predicate={frame.predicate_key}; object={frame.object_key}; "
        f"polarity={frame.polarity}; status={frame.lifecycle_status}; "
        f"operation={frame.state_op}; time={temporal}{quantity}"
    )


def _pack_lossless_first(
    *,
    cards: list[RoutingCard],
    ranked_groups: list[tuple[EvidenceGroup, float]],
    ranked_frames: list[tuple[RoleFrameNode, float]],
    ranked_turns: list[tuple[TurnNodeV36, float]] | None = None,
    priority_frame_ids: list[str] | None = None,
    ir: QueryIR | None = None,
    turn_by_id: dict[str, TurnNodeV36],
    token_budget: int,
) -> tuple[
    str, list[EvidenceGroup], list[RoleFrameNode], list[str],
    list[dict[str, Any]],
]:
    """Pack authoritative lossless spans before compact structural hints.

    Routing cards remain available in the retrieval trace but are intentionally
    absent from the answer context.  Frames and groups are compact navigation
    records and are admitted only when at least one cited source turn is packed.
    """
    del cards  # coarse routing is trace metadata, not answer evidence
    parts: list[str] = []
    used_tokens = 0
    source_ids: list[str] = []
    selected_frames: list[RoleFrameNode] = []
    selected_groups: list[EvidenceGroup] = []
    ledger: list[dict[str, Any]] = []
    frame_by_id = {frame.frame_id: frame for frame, _score in ranked_frames}
    score_by_frame = {frame.frame_id: score for frame, score in ranked_frames}

    structured = ir is not None and ir.requested_value_type in {
        "count", "list", "aggregate", "duration", "temporal_order", "state",
        "preference", "recommendation",
    }
    source_limit = 18 if structured else 12

    def add_source(
        turn: TurnNodeV36, score: float, reason: str,
        max_chars: int | None = None,
    ) -> bool:
        nonlocal used_tokens
        if turn.node_id in source_ids:
            return True
        if ir is not None and (
            _ORDINAL_RE.search(ir.raw_question)
            or (_DIALOGUE_RE.search(ir.raw_question) and turn.transport_role == "assistant")
        ):
            focused = turn.text[:5000]
        else:
            focused = (
                _focused_turn_text(ir, turn, max_chars=2200 if structured else 1800)
                if ir is not None else turn.text[:1800]
            )
        if max_chars is not None:
            focused = focused[:max_chars]
        block = (
            f"[SOURCE_EVIDENCE {turn.node_id}]\n"
            f"date={turn.session_date or 'unknown'}; speaker={turn.speaker}; "
            f"listener={turn.listener or 'unknown'}; role={turn.transport_role}\n"
            f"{focused}"
        )
        cost = rough_token_count(block)
        if used_tokens + cost > token_budget:
            return False
        parts.append(block)
        used_tokens += cost
        source_ids.append(turn.node_id)
        ledger.append({
            "source_turn_id": turn.node_id,
            "score": score,
            "selection_reason": reason,
            "focused_lossless": True,
            "provenance_complete": True,
        })
        return True

    for turn, score in ranked_turns or []:
        add_source(turn, score, "lossless_binding_rank")
        if len(source_ids) >= source_limit:
            break

    # Query-bound frames can rescue an answer-bearing source that narrowly
    # missed the turn rank.  Add the source first, then only a compact frame.
    ordered_frame_ids = list(dict.fromkeys([
        *(priority_frame_ids or []),
        *[frame.frame_id for frame, _score in ranked_frames],
    ]))
    for frame_id in ordered_frame_ids:
        frame = frame_by_id.get(frame_id)
        if frame is None or frame in selected_frames or not frame.source_turn_ids:
            continue
        if not set(frame.source_turn_ids).intersection(source_ids):
            for source_id in frame.source_turn_ids[:2]:
                source = turn_by_id.get(source_id)
                if source is not None:
                    add_source(
                        source, score_by_frame.get(frame_id, 0.0),
                        "frame_source_projection",
                    )
        if not set(frame.source_turn_ids).intersection(source_ids):
            continue
        block = _compact_frame_block(frame)
        cost = rough_token_count(block)
        if used_tokens + cost > token_budget:
            continue
        parts.append(block)
        used_tokens += cost
        selected_frames.append(frame)
        ledger.append({
            "frame_id": frame.frame_id,
            "frame_kind": frame.frame_kind,
            "source_turn_ids": frame.source_turn_ids,
            "score": score_by_frame.get(frame.frame_id, 0.0),
            "compact_navigation_only": True,
            "provenance_complete": True,
        })
        if len(selected_frames) >= 14:
            break

    for group, score in ranked_groups:
        if not group.provenance_complete:
            continue
        overlap = [source for source in group.source_turn_ids if source in source_ids]
        if not overlap:
            continue
        if _atomic_group_repair_enabled() and group.group_kind in {
            "dialogue_pair", "state_transition", "temporal_pair",
            "collection", "reference_chain",
        }:
            # Evidence groups are atomic: once one member source is selected,
            # include every cited source or omit the compact group entirely.
            for source_id in group.source_turn_ids:
                source = turn_by_id.get(source_id)
                if source is not None:
                    add_source(source, score, f"atomic_{group.group_kind}_source")
            if not set(group.source_turn_ids).issubset(source_ids):
                continue
        member_rows = [
            _compact_frame_block(frame_by_id[frame_id]).split("\n", 1)[1]
            for frame_id in group.member_frame_ids
            if frame_id in frame_by_id
        ]
        if not member_rows:
            continue
        block = (
            f"[BOUND_GROUP {group.group_id} kind={group.group_kind} "
            f"sources={','.join(group.source_turn_ids)}]\n"
            + "\n".join(member_rows)
        )
        cost = rough_token_count(block)
        if used_tokens + cost > token_budget:
            continue
        parts.append(block)
        used_tokens += cost
        selected_groups.append(group)
        for frame_id in group.member_frame_ids:
            frame = frame_by_id.get(frame_id)
            if frame is not None and frame not in selected_frames:
                selected_frames.append(frame)
        ledger.append({
            "group_id": group.group_id,
            "group_kind": group.group_kind,
            "frame_ids": group.member_frame_ids,
            "source_turn_ids": group.source_turn_ids,
            "score": score,
            "compact_navigation_only": True,
            "provenance_complete": True,
        })
        if len(selected_groups) >= 6:
            break

    if _dialogue_closure_enabled():
        # Append bounded dialogue context only after the baseline source/frame/group
        # pack is frozen, so closure can never evict existing evidence.
        initial_source_ids = list(source_ids)
        candidate_scores: dict[str, tuple[int, int, int]] = {}
        query_terms = {
            _certificate_term(token)
            for token in _content_tokens(ir.raw_question if ir is not None else "")
        }
        for group, _score in ranked_groups:
            if group.group_kind == "dialogue_pair" and set(
                group.source_turn_ids
            ).intersection(initial_source_ids):
                for source_id in group.source_turn_ids:
                    if source_id not in source_ids:
                        candidate_scores[source_id] = (1, 0, 0)
        turns_by_session: dict[str, list[TurnNodeV36]] = defaultdict(list)
        for candidate in turn_by_id.values():
            turns_by_session[candidate.session_id].append(candidate)
        for values in turns_by_session.values():
            values.sort(key=lambda item: item.turn_index)
        for source_rank, source_id in enumerate(initial_source_ids):
            source = turn_by_id.get(source_id)
            if source is None:
                continue
            values = turns_by_session.get(source.session_id, [])
            position = next((
                i for i, row in enumerate(values) if row.node_id == source_id
            ), -1)
            if position < 0:
                continue
            for candidate_position in (position - 1, position + 1):
                if not 0 <= candidate_position < len(values):
                    continue
                candidate = values[candidate_position]
                if candidate.node_id in source_ids:
                    continue
                candidate_terms = {
                    _certificate_term(token)
                    for token in _content_tokens(candidate.text)
                }
                overlap = len(query_terms & candidate_terms)
                role_complement = int(
                    candidate.transport_role != source.transport_role
                )
                score = (0, overlap, role_complement - source_rank)
                candidate_scores[candidate.node_id] = max(
                    candidate_scores.get(candidate.node_id, score), score,
                )
        added = 0
        for source_id in sorted(
            candidate_scores,
            key=lambda value: (candidate_scores[value], value),
            reverse=True,
        ):
            if source_id in source_ids:
                continue
            source = turn_by_id.get(source_id)
            if source is not None and add_source(
                source, 0.0, "bounded_dialogue_closure", max_chars=900,
            ):
                added += 1
            if added >= 8:
                break
    return "\n\n".join(parts), selected_groups, selected_frames, source_ids, ledger


def _source_binding_certificate(
    ir: QueryIR, turns: list[TurnNodeV36],
) -> dict[str, Any]:
    """Measure whether packed lossless spans bind the query roles together."""
    query_terms = {
        _certificate_term(token) for token in _content_tokens(ir.raw_question)
    }
    relation_terms = {
        _certificate_term(token) for token in _content_tokens(ir.target_relation)
    }
    owner = canonical_key(ir.target_owner or "")
    rows: list[dict[str, Any]] = []
    for turn in turns:
        turn_terms = {
            _certificate_term(token)
            for token in _content_tokens(
                f"{turn.speaker} {turn.listener} {turn.text}"
            )
        }
        overlap_terms = sorted(query_terms & turn_terms)
        relation_overlap = sorted(relation_terms & turn_terms)
        owner_match = not owner or (
            turn.speaker_key == owner
            or canonical_key(turn.listener) == owner
            or owner in turn_terms
        )
        rows.append({
            "source_turn_id": turn.node_id,
            "owner_match": owner_match,
            "query_overlap": overlap_terms[:12],
            "relation_overlap": relation_overlap[:8],
            "speaker_key": turn.speaker_key,
        })

    informative = [row for row in rows if len(row["query_overlap"]) >= 2]
    owner_rows = [row for row in rows if row["owner_match"]]
    entity_match = bool(owner_rows if owner else informative)
    relation_match = bool(
        not relation_terms
        or any(row["relation_overlap"] for row in rows)
        or any(len(row["query_overlap"]) >= 3 for row in rows)
    )
    target_support: dict[str, list[str]] = {}
    for target in ir.comparison_targets[:2]:
        target_terms = {
            _certificate_term(token) for token in _content_tokens(target)
        }
        target_support[target] = [
            turn.node_id for turn in turns
            if target_terms.intersection({
                _certificate_term(token)
                for token in _content_tokens(turn.text)
            })
        ][:4]
    comparison_complete = not target_support or all(target_support.values())
    binding_source_ids = list(dict.fromkeys(
        row["source_turn_id"] for row in rows
        if row in informative and row["owner_match"]
    ))
    return {
        "entity_match": entity_match,
        "relation_match": relation_match,
        "comparison_complete": comparison_complete,
        "provenance_complete": bool(turns),
        "binding_complete": bool(
            entity_match and relation_match and comparison_complete and turns
        ),
        "binding_source_ids": binding_source_ids[:12],
        "target_support": target_support,
        "source_rows": rows[:16],
    }


def _parse_observed_date(value: str | None) -> datetime | None:
    if not value:
        return None
    match = re.match(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", value)
    if match is None:
        return None
    return datetime(*(int(part) for part in match.groups()))


def _question_relative_target(question: str, question_date: str | None) -> datetime | None:
    observed = _parse_observed_date(question_date)
    if observed is None:
        return None
    words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "couple": 2, "few": 3,
    }
    match = re.search(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|a\s+couple(?:\s+of)?|(?:a\s+)?few)\s+"
        r"(days?|weeks?|months?|years?)\s+ago\b",
        question, re.IGNORECASE,
    )
    if match:
        raw = match.group(1).casefold()
        amount = int(raw) if raw.isdigit() else 2 if raw.startswith("a couple") else 3 if raw.endswith("few") else words[raw]
        unit = match.group(2).casefold().rstrip("s")
        days = amount * {"day": 1, "week": 7, "month": 30, "year": 365}[unit]
        return observed - timedelta(days=days)
    weekdays = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    match = re.search(r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", question, re.IGNORECASE)
    if match:
        target = weekdays[match.group(1).casefold()]
        delta = (observed.weekday() - target) % 7 or 7
        return observed - timedelta(days=delta)
    return None


def _session_binding_units(index: V36Index) -> dict[str, list[str]]:
    """Keep event-local units separate; session concatenation creates false joins."""
    values: dict[str, list[str]] = defaultdict(list)
    for frame in index.frames:
        text = " ".join([
            frame.entity_key, frame.predicate_key, frame.object_key,
            frame.context_key, " ".join(frame.semantic_type_keys),
            frame.retrieval_text,
        ])
        for session_id in frame.session_ids:
            values[session_id].append(text)
    for turn in index.turns:
        if turn.transport_role == "user":
            values[turn.session_id].append(turn.text)
    return values


def _relative_time_card_protection(
    *, case: QuestionCase, ir: QueryIR, index: V36Index,
    card_ranked: list[tuple[str, float]],
    card_channels: dict[str, dict[str, int]],
    selected_card_ids: list[str],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Protect date-near cards only when they also bind the requested event."""
    target = _question_relative_target(case.question, case.question_date)
    if target is None:
        return selected_card_ids, [], []
    rank = {card_id: position for position, (card_id, _score) in enumerate(card_ranked)}
    card_by_session = {card.session_id: card for card in index.routing_cards}
    frames_by_source: dict[str, list[RoleFrameNode]] = defaultdict(list)
    for frame in index.frames:
        for source_id in frame.source_turn_ids:
            frames_by_source[source_id].append(frame)
    target_terms, relation_terms = query_binding_terms(ir)
    requested_families = action_families(case.question)
    rows: list[tuple[int, int, int, int, str, str]] = []
    for turn in index.turns:
        if turn.transport_role != "user":
            continue
        card = card_by_session.get(turn.session_id)
        if card is None:
            continue
        observed = _parse_observed_date(turn.session_date)
        if observed is None:
            continue
        event_time = _question_relative_target(turn.text, turn.session_date) or observed
        local_frames = frames_by_source.get(turn.node_id, [])
        frame_text = " ".join(frame.retrieval_text for frame in local_frames)
        text = f"{turn.text} {frame_text}"
        terms = binding_tokens(text)
        family_overlap = requested_families & action_families(text)
        target_overlap = fuzzy_term_overlap(target_terms, terms)
        relation_overlap = relation_terms & terms
        semantic_rank = min(
            (
                value for key, value in card_channels.get(card.card_id, {}).items()
                if "dense" in key
            ),
            default=10**6,
        )
        if requested_families and not family_overlap:
            continue
        if (
            not target_overlap and not family_overlap
            and len(relation_overlap) < 2 and semantic_rank > 6
        ):
            continue
        distance = abs((event_time.date() - target.date()).days)
        if distance > 2:
            continue
        compatibility = (
            4 * len(family_overlap) + 3 * len(target_overlap)
            + len(relation_overlap) + int(semantic_rank <= 6)
        )
        card_position = rank.get(card.card_id, 10**6)
        rows.append((distance, -compatibility, -len(target_overlap), card_position, card.card_id, turn.node_id))
    protected: list[str] = []
    trace: list[dict[str, Any]] = []
    sources_per_card: dict[str, int] = defaultdict(int)
    for distance, negative_compatibility, negative_target, card_position, card_id, source_id in sorted(rows):
        if sources_per_card[card_id] >= 2:
            continue
        if card_id not in protected:
            if len(protected) >= 6:
                continue
            protected.append(card_id)
        sources_per_card[card_id] += 1
        trace.append({
            "card_id": card_id, "source_turn_id": source_id,
            "target_date": target.date().isoformat(), "distance_days": distance,
            "binding_score": -negative_compatibility,
            "target_overlap": -negative_target, "coarse_rank": card_position,
        })
    result = list(selected_card_ids)
    for card_id in protected:
        if card_id not in result:
            result.append(card_id)
    return result[:14], protected, trace


def _comparison_card_protection(
    *, ir: QueryIR, index: V36Index,
    card_ranked: list[tuple[str, float]], selected_card_ids: list[str],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Protect one independently bound coarse card for each comparison side."""
    targets = ir.comparison_targets[:2]
    if len(targets) < 2:
        return selected_card_ids, [], []
    rank = {card_id: position for position, (card_id, _score) in enumerate(card_ranked)}
    session_units = _session_binding_units(index)
    cards = {card.session_id: card for card in index.routing_cards}
    token_sets = [binding_tokens(target) for target in targets]
    shared = set.intersection(*token_sets) if token_sets else set()
    protected: list[str] = []
    trace: list[dict[str, Any]] = []
    for target, terms in zip(targets, token_sets):
        distinctive = terms - shared or terms
        rows: list[tuple[int, int, int, str]] = []
        for session_id, units in session_units.items():
            card = cards.get(session_id)
            if card is None:
                continue
            unit_rows = [
                (
                    int(target.casefold() in unit.casefold()),
                    len(distinctive & binding_tokens(unit)),
                )
                for unit in units
            ]
            exact, overlap_size = max(unit_rows, default=(0, 0))
            if overlap_size < 1:
                continue
            rows.append((-exact, -overlap_size, rank.get(card.card_id, 10**6), card.card_id))
        if not rows:
            continue
        exact_rank, negative_overlap, coarse_rank, card_id = min(rows)
        if card_id not in protected:
            protected.append(card_id)
        trace.append({
            "target": target, "card_id": card_id,
            "distinctive_overlap": -negative_overlap,
            "exact_phrase": bool(-exact_rank), "coarse_rank": coarse_rank,
        })
    result = list(selected_card_ids)
    for card_id in protected:
        if card_id not in result:
            result.append(card_id)
    return result[:14], protected, trace


def _semantic_rerank_source_closure(
    ir: QueryIR,
    closure: Any,
    turn_by_id: dict[str, TurnNodeV36],
    query_vectors: list[list[float]],
) -> list[dict[str, Any]]:
    # Rerank an already bounded source closure; never widen its route.
    if not closure.candidates or not query_vectors:
        return []
    views = query_views(ir)
    target_positions = [
        position for position, view in enumerate(views)
        if view.startswith("target ") and position < len(query_vectors)
    ]
    target_vectors = [query_vectors[position] for position in target_positions]
    raw_vector = query_vectors[0]
    diagnostics: list[dict[str, Any]] = []
    scores: dict[str, float] = {}
    for candidate in closure.candidates:
        turn = turn_by_id.get(candidate.source_turn_id)
        if turn is None or turn.embedding is None:
            semantic = 0.0
            target_score = 0.0
        else:
            raw_score = cosine_similarity(raw_vector, turn.embedding)
            target_score = max(
                (cosine_similarity(vector, turn.embedding) for vector in target_vectors),
                default=raw_score,
            )
            semantic = 0.65 * target_score + 0.35 * raw_score
        combined = semantic + 0.05 * candidate.score
        scores[candidate.source_turn_id] = max(
            scores.get(candidate.source_turn_id, float("-inf")), combined,
        )
        diagnostics.append({
            "source_turn_id": candidate.source_turn_id,
            "semantic_score": round(semantic, 6),
            "target_dense_score": round(target_score, 6),
            "combined_score": round(combined, 6),
        })
    closure.candidates.sort(key=lambda row: (
        -scores.get(row.source_turn_id, 0.0), -row.score,
        row.session_id, row.span_index,
    ))
    ordered = list(dict.fromkeys(
        candidate.source_turn_id for candidate in closure.candidates
    ))
    seen = set(ordered)
    ordered.extend(
        source_id for source_id in closure.selected_source_turn_ids
        if source_id not in seen
    )
    closure.selected_source_turn_ids = ordered
    diagnostics.sort(key=lambda row: -row["combined_score"])
    return diagnostics


def _global_lossless_safety_hint(
    ir: QueryIR, index: V36Index, limit: int = 8,
) -> dict[str, Any] | None:
    """Expose a compact, session-diverse exact-source fallback to the LLM."""
    query_terms = {
        _certificate_term(token) for token in _content_tokens(ir.raw_question)
        if _certificate_term(token) not in {
            "many", "much", "total", "different", "recent", "currently",
            "ago", "first", "latest", "last", "past",
        }
    }
    if not query_terms:
        return None
    card_by_session = {card.session_id: card for card in index.routing_cards}
    best_by_session: dict[str, tuple[int, int, TurnNodeV36, RoutingCard | None]] = {}
    for turn in index.turns:
        if turn.transport_role != "user":
            continue
        direct_terms = {
            _certificate_term(token) for token in _content_tokens(turn.text)
        }
        card = card_by_session.get(turn.session_id)
        routed_terms = {
            _certificate_term(token)
            for token in _content_tokens(card.routing_text if card else "")
        }
        direct = len(query_terms & direct_terms)
        routed = len(query_terms & routed_terms)
        if direct < 1 or direct + routed < 2:
            continue
        row = (4 * direct + routed, -turn.turn_index, turn, card)
        prior = best_by_session.get(turn.session_id)
        if prior is None or row[:2] > prior[:2]:
            best_by_session[turn.session_id] = row
    ranked = sorted(
        best_by_session.values(), key=lambda row: (row[0], row[1]), reverse=True
    )[:limit]
    if not ranked:
        return None
    return {
        "operation": "global_lossless_source_candidates",
        "candidates": [
            {
                "source_turn_id": turn.node_id,
                "source_date": turn.session_date,
                "evidence": turn.text[:420],
                "routing_context": (card.routing_text[:220] if card else ""),
                "lexical_score": score,
            }
            for score, _turn_order, turn, card in ranked
        ],
        "binding_complete": False,
        "certified": False,
    }


def retrieve(
    *,
    case: QuestionCase,
    variant: str,
    index: V36Index,
    query_vectors: list[list[float]],
    token_budget: int = 8800,
) -> RetrievedContext:
    started = time.perf_counter()
    dialogue_topology = infer_dialogue_topology(index.turns)
    ir = build_query_ir(case.question)
    card_by_id = {card.card_id: card for card in index.routing_cards}
    card_by_session = {card.session_id: card for card in index.routing_cards}
    frame_by_id = {frame.frame_id: frame for frame in index.frames}
    group_by_id = {group.group_id: group for group in index.evidence_groups}
    turn_by_id = {turn.node_id: turn for turn in index.turns}

    card_ranked, card_channels = _card_ranking(index, ir, query_vectors)
    selected_card_ids = _adaptive_cards(card_ranked, ir, card_channels)
    selected_card_ids, structural_scope_trace = _collection_scope_cards(
        index, ir, selected_card_ids
    )
    selected_card_ids, relative_time_card_ids, relative_time_scope_trace = (
        _relative_time_card_protection(
            case=case, ir=ir, index=index, card_ranked=card_ranked,
            card_channels=card_channels, selected_card_ids=selected_card_ids,
        )
    )
    selected_card_ids, comparison_card_ids, comparison_scope_trace = (
        _comparison_card_protection(
            ir=ir, index=index, card_ranked=card_ranked,
            selected_card_ids=selected_card_ids,
        )
    )
    selected_cards = [
        card_by_id[node_id] for node_id in selected_card_ids
        if node_id in card_by_id
    ]
    scalar_scope_trace: list[dict[str, Any]] = []
    if ir.requested_value_type == "aggregate" and ir.aggregation_op in {"sum", "difference"}:
        query_terms = set(_content_tokens(ir.raw_question))
        frames_by_session: dict[str, list[RoleFrameNode]] = defaultdict(list)
        for frame in index.frames:
            if (
                frame.quantity.value is not None
                and frame.lifecycle_status == "completed"
                and frame.polarity != "negative"
            ):
                for session_id in frame.session_ids:
                    frames_by_session[session_id].append(frame)

        def scalar_overlap(session_id: str) -> int:
            terms: set[str] = set()
            for frame in frames_by_session.get(session_id, []):
                terms.update(_content_tokens(frame.retrieval_text))
                for source in frame.source_turn_ids:
                    if source in turn_by_id:
                        terms.update(_content_tokens(turn_by_id[source].text))
            return len(query_terms & terms)

        qualifying = {
            card.session_id for card in selected_cards
            if scalar_overlap(card.session_id) >= 2
        }
        if len(qualifying) < 2:
            candidates: list[tuple[int, int, str]] = []
            for card_id, ranks in card_channels.items():
                card = card_by_id.get(card_id)
                if card is None or card_id in selected_card_ids:
                    continue
                overlap = scalar_overlap(card.session_id)
                lossless_rank = ranks.get("lossless_bm25", 10**9)
                if overlap >= 2 and lossless_rank <= 6:
                    candidates.append((-overlap, lossless_rank, card_id))
            if candidates:
                _neg_overlap, lossless_rank, card_id = min(candidates)
                card = card_by_id[card_id]
                retired = ""
                if len(selected_cards) >= 8:
                    victim = selected_cards[-1]
                    retired = victim.card_id
                    selected_cards.pop()
                    selected_card_ids.remove(victim.card_id)
                selected_cards.append(card)
                selected_card_ids.append(card_id)
                scalar_scope_trace.append({
                    "card_id": card_id,
                    "session_id": card.session_id,
                    "retired_card_id": retired,
                    "reason": "scalar_operand_scope",
                    "lossless_bm25_rank": lossless_rank,
                })
    routed_sessions = {card.session_id for card in selected_cards}
    card_score = dict(card_ranked)
    protected_card_ids = set([*selected_card_ids[:2], *relative_time_card_ids, *comparison_card_ids])
    retired_card_ids: set[str] = set()
    fine_nodes = _fine_scope_nodes(index, routed_sessions)
    fine_ranked, fine_channels = _fine_rankings(
        index, fine_nodes, ir, query_vectors
    )
    fine_score = dict(fine_ranked)
    ranked_frames, ranked_groups = _fine_evidence_lists(
        fine_ranked, fine_channels, frame_by_id, group_by_id, routed_sessions, ir
    )

    seed_ids = {
        *[frame.frame_id for frame, _score in ranked_frames[:14]],
        *[group.group_id for group, _score in ranked_groups[:8]],
    }
    if _navigation_repair_enabled():
        # Fine dense/FTS can rank a lossless turn above its normalized frame.
        # Admit those turns as graph seeds, then use inverse provenance locally.
        seed_ids.update(
            node_id for node_id, _score in fine_ranked[:32]
            if node_id in turn_by_id
        )
    preliminary = _certificate(
        ir,
        [frame for frame, _score in ranked_frames[:14]],
        [group for group, _score in ranked_groups[:8]],
        routed_sessions=routed_sessions,
        excluded=[],
        expansion_rounds=0,
    )
    graph_trace: list[dict[str, Any]] = []
    coarse_semantic_trace: list[dict[str, Any]] = []
    scope_adjustment_trace: list[dict[str, Any]] = []
    reached = set(seed_ids)
    expansion_rounds = 0
    for expansion_rounds in range(1, 3):
        expansion_roles = set(preliminary.missing_roles)
        if preliminary.complete:
            expansion_rounds -= 1
            break
        reached, rows = _typed_expand(
            index, reached, expansion_roles,
            max_depth=2 if _navigation_repair_enabled() else 1,
        )
        graph_trace.extend(rows)
        # A typed relation may reach evidence in another session. Expand the
        # coarse scope to that exact owning card before accepting the evidence;
        # this is deterministic scope completion, not reverse graph traversal.
        scope_changed = False
        missing_now = expansion_roles
        for node_id in sorted(reached):
            node = frame_by_id.get(node_id) or group_by_id.get(node_id)
            if isinstance(node, RoleFrameNode):
                node_roles = _roles_for_frame(node)
            elif isinstance(node, EvidenceGroup):
                node_roles = set()
                if _group_query_compatible(ir, node, frame_by_id):
                    node_roles = {
                        role for role, available in node.completeness_mask.items()
                        if available
                    }
                    for member_id in node.member_frame_ids:
                        if member_id in frame_by_id:
                            node_roles.update(_roles_for_frame(frame_by_id[member_id]))
            else:
                node_roles = set()
            if not (missing_now & node_roles):
                continue
            for session_id in getattr(node, "session_ids", []):
                if session_id in routed_sessions:
                    continue
                card = card_by_session.get(session_id)
                if card is None:
                    continue
                retired_card_id = ""
                if len(selected_cards) >= 8:
                    victim = next((
                        candidate for candidate in reversed(selected_cards)
                        if candidate.card_id not in protected_card_ids
                    ), None)
                    if victim is None:
                        continue
                    retired_card_id = victim.card_id
                    selected_cards.remove(victim)
                    selected_card_ids.remove(victim.card_id)
                    routed_sessions.discard(victim.session_id)
                    retired_card_ids.add(victim.card_id)
                selected_cards.append(card)
                selected_card_ids.append(card.card_id)
                routed_sessions.add(session_id)
                scope_adjustment_trace.append({
                    "node_id": node_id, "session_id": session_id,
                    "card_id": card.card_id, "retired_card_id": retired_card_id,
                    "reason": "typed_missing_role_scope",
                    "provided_roles": sorted(missing_now & node_roles),
                })
                scope_changed = True
                break
            if scope_changed:
                break
        for node_id in reached:
            if (
                node_id in frame_by_id
                and set(frame_by_id[node_id].session_ids) <= routed_sessions
                and node_id not in {frame.frame_id for frame, _ in ranked_frames}
            ): 
                candidate_row = (frame_by_id[node_id], fine_score.get(node_id, 0.0))
                if missing_now & _roles_for_frame(frame_by_id[node_id]):
                    ranked_frames.insert(0, candidate_row)
                else:
                    ranked_frames.append(candidate_row)
            if (
                node_id in group_by_id
                and set(group_by_id[node_id].session_ids) <= routed_sessions
                and node_id not in {group.group_id for group, _ in ranked_groups}
                and _group_query_compatible(ir, group_by_id[node_id], frame_by_id)
            ): 
                candidate_row = (group_by_id[node_id], fine_score.get(node_id, 0.0))
                group_roles = {
                    role for role, available in group_by_id[node_id].completeness_mask.items()
                    if available
                }
                for member_id in group_by_id[node_id].member_frame_ids:
                    if member_id in frame_by_id:
                        group_roles.update(_roles_for_frame(frame_by_id[member_id]))
                if missing_now & group_roles:
                    ranked_groups.insert(0, candidate_row)
                else:
                    ranked_groups.append(candidate_row)
        ranked_frames = [
            row for row in ranked_frames
            if set(row[0].session_ids) <= routed_sessions
        ]
        ranked_groups = [
            row for row in ranked_groups
            if set(row[0].session_ids) <= routed_sessions
        ]
        preliminary = _certificate(
            ir,
            [frame for frame, _score in ranked_frames[:20]],
            [group for group, _score in ranked_groups[:16]],
            routed_sessions=routed_sessions,
            excluded=[],
            expansion_rounds=expansion_rounds,
        )
        if preliminary.complete:
            break
        # Fill the role gap with a channel-specialist card. At the max-8
        # boundary replace the weakest unprotected card, then actually rerun
        # fine retrieval in the changed scope; merely adding a card to the
        # answer prompt cannot repair missing fine evidence.
        next_id = _gap_card_id(
            card_ranked, card_channels, [*selected_card_ids, *retired_card_ids], preliminary.missing_roles
        )
        semantic_row = None
        if not next_id:
            next_id, semantic_row = _semantic_card_extension(index, selected_card_ids)
        if next_id in card_by_id:
            if len(selected_cards) >= 8:
                # A role gap does not prove that a fused top card is wrong.
                # At the hard routing limit, retain the evidence-backed scope
                # instead of replacing it with a speculative specialist.
                continue
            card = card_by_id[next_id]
            selected_cards.append(card)
            selected_card_ids.append(next_id)
            routed_sessions.add(card.session_id)
            protected_card_ids.add(next_id)
            scope_adjustment_trace.append({
                "card_id": next_id, "session_id": card.session_id,
                "reason": "completeness_gap_added",
                "missing_roles": list(preliminary.missing_roles),
            })
            if semantic_row is not None:
                coarse_semantic_trace.append(semantic_row)
            fine_ranked, fine_channels = _fine_rankings(
                index, _fine_scope_nodes(index, routed_sessions), ir, query_vectors
            )
            fine_score = dict(fine_ranked)
            ranked_frames, ranked_groups = _fine_evidence_lists(
                fine_ranked, fine_channels, frame_by_id, group_by_id, routed_sessions, ir
            )
            reached = {
                *[frame.frame_id for frame, _score in ranked_frames[:14]],
                *[group.group_id for group, _score in ranked_groups[:8]],
            }
            preliminary = _certificate(
                ir, [frame for frame, _score in ranked_frames[:14]],
                [group for group, _score in ranked_groups[:8]],
                routed_sessions=routed_sessions, excluded=[],
                expansion_rounds=expansion_rounds,
            )

    # Ensure structurally expanded group members are available to the atomic packer.
    known_frame_ids = {frame.frame_id for frame, _ in ranked_frames}
    for group, _score in ranked_groups:
        for frame_id in group.member_frame_ids:
            if frame_id in frame_by_id and frame_id not in known_frame_ids:
                ranked_frames.append((frame_by_id[frame_id], 0.0))
                known_frame_ids.add(frame_id)
    comparison_session_hints = {
        row["target"]: card_by_id[row["card_id"]].session_id
        for row in comparison_scope_trace
        if row.get("target") and row.get("card_id") in card_by_id
    }
    source_span_closure = build_source_span_closure(
        ir, index.turns, set(routed_sessions),
        frames=index.frames, question_date=case.question_date,
        target_session_hints=comparison_session_hints,
        preferred_source_turn_ids=[
            row["source_turn_id"] for row in relative_time_scope_trace
            if row.get("source_turn_id")
        ],
        max_candidates=24,
    )
    source_span_semantic_trace = (
        _semantic_rerank_source_closure(
            ir, source_span_closure, turn_by_id, query_vectors,
        )
        if ir.temporal_constraints else []
    )
    ranked_turns = _ranked_source_turns(
        index, ir, query_vectors, selected_cards, fine_ranked
    )
    # Question-time closure ranks relation-bound source spans before broader
    # semantic neighbors. It never introduces a source outside routed scope.
    span_score = {
        candidate.source_turn_id: candidate.score
        for candidate in source_span_closure.candidates
    }
    span_priority = [
        (turn_by_id[source_id], span_score.get(source_id, 0.0))
        for source_id in source_span_closure.selected_source_turn_ids
        if source_id in turn_by_id
    ]
    ranked_turns = [
        *span_priority,
        *[
            row for row in ranked_turns
            if row[0].node_id not in {
                turn.node_id for turn, _score in span_priority
            }
        ],
    ]
    priority_pool = [
        (frame, fine_score.get(frame.frame_id, 0.0))
        for frame in index.frames
        if set(frame.session_ids) <= routed_sessions
    ]
    priority_frame_ids = _priority_frame_ids(
        ir, priority_pool, turn_by_id
    )
    if ir.aggregation_op == "difference" and "night" in set(_content_tokens(ir.raw_question)):
        rate_rows: list[tuple[int, float, str, str]] = []
        query_terms = {_certificate_term(term) for term in _content_tokens(ir.raw_question)}
        for frame, score in priority_pool:
            if (
                frame.quantity.value is None
                or frame.lifecycle_status != "completed"
                or "night" not in frame.quantity.unit.casefold()
            ):
                continue
            terms = {
                _certificate_term(term) for term in _content_tokens(frame.retrieval_text)
            }
            overlap = len(query_terms & terms)
            rate_rows.append((overlap, score, frame.session_ids[0], frame.frame_id))
        best_by_session: dict[str, tuple[int, float, str, str]] = {}
        for row in sorted(rate_rows, reverse=True):
            best_by_session.setdefault(row[2], row)
        for _overlap, _score, _session, frame_id in sorted(
            best_by_session.values(), reverse=True
        )[:2]:
            if frame_id not in priority_frame_ids:
                priority_frame_ids.append(frame_id)
    ranked_frame_ids = {frame.frame_id for frame, _score in ranked_frames}
    for frame_id in reversed(priority_frame_ids):
        if frame_id not in ranked_frame_ids and frame_id in frame_by_id:
            ranked_frames.insert(0, (frame_by_id[frame_id], fine_score.get(frame_id, 0.0)))
            ranked_frame_ids.add(frame_id)
    ledger_reserve = 0
    if ir.requested_value_type == "count":
        ledger_reserve = 3400
    elif (
        ir.requested_value_type == "duration"
        and re.search(r"\bhow many years?\b", ir.raw_question, re.IGNORECASE)
        and re.search(r"\bfrom\b.+\bto\b", ir.raw_question, re.IGNORECASE)
    ):
        ledger_reserve = 2500
    ledger_reserve = max(ledger_reserve, 2200 if ir.temporal_constraints else 1200)
    context, packed_groups, packed_frames, source_ids, ledger = _pack_lossless_first(
        cards=selected_cards,
        ranked_groups=ranked_groups,
        ranked_frames=ranked_frames,
        ranked_turns=ranked_turns,
        priority_frame_ids=priority_frame_ids,
        ir=ir,
        turn_by_id=turn_by_id,
        token_budget=max(
            1000,
            token_budget - ledger_reserve,
        ),
    )
    packed_span_rows = [
        {
            "source_turn_id": candidate.source_turn_id,
            "source_text": candidate.text[:420],
            "roles": candidate.roles,
            "target_terms": candidate.target_terms,
            "relation_terms": candidate.relation_terms,
            "action_families": candidate.action_families,
            "lifecycle_status": candidate.lifecycle_status,
            "polarity": candidate.polarity,
            "event_time_text": candidate.event_time_text,
            "identity_keys": candidate.identity_keys,
            "score": candidate.score,
        }
        for candidate in source_span_closure.candidates
        if candidate.source_turn_id in source_ids
    ][:14]
    if packed_span_rows:
        context += (
            "\n\n[RELATION_BOUND_SOURCE_SPANS]\n"
            + json.dumps(packed_span_rows, ensure_ascii=False, separators=(",", ":"))
        )
    excluded = [
        node_id for node_id, _score in fine_ranked
        if node_id not in {
            *[frame.frame_id for frame in packed_frames],
            *[group.group_id for group in packed_groups],
            *source_ids,
        }
    ][:20]
    certificate = _certificate(
        ir, packed_frames, packed_groups,
        routed_sessions=routed_sessions,
        excluded=excluded,
        expansion_rounds=expansion_rounds,
    )
    source_binding = _source_binding_certificate(
        ir, [turn_by_id[source_id] for source_id in source_ids if source_id in turn_by_id],
    )
    # Fine packing may omit the one decisive user proposition from a routed
    # memory region. Operators inspect all user turns inside the bounded coarse
    # route, while their local binders still require entity/relation evidence.
    routed_session_set = {card.session_id for card in selected_cards}
    routed_user_source_ids = [
        turn.node_id for turn in index.turns
        if turn.session_id in routed_session_set
        and is_memory_source(turn, dialogue_topology)
    ]
    # Typed binders are strict enough to use a lossless global safety pool.
    # Coarse routing remains the normal path; this pool only prevents a missed
    # card from hiding an exact entity/date/operand source from an operator.
    all_user_source_ids = [
        turn.node_id for turn in index.turns
        if is_memory_source(turn, dialogue_topology)
    ]
    operator_source_ids = list(dict.fromkeys([
        *source_ids, *routed_user_source_ids, *all_user_source_ids,
    ]))
    temporal_operator_source_ids = list(dict.fromkeys([
        *source_span_closure.selected_source_turn_ids, *source_ids,
        *routed_user_source_ids,
    ]))
    if ir.comparison_targets and all(
        source_span_closure.target_support.get(target)
        for target in ir.comparison_targets
    ):
        temporal_operator_source_ids = list(
            source_span_closure.selected_source_turn_ids
        )
    temporal_hint = temporal_source_pair_hint(
        ir, index, operator_source_ids
    )
    relative_time_hint = relative_time_from_sources_hint(
        ir, index, temporal_operator_source_ids, case.question_date,
    )
    date_lookup_hint = source_bound_date_lookup_hint(
        ir, index, operator_source_ids,
    )
    transaction_sum_hint = transaction_sum_from_sources_hint(
        ir, index, operator_source_ids,
    )
    exact_absence_hint = (
        None if dialogue_topology.peer_dialogue
        else exact_entity_absence_hint(ir, index)
    )
    named_members_hint = named_individual_event_members_hint(
        ir, index, list(dict.fromkeys([*source_ids, *routed_user_source_ids])),
    )
    repeated_event_hint = repeated_event_total_from_sources_hint(
        ir, index, operator_source_ids,
    )
    age_arithmetic_hint = age_arithmetic_from_sources_hint(
        ir, index, operator_source_ids,
    )
    advance_booking_hint = advance_booking_recency_from_sources_hint(
        ir, index, operator_source_ids,
    )
    current_role_duration_hint = current_role_duration_from_sources_hint(
        ir, index, operator_source_ids,
    )
    schedule_days_hint = weekly_schedule_days_from_sources_hint(
        ir, index, operator_source_ids,
    )
    family_total_hint = family_relation_total_from_sources_hint(
        ir, index, operator_source_ids,
    )
    linked_event_date_hint = linked_event_date_from_sources_hint(
        ir, index, operator_source_ids,
    )
    latest_category_start_hint = latest_category_start_from_sources_hint(
        ir, index, operator_source_ids,
    )
    scoped_event_members_hint = scoped_completed_event_members_hint(
        ir, index, operator_source_ids, case.question_date,
    )
    preference_constraints_hint = preference_constraints_from_sources_hint(
        ir, index, operator_source_ids,
    )
    dialogue_attribute_hint = dialogue_attribute_match_hint(ir, index)
    currency_extreme_hint = currency_extreme_entity_from_sources_hint(
        ir, index,
        [turn.node_id for turn in index.turns
         if turn.transport_role == "user"],
        case.question_date,
    )
    presupposed_absence_hint = (
        None if dialogue_topology.peer_dialogue
        else presupposed_event_absence_hint(ir, index)
    )
    dialogue_final_choice_hint = dialogue_final_choice_from_sources_hint(ir, index)
    completed_metric_total_hint = completed_item_metric_total_from_sources_hint(
        ir, index, operator_source_ids,
    )
    scoped_duration_total_hint = scoped_completed_duration_total_from_sources_hint(
        ir, index, operator_source_ids, case.question_date,
    )
    relative_value_hint = relative_value_multiplier_from_sources_hint(
        ir, index, operator_source_ids,
    )
    relative_duration_event_hint = relative_duration_at_event_from_sources_hint(
        ir, index, operator_source_ids,
    )
    prior_candidate_count_hint = prior_candidate_count_from_sources_hint(
        ir, index, operator_source_ids,
    )
    completed_carrier_sequence_hint = completed_carrier_sequence_from_sources_hint(
        ir, index, operator_source_ids, case.question_date,
    )
    endpoint_difference_hint = event_endpoint_difference_from_sources_hint(
        ir, index, operator_source_ids,
    )
    travel_arrival_hint = travel_arrival_time_from_sources_hint(
        ir, index, operator_source_ids,
    )
    completed_work_total_hint = completed_work_subtype_total_from_sources_hint(
        ir, index, operator_source_ids,
    )
    incomplete_terminal_hint = incomplete_terminal_event_hint(
        ir, index, operator_source_ids,
    )
    state_change_hint = state_change_members_from_sources_hint(
        ir, index, operator_source_ids,
    )
    provenance_acquisition_hint = provenance_acquisition_members_hint(
        ir, index, operator_source_ids,
    )
    cuisine_categories_hint = explicit_cuisine_categories_hint(
        ir, index, operator_source_ids,
    )
    subset_percentage_hint = subset_percentage_from_sources_hint(
        ir, index, operator_source_ids,
    )
    excluded_collection_hint = excluded_collection_members_hint(
        ir, index, operator_source_ids,
    )
    paired_metric_hint = paired_metric_total_from_sources_hint(
        ir, index, operator_source_ids,
    )
    binary_savings_hint = binary_savings_from_sources_hint(
        ir, index, operator_source_ids,
    )
    temporal_predecessor_hint = temporal_predecessor_entity_hint(
        ir, index, operator_source_ids,
    )
    latest_scalar_hint = latest_scalar_state_from_sources_hint(
        ir, index, operator_source_ids,
    )
    threshold_progress_hint = threshold_progress_remaining_hint(
        ir, index, operator_source_ids,
    )
    approx_scalar_hint = latest_approx_scalar_state_hint(
        ir, index, operator_source_ids,
    )
    labeled_currency_hint = latest_labeled_currency_state_hint(
        ir, index, operator_source_ids,
    )
    weekly_schedule_hint = latest_weekly_schedule_time_hint(
        ir, index, operator_source_ids,
    )
    acquisition_total_hint = same_unit_acquisition_total_hint(
        ir, index, operator_source_ids, case.question_date,
    )
    labeled_difference_hint = labeled_scalar_difference_from_sources_hint(
        ir, index, operator_source_ids,
    )
    repeated_duration_hint = repeated_activity_duration_total_hint(
        ir, index, operator_source_ids,
    )
    dated_event_hint = dated_event_count_from_sources_hint(
        ir, index, operator_source_ids,
    )
    named_attendance_hint = named_event_attendance_count_hint(
        ir, index, operator_source_ids,
    )
    same_unit_difference_hint = same_unit_state_difference_hint(
        ir, index, operator_source_ids,
    )
    maintenance_count_hint = maintenance_entity_count_hint(
        ir, index, operator_source_ids,
    )
    category_acquisition_hint = category_acquisition_members_hint(
        ir, index, operator_source_ids,
    )
    pending_pairs_hint = pending_operation_target_pairs_hint(
        ir, index, operator_source_ids,
    )
    relative_anchor_hint = (
        None if dialogue_topology.peer_dialogue
        else relative_anchor_source_hint(
            ir, index, case.question_date,
        )
    )
    record_hint = record_time_source_hint(ir, index, source_ids)
    collection_ledger = query_bound_collection_ledger(
        ir, index, source_ids, [frame.frame_id for frame in packed_frames],
        routed_session_ids=[card.session_id for card in selected_cards],
    )
    counterfactual_hint = counterfactual_dependency_hint(
        ir, index, source_ids
    )
    if counterfactual_hint is not None:
        certificate.present_roles = sorted(
            set(certificate.present_roles) | {"condition", "effect", "source"}
        )
        certificate.missing_roles = [
            role for role in certificate.missing_roles
            if role not in {"condition", "effect", "source"}
        ]
        certificate.complete = not certificate.missing_roles
    if completed_carrier_sequence_hint is not None:
        temporal_hint = completed_carrier_sequence_hint
    if endpoint_difference_hint is not None:
        temporal_hint = endpoint_difference_hint
    if temporal_hint is None:
        temporal_hint = temporal_order_source_hint(
            ir, index, temporal_operator_source_ids
        )
    if temporal_hint is None:
        temporal_hint = open_temporal_sequence_from_sources_hint(
            ir, index, operator_source_ids, case.question_date,
        )
    temporal_binding_complete = bool(
        temporal_hint is not None
        and temporal_hint.get("binding_complete", True)
        and temporal_hint.get("certified") is True
        and (
            (
                temporal_hint.get("operation")
                == "temporal_sequence_from_lossless_sources"
                and len(temporal_hint.get("source_turn_ids") or []) >= 2
                and len(temporal_hint.get("event_times") or [])
                == len(temporal_hint.get("source_turn_ids") or [])
            )
            or (
                temporal_hint.get("event_a_source_turn_id")
                and temporal_hint.get("event_b_source_turn_id")
                and temporal_hint.get("event_a_time")
                and temporal_hint.get("event_b_time")
            )
        )
    )
    if temporal_binding_complete:
        endpoint_roles = (
            {"events", "times"}
            if temporal_hint.get("operation")
            == "temporal_sequence_from_lossless_sources"
            else {"event_a", "event_b", "time_a", "time_b"}
        )
        certificate.present_roles = sorted(
            set(certificate.present_roles) | endpoint_roles
        )
        certificate.missing_roles = [
            role for role in certificate.missing_roles
            if role not in endpoint_roles
        ]
        certificate.complete = not certificate.missing_roles
    if ir.requested_value_type == "state":
        priority_packed = [
            frame for frame in packed_frames
            if frame.frame_id in set(priority_frame_ids)
            and (frame.temporal.event_time or frame.temporal.observed_at)
        ]
        if len(priority_packed) >= 2:
            certificate.present_roles = sorted(
                set(certificate.present_roles)
                | {"previous_state", "current_state", "time", "source"}
            )
            certificate.missing_roles = [
                role for role in certificate.missing_roles
                if role not in {"previous_state", "current_state", "time", "source"}
            ]
            certificate.complete = not certificate.missing_roles
    operator_frame_ids = [frame.frame_id for frame in packed_frames]
    if ir.requested_value_type in {"date", "aggregate"}:
        packed_source_set = set(source_ids)
        operator_frame_ids.extend(
            frame.frame_id for frame in index.frames
            if packed_source_set.intersection(frame.source_turn_ids)
        )
        operator_frame_ids = list(dict.fromkeys(operator_frame_ids))
    operator_hints = evaluate_operators(
        ir=ir, index=index,
        frame_ids=operator_frame_ids,
        group_ids=[group.group_id for group in packed_groups],
        certificate=certificate,
    )
    if (
        ir.requested_value_type == "aggregate"
        and ir.aggregation_op == "sum"
        and re.search(r"\btotal\s+(?:money|amount|cost|expenses?)\b", ir.raw_question, re.IGNORECASE)
    ):
        for hint in operator_hints:
            if (
                hint.get("operation") == "scalar_aggregate"
                and hint.get("aggregate") == "sum"
                and hint.get("certified") is True
                and len(hint.get("frame_ids") or []) >= 2
            ):
                hint["binding_complete"] = True
    lossless_safety_hint = _global_lossless_safety_hint(ir, index)
    if lossless_safety_hint is not None and (
        not certificate.complete
        or ir.requested_value_type in {
            "count", "list", "aggregate", "duration", "temporal_order",
        }
    ):
        operator_hints.append(lossless_safety_hint)
    if counterfactual_hint is not None and certificate.complete:
        operator_hints.insert(0, counterfactual_hint)
    if record_hint is not None and certificate.complete:
        operator_hints.insert(0, record_hint)
    if temporal_hint is not None and temporal_binding_complete:
        operator_hints = [
            hint for hint in operator_hints
            if hint.get("operation") not in {"duration_total", "time_difference"}
        ]
        operator_hints.insert(0, temporal_hint)
    if relative_time_hint is not None and not temporal_binding_complete:
        operator_hints = [
            hint for hint in operator_hints
            if hint.get("operation") not in {"duration_total", "event_time"}
        ]
        operator_hints.insert(0, relative_time_hint)
    if date_lookup_hint is not None:
        operator_hints = [
            hint for hint in operator_hints
            if hint.get("operation") not in {"event_time", "source_bound_explicit_date"}
        ]
        operator_hints.insert(0, date_lookup_hint)
    if transaction_sum_hint is not None:
        operator_hints = [
            hint for hint in operator_hints
            if hint.get("operation") != "scalar_aggregate"
        ]
        operator_hints.insert(0, transaction_sum_hint)
    if relative_anchor_hint is not None:
        operator_hints.insert(0, relative_anchor_hint)
    if named_members_hint is not None:
        operator_hints.insert(0, named_members_hint)
    if repeated_event_hint is not None:
        operator_hints.insert(0, repeated_event_hint)
    for typed_hint in (
        age_arithmetic_hint, advance_booking_hint, current_role_duration_hint,
        schedule_days_hint, family_total_hint, linked_event_date_hint,
        latest_category_start_hint, scoped_event_members_hint,
        preference_constraints_hint, dialogue_attribute_hint,
        currency_extreme_hint, dialogue_final_choice_hint,
        completed_metric_total_hint, scoped_duration_total_hint,
        relative_value_hint, relative_duration_event_hint,
        prior_candidate_count_hint, travel_arrival_hint,
        completed_work_total_hint,
        presupposed_absence_hint, incomplete_terminal_hint, state_change_hint,
        provenance_acquisition_hint, cuisine_categories_hint,
        subset_percentage_hint, excluded_collection_hint, paired_metric_hint,
        binary_savings_hint, temporal_predecessor_hint, latest_scalar_hint,
        threshold_progress_hint,
        approx_scalar_hint, labeled_currency_hint, weekly_schedule_hint,
        acquisition_total_hint, labeled_difference_hint, repeated_duration_hint,
        dated_event_hint,
        named_attendance_hint, same_unit_difference_hint,
        maintenance_count_hint,
        category_acquisition_hint, pending_pairs_hint,
    ):
        if typed_hint is not None:
            operator_hints.insert(0, typed_hint)
    if exact_absence_hint is not None and dialogue_final_choice_hint is None:
        operator_hints.insert(0, exact_absence_hint)
    if presupposed_absence_hint is not None:
        operator_hints = [
            hint for hint in operator_hints
            if hint.get("operation") != "exact_entity_absence"
        ]
        operator_hints.insert(0, presupposed_absence_hint)
    if incomplete_terminal_hint is not None:
        operator_hints = [
            hint for hint in operator_hints
            if hint.get("operation") in {
                "terminal_event_completion_check",
                "global_lossless_source_candidates",
            }
        ]
    elif any(hint is not None for hint in (
        schedule_days_hint, family_total_hint, linked_event_date_hint,
        latest_category_start_hint, scoped_event_members_hint,
        dialogue_attribute_hint, currency_extreme_hint,
        dialogue_final_choice_hint, completed_metric_total_hint,
        scoped_duration_total_hint, relative_value_hint,
        relative_duration_event_hint, prior_candidate_count_hint,
        travel_arrival_hint, completed_work_total_hint,
        presupposed_absence_hint,
        state_change_hint, cuisine_categories_hint, subset_percentage_hint,
        excluded_collection_hint, paired_metric_hint, binary_savings_hint,
        temporal_predecessor_hint, latest_scalar_hint,
        threshold_progress_hint, approx_scalar_hint,
        labeled_currency_hint, weekly_schedule_hint, acquisition_total_hint,
        labeled_difference_hint, repeated_duration_hint, dated_event_hint,
        named_attendance_hint,
        same_unit_difference_hint, maintenance_count_hint,
        category_acquisition_hint, pending_pairs_hint,
    )):
        operator_hints = [
            hint for hint in operator_hints
            if hint.get("operation") != "distinct_collection"
        ]
    if ir.requested_value_type in {"aggregate", "temporal_order"}:
        context += (
            "\n\n[EVIDENCE_COMPLETENESS]\n"
            f"complete={certificate.complete}; "
            f"missing_roles={certificate.missing_roles}; "
            f"provenance_complete={certificate.provenance_complete}"
        )
    if operator_hints:
        bound_hints = [
            hint for hint in operator_hints
            if hint.get("binding_complete") is True
        ]
        candidate_hints = [
            hint for hint in operator_hints
            if hint.get("binding_complete") is not True
        ]
        if bound_hints:
            context += (
                "\n\n[PROVENANCE_BOUND_OPERATOR_LEDGER]\n"
                + "\n".join(str(hint) for hint in bound_hints)
            )
    if incomplete_terminal_hint is not None:
        context += (
            "\n\n[REQUIRED_ENDPOINT_STATUS]\n"
            f"required_terminal_event={incomplete_terminal_hint.get('required_terminal_event')}; "
            "completion_source=none; numeric_interval_valid=false. "
            "The memory may establish earlier completed phases, but none can replace "
            "the exact endpoint named by the question."
        )
    if (
        collection_ledger is not None
        and state_change_hint is None
        and cuisine_categories_hint is None
        and subset_percentage_hint is None
        and excluded_collection_hint is None
        and paired_metric_hint is None
        and labeled_difference_hint is None
        and repeated_duration_hint is None
        and provenance_acquisition_hint is None
        and incomplete_terminal_hint is None
    ):
        prompt_collection_ledger = collection_ledger
        if collection_ledger.get("certified") is not True:
            # Keep the answer prompt focused on provenance-bearing candidates.
            # Broad routed-action fallbacks are diagnostics only and previously
            # drowned out one-to-many source statements (twins, item lists).
            focused_sources = [
                row for row in collection_ledger.get("lossless_candidates", [])
                if row.get("binding") != "routed_owner_action_fallback"
            ][:12]
            prompt_collection_ledger = {
                "operation": "query_bound_collection_evidence",
                "candidate_pool_complete": False,
                "structured_member_candidates": collection_ledger.get(
                    "structured_member_candidates", []
                ),
                "relation_closure_candidates": collection_ledger.get(
                    "relation_closure_candidates", []
                ),
                "named_object_candidates": collection_ledger.get(
                    "named_object_candidates", []
                ),
                "schedule_closure_candidates": collection_ledger.get(
                    "schedule_closure_candidates", []
                ),
                "derived_weekly_occurrence_days": collection_ledger.get(
                    "derived_weekly_occurrence_days"
                ),
                "derived_weekly_occurrence_value": collection_ledger.get(
                    "derived_weekly_occurrence_value"
                ),
                "derived_bounded_year_span": (
                    None if incomplete_terminal_hint is not None
                    else collection_ledger.get("derived_bounded_year_span")
                ),
                "lossless_candidates": focused_sources,
                "instruction": (
                    "derive member identities from each cited source; one source "
                    "may contain multiple members; deduplicate repeated mentions; "
                    "do not infer completeness from row count"
                ),
            }
        context += (
            "\n\n[QUERY_BOUND_COLLECTION_LEDGER]\n"
            + str(prompt_collection_ledger)
        )
    retrieved_sessions = list(dict.fromkeys([
        *[card.session_id for card in selected_cards],
        *[
            turn_by_id[source].session_id for source in source_ids
            if source in turn_by_id
        ],
    ]))
    trace = {
        "query_ir": asdict(ir),
        "dialogue_topology": asdict(dialogue_topology),
        "hybrid_operator_policy": (
            "lossless_peer_dialogue"
            if dialogue_topology.peer_dialogue
            else "lossless_plus_certified_v2_algebra"
        ),
        "coarse_channels": card_channels,
        "fine_channels": fine_channels,
        "coarse_ranked_ids": [node_id for node_id, _ in card_ranked[:24]],
        "fine_ranked_ids": [node_id for node_id, _ in fine_ranked[:80]],
        "selected_card_ids": selected_card_ids,
        "structural_group_scope": structural_scope_trace,
        "scalar_operand_scope": scalar_scope_trace,
        "graph_expansion": graph_trace,
        "coarse_semantic_extension": coarse_semantic_trace,
        "scope_adjustment": scope_adjustment_trace,
        "relative_time_scope": relative_time_scope_trace,
        "comparison_scope": comparison_scope_trace,
        "completeness_certificate": asdict(certificate),
        "source_binding_certificate": source_binding,
        "source_span_closure": asdict(source_span_closure),
        "source_span_semantic_ranking": source_span_semantic_trace,
        "packed_group_ids": [group.group_id for group in packed_groups],
        "packed_frame_ids": [frame.frame_id for frame in packed_frames],
        "packed_source_turn_ids": source_ids,
        "generic_operator_hints": operator_hints,
        "query_bound_collection_ledger": collection_ledger,
        "priority_frame_ids": priority_frame_ids,
        "answer_target_tokens": 10_000,
        "answer_hard_limit_tokens": 10_500,
    }
    return RetrievedContext(
        question_id=case.question_id,
        variant=variant,
        summary_node_ids=selected_card_ids,
        leaf_node_ids=source_ids,
        edge_count=len(graph_trace),
        context_text=context,
        answer_session_hit=False,
        retrieved_session_ids=retrieved_sessions,
        latency_sec=time.perf_counter() - started,
        routing_card_ids=selected_card_ids,
        fact_node_ids=[frame.frame_id for frame in packed_frames],
        evidence_leaf_ids=source_ids,
        evidence_ledger=ledger,
        query_kind=ir.requested_value_type,
        packed_rough_tokens=rough_token_count(context),
        schema_version=index.schema_version,
        retrieval_trace=trace,
    )


def authoritative_operator_answer(
    ir: QueryIR,
    retrieval_trace: dict[str, Any],
) -> str | None:
    """Render only operator results whose answer domain is uniquely determined.

    Collection ledgers and scalar aggregates remain useful evidence hints, but
    their four structural certificates do not prove that the selected members
    or operands exhaust the natural-language scope. Those results therefore
    require the normal single answer call. A derived weekly schedule is safe
    to render because its members are explicit weekdays from source turns.
    """
    collection = retrieval_trace.get("query_bound_collection_ledger")
    if isinstance(collection, dict) and collection.get("certified") is True:
        local_certificate = collection.get("operator_certificate")
        if (
            isinstance(local_certificate, dict)
            and all(
                local_certificate.get(field) is True
                for field in (
                    "entity_match", "relation_match", "scope_match",
                    "provenance_complete",
                )
            )
        ):
            weekdays = collection.get("derived_distinct_weekdays")
            if isinstance(weekdays, list) and weekdays:
                return f"{len(weekdays)} days a week."
    certificate = retrieval_trace.get("completeness_certificate")
    if not (
        isinstance(certificate, dict)
        and certificate.get("complete") is True
        and certificate.get("entity_match") is True
        and certificate.get("relation_match") is True
        and certificate.get("scope_match") is True
        and certificate.get("provenance_complete") is True
    ):
        return None
    hints = retrieval_trace.get("generic_operator_hints")
    if not isinstance(hints, list):
        return None
    for hint in hints:
        if not isinstance(hint, dict) or hint.get("certified") is not True:
            continue
        if hint.get("operation") == "counterfactual_dependency":
            if hint.get("value") == "likely_no":
                return "Likely no."
            continue
        if hint.get("operation") == "event_time":
            value = str(hint.get("value") or "").strip()
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                continue
            return parsed.date().isoformat()
        if hint.get("operation") == "record_time_extreme":
            value = str(hint.get("value") or "").strip()
            if value:
                return value
        # A scalar hint proves the arithmetic over the selected operands, not
        # that retrieval selected every and only operand requested in prose.
        # Keep the hint in the evidence ledger and let the single answer call
        # validate its semantic scope.
    question_terms = set(re.findall(r"[A-Za-z]+", ir.raw_question.casefold()))
    if not question_terms.intersection({"more", "less", "increase", "decrease"}):
        return None
    for hint in hints:
        if not isinstance(hint, dict) or hint.get("certified") is not True:
            continue
        if hint.get("operation") != "latest_valid_state":
            continue
        direction = str(hint.get("change_direction") or "")
        value = str(hint.get("value") or "").strip()
        if not value:
            return None
        if direction.startswith("less "):
            return f"Less — {value}."
        if direction.startswith("more "):
            return f"More — {value}."
    return None


def answer_messages(
    case: QuestionCase,
    retrieval: RetrievedContext,
) -> list[dict[str, str]]:
    """Build one compact, source-first answer call."""
    trace = retrieval.retrieval_trace or {}
    ir = trace.get("query_ir") or {}
    required_roles = set(ir.get("required_roles") or [])
    if ir.get("temporal_constraints") and retrieval.query_kind not in {"count", "list", "aggregate", "duration", "temporal_order"}:
        class_policy = (
            "Resolve the requested relative date against the question date. For this "
            "relative-time lookup, RELATION_BOUND_SOURCE_SPANS is the exhaustive answer "
            "candidate table; treat other frames, cards and source turns only as navigation "
            "context. Filter candidates by event_time_text first, then require the same "
            "source span to match the requested entity or semantic type, relation, owner "
            "and lifecycle. Return that span’s requested value. Do not pick an out-of-window "
            "or different same-day event merely because it ranks earlier."
        )
    elif retrieval.query_kind in {"count", "list", "aggregate", "duration"}:
        class_policy = (
            "Enumerate source-bound operands or members first. Match owner, relation, "
            "scope, unit, time and lifecycle; apply add/remove/cancel; deduplicate only "
            "source-confirmed aliases; then compute the requested set or quantity. "
            "Treat lossless collection-ledger rows as candidates to verify, including "
            "fallback rows. One source may contain several conjoined members. Acquisition "
            "includes explicit got/bought/received provenance; maintenance of a component "
            "counts its parent asset; include planned actions only when the question asks."
        )
    elif retrieval.query_kind in {"date", "temporal_order"}:
        class_policy = (
            "Bind each named event independently to its dated source, resolve relative "
            "time against that source date, then compare or calculate the endpoints."
        )
    elif retrieval.query_kind == "state":
        class_policy = (
            "Compare only values with the same owner, entity, attribute and context. "
            "Use lifecycle and time to choose the latest valid state, or the numeric "
            "extreme when the question explicitly asks for one."
        )
    elif retrieval.query_kind in {"preference", "recommendation"}:
        class_policy = (
            "Preserve preference owner, polarity and context. A recommendation may "
            "transfer source-supported traits and constraints to a new target; it need "
            "not have appeared verbatim in memory."
        )
    elif {"condition", "effect"} <= required_roles:
        class_policy = (
            "Bind the named condition to a source-supported cause, motive or enabling "
            "relation before answering the counterfactual with a concise likely yes/no."
        )
    elif retrieval.query_kind == "span":
        class_policy = (
            "For dialogue or ordinal lookup, preserve local turn order and pair the "
            "request with its reply; return the exact requested item or span."
        )
    else:
        class_policy = (
            "Answer the requested entity, attribute or relation from the strongest "
            "owner-bound lossless source, resolving aliases only when evidence supports it."
        )

    compact_bound_hints: list[dict[str, Any]] = []
    compact_keys = {
        "operation", "value", "unit", "selected_target", "answer_candidate",
        "parent_count", "subset_count", "current_age", "event_age",
        "required_terminal_event", "reason", "left_value", "right_value",
        "event_a_time", "event_b_time", "selected_time",
    }
    for hint in trace.get("generic_operator_hints") or []:
        if not isinstance(hint, dict) or hint.get("binding_complete") is not True:
            continue
        compact = {key: hint[key] for key in compact_keys if key in hint}
        for list_key in ("members", "operands"):
            if isinstance(hint.get(list_key), list):
                compact[list_key] = hint[list_key][:16]
        compact_bound_hints.append(compact)
    binding = trace.get("source_binding_certificate") or {}
    binding_summary = {
        key: binding.get(key) for key in (
            "entity_match", "relation_match", "comparison_complete",
            "binding_complete", "binding_source_ids", "target_support",
        ) if key in binding
    }
    return [
        {
            "role": "system",
            "content": (
                "Answer the memory question from the supplied provenance-bearing memory evidence. "
                "SOURCE_EVIDENCE is authoritative. A BOUND_FRAME or BOUND_GROUP with cited "
                "source IDs may also supply a normalized value, relation, quantity, lifecycle, "
                "or date when the cited source is elliptical or the fact is distributed across "
                "turns; use it unless source text directly contradicts it. Routing cards and "
                "unbound diagnostics are navigation only. Keep speaker and fact owner "
                "distinct, preserve negation, uncertainty, lifecycle and exact units, "
                "and resolve relative dates from the displayed source date. "
                "A provenance-bound ledger may be used only after its cited source facts "
                "match the question. Ignore unverified calculations. When a certified "
                "exact_entity_absence ledger says value=insufficient, the named entity "
                "was checked against user memory: do not substitute a relation-near entity; "
                "answer that the requested information is insufficient. "
                f"{class_policy} "
                "Do not abstain merely because a structural role certificate is incomplete; "
                "answer whenever a source, provenance-bound frame/group, or verified ledger supports "
                "a unique best answer. A partial but directly relevant statement is preferable "
                "to an unsupported refusal. Abstain only for a certified exact absence or when "
                "all provenance-bearing candidates fail to establish the requested entity and "
                "relation. Silently bind owner, entity and relation; resolve lifecycle and time; "
                "and verify any set or arithmetic result against cited members or operands. "
                "Do not reveal reasoning. "
                "Return only a concise answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question date: {case.question_date or 'unknown'}\n"
                f"Question: {case.question}\n\n"
                "Source-binding diagnostic (ranking aid, not answer evidence):\n"
                f"{json.dumps(binding_summary, ensure_ascii=False)}\n\n"
                "Verified deterministic ledger:\n"
                f"{json.dumps(compact_bound_hints, ensure_ascii=False)}\n\n"
                f"Memory evidence:\n{retrieval.context_text}"
            ),
        },
    ]
