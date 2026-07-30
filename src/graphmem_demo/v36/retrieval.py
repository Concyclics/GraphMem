from __future__ import annotations

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
from ..v3.build import canonical_key
from .operators import (
    evaluate_operators, query_bound_collection_ledger, counterfactual_dependency_hint, record_time_source_hint, temporal_order_source_hint,
    temporal_source_pair_hint, relative_time_from_sources_hint,
    transaction_sum_from_sources_hint, exact_entity_absence_hint,
    named_individual_event_members_hint, repeated_event_total_from_sources_hint,
    age_arithmetic_from_sources_hint, incomplete_terminal_event_hint,
    state_change_members_from_sources_hint, provenance_acquisition_members_hint,
    explicit_cuisine_categories_hint, subset_percentage_from_sources_hint,
    excluded_collection_members_hint, paired_metric_total_from_sources_hint,
    labeled_scalar_difference_from_sources_hint,
    repeated_activity_duration_total_hint, relative_anchor_source_hint,
    dated_event_count_from_sources_hint, same_unit_state_difference_hint,
    maintenance_entity_count_hint, category_acquisition_members_hint,
    pending_operation_target_pairs_hint,
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


V36_RETRIEVAL_VERSION = "graphmem_v36_generic_evidence_closure_20260730_locked"

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

def build_query_ir(question: str) -> QueryIR:
    content = _content_tokens(question)
    lowered = question.casefold()
    comparison_targets = _comparison_targets(question)
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
    elif _TEMPORAL_AFTER_FIRST_RE.search(question):
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
    owner_match = re.search(
        r"\b(?:did|does|is|was|has|have|can|could|would|will|should)\s+([A-Z][\w'-]+)\b",
        question, re.IGNORECASE,
    )
    target_owner = (
        canonical_key(owner_match.group(1)) if owner_match else ""
    )
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
    return views


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
        if edge.relation != "source":
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
        anchors = [0]
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


def _ranked_source_turns(
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
    protected_card_ids = set(selected_card_ids[:2])
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
        if preliminary.complete:
            expansion_rounds -= 1
            break
        reached, rows = _typed_expand(
            index, reached, set(preliminary.missing_roles), max_depth=1
        )
        graph_trace.extend(rows)
        # A typed relation may reach evidence in another session. Expand the
        # coarse scope to that exact owning card before accepting the evidence;
        # this is deterministic scope completion, not reverse graph traversal.
        scope_changed = False
        missing_now = set(preliminary.missing_roles)
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
    ranked_turns = _ranked_source_turns(
        index, ir, query_vectors, selected_cards, fine_ranked
    )
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
    context, packed_groups, packed_frames, source_ids, ledger = _pack(
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
    # Fine packing may omit the one decisive user proposition from a routed
    # memory region. Operators inspect all user turns inside the bounded coarse
    # route, while their local binders still require entity/relation evidence.
    routed_session_set = {card.session_id for card in selected_cards}
    routed_user_source_ids = [
        turn.node_id for turn in index.turns
        if turn.session_id in routed_session_set
        and turn.transport_role == "user"
    ]
    # Typed binders are strict enough to use a lossless global safety pool.
    # Coarse routing remains the normal path; this pool only prevents a missed
    # card from hiding an exact entity/date/operand source from an operator.
    all_user_source_ids = [
        turn.node_id for turn in index.turns
        if turn.transport_role == "user"
    ]
    operator_source_ids = list(dict.fromkeys([
        *source_ids, *routed_user_source_ids, *all_user_source_ids,
    ]))
    temporal_hint = temporal_source_pair_hint(ir, index, operator_source_ids)
    relative_time_hint = relative_time_from_sources_hint(
        ir, index, operator_source_ids, case.question_date,
    )
    transaction_sum_hint = transaction_sum_from_sources_hint(
        ir, index, operator_source_ids,
    )
    exact_absence_hint = exact_entity_absence_hint(ir, index)
    named_members_hint = named_individual_event_members_hint(
        ir, index, list(dict.fromkeys([*source_ids, *routed_user_source_ids])),
    )
    repeated_event_hint = repeated_event_total_from_sources_hint(
        ir, index, operator_source_ids,
    )
    age_arithmetic_hint = age_arithmetic_from_sources_hint(
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
    labeled_difference_hint = labeled_scalar_difference_from_sources_hint(
        ir, index, operator_source_ids,
    )
    repeated_duration_hint = repeated_activity_duration_total_hint(
        ir, index, operator_source_ids,
    )
    dated_event_hint = dated_event_count_from_sources_hint(
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
    relative_anchor_hint = relative_anchor_source_hint(
        ir, index, case.question_date,
        allowed_session_ids={card.session_id for card in selected_cards},
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
    if temporal_hint is None:
        temporal_hint = temporal_order_source_hint(
            ir, index, operator_source_ids
        )
    if (
        temporal_hint is not None
        and certificate.entity_match
        and certificate.relation_match
        and certificate.scope_match
        and certificate.provenance_complete
    ):
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
    if temporal_hint is not None and certificate.complete:
        operator_hints = [
            hint for hint in operator_hints
            if hint.get("operation") not in {"duration_total", "time_difference"}
        ]
        operator_hints.insert(0, temporal_hint)
    if relative_time_hint is not None:
        operator_hints = [
            hint for hint in operator_hints
            if hint.get("operation") not in {"duration_total", "event_time"}
        ]
        operator_hints.insert(0, relative_time_hint)
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
        age_arithmetic_hint, incomplete_terminal_hint, state_change_hint,
        provenance_acquisition_hint, cuisine_categories_hint,
        subset_percentage_hint, excluded_collection_hint, paired_metric_hint,
        labeled_difference_hint, repeated_duration_hint, dated_event_hint,
        same_unit_difference_hint, maintenance_count_hint,
        category_acquisition_hint, pending_pairs_hint,
    ):
        if typed_hint is not None:
            operator_hints.insert(0, typed_hint)
    if exact_absence_hint is not None:
        operator_hints.insert(0, exact_absence_hint)
    if incomplete_terminal_hint is not None:
        operator_hints = [
            hint for hint in operator_hints
            if hint.get("operation") in {
                "terminal_event_completion_check",
                "global_lossless_source_candidates",
            }
        ]
    elif any(hint is not None for hint in (
        state_change_hint, cuisine_categories_hint, subset_percentage_hint,
        excluded_collection_hint, paired_metric_hint, labeled_difference_hint,
        repeated_duration_hint, dated_event_hint, same_unit_difference_hint,
        maintenance_count_hint, category_acquisition_hint, pending_pairs_hint,
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
        if candidate_hints:
            context += (
                "\n\n[UNVERIFIED_OPERATOR_CANDIDATES]\n"
                "These calculations do not prove operand scope closure. "
                "Do not copy their value; recompute from matching source facts.\n"
                + "\n".join(str(hint) for hint in candidate_hints)
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
        "completeness_certificate": asdict(certificate),
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
    trace = retrieval.retrieval_trace or {}
    ir = trace.get("query_ir") or {}
    terminal_incomplete = any(
        hint.get("operation") == "terminal_event_completion_check"
        and hint.get("value") == "insufficient"
        for hint in (trace.get("generic_operator_hints") or [])
        if isinstance(hint, dict)
    )
    if terminal_incomplete:
        class_policy = (
            "This is a bounded-interval question whose explicitly named terminal event "
            "was not completed. The provenance-bound terminal_event_completion_check is "
            "decisive: answer that the information is insufficient. Do not calculate to a "
            "different degree, phase, current date, or most recent completed endpoint. "
        )
    elif retrieval.query_kind == "recommendation":
        class_policy = (
            "This is a personalized recommendation request. Bind the answer to "
            "every explicit owner, place, object, activity and other constraint "
            "in the question. Never substitute a similar option in a different "
            "city or for a different entity. Infer useful constraints from "
            "user-owned experiences, successes, dislikes, skills, resources and "
            "current needs. A recommendation need not have appeared verbatim in "
            "memory, and it need not claim live availability; give a concise "
            "memory-grounded suggestion when the supported constraints suffice. "
            "If the question introduces a new place or future situation, transfer "
            "the supported feature preferences without claiming live availability. "
            "For a future recommendation, the requested destination or target is "
            "where the recommendation will be used, not an evidence entity that must "
            "already occur in memory. Transfer supported preference attributes from "
            "analogous past choices, but never return the old target as the answer. "
            "If no named option is grounded, recommend a property or option profile "
            "that satisfies those supported features instead of abstaining. Explicitly "
            "mention the relevant contrasted options or features found in evidence. "
            "When prior evidence compares named candidate options, name those options "
            "and compare their supported attributes rather than giving generic tips. "
        )
    elif retrieval.query_kind in {"count", "list", "aggregate", "duration"}:
        class_policy = (
            "This is a structured quantity or collection question. Bind every operand "
            "to the requested owner, entity, relation, time range and lifecycle. "
            "When a PROVENANCE_BOUND_OPERATOR_LEDGER has binding_complete=true, its "
            "source-cited operand/member set and arithmetic are the closed result for that "
            "operation; verify the cited rows, then use it instead of a noisier incomplete "
            "frame count. A terminal_event_completion_check with value=insufficient means "
            "the named endpoint was never completed, so do not substitute a different degree "
            "or interval endpoint. "
            "Deduplicate alternate frames from the same source fact. Exclude budgets, "
            "goals, examples, suggestions, cancelled items, and quantities with a wrong "
            "unit or relation. Preserve whether a duration is per direction, per "
            "occurrence, or already a total. For how-many-days-ago, subtract the named "
            "event date from the Question date, not from the source session date. For "
            "how long something had lasted when another event happened, subtract the "
            "event recency from the current tenure. For time worked before a current "
            "job, subtract current-job tenure from total professional tenure. Enumerate "
            "valid operands before answering. For how-many-units older than an "
            "earlier event, the answer is the elapsed time since that event. "
            "For frequency, distinguish distinct days from the number of classes "
            "or sessions held on those days. A query-bound collection ledger is "
            "a compact provenance view, not proof of scope closure: enumerate every "
            "structured member candidate first, then inspect every unframed lossless "
            "candidate and every explicit relation-closure candidate for an omitted "
            "member. When a worked-on project explicitly features, contains, or includes "
            "a collection-typed artifact, treat that artifact as the operation target. "
            "A single source can establish multiple named members. Treat every "
            "ledger row as a candidate rather than a complete checklist or certified "
            "count. Re-read the cited owner source, derive the concrete member identity, "
            "and match the requested counting unit: when the question counts people or "
            "other individuals, every separately named member of a plural group (including "
            "twins or siblings) is one member, not one group event. For acquire, receive, "
            "or inherit relations, a direct ownership-provenance statement such as an item "
            "being from or having belonged to a named source establishes acquisition when "
            "the owner and requested provenance scope match, even if the same sentence does "
            "not repeat the query verb. Then reject category labels, background activities, "
            "goals, contexts, and "
            "nearby entities that do not themselves satisfy the requested operation. "
            "A member name need not repeat the collection head when its source or routing "
            "context establishes the semantic type. For distinct taxonomy or category "
            "counts, map each source-confirmed named instance to its category (for example, "
            "a named dish to its cuisine) even when the sentence omits the category label; "
            "show the instance-to-category mapping before deduplication. Build the final set across every "
            "routed owner source, then deduplicate source-confirmed aliases by entity "
            "identity; never answer from the number of ledger rows. "
            "Merge only source-confirmed aliases, apply "
            "add/remove/cancel and lifecycle semantics, then answer from the resulting "
            "operation-target or distinct-member set. Count pending operation-target pairs, "
            "not merely distinct entity labels: when evidence distinguishes an old "
            "item to return from its replacement to pick up, they remain separate "
            "members. If the ledger supplies a "
            "derived weekly-occurrence or bounded-interval value, verify its cited "
            "days or endpoints and prefer it over counting distinct labels or adding "
            "overlapping phase durations. "
        )
    elif "members" in set(ir.get("required_roles") or []):
        class_policy = (
            "This is a scoped collection question. Enumerate every distinct member "
            "supported by owner-bound frames and lossless sources across all routed "
            "sessions. A named member remains valid when its source establishes the "
            "requested relation even if the frame uses a paraphrase. Preserve positive "
            "and negative preference polarity and the stated owner/subject. Merge only "
            "source-confirmed aliases, preserve provenance, and omit suggestions or "
            "hypothetical examples. "
        )
    elif retrieval.query_kind == "span" and bool(ir.get("temporal_constraints")):
        class_policy = (
            "This is a source-dated past-event lookup. Resolve relative time against "
            "the Question date and source date, and select the closest supported "
            "event. Natural-language week counts are rounded calendar intervals; "
            "four weeks ago may be 27-29 days earlier. "
        )
    elif retrieval.query_kind in {"entity", "date"}:
        class_policy = (
            "This is a fact lookup with possible relative time. Treat today, "
            "yesterday, or ago in a source as anchored to that source date, then "
            "compare it with the Question date. Return the entity from the source "
            "whose anchored event satisfies the requested interval. "
        )
    elif {"condition", "effect"} <= set(ir.get("required_roles") or []):
        class_policy = (
            "This is a counterfactual dependency question. Identify the stated "
            "condition and the supported motive, cause, or enabling relation for the "
            "effect. Return a likely yes/no only when the sources establish that "
            "dependency. Explicit causal language such as because, made, motivated, "
            "led to, instrumental, or so I started establishes a dependency when it "
            "connects the named condition and effect; for a positive cause of the effect, "
            "negating that cause implies a likely no. Reverse the effect under the "
            "negated condition without "
            "inventing a new cause. Otherwise state that the memory is insufficient. "
        )
    elif retrieval.query_kind == "state":
        class_policy = (
            "This is a state question. Compare only values for the same entity and "
            "attribute. For highest, lowest, best, or record questions use the supported "
            "numeric extreme, not merely the latest mention. For a ratio A per B, a "
            "smaller B denominator means less B per A. Ignore same-topic evidence with "
            "a different predicate. "
        )
    else:
        class_policy = (
            "For previous-chat, ordinal, or sequence questions, use the lossless source "
            "and preserve list position and turn order exactly: locate the requested "
            "numbered item, requested version, or item immediately after the anchor. "
            "When asked what comes after a quoted sequence anchor, return the next item, "
            "not the final token inside the anchor. Treat an activity or entity name with "
            "an extra lexical modifier as distinct unless evidence explicitly aliases it; "
            "for example, a compound activity is not established by its head word alone. "
            "Exclude hypothetical examples and unexecuted assistant suggestions. "
            "If the question says initially, first, or earlier, use the earliest "
            "valid assertion in the requested scope rather than the latest state. "
        )
    compact_bound_hints = []
    compact_keys = {
        "operation", "value", "unit", "selected_target", "answer_candidate",
        "parent_count", "subset_count", "current_age", "event_age",
        "required_terminal_event", "reason", "left_value", "right_value",
        "event_a_time", "event_b_time", "selected_time",
    }
    for hint in (trace.get("generic_operator_hints") or []):
        if not isinstance(hint, dict) or hint.get("binding_complete") is not True:
            continue
        compact = {key: hint[key] for key in compact_keys if key in hint}
        if "selected_target" in hint:
            compact["answer_value"] = hint["selected_target"]
        elif "answer_candidate" in hint:
            compact["answer_value"] = hint["answer_candidate"]
        elif "value" in hint:
            compact["answer_value"] = hint["value"]
        for list_key in ("members", "operands"):
            if isinstance(hint.get(list_key), list):
                compact[list_key] = [
                    {key: value for key, value in row.items() if key in {
                        "identity", "category", "target", "value", "value_minutes",
                        "source_turn_id", "source_turn_ids", "derivation",
                    }} if isinstance(row, dict) else row
                    for row in hint[list_key][:16]
                ]
        compact_bound_hints.append(compact)
    decisive_block = (
        json.dumps(compact_bound_hints, ensure_ascii=False)
        if compact_bound_hints else "none"
    )
    return [
        {
            "role": "system",
            "content": (
                "Answer the memory question using only the supplied routing cards, "
                "role frames, complete evidence groups, and source turns. Routing "
                "cards locate evidence but are not themselves proof. Respect fact "
                "ownership, negation, lifecycle status, collection operations and "
                "time. Resolve relative dates against the date shown on their source turn. "
                "An operator row with binding_complete=True is a provenance-bound "
                "deterministic calculation whose cited operands have been matched to "
                "the requested relation and scope. Prefer its computed result unless "
                "a cited direct source contradicts the binding. When a bound temporal "
                "anchor row includes answer_candidate, that field is the requested role "
                "extracted from its date-matched source. Operator candidates "
                "without binding_complete=True are diagnostic only: never copy their "
                "value, and recompute from matching source facts instead. "
                "Direct lossless sources override a duplicated or mismatched operator. For aggregate, comparison, or "
                "ordering questions, do not guess when the evidence-completeness "
                "record lists a required role as missing. Match the complete requested "
                "entity expression; do not infer identity from a partial lexical "
                "match alone. A common unambiguous alias or containment relation may "
                "be used when the cited evidence supports that interpretation. "
                "For a temporal-order comparison, compare only the two named "
                "alternatives and bind each one to its own dated source. "
                "For count or list questions, treat each supported operation-target "
                "pair as a distinct candidate; different operations and explicitly "
                "old/new versions remain distinct unless the source identifies the "
                "same outstanding item. "
                f"{class_policy}When dated user-owned values conflict, prefer the latest "
                "valid direct assertion. A value presupposed as the baseline of a goal "
                "still establishes that baseline, while the goal itself is not a "
                "completed result. A concrete fact used as the premise of a user "
                "request (for example, an owned entity described by an attribute) "
                "is a user assertion and may establish that fact. "
                "Prefer a complete evidence group over an isolated similar "
                "sentence. Do not reveal hidden reasoning. If the evidence and "
                "provenance are insufficient, say that the memory does not establish "
                "the answer. Return only a concise answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question date:\n{case.question_date or 'unknown'}\n\n"
                f"Question:\n{case.question}\n\n"
                "Decisive provenance-bound ledger (verify cited sources, then use it):\n"
                f"{decisive_block}\n\n"
                f"Memory evidence:\n{retrieval.context_text}"
            ),
        },
    ]
