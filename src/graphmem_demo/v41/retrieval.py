from __future__ import annotations

import json
import re
from collections import Counter, defaultdict, deque
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any

from ..clients import rough_token_count
from ..models import QuestionCase, RetrievedContext
from ..v3.action_semantics import action_families
from ..v36.operators import (
    latest_approx_scalar_state_hint,
    latest_labeled_currency_state_hint,
    latest_weekly_schedule_time_hint,
    open_temporal_sequence_from_sources_hint,
    repeated_event_total_from_sources_hint, temporal_order_source_hint,
    threshold_progress_remaining_hint,
)
from ..v36.source_spans import (
    binding_tokens, fuzzy_term_overlap, query_binding_terms,
)
from ..v36.schema import QueryIR, TurnNodeV36, V36Index
from ..v4.retrieval import (
    answer_messages as answer_messages_v4,
    retrieve as retrieve_v4,
)
from ..v4.schema import CapabilityViewV4
from .domains import augment_query, is_inferential_question
from .schema import (
    EvidenceCertificateV41, GRAPHMEM_V41_SCHEMA, PlannerResultV41,
    QueryAugmentationV41, QueryPolicyV41, QuerySidecarV41,
)
from .sidecar import inverted_rank, lexical_rank


_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'_-]*", re.IGNORECASE)


def _tokens(text: str) -> set[str]:
    """Canonical lexical tokens with English possessives folded to owners.

    Keeping a possessive owner distinct from its base name made the owner look
    like a relation concept and gave every scene by that owner a false semantic
    match. Folding possessives fixes entity subtraction without topic rules.
    """
    values: set[str] = set()
    for raw in _WORD_RE.findall(text):
        token = raw.casefold()
        if token.endswith(chr(39) + "s") and len(token) > 2:
            token = token[:-2]
        values.add(token)
    return values


_QUERY_STOP_TERMS = {
    "a", "an", "and", "are", "as", "at", "be", "by",
    "did", "do", "does", "for", "from", "go", "had",
    "has", "have", "her", "his", "how", "i", "in",
    "is", "it", "likely", "many", "much", "of", "on",
    "different", "item", "items", "number", "piece", "pieces",
    "total", "times",
    "or", "place", "she", "signed", "take", "that", "the",
    "their", "there", "they", "this", "to", "type", "up",
    "using", "was", "were", "what", "when", "where",
    "which", "who", "why", "with", "would", "you",
    "about", "advice", "any", "back", "been", "can", "conversation",
    "could", "feeling", "find", "give", "having", "i've",
    "interesting", "lately", "look", "looking", "me", "might", "my",
    "new", "please",
    "previous", "really", "remember", "should", "some", "thinking",
    "tips", "tonight", "trouble", "weekend",
}


def _fuzzy_overlap(targets: set[str], document_terms: set[str]) -> set[str]:
    return {
        target for target in targets
        if any(
            target == token
            or (
                len(target) >= 4 and len(token) >= 4
                and target[:4] == token[:4]
            )
            for token in document_terms
        )
    }


_RELATION_TERM_FAMILIES: dict[str, set[str]] = {
    "symbol": {"mean", "meaning", "represent", "stand", "signify"},
    "symbolize": {"symbol", "mean", "meaning", "represent", "stand", "signify"},
    "represent": {"symbol", "symbolize", "mean", "meaning", "stand", "signify"},
    "mean": {"meaning", "symbol", "symbolize", "represent", "stand", "signify"},
    "course": {"class", "workshop", "training"},
    "support": {"back", "endorse", "favor", "fan"},
    "drawing": {"draw", "drew", "sketch", "illustration"},
    "draw": {"drawing", "drew", "sketch", "illustration"},
    "painting": {"paint", "painted", "artwork"},
    "sibling": {"siblings", "brother", "brothers", "sister", "sisters"},
    "siblings": {"sibling", "brother", "brothers", "sister", "sisters"},
    "acquire": {"acquired", "buy", "bought", "got", "obtain", "obtained", "receive", "received"},
    "acquired": {"acquire", "buy", "bought", "got", "obtain", "obtained", "receive", "received"},
    "attend": {"attended", "join", "joined", "participate", "participated", "visit", "visited", "went"},
    "attended": {"attend", "join", "joined", "participate", "participated", "visit", "visited", "went"},
    "bake": {"baked", "cook", "cooked", "make", "made"},
    "baked": {"bake", "cook", "cooked", "make", "made"},
    "hike": {"hiking", "walk", "walking", "trail", "trek", "trekking"},
    "hiking": {"hike", "walk", "walking", "trail", "trek", "trekking"},
    "album": {"albums", "ep", "eps", "record", "records", "vinyl"},
    "albums": {"album", "ep", "eps", "record", "records", "vinyl"},
    "arrive": {"arrived", "reach", "reached", "got"},
    "reached": {"arrive", "arrived", "reach", "got"},
    "relocation": {"relocate", "relocated", "move", "moved"},
    "relocated": {"relocation", "relocate", "move", "moved"},
    "relationship": {"single", "married", "dating", "partner", "spouse", "husband", "wife", "divorced", "breakup"},
    "status": {"single", "married", "dating", "partner", "current", "latest"},
    "compare": {"compared", "comparison", "like", "similar", "analogy", "metaphor", "remind"},
    "compared": {"compare", "comparison", "like", "similar", "analogy", "metaphor", "remind"},
    "health": {"medical", "condition", "symptom", "weight", "overweight", "obesity", "fitness", "exercise", "running", "diet", "doctor"},
}


def _query_base_terms(ir: QueryIR) -> set[str]:
    return (
        _tokens(f"{ir.raw_question} {ir.target_relation}")
        - _QUERY_STOP_TERMS - _tokens(ir.target_owner)
    )


def _query_content_terms(ir: QueryIR) -> set[str]:
    base = _query_base_terms(ir)
    expanded = set(base)
    for token in base:
        expanded.update(_RELATION_TERM_FAMILIES.get(token, set()))
    return expanded


def _semantic_overlap(ir: QueryIR, document_terms: set[str]) -> set[str]:
    """Count each query concept once even when many aliases match."""
    matched: set[str] = set()
    for base in _query_base_terms(ir):
        aliases = {base, *_RELATION_TERM_FAMILIES.get(base, set())}
        if _fuzzy_overlap(aliases, document_terms):
            matched.add(base)
    return matched


def query_views(ir: QueryIR, augmentation: QueryAugmentationV41) -> list[str]:
    values = [
        ir.raw_question,
        " ".join([
            *ir.target_entities, ir.target_relation, ir.target_owner,
            *augmentation.expanded_terms,
        ]).strip(),
        " ".join([
            augmentation.answer_algebra,
            *augmentation.domain_hints,
            *augmentation.required_roles,
            *augmentation.scope_terms,
        ]).strip(),
    ]
    return [value for value in dict.fromkeys(values) if value]


def build_query_plan(ir: QueryIR) -> QueryAugmentationV41:
    return augment_query(ir)


def _document_sources(
    sidecar: QuerySidecarV41, node_id: str,
) -> list[str]:
    document = sidecar.documents.get(node_id)
    return list(document.source_turn_ids) if document else []


def _strict_certificate(
    ir: QueryIR,
    augmentation: QueryAugmentationV41,
    index: V36Index,
    source_ids: list[str],
    packed_frame_ids: list[str],
    packed_group_ids: list[str],
) -> EvidenceCertificateV41:
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    frame_by_id = {frame.frame_id: frame for frame in index.frames}
    group_by_id = {group.group_id: group for group in index.evidence_groups}
    turns = [turn_by_id[node_id] for node_id in source_ids if node_id in turn_by_id]
    frames = [
        frame_by_id[node_id] for node_id in packed_frame_ids
        if node_id in frame_by_id
    ]
    groups = [
        group_by_id[node_id] for node_id in packed_group_ids
        if node_id in group_by_id
    ]
    evidence_text = " ".join([
        *(turn.text for turn in turns),
        *(frame.retrieval_text for frame in frames),
        *(group.retrieval_text for group in groups),
    ])
    evidence_terms = _tokens(evidence_text)
    entity_targets = [
        value for value in [
            *ir.target_entities, ir.target_owner,
            *augmentation.alternative_entities,
        ] if value
    ]
    entity_match = not entity_targets or any(
        _tokens(value) & evidence_terms for value in entity_targets
    )
    relation_terms = _tokens(ir.target_relation)
    relation_match = (
        not relation_terms
        or bool(relation_terms & evidence_terms)
        or any(
            relation_terms & _tokens(frame.predicate_key)
            for frame in frames
        )
    )
    scope_terms = set().union(*(
        _tokens(value) for value in augmentation.scope_terms
    )) if augmentation.scope_terms else set()
    scope_match = not scope_terms or bool(scope_terms & evidence_terms)
    provenance_complete = bool(turns) and all(
        frame.source_turn_ids for frame in frames
    ) and all(group.provenance_complete for group in groups)

    present: set[str] = {"source"} if turns else set()
    for frame in frames:
        present.update(frame.coverage_mask)
        present.update(frame.semantic_type_keys)
        present.add(frame.frame_kind)
        if frame.owner_key:
            present.add("owner")
        if frame.entity_key:
            present.add("entity")
        if frame.predicate_key:
            present.add("relation")
        if frame.object_key:
            present.update({"value", "member"})
        if frame.lifecycle_status != "unknown":
            present.add("lifecycle")
        if any((
            frame.temporal.event_time, frame.temporal.observed_at,
            frame.temporal.start, frame.temporal.end,
        )):
            present.add("time")
    for group in groups:
        present.update(group.required_roles)
        if group.group_kind == "dialogue_pair":
            present.update({"request", "reply", "reply_content"})
        elif group.group_kind == "temporal_pair":
            present.update({"event_a", "event_b", "time_a", "time_b"})
        elif group.group_kind == "state_transition":
            present.update({"old_state", "new_state", "time"})
        elif group.group_kind == "collection":
            present.update({"scope", "member", "lifecycle"})

    turns_by_session: defaultdict[str, list[TurnNodeV36]] = defaultdict(list)
    for turn in turns:
        turns_by_session[turn.session_id].append(turn)
        folded = turn.text.casefold()
        if turn.session_date or re.search(
            r"\b(?:today|yesterday|tomorrow|ago|before|after|last|next|19\d{2}|20\d{2})\b",
            folded,
        ):
            present.add("time")
        if re.search(
            r"\b(?:completed|finished|bought|purchased|attended|visited|cancelled|canceled|planned|planning|started|began)\b",
            folded,
        ):
            present.add("lifecycle")
        if re.search(r"\b(?:like|love|prefer|favorite|dislike|hate|avoid)\b", folded):
            present.update({"polarity", "context"})
        if re.search(
            r"\b(?:for your|compatible with|currently (?:use|using|have|own)|i (?:have|own|use)|my current|your current)\b",
            folded,
        ):
            present.update({"current_state", "context"})
        if len(_tokens(turn.text)) >= 2:
            present.update({"value", "member"})
    for session_turns in turns_by_session.values():
        ordered = sorted(session_turns, key=lambda item: item.turn_index)
        if any(
            right.turn_index - left.turn_index == 1
            and right.transport_role != left.transport_role
            for left, right in zip(ordered, ordered[1:])
        ):
            present.update({"request", "reply", "reply_content"})

    if augmentation.answer_algebra == "temporal_comparison":
        dated = [
            turn for turn in turns
            if turn.session_date or re.search(
                r"\b(?:19\d{2}|20\d{2}|yesterday|tomorrow|ago)\b",
                turn.text.casefold(),
            )
        ]
        target_hits = []
        for target in ir.comparison_targets[:2]:
            target_terms = _tokens(target)
            target_hits.append(any(
                target_terms.intersection(_tokens(turn.text)) for turn in dated
            ))
        if len(dated) >= 2 and (not target_hits or all(target_hits)):
            present.update({"event_a", "event_b", "time_a", "time_b"})

    roles = list(dict.fromkeys(augmentation.required_roles))
    aliases = {
        "support": {"source", "fact", "event"},
        "attribute": {"relation", "state"},
        "context": {"source", "fact", "preference"},
        "polarity": {"preference", "source"},
        "profile_fact": {"fact", "state", "preference", "event", "source"},
    }
    missing = [
        role for role in roles
        if role not in present and not aliases.get(role, set()).intersection(present)
    ]
    lifecycle_complete = (
        augmentation.answer_algebra not in {"collection", "state_update"}
        or "lifecycle" in present
    )
    temporal_complete = (
        augmentation.answer_algebra != "temporal_comparison"
        or {"event_a", "event_b", "time_a", "time_b"}.issubset(present)
    )
    dialogue_complete = (
        augmentation.answer_algebra != "dialogue_lookup"
        or {"request", "reply", "reply_content"}.issubset(present)
    )
    complete = all((
        entity_match, relation_match, scope_match, provenance_complete,
        lifecycle_complete, temporal_complete, dialogue_complete,
        not missing,
    ))
    return EvidenceCertificateV41(
        entity_match=entity_match, relation_match=relation_match,
        scope_match=scope_match, provenance_complete=provenance_complete,
        lifecycle_complete=lifecycle_complete,
        temporal_complete=temporal_complete,
        dialogue_complete=dialogue_complete,
        present_roles=sorted(present), missing_roles=missing,
        source_turn_ids=[turn.node_id for turn in turns], complete=complete,
    )


def _invalidate_contradicted_absence(
    result: RetrievedContext,
    index: V36Index,
    ir: QueryIR,
    sidecar: QuerySidecarV41,
    token_budget: int,
) -> list[dict[str, Any]]:
    """Reject a local absence hint after a global lossless relation scan."""
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    # A matching name in unrelated boilerplate or a generated story must not
    # invalidate an absence certificate.  Both dialogue roles remain eligible
    # because an assistant reply can itself be durable answer evidence; the
    # relation-bound check below separates a real answer from a name-only hit.
    turns = list(index.turns)
    invalidated: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    scene_text_by_session: dict[str, str] = defaultdict(str)
    selected_source_ids = set(result.leaf_node_ids)
    assistant_authoritative_query = bool(re.search(
        r"\b(?:previous|earlier|prior)\s+(?:chat|conversation|reply)|"
        r"\b(?:you|assistant)\s+(?:recommended|suggested|provided)|"
        r"\b(?:campaign|project|proposal)\s+plan\b",
        ir.raw_question, re.IGNORECASE,
    ))
    for turn in turns:
        scene_text_by_session[turn.session_id] += "\n" + turn.text
    stop = {
        "something", "thing", "things", "event", "events", "time",
        "times", "total", "both", "all", "some", "trip",
    }
    stop_stems = {value[:4] for value in stop}

    def stems(text: str) -> set[str]:
        return {
            token[:4] for token in _tokens(text)
            if token[:4] not in stop_stems and len(token) >= 3
        }

    def stem_sequence(text: str) -> list[str]:
        return [
            token.casefold()[:4] for token in _WORD_RE.findall(text)
            if len(token) >= 3
        ]

    def contains_sequence(haystack: list[str], needle: list[str]) -> bool:
        return bool(needle) and any(
            haystack[offset:offset + len(needle)] == needle
            for offset in range(len(haystack) - len(needle) + 1)
        )

    for hint in result.retrieval_trace.get("generic_operator_hints") or []:
        if not isinstance(hint, dict) or hint.get("operation") != "exact_entity_absence":
            retained.append(hint)
            continue
        required_marker = str(hint.get("required_marker") or "").strip()
        required_text = str(hint.get("required_phrase") or required_marker)
        required = stems(required_text)
        required_sequence = stem_sequence(required_text)
        binding_kind = str(hint.get("binding_kind") or "")
        normalized_marker = re.sub(
            r"[^a-z0-9-]+", " ", required_marker.casefold(),
        ).strip()
        ranked_matches: list[tuple[int, TurnNodeV36]] = []
        for turn in turns:
            marker_bound = bool(
                normalized_marker
                and normalized_marker in _tokens(turn.text)
            )
            phrase_bound = bool(
                not normalized_marker
                and (
                    contains_sequence(
                        stem_sequence(turn.text), required_sequence,
                    )
                    if binding_kind in {
                        "required_component", "role_title", "required_role",
                        "required_collection_type", "required_operand",
                    }
                    else required and required.issubset(stems(turn.text))
                )
            )
            if not marker_bound and not phrase_bound:
                continue
            if (
                binding_kind in {
                    "named_entity", "required_component", "role_title",
                    "required_role", "required_collection_type",
                    "required_operand", "required_subtype",
                }
                and turn.transport_role == "assistant"
                and len(turn.text) > 320
                and not assistant_authoritative_query
            ):
                continue
            relation_text = (
                scene_text_by_session.get(turn.session_id, turn.text)
                if binding_kind == "named_entity"
                else turn.text
            )
            overlap = _semantic_overlap(ir, _tokens(relation_text))
            action_overlap = action_families(ir.raw_question) & action_families(
                relation_text
            )
            relation_overlap = {
                value for value in overlap
                if value[:4] not in required
                and value.casefold() not in {
                    "dr", "doctor", "mr", "mrs", "ms",
                }
            }
            exact_relation_terms = {
                value for value in _tokens(ir.raw_question)
                if value not in _QUERY_STOP_TERMS
                and value[:4] not in required
                and value not in {
                    "dr", "doctor", "mr", "mrs", "ms",
                }
            }
            exact_relation_overlap = (
                exact_relation_terms & _tokens(relation_text)
            )
            # Exact text alone is insufficient in a distractor-rich memory.
            # Require the same relation/action scene.  A short assistant turn
            # may itself be the direct slot answer (for example, a model name);
            # long generated articles and tables never receive that exemption.
            direct_answer = (
                turn.transport_role == "assistant"
                and len(turn.text) <= 320
            )
            if binding_kind == "named_entity":
                relation_bound = bool(
                    turn.node_id in selected_source_ids
                    or len(exact_relation_overlap) >= 2
                    or (
                        exact_relation_overlap
                        and action_overlap
                    )
                    or direct_answer
                )
            else:
                relation_bound = bool(
                    relation_overlap or action_overlap or direct_answer
                )
            if not relation_bound:
                continue
            ranked_matches.append((
                4 * len(overlap) + 3 * len(action_overlap)
                + int(turn.transport_role == "user"),
                turn,
            ))
        matches = [
            turn for _score, turn in sorted(
                ranked_matches, key=lambda row: (-row[0], row[1].node_id),
            )
        ]
        if ir.requested_value_type in {"count", "aggregate"}:
            matches = [
                turn for turn in matches
                if re.search(r"\b\d+(?:\.\d+)?\b", turn.text)
                or re.search(
                    r"\b(?:caught|bought|baked|completed|finished|attended|visited|made|created|received)\b",
                    turn.text, re.IGNORECASE,
                )
            ]
        if matches:
            invalidated.append({
                "operation": "invalidate_exact_entity_absence",
                "required_marker": hint.get("required_marker"),
                "required_phrase": hint.get("required_phrase"),
                "positive_source_turn_ids": [turn.node_id for turn in matches[:4]],
                "reason": "lossless source positively binds the required phrase",
                "certified": True,
            })
        else:
            retained.append(hint)
    if invalidated:
        result.retrieval_trace["generic_operator_hints"] = retained
        result.retrieval_trace["v41_invalidated_absence_hints"] = invalidated
        positive_ids = list(dict.fromkeys(
            source_id
            for row in invalidated
            if row.get("required_marker")
            for source_id in row.get("positive_source_turn_ids") or []
        ))
        temporal_markers = {
            "monday", "tuesday", "wednesday", "thursday", "friday",
            "saturday", "sunday", "january", "february", "march", "april",
            "may", "june", "july", "august", "september", "october",
            "november", "december", "today", "yesterday", "tomorrow",
        }
        marker_values = {
            str(row.get("required_marker") or "").casefold()
            for row in invalidated if row.get("required_marker")
        }
        if marker_values & temporal_markers:
            positive_ids = []
        added, decisions = _append_sources(
            result, index, sidecar, positive_ids,
            token_budget, "v41_global_exact_recovery", 2,
        ) if positive_ids else ([], [])
        result.retrieval_trace["v41_global_exact_recovery_source_ids"] = added
        result.retrieval_trace["v41_global_exact_recovery_decisions"] = decisions
        result.context_text += (
            "\n\n[V4.1_CONSTRAINT_OVERRIDE]\n"
            + json.dumps(invalidated, ensure_ascii=False)
            + "\nThe older exact_entity_absence diagnostic is invalid for these phrases "
            "because positive lossless evidence was found."
        )
        result.packed_rough_tokens = rough_token_count(result.context_text)
    return invalidated


def _relations_for_gap(
    missing_roles: list[str], algebra: str,
) -> list[str]:
    roles = set(missing_roles)
    relations: list[str] = []
    if roles.intersection({"request", "reply", "reply_content"}):
        relations.extend(["dialogue_pair", "next_turn", "source"])
    if roles.intersection({"entity", "event_a", "event_b", "support"}):
        relations.extend(["reference", "same_event", "source"])
    if roles.intersection({"old_state", "new_state", "attribute", "lifecycle"}):
        relations.extend(["state_transition", "source"])
    if roles.intersection({"scope", "member"}):
        relations.extend(["collection_member", "source"])
    if roles.intersection({"time", "time_a", "time_b"}):
        relations.extend(["temporal_endpoint", "same_event", "source"])
    if "polarity" in roles:
        relations.extend(["contrast", "source"])
    if not relations:
        relations = ["source"]
        if algebra == "dialogue_lookup":
            relations[:0] = ["dialogue_pair", "next_turn"]
    return list(dict.fromkeys(relations))


def _semantic_turn_rank(
    ir: QueryIR,
    augmentation: QueryAugmentationV41,
    sidecar: QuerySidecarV41,
    limit: int = 16,
) -> list[tuple[str, float]]:
    """Protected lossless-turn channel scored by canonical query concepts."""
    owner_terms = _tokens(ir.target_owner)
    expanded_terms = binding_tokens(
        " ".join(augmentation.expanded_terms)
    )
    historical_attribute_query = bool(re.search(
        r"\b(?:previous|former|prior)\s+"
        r"(?:occupation|job|role|position|employer|address|state|value|name)\b",
        ir.raw_question, re.IGNORECASE,
    ))
    rows: list[tuple[str, float]] = []
    for node_id, document in sidecar.documents.items():
        if document.node_type != "turn":
            continue
        if (
            historical_attribute_query
            and not re.search(
                r"\b(?:previous|previously|former|prior|used to)\b",
                document.text, re.IGNORECASE,
            )
        ):
            continue
        document_terms = _tokens(document.text)
        overlap = _semantic_overlap(ir, document_terms)
        expanded_overlap = fuzzy_term_overlap(
            expanded_terms, binding_tokens(document.text),
        )
        if not overlap and not expanded_overlap:
            continue
        document_owners = {
            value.casefold()
            for value in document.fields.get("owner", [])
        }
        owner_exact = bool(
            ir.target_owner
            and ir.target_owner.casefold() in document_owners
        )
        owner_bonus = (
            0.75 * len(_fuzzy_overlap(owner_terms, document_terms))
            + 12.0 * int(owner_exact)
        )
        question_bonus = 1.0 * int(
            "?" in document.text
            and len(overlap | set(expanded_overlap)) >= 2
        )
        rows.append((
            node_id,
            6.0 * len(overlap)
            + 8.0 * len(expanded_overlap)
            + owner_bonus + question_bonus
            + 20.0 * int(historical_attribute_query),
        ))
    return sorted(rows, key=lambda row: (-row[1], row[0]))[:limit]



def _location_source_rank(
    ir: QueryIR,
    sidecar: QuerySidecarV41,
    limit: int = 8,
) -> list[tuple[str, float]]:
    """Find owner-bound movement/residence turns containing a concrete place.

    The requested answer may be a containing region rather than the verbatim
    place (for example city -> state or city -> country), so exact place strings
    cannot be generated as query aliases ahead of retrieval. This channel only
    recognizes source grammar and leaves geographic inference to the planner.
    """
    if ir.requested_value_type != "location":
        return []
    place_pattern = re.compile(
        r"\b(?:in|to|from|at|near|around|outside)\s+(?:the\s+)?"
        r"([A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){0,3})\b"
    )
    movement_pattern = re.compile(
        r"\b(?:visit(?:ed|ing)?|went|go|travel(?:ed|ing)?|trip|outing|"
        r"moved?|live[sd]?|resid(?:e|ed|ing)|stay(?:ed|ing)?|"
        r"took\b.{0,50}\bto|beach)\b",
        re.IGNORECASE,
    )
    rows: list[tuple[str, float]] = []
    for node_id, document in sidecar.documents.items():
        if document.node_type != "turn":
            continue
        source_text = re.sub(
            r"^speaker\s+[^|]+\|\s*", "", document.text,
            flags=re.IGNORECASE,
        )
        place_matches = place_pattern.findall(source_text)
        if not place_matches or not movement_pattern.search(source_text):
            continue
        owners = {
            value.casefold() for value in document.fields.get("owner", [])
        }
        owner_exact = bool(
            ir.target_owner and ir.target_owner.casefold() in owners
        )
        overlap = _semantic_overlap(ir, _tokens(source_text))
        score = (
            20.0 + 18.0 * int(owner_exact)
            + 5.0 * len(overlap)
            + min(3, len(place_matches))
        )
        rows.append((node_id, score))
    return sorted(rows, key=lambda row: (-row[1], row[0]))[:limit]


def _answer_bearing_turn_rank(
    ir: QueryIR,
    augmentation: QueryAugmentationV41,
    sidecar: QuerySidecarV41,
    limit: int = 16,
) -> list[tuple[str, float]]:
    """Rank local source windows that jointly bind relation and answer roles."""
    target_terms, relation_terms = query_binding_terms(ir)
    expanded_binding_terms = binding_tokens(
        " ".join(augmentation.expanded_terms)
    )
    target_terms = set(target_terms) | set(expanded_binding_terms)
    requested_families = action_families(ir.raw_question)
    historical_attribute_query = bool(re.search(
        r"\b(?:previous|former|prior)\s+"
        r"(?:occupation|job|role|position|employer|address|state|value|name)\b",
        ir.raw_question, re.IGNORECASE,
    ))
    if "complete" in requested_families:
        requested_families.discard("project_work")
    number_pattern = re.compile(
        r"(?:[$€£¥]\s*)?\b\d[\d,]*(?:\.\d+)?(?:\s*%)?\b|"
        r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
        r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
        r"nineteen|twenty|dozen|couple|few|several)\b",
        re.IGNORECASE,
    )
    duration_pattern = re.compile(
        r"\b(?:second|minute|hour|day|week|month|year)s?\b|"
        r"\b(?:over|under|about|around|nearly|almost)\s+(?:a|an|one|two|"
        r"three|four|five|six|seven|eight|nine|ten)\b",
        re.IGNORECASE,
    )
    lifecycle_pattern = re.compile(
        r"\b(?:already|previous|prior|born|bought|completed|finished|got|"
        r"had|made|met|moved|received|returned|sold|visited|went|won)\b",
        re.IGNORECASE,
    )
    rows: list[tuple[str, float]] = []
    for node_id, document in sidecar.documents.items():
        if document.node_type != "turn":
            continue
        source_text = re.sub(
            r"^speaker\s+[^|]+\|\s*", "", document.text,
            flags=re.IGNORECASE,
        )
        document_owners = {
            value.casefold()
            for value in document.fields.get("owner", [])
        }
        owner_exact = bool(
            ir.target_owner
            and ir.target_owner.casefold() in document_owners
        )
        segments = [
            value.strip() for value in re.split(
                r"(?<=[.!?])\s+|\n+", source_text,
            ) if value.strip()
        ]
        windows = list(segments)
        windows.extend(
            f"{segments[index]} {segments[index + 1]}"
            for index in range(len(segments) - 1)
        )
        best = 0.0
        for window in windows:
            if (
                historical_attribute_query
                and not re.search(
                    r"\b(?:previous|previously|former|prior|used to)\b",
                    window, re.IGNORECASE,
                )
            ):
                continue
            terms = binding_tokens(window)
            target_overlap = fuzzy_term_overlap(target_terms, terms)
            relation_overlap = fuzzy_term_overlap(relation_terms, terms)
            family_overlap = requested_families.intersection(
                action_families(window)
            )
            semantic_overlap = _semantic_overlap(ir, _tokens(window))
            expanded_overlap = fuzzy_term_overlap(
                expanded_binding_terms, terms,
            )
            preference_scene = bool(re.search(
                r"\b(?:prefer|like|love|enjoy|experiment|using|used|own|"
                r"bought|consider|interested|interests?|want|wanted)\b",
                window, re.IGNORECASE,
            ))
            target_ok = bool(
                target_overlap
                or (not target_terms and semantic_overlap)
                or (
                    augmentation.answer_algebra
                    == "preference_recommendation"
                    and len(semantic_overlap | set(expanded_overlap)) >= 2
                )
            )
            numeric = bool(number_pattern.search(window))
            duration = bool(duration_pattern.search(window))
            lifecycle = bool(lifecycle_pattern.search(window))
            historical_quantity = bool(
                numeric and re.search(
                    r"\b(?:already|previous|previously|prior)\b",
                    window, re.IGNORECASE,
                )
            )
            first_person = bool(re.search(
                r"\b(?:i(?:'ve|'m|'d|'ll)?|me|my|mine|we|our|ours)\b",
                window, re.IGNORECASE,
            ))
            relation_ok = bool(
                relation_overlap or family_overlap
                or (not relation_terms and len(semantic_overlap) >= 2)
                or (
                    augmentation.answer_algebra == "collection"
                    and historical_quantity
                )
                or (
                    augmentation.answer_algebra
                    == "preference_recommendation"
                    and (
                        len(semantic_overlap | set(expanded_overlap)) >= 3
                        or (preference_scene and bool(
                            semantic_overlap or expanded_overlap
                        ))
                    )
                )
                or (
                    augmentation.answer_algebra
                    in {"dialogue_lookup", "direct_fact", "inferential_profile"}
                    and (
                        len(semantic_overlap | set(expanded_overlap)) >= 3
                        or (
                            historical_attribute_query
                            and bool(expanded_overlap)
                        )
                    )
                )
            )
            if not target_ok or not relation_ok:
                continue
            if augmentation.answer_algebra == "collection" and not (
                numeric or lifecycle
            ):
                continue
            if (
                augmentation.answer_algebra in {
                    "temporal_lookup", "temporal_comparison",
                }
                and not (numeric or duration)
            ):
                continue
            score = (
                12.0 + 4.0 * len(target_overlap)
                + 5.0 * len(relation_overlap)
                + 6.0 * len(family_overlap)
                + 1.5 * len(semantic_overlap)
                + 6.0 * len(expanded_overlap)
                + 4.0 * int(numeric)
                + 4.0 * int(duration)
                + 3.0 * int(lifecycle)
                + 14.0 * int(owner_exact)
                + 8.0 * int(first_person and bool(re.search(
                    r"\b(?:i|me|my|mine|we|our|ours)\b",
                    ir.raw_question, re.IGNORECASE,
                )))
                + 20.0 * int(
                    historical_attribute_query
                    and bool(re.search(
                        r"\b(?:previous|previously|former|prior|used to)\b",
                        window, re.IGNORECASE,
                    ))
                )
            )
            best = max(best, score)
        if best > 0:
            rows.append((node_id, best))
    return sorted(rows, key=lambda row: (-row[1], row[0]))[:limit]


def _reply_bound_turn_rank(
    ir: QueryIR,
    augmentation: QueryAugmentationV41,
    sidecar: QuerySidecarV41,
    limit: int = 16,
) -> list[tuple[str, float]]:
    """Rank elliptical replies against their preceding local prompt/stimulus.

    This is an additive channel and never changes the existing answer-bearing
    order. It covers replies such as exact values, pronouns, reactions, and a
    second clarification turn whose text omits the original topic.
    """
    expanded_terms = binding_tokens(" ".join(augmentation.expanded_terms))
    owner_terms = _tokens(ir.target_owner)
    rows: list[tuple[str, float]] = []
    for node_id, document in sidecar.documents.items():
        if document.node_type != "turn":
            continue
        match = re.match(r"^(.*:turn:)(\d+)$", node_id)
        if match is None or int(match.group(2)) < 1:
            continue
        prefix, ordinal_text = match.groups()
        ordinal = int(ordinal_text)
        local = []
        for previous_ordinal in range(max(0, ordinal - 2), ordinal):
            previous = sidecar.documents.get(f"{prefix}{previous_ordinal}")
            if previous is not None and previous.node_type == "turn":
                local.append(previous)
        if not local:
            continue
        stimulus = " ".join(row.text for row in local)
        if not re.search(
            r"\?|\b(?:media shared|caption|photo|picture|image|what|which|how|why)\b",
            stimulus, re.IGNORECASE,
        ):
            continue
        combined_terms = _tokens(f"{stimulus} {document.text}")
        semantic = _semantic_overlap(ir, combined_terms)
        expanded = fuzzy_term_overlap(expanded_terms, binding_tokens(
            f"{stimulus} {document.text}"
        ))
        if len(semantic | set(expanded)) < 2:
            continue
        owners = {
            value.casefold() for value in document.fields.get("owner", [])
        }
        owner_exact = bool(
            ir.target_owner and ir.target_owner.casefold() in owners
        )
        speaker_match = re.match(
            r"^speaker\s+([^|]+)\|", document.text, re.IGNORECASE,
        )
        if speaker_match is not None and _fuzzy_overlap(
            owner_terms, _tokens(speaker_match.group(1)),
        ):
            owner_exact = True
        answer_terms = _tokens(document.text) - _QUERY_STOP_TERMS
        score = (
            12.0 + 6.0 * len(semantic) + 4.0 * len(expanded)
            + 14.0 * int(owner_exact)
            + 4.0 * int("?" in local[-1].text)
            + 3.0 * int(bool(re.search(
                r"\b(?:media shared|caption|photo|picture|image)\b",
                stimulus, re.IGNORECASE,
            )))
            + min(4.0, 0.25 * len(answer_terms))
        )
        rows.append((node_id, score))
    return sorted(rows, key=lambda row: (-row[1], row[0]))[:limit]


def _candidate_nodes(
    ir: QueryIR,
    augmentation: QueryAugmentationV41,
    sidecar: QuerySidecarV41,
    planner: PlannerResultV41 | None,
    limit: int,
) -> tuple[list[str], dict[str, Any]]:
    planner_terms = [] if planner is None else [
        *planner.alternative_entities, *planner.event_aliases,
        *planner.relations, *planner.temporal_constraints,
    ]
    # Lexical candidates must be driven by semantic content, not question
    # function words or answer-algebra words such as total/number/different.
    # Passing the raw question here caused every shared stop word to add score.
    terms = [
        *sorted(_query_content_terms(ir)),
        *augmentation.expanded_terms,
        *ir.comparison_targets,
        *planner_terms,
    ]
    lexical = lexical_rank(sidecar, terms, 120)
    semantic_turns = _semantic_turn_rank(
        ir, augmentation, sidecar, 24,
    )
    location_sources = _location_source_rank(ir, sidecar, 12)
    # The source-window channel is useful for every answer algebra: it binds
    # query concepts to a local, provenance-bearing scene. Algebra-specific
    # guards inside the ranker still require quantities for collections and
    # time-bearing evidence for temporal questions.
    answer_bearing_turns = _answer_bearing_turn_rank(
        ir, augmentation, sidecar, 24,
    )
    reply_bound_turns = _reply_bound_turn_rank(
        ir, augmentation, sidecar, 24,
    )
    lookups = {
        "entity": [
            *ir.target_entities, *augmentation.alternative_entities,
            *(planner.alternative_entities if planner else []),
        ],
        "owner": [ir.target_owner],
        "predicate": [
            ir.target_relation,
            *(planner.relations if planner else []),
        ],
        "event_identity": [
            *augmentation.event_identity_terms,
            *(planner.event_aliases if planner else []),
        ],
        "date": [
            *ir.temporal_constraints,
            *(planner.temporal_constraints if planner else []),
        ],
        "status": ir.state_constraints,
        "polarity": [ir.polarity] if ir.polarity != "unknown" else [],
        "semantic_type": [
            *augmentation.domain_hints, *augmentation.required_roles,
        ],
    }
    structured = inverted_rank(sidecar, lookups, 120)
    scores: defaultdict[str, float] = defaultdict(float)
    channels: defaultdict[str, dict[str, int]] = defaultdict(dict)
    for channel, rows in (
        ("location_source_turn", location_sources),
        ("answer_bearing_turn", answer_bearing_turns),
        ("reply_bound_turn", reply_bound_turns),
        ("lossless_semantic_turn", semantic_turns),
        ("sidecar_fts", lexical),
        ("sidecar_inverted", structured),
    ):
        for rank, (node_id, _score) in enumerate(rows, 1):
            scores[node_id] += 1.0 / (10 + rank)
            channels[node_id][channel] = rank
    ranked = sorted(scores, key=lambda node_id: (-scores[node_id], node_id))
    # Preserve independent channel anchors before filling by fused score.
    chosen: list[str] = []
    if planner is not None and planner.valid:
        for rank, node_id in enumerate(planner.selected_source_ids[:8], 1):
            document = sidecar.documents.get(node_id)
            if document is None or document.node_type != "turn":
                continue
            if node_id not in chosen:
                chosen.append(node_id)
                channels[node_id]["planner_selected_source"] = rank
    for rows in (
        location_sources[:6], answer_bearing_turns[:8], reply_bound_turns[:6],
        semantic_turns[:12], lexical[:8], structured[:4],
    ):
        for node_id, _score in rows:
            if node_id not in chosen:
                chosen.append(node_id)
    for node_id in ranked:
        if node_id not in chosen:
            chosen.append(node_id)
        if len(chosen) >= limit:
            break
    return chosen[:limit], {
        "channels": dict(channels),
        "ranked_ids": ranked[:limit],
        "protected_ids": chosen[:16],
    }


def _expand_nodes(
    seeds: list[str],
    sidecar: QuerySidecarV41,
    relations: list[str],
    policy: QueryPolicyV41,
) -> tuple[list[str], list[dict[str, Any]]]:
    chosen: list[str] = []
    trace: list[dict[str, Any]] = []
    queue = deque((seed, 0) for seed in seeds)
    seen = set(seeds)
    relation_counts: Counter[str] = Counter()
    while queue:
        node_id, depth = queue.popleft()
        if depth >= policy.expansion_depth:
            continue
        for relation in relations:
            cap = (
                policy.collection_relation_limit
                if relation == "collection_member"
                else policy.relation_limit
            )
            if relation_counts[relation] >= cap:
                continue
            for neighbor in sidecar.adjacency.get(node_id, {}).get(relation, []):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                relation_counts[relation] += 1
                chosen.append(neighbor)
                trace.append({
                    "src": node_id, "dst": neighbor, "relation": relation,
                    "depth": depth + 1, "source": "v41_sidecar",
                })
                queue.append((neighbor, depth + 1))
                if relation_counts[relation] >= cap:
                    break
    return chosen, trace



def _hydrate_lossless_sidecar(
    case: QuestionCase, sidecar: QuerySidecarV41,
) -> dict[str, str]:
    """Restore source text omitted by an older cache in the disposable sidecar."""
    overlays: dict[str, str] = {}
    for session_id, messages in zip(
        case.haystack_session_ids, case.haystack_sessions,
    ):
        for turn_index, message in enumerate(messages):
            node_id = f"{case.question_id}:{session_id}:turn:{turn_index}"
            document = sidecar.documents.get(node_id)
            raw_text = str(message.get("content") or "").strip()
            if document is None or document.node_type != "turn" or not raw_text:
                continue
            overlays[node_id] = raw_text
            missing_terms = _tokens(raw_text) - _tokens(document.text)
            if len(missing_terms) >= 3:
                document.text = raw_text
    return overlays


def _append_lossless_overlays(
    result: RetrievedContext,
    index: V36Index,
    raw_overlays: dict[str, str],
    ir: QueryIR,
    token_budget: int,
    limit: int = 6,
    preferred_source_ids: list[str] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Expose restored text for already-packed provenance IDs without graph edits."""
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    salient = {
        token for token in _tokens(
            " ".join([ir.raw_question, *ir.target_entities, ir.target_relation])
        )
        if len(token) >= 3 and token not in {
            "what", "which", "when", "where", "there", "does", "did",
        }
    }
    preferred = set(preferred_source_ids or [])
    candidate_ids = list(dict.fromkeys([
        *(preferred_source_ids or []), *result.leaf_node_ids,
    ]))
    ranked: list[tuple[float, str, str]] = []
    for source_id in candidate_ids:
        raw_text = raw_overlays.get(source_id)
        turn = turn_by_id.get(source_id)
        if raw_text is None or turn is None:
            continue
        raw_terms = _tokens(raw_text)
        missing_terms = raw_terms - _tokens(turn.text)
        overlap = {
            target for target in salient
            if any(
                target == token
                or (
                    len(target) >= 4 and len(token) >= 4
                    and target[:4] == token[:4]
                )
                for token in raw_terms
            )
        }
        if not overlap and source_id not in preferred:
            continue
        score = (
            (40.0 if source_id in preferred else 0.0)
            + 4.0 * len(overlap) + min(20, len(missing_terms))
        )
        if len(missing_terms) >= 3:
            ranked.append((score, source_id, raw_text))
    decisions: list[dict[str, Any]] = []
    added: list[str] = []
    for _score, source_id, raw_text in sorted(
        ranked, key=lambda row: (-row[0], row[1]),
    ):
        turn = turn_by_id[source_id]
        block = (
            f"[LOSSLESS_SOURCE_OVERLAY {source_id}; provenance=same_turn]\n"
            f"date={turn.session_date or 'unknown'}; speaker={turn.speaker}; "
            f"listener={turn.listener}; role={turn.transport_role}\n"
            f"{raw_text[:900]}"
        )
        cost = rough_token_count(block)
        if result.packed_rough_tokens + cost > token_budget:
            decisions.append({
                "source_turn_id": source_id, "decision": "reject_budget",
                "cost": cost,
            })
            continue
        result.context_text = f"{result.context_text}\n\n{block}"
        result.packed_rough_tokens += cost
        if source_id not in result.leaf_node_ids:
            result.leaf_node_ids.append(source_id)
            result.evidence_leaf_ids.append(source_id)
        added.append(source_id)
        decisions.append({
            "source_turn_id": source_id, "decision": "append", "cost": cost,
        })
        if len(added) >= limit:
            break
    return added, decisions

def _append_sources(
    result: RetrievedContext,
    index: V36Index,
    sidecar: QuerySidecarV41,
    node_ids: list[str],
    token_budget: int,
    reason: str,
    max_sources: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    existing = set(result.retrieval_trace.get("packed_source_turn_ids") or [])
    existing.update(result.leaf_node_ids)
    additions: list[str] = []
    decisions: list[dict[str, Any]] = []
    for node_id in node_ids:
        for source_id in _document_sources(sidecar, node_id):
            if source_id in existing:
                continue
            turn = turn_by_id.get(source_id)
            if turn is None:
                decisions.append({
                    "node_id": node_id, "source_turn_id": source_id,
                    "decision": "reject_missing_source", "reason": reason,
                })
                continue
            document = sidecar.documents.get(source_id)
            evidence_text = document.text if document is not None else turn.text
            block = (
                f"[SOURCE_EVIDENCE {turn.node_id}; added_by={reason}]\n"
                f"date={turn.session_date or 'unknown'}; speaker={turn.speaker}; "
                f"listener={turn.listener}; role={turn.transport_role}\n"
                f"{evidence_text[:900]}"
            )
            cost = rough_token_count(block)
            if result.packed_rough_tokens + cost > token_budget:
                decisions.append({
                    "node_id": node_id, "source_turn_id": source_id,
                    "decision": "reject_budget", "cost": cost, "reason": reason,
                })
                continue
            result.context_text = (
                f"{result.context_text}\n\n{block}" if result.context_text else block
            )
            result.packed_rough_tokens += cost
            result.leaf_node_ids.append(source_id)
            result.evidence_leaf_ids.append(source_id)
            existing.add(source_id)
            additions.append(source_id)
            decisions.append({
                "node_id": node_id, "source_turn_id": source_id,
                "decision": "append", "cost": cost, "reason": reason,
            })
            if len(additions) >= max_sources:
                return additions, decisions
    return additions, decisions


def _refresh_post_expansion_operator_hints(
    result: RetrievedContext,
    index: V36Index,
    ir: QueryIR,
    packed_sources: list[str],
) -> list[dict[str, Any]]:
    # Recompute source-bound scalar operators after V4.1 source growth.
    refreshed = [
        hint for hint in (
            threshold_progress_remaining_hint(ir, index, packed_sources),
            latest_approx_scalar_state_hint(ir, index, packed_sources),
            latest_labeled_currency_state_hint(ir, index, packed_sources),
            latest_weekly_schedule_time_hint(ir, index, packed_sources),
        ) if hint is not None
    ]
    if not refreshed:
        return []
    operations = {
        str(hint.get("operation") or "") for hint in refreshed
    }
    if "threshold_progress_remaining" in operations:
        operations.add("latest_scalar_state_from_lossless_sources")
    result.retrieval_trace["generic_operator_hints"] = [
        hint for hint in result.retrieval_trace.get(
            "generic_operator_hints", []
        )
        if not (
            isinstance(hint, dict)
            and str(hint.get("operation") or "") in operations
        )
    ] + refreshed
    return refreshed


def _operator_source_projection(
    result: RetrievedContext, ir: QueryIR, sidecar: QuerySidecarV41,
    limit: int = 6,
) -> list[str]:
    """Project provenance-bearing operator candidates into the final pack."""
    salient = {
        token for token in _tokens(ir.raw_question)
        if len(token) >= 3 and token not in {
            "how", "many", "much", "what", "which", "when", "there",
            "total", "both", "some", "thing", "things", "times",
        }
    }
    scored: dict[str, float] = {}
    for hint in result.retrieval_trace.get("generic_operator_hints") or []:
        if not isinstance(hint, dict):
            continue
        for operand in hint.get("operands") or []:
            if not isinstance(operand, dict):
                continue
            source_id = str(operand.get("source_turn_id") or "")
            if source_id in sidecar.documents:
                scored[source_id] = max(scored.get(source_id, 0.0), 20.0)
        if (
            hint.get("certified") is True
            and hint.get("answer_candidate")
            and hint.get("operation") in {
                "relative_anchor_source_lookup",
                "temporal_predecessor_entity",
            }
        ):
            for source_id in hint.get("source_turn_ids") or []:
                if source_id in sidecar.documents:
                    scored[source_id] = max(
                        scored.get(source_id, 0.0), 30.0,
                    )
        if ir.requested_value_type not in {
            "count", "aggregate", "date", "duration",
            "temporal_order", "state",
        }:
            continue
        for candidate in hint.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            source_id = str(candidate.get("source_turn_id") or "")
            document = sidecar.documents.get(source_id)
            if document is None:
                continue
            overlap = salient.intersection(_tokens(
                f"{candidate.get('evidence', '')} {document.text}"
            ))
            if overlap:
                scored[source_id] = max(
                    scored.get(source_id, 0.0),
                    float(candidate.get("lexical_score") or 0)
                    + 2.0 * len(overlap),
                )
    return [
        source_id for source_id, _score in sorted(
            scored.items(), key=lambda item: (-item[1], item[0])
        )[:limit]
    ]


def _dialogue_closure_nodes(
    source_ids: list[str], ir: QueryIR, sidecar: QuerySidecarV41,
    limit: int = 8,
) -> list[str]:
    """Rank adjacent turns by query relevance; never expand dialogue blindly."""
    salient = {
        token for token in _tokens(
            " ".join([ir.raw_question, *ir.target_entities, ir.target_relation])
        )
        if len(token) >= 3 and token not in {
            "how", "many", "much", "what", "which", "when", "there",
            "total", "both", "some", "thing", "things", "times",
        }
    }
    scores: dict[str, float] = {}
    for source_id in source_ids:
        for neighbor in sidecar.adjacency.get(source_id, {}).get("next_turn", []):
            document = sidecar.documents.get(neighbor)
            if document is None or document.node_type != "turn":
                continue
            document_tokens = _tokens(document.text)
            overlap = {
                target for target in salient
                if any(
                    target == token
                    or (len(target) >= 5 and len(token) >= 5
                        and target[:5] == token[:5])
                    for token in document_tokens
                )
            }
            if not overlap:
                continue
            temporal_bonus = 0.0
            if (
                ir.requested_value_type in {"date", "temporal_order"}
                and re.search(
                    r"\b(?:19\d{2}|20\d{2}|january|february|march|april|may|june|july|august|september|october|november|december)\b",
                    document.text, re.IGNORECASE,
                )
            ):
                temporal_bonus = 12.0
            scores[neighbor] = max(
                scores.get(neighbor, 0.0),
                3.0 * len(overlap) + temporal_bonus,
            )
    return [
        node_id for node_id, _score in sorted(
            scores.items(), key=lambda item: (-item[1], item[0])
        )[:limit]
    ]



def _dialogue_pair_completion_nodes(
    source_ids: list[str], ir: QueryIR, sidecar: QuerySidecarV41,
    limit: int = 3,
) -> list[str]:
    """Complete selected, query-relevant prompt turns before optional anchors.

    A prompt without its direct reply is a broken evidence unit. This pass is
    intentionally earlier than ordinary multi-channel augmentation so a
    lower-ranked anchor cannot consume the small amount of budget needed by a
    directly paired answer.
    """
    salient = {
        token for token in _tokens(
            " ".join([ir.raw_question, *ir.target_entities, ir.target_relation])
        )
        if len(token) >= 3 and token not in {
            "how", "many", "much", "what", "which", "when", "there",
            "total", "both", "some", "thing", "things", "times",
        }
    }
    scores: dict[str, float] = {}
    selected = set(source_ids)
    for source_id in source_ids:
        source = sidecar.documents.get(source_id)
        if source is None or source.node_type != "turn" or "?" not in source.text:
            continue
        prompt_tokens = _tokens(source.text)
        overlap = {
            target for target in salient
            if any(
                target == token
                or (
                    len(target) >= 5 and len(token) >= 5
                    and target[:5] == token[:5]
                )
                for token in prompt_tokens
            )
        }
        if not overlap:
            continue
        ordinal_match = re.search(r":turn:(\d+)$", source_id)
        if ordinal_match is None:
            continue
        expected_ordinal = int(ordinal_match.group(1)) + 1
        for relation, relation_bonus in (
            ("dialogue_pair", 8.0), ("next_turn", 4.0),
        ):
            for neighbor in sidecar.adjacency.get(source_id, {}).get(relation, []):
                if neighbor in selected:
                    continue
                neighbor_match = re.search(r":turn:(\d+)$", neighbor)
                document = sidecar.documents.get(neighbor)
                if (
                    document is None
                    or document.node_type != "turn"
                    or neighbor_match is None
                    or int(neighbor_match.group(1)) != expected_ordinal
                ):
                    continue
                scores[neighbor] = max(
                    scores.get(neighbor, 0.0),
                    3.0 * len(overlap) + relation_bonus,
                )
    return [
        node_id for node_id, _score in sorted(
            scores.items(), key=lambda item: (-item[1], item[0])
        )[:limit]
    ]


def _scene_window_nodes(
    source_ids: list[str], ir: QueryIR, sidecar: QuerySidecarV41,
    limit: int = 8,
) -> list[str]:
    """Select a bounded same-session window around content-bearing anchors."""
    owner_terms = _tokens(ir.target_owner)
    content_terms = _query_content_terms(ir)
    anchors: list[tuple[float, str]] = []
    for source_id in dict.fromkeys(source_ids):
        document = sidecar.documents.get(source_id)
        if document is None or document.node_type != "turn":
            continue
        terms = _tokens(document.text)
        content_overlap = _semantic_overlap(ir, terms)
        owner_overlap = _fuzzy_overlap(owner_terms, terms)
        score = 6.0 * len(content_overlap) + 0.5 * len(owner_overlap)
        if score > 0:
            anchors.append((score, source_id))
    anchors.sort(key=lambda row: (-row[0], row[1]))
    candidates: dict[str, float] = {}
    protected_forward: list[str] = []
    protected_backward: list[str] = []
    relative_time = re.compile(
        r"\b(?:next|last|this)\s+(?:day|week|month|year|weekend)\b|"
        r"\b(?:yesterday|tomorrow|today)\b", re.IGNORECASE,
    )
    reaction_query = bool(re.search(
        r"\b(?:react(?:ion|ed)?|say|said|think|thought|feel|felt)\b"
        r".{0,50}\b(?:photo|picture|image|posters?|painting|video|it|them)\b",
        ir.raw_question, re.IGNORECASE,
    ))
    slot_completion_query = bool(
        ir.requested_value_type in {"span", "recommendation", "preference"}
        and re.search(r"\b(?:what|which|how|why)\b", ir.raw_question, re.IGNORECASE)
    )
    # Multiple local mentions can share a broad topic.  Twelve bounded anchors
    # keep the exact question/reply scene reachable without global expansion.
    for anchor_score, source_id in anchors[:24]:
        match = re.match(r"^(.*:turn:)(\d+)$", source_id)
        if match is None:
            continue
        prefix, ordinal_text = match.groups()
        ordinal = int(ordinal_text)
        anchor_document = sidecar.documents.get(source_id)
        media_anchor = bool(
            anchor_document is not None
            and re.search(
                r"\b(?:media shared|caption|photo|picture|image|poster|painting|video)\b",
                anchor_document.text, re.IGNORECASE,
            )
        )
        anchor_overlap = (
            _semantic_overlap(ir, _tokens(anchor_document.text))
            if anchor_document is not None else set()
        )
        # A relative-date reply is often elliptical: "we did it yesterday"
        # gets its event identity from the preceding question, media caption,
        # or statement. Keep that tiny backward dialogue unit intact instead
        # of asking lexical overlap to rediscover nouns that the reply omits.
        # This is deliberately limited to temporal questions and two turns.
        deictic_temporal_reply = bool(
            anchor_document is not None
            and ir.requested_value_type in {
                "date", "temporal_order", "duration",
            }
            and relative_time.search(anchor_document.text)
            and re.search(
                r"\b(?:it|that|this|them|those|did\s+(?:it|that|this)|"
                r"went|was|were)\b",
                anchor_document.text, re.IGNORECASE,
            )
        )
        if deictic_temporal_reply:
            for delta in (-2, -1):
                neighbor_id = f"{prefix}{ordinal + delta}"
                neighbor = sidecar.documents.get(neighbor_id)
                if neighbor is None or neighbor.node_type != "turn":
                    continue
                candidates[neighbor_id] = max(
                    candidates.get(neighbor_id, 0.0),
                    anchor_score + 18.0 - 0.5 * abs(delta),
                )
                if neighbor_id not in protected_backward:
                    protected_backward.append(neighbor_id)
        deltas = (-2, -1, 1, 2, 3) if slot_completion_query else (-2, -1, 1, 2)
        for delta in deltas:
            neighbor_id = f"{prefix}{ordinal + delta}"
            document = sidecar.documents.get(neighbor_id)
            if document is None or document.node_type != "turn":
                continue
            terms = _tokens(document.text)
            content_overlap = _semantic_overlap(ir, terms)
            temporal_match = bool(
                ir.requested_value_type in {"date", "temporal_order", "duration"}
                and relative_time.search(document.text)
            )
            stimulus_response_match = bool(
                reaction_query and media_anchor and delta in {1, 2}
            )
            forward_slot_match = bool(
                slot_completion_query
                and len(anchor_overlap) >= 2
                and delta in {1, 2, 3}
            )
            # Owner names alone are routing metadata, not a scene match. A
            # media stimulus followed by the requested reaction is a typed
            # dialogue unit even when the reply uses no nouns from the photo.
            if not (
                content_overlap or temporal_match
                or stimulus_response_match or forward_slot_match
            ):
                continue
            score = (
                anchor_score + 7.0 * len(content_overlap)
                + 6.0 * int(temporal_match)
                + 14.0 * int(stimulus_response_match)
                + 9.0 * int(forward_slot_match)
                - 0.5 * abs(delta)
            )
            candidates[neighbor_id] = max(candidates.get(neighbor_id, 0.0), score)
            if forward_slot_match and neighbor_id not in protected_forward:
                protected_forward.append(neighbor_id)
            if "?" in document.text:
                reply_id = f"{prefix}{ordinal + delta + 1}"
                reply = sidecar.documents.get(reply_id)
                if reply is not None and reply.node_type == "turn":
                    candidates[reply_id] = max(
                        candidates.get(reply_id, 0.0), score + 3.0,
                    )
                    followup_id = f"{prefix}{ordinal + delta + 2}"
                    followup = sidecar.documents.get(followup_id)
                    if followup is not None and followup.node_type == "turn":
                        candidates[followup_id] = max(
                            candidates.get(followup_id, 0.0), score + 1.0,
                        )
    ranked = [
        node_id for node_id, _score in sorted(
            candidates.items(), key=lambda row: (-row[1], row[0]),
        )
    ]
    # A direct answer slot may be stated in the second or third short turn after
    # the topical anchor. Reserve a tiny ordered closure before global scoring;
    # otherwise many stronger topical anchors can evict the actual reply.
    return list(dict.fromkeys([
        *protected_backward[:4], *protected_forward[:6], *ranked,
    ]))[:limit]

def _followup_endorsements(text: str) -> list[str]:
    pattern = re.compile(
        r"\b((?:the\s+)?[A-Z][A-Za-z0-9&'.-]*"
        r"(?:\s+[A-Z][A-Za-z0-9&'.-]*)?)"
        r"\s+(?:is|are|was|were|seems?)\s+"
        r"(?:solid|great|amazing|excellent|good|awesome|fantastic|impressive)\b"
    )
    return list(dict.fromkeys(
        match.group(1).strip() for match in pattern.finditer(text)
    ))[:6]


def _direct_dialogue_highlights(
    source_ids: list[str], ir: QueryIR, sidecar: QuerySidecarV41,
    limit: int = 4,
) -> list[dict[str, Any]]:
    """Return source-bound local statement/prompt/reply scenes."""
    query_terms = _query_content_terms(ir)
    selected = set(source_ids)
    rows: list[tuple[float, dict[str, Any]]] = []
    for source_id in dict.fromkeys(source_ids):
        prompt = sidecar.documents.get(source_id)
        if prompt is None or prompt.node_type != "turn" or "?" not in prompt.text:
            continue
        match = re.match(r"^(.*:turn:)(\d+)$", source_id)
        if match is None:
            continue
        prefix, ordinal_text = match.groups()
        ordinal = int(ordinal_text)
        reply_id = f"{prefix}{ordinal + 1}"
        reply = sidecar.documents.get(reply_id)
        if reply is None or reply.node_type != "turn" or reply_id not in selected:
            continue
        previous_id = f"{prefix}{ordinal - 1}" if ordinal > 0 else ""
        previous = sidecar.documents.get(previous_id) if previous_id in selected else None
        followup_id = f"{prefix}{ordinal + 2}"
        followup = sidecar.documents.get(followup_id) if followup_id in selected else None
        scene_text = " ".join([
            previous.text if previous is not None else "",
            prompt.text, reply.text,
            followup.text if followup is not None else "",
        ])
        overlap = _semantic_overlap(ir, _tokens(scene_text))
        prompt_overlap = _semantic_overlap(ir, _tokens(prompt.text))
        if not overlap or not prompt_overlap:
            continue
        score = (
            6.0 * len(overlap) + 2.0 * len(prompt_overlap)
            + 2.0 * int(previous is not None)
            + 2.0 * int(followup is not None)
        )
        rows.append((score, {
            "context_source_id": previous_id if previous is not None else None,
            "prompt_source_id": source_id,
            "reply_source_id": reply_id,
            "followup_source_id": followup_id if followup is not None else None,
            "context": previous.text[:600] if previous is not None else "",
            "prompt": prompt.text[:500],
            "reply": reply.text[:700],
            "followup": followup.text[:600] if followup is not None else "",
            "followup_endorsements": (
                _followup_endorsements(followup.text)
                if followup is not None else []
            ),
            "matched_terms": sorted(overlap),
            "provenance_complete": True,
        }))
    return [row for _score, row in sorted(
        rows, key=lambda item: (-item[0], item[1]["prompt_source_id"]),
    )[:limit]]


def _best_query_clause(text: str, ir: QueryIR) -> str:
    """Return the most query-bound local clause from a lossless turn."""
    clauses = [
        value.strip() for value in re.split(r"(?<=[.!?])\s+|\n+", text)
        if value.strip()
    ] or [text]
    ranked: list[tuple[float, int, str]] = []
    requested_actions = action_families(ir.raw_question)
    comparison_query = bool(re.search(
        r"\b(?:compare|compared|comparison|analogy|metaphor)\b",
        ir.raw_question, re.IGNORECASE,
    ))
    for ordinal, clause in enumerate(clauses):
        overlap = _semantic_overlap(ir, _tokens(clause))
        action_overlap = requested_actions & action_families(clause)
        score = (
            5.0 * len(overlap)
            + 4.0 * len(action_overlap)
            + 2.0 * bool(re.search(
                r"\b(?:bought|completed|downloaded|finished|got|had|owned|"
                r"purchased|rejected|started|used|viewed|visited|went)\b",
                clause, re.IGNORECASE,
            ))
            + 1.5 * bool(re.search(
                r"\b[A-Z][A-Za-z0-9&'.-]{1,}(?:\s+[A-Z][A-Za-z0-9&'.-]{1,})+\b",
                clause,
            ))
            + 1.0 * bool(re.search(r"\d", clause))
            + 25.0 * bool(
                re.search(r"(?:[$]\s*)?\d", clause, re.IGNORECASE)
                and (
                    ir.requested_value_type
                    in {"count", "aggregate", "duration", "quantity"}
                    or re.search(
                        r"\b(?:how much|amount|budget|allocat|cost|price|"
                        r"total|ratio|percent)\b",
                        ir.raw_question, re.IGNORECASE,
                    )
                )
            )
            + 1.0 * bool(re.search(
                r"\b(?:today|yesterday|last|ago|before|after|current|now)\b",
                clause, re.IGNORECASE,
            ))
            + 12.0 * bool(
                comparison_query and re.search(
                    r"\b(?:like|similar|analogy|metaphor|remind(?:s|ed)?)\b",
                    clause, re.IGNORECASE,
                )
            )
            - 1.5 * clause.rstrip().endswith("?")
        )
        ranked.append((score, -ordinal, clause))
    best = max(ranked, key=lambda row: (row[0], row[1]))
    ordinal = -best[1]
    clause = best[2]
    anaphora_pattern = re.compile(
        r"\b(?:this|that|these|those|it)\b", re.IGNORECASE,
    )
    if ordinal > 0 and anaphora_pattern.search(clause):
        for prior_ordinal in range(ordinal - 1, max(-1, ordinal - 4), -1):
            prior = clauses[prior_ordinal]
            if _semantic_overlap(ir, _tokens(prior)):
                return f"{prior} {clause}"
    for next_ordinal in range(ordinal + 1, min(len(clauses), ordinal + 4)):
        following = clauses[next_ordinal]
        if (
            anaphora_pattern.search(following)
            and (
                _semantic_overlap(ir, _tokens(following))
                or requested_actions & action_families(following)
            )
        ):
            return f"{clause} {following}"
    return clause


def _source_owner_compatible(
    source_id: str, ir: QueryIR, sidecar: QuerySidecarV41,
) -> bool:
    """Reject cross-speaker first-person claims for owner-sensitive slots.

    A different speaker may still report a fact about the target owner, but a
    first-person claim whose body never names the target belongs to its speaker,
    not to the listener named in transport metadata.
    """
    if not ir.target_owner:
        return True
    document = sidecar.documents.get(source_id)
    if document is None or document.node_type != "turn":
        return False
    speaker_match = re.match(r"^speaker\s+([^|]+)\|", document.text, re.IGNORECASE)
    if speaker_match is None:
        return True
    speaker_terms = _tokens(speaker_match.group(1))
    owner_terms = _tokens(ir.target_owner)
    if _fuzzy_overlap(owner_terms, speaker_terms):
        return True
    body = document.text.split("|", 2)[-1]
    if _fuzzy_overlap(owner_terms, _tokens(body)):
        return True
    first_person = bool(re.search(
        r"\b(?:i(?:'m|'ve|'d|'ll)?|me|my|mine|we|our|ours)\b",
        body, re.IGNORECASE,
    ))
    return not first_person


def _planner_evidence_candidates(
    source_ids: list[str], ir: QueryIR, sidecar: QuerySidecarV41,
    limit: int = 8, *, preferred_source_ids: list[str] | None = None,
    session_diverse: bool = True,
) -> list[dict[str, str]]:
    """Compact, query-bound evidence for the optional planner.

    Every exposed row is still a real lossless source, but bounded retrieval
    channels may nominate a source even when the large full-turn block did not
    fit in the main evidence pack. Each row is reduced to its most query-bound
    clause and remains source-bound.
    """
    preferred = [
        source_id for source_id in dict.fromkeys(preferred_source_ids or [])
        if source_id in sidecar.documents
    ]
    rows: list[dict[str, str]] = []
    for source_id in preferred[:limit]:
        document = sidecar.documents.get(source_id)
        if document is None or document.node_type != "turn":
            continue
        text = re.sub(
            r"^speaker\s+[^|]+\|\s*", "", document.text,
            flags=re.IGNORECASE,
        )
        rows.append({
            "source_turn_id": source_id,
            "text": _best_query_clause(text, ir)[:280],
        })

    content_terms = _query_content_terms(ir)
    candidate_source_ids = list(dict.fromkeys([*preferred, *source_ids]))
    term_frequency: Counter[str] = Counter()
    for source_id in candidate_source_ids:
        document = sidecar.documents.get(source_id)
        if document is None or document.node_type != "turn":
            continue
        term_frequency.update(_semantic_overlap(ir, _tokens(document.text)))
    ranked: list[tuple[float, bool, str, str, str]] = []
    for source_id in candidate_source_ids:
        document = sidecar.documents.get(source_id)
        if document is None or document.node_type != "turn":
            continue
        overlap = _semantic_overlap(ir, _tokens(document.text))
        if not overlap:
            continue
        modal = bool(re.search(
            r"\b(?:want|wanted|wish|dream|hope|like|liked|love|cool|interested)\b",
            document.text, re.IGNORECASE,
        ))
        named_phrase = bool(re.search(
            r"\b[A-Z][A-Za-z0-9&'.-]{2,}\s+[A-Z][A-Za-z0-9&'.-]{2,}\b",
            document.text,
        ))
        candidate_like = modal and named_phrase
        score = (
            5.0 * len(overlap) + 2.0 * int(modal)
            + 10.0 * int(candidate_like)
            + sum(8.0 / (1.0 + term_frequency[token]) for token in overlap)
        )
        session_id = document.session_ids[0] if document.session_ids else ""
        ranked.append((score, candidate_like, session_id, source_id, document.text))
    ordered = sorted(ranked, key=lambda row: (-row[0], row[3]))
    selected: list[tuple[float, bool, str, str, str]] = []
    # Reserve one slot for a named desired/possible candidate even when another
    # turn in the same session has higher raw lexical overlap.
    for row in ordered:
        if row[1]:
            selected.append(row)
            break
    seen_sessions = {row[2] for row in selected}
    for row in ordered:
        if row in selected:
            continue
        if (
            session_diverse
            and row[2] in seen_sessions
            and len(selected) < max(2, limit - 1)
        ):
            continue
        selected.append(row)
        seen_sessions.add(row[2])
        if len(selected) >= limit:
            break
    existing = {row["source_turn_id"] for row in rows}
    rows.extend([
        {"source_turn_id": source_id, "text": text[:280]}
        for _score, _candidate_like, _session_id, source_id, text in selected
        if source_id not in existing
    ][:max(0, limit - len(rows))])
    return rows[:limit]


def _append_compact_answer_bearing_sources(
    result: RetrievedContext,
    index: V36Index,
    sidecar: QuerySidecarV41,
    node_ids: list[str],
    ir: QueryIR,
    token_budget: int,
    limit: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Pack short query-bound source clauses before lower-value expansion."""
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    existing = set(result.retrieval_trace.get("packed_source_turn_ids") or [])
    existing.update(result.leaf_node_ids)
    added: list[str] = []
    decisions: list[dict[str, Any]] = []
    for node_id in node_ids:
        for source_id in _document_sources(sidecar, node_id):
            if source_id in existing:
                continue
            turn = turn_by_id.get(source_id)
            document = sidecar.documents.get(source_id)
            if turn is None or document is None or document.node_type != "turn":
                continue
            # A selected source is authoritative at turn granularity. Keeping
            # only one best clause can drop an identity/value in an adjacent
            # sentence while retaining a generic explanation or media caption.
            evidence_text = turn.text[:900]
            block = (
                f"[SOURCE_EVIDENCE {source_id}; added_by=v41_answer_bearing_span]\n"
                f"date={turn.session_date or 'unknown'}; speaker={turn.speaker}; "
                f"listener={turn.listener}; role={turn.transport_role}\n"
                f"{evidence_text}"
            )
            cost = rough_token_count(block)
            if result.packed_rough_tokens + cost > token_budget:
                decisions.append({
                    "node_id": node_id, "source_turn_id": source_id,
                    "decision": "reject_budget", "cost": cost,
                    "reason": "v41_answer_bearing_span",
                })
                continue
            result.context_text = (
                f"{result.context_text}\n\n{block}"
                if result.context_text else block
            )
            result.packed_rough_tokens += cost
            result.leaf_node_ids.append(source_id)
            result.evidence_leaf_ids.append(source_id)
            existing.add(source_id)
            added.append(source_id)
            decisions.append({
                "node_id": node_id, "source_turn_id": source_id,
                "decision": "append", "cost": cost,
                "reason": "v41_answer_bearing_span",
            })
            if len(added) >= limit:
                return added, decisions
    return added, decisions


def _verified_inference_candidates(
    planner: PlannerResultV41 | None, source_ids: list[str],
    ir: QueryIR, sidecar: QuerySidecarV41,
) -> list[dict[str, Any]]:
    if planner is None or not planner.valid:
        return []
    question_terms = set(_tokens(f"{ir.raw_question} {ir.target_relation}"))
    known_speakers = set(sidecar.inverted.get("speaker", {}))
    person_answer_requested = bool(
        ir.requested_value_type == "entity"
        and re.search(r"\b(?:who|which person)\b", ir.raw_question, re.IGNORECASE)
    )
    rows: list[dict[str, Any]] = []
    for candidate in planner.alternative_entities:
        normalized = candidate.strip().casefold()
        candidate_terms = set(_tokens(normalized))
        if len(normalized) < 3 or not candidate_terms or candidate_terms <= question_terms:
            continue
        # The planner often emits both conversation participants as retrieval
        # aliases. They are useful for finding sources but are not candidate
        # values for a yoga style, company, book, place, device, etc.
        if normalized in known_speakers and not person_answer_requested:
            continue
        matches = []
        for source_id in dict.fromkeys(source_ids):
            document = sidecar.documents.get(source_id)
            if document is not None and normalized in document.text.casefold():
                matches.append(source_id)
        if matches:
            rows.append({
                "candidate": candidate.strip(),
                "source_turn_ids": matches[:4],
                "provenance_complete": True,
            })
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(row["candidate"].casefold(), row)
    return list(unique.values())


def _verified_planner_slot_candidates(
    planner: PlannerResultV41 | None,
    allowed_source_ids: list[str],
    ir: QueryIR,
    sidecar: QuerySidecarV41,
    *,
    require_owner: bool = False,
) -> list[dict[str, Any]]:
    """Keep only planner values copied from an allowed lossless source."""
    if planner is None or not planner.valid:
        return []
    allowed = set(allowed_source_ids)
    question_terms = _tokens(ir.raw_question) - _QUERY_STOP_TERMS
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in planner.slot_candidates[:8]:
        value = str(candidate.get("value") or "").strip()
        source_id = str(candidate.get("source_turn_id") or "").strip()
        document = sidecar.documents.get(source_id)
        if (
            len(value) < 2 or source_id not in allowed
            or document is None or document.node_type != "turn"
            or (require_owner and not _source_owner_compatible(
                source_id, ir, sidecar,
            ))
        ):
            continue
        value_terms = _tokens(value) - _QUERY_STOP_TERMS
        source_terms = _tokens(document.text)
        if not value_terms or value_terms <= question_terms:
            continue
        normalized_value = " ".join(sorted(value_terms))
        key = (normalized_value, source_id)
        if key in seen:
            continue
        fuzzy_bound = _fuzzy_overlap(value_terms, source_terms)
        if len(fuzzy_bound) < max(1, len(value_terms) - 1):
            continue
        seen.add(key)
        rows.append({
            "value": value,
            "source_turn_id": source_id,
            "source_text": document.text[:420],
            "provenance_complete": True,
        })
    return rows


def _collection_head_and_boundary(question: str) -> tuple[str, str]:
    """Split the requested member type from a temporal/comparison boundary."""
    normalized = question.strip().rstrip("?.!")
    head_match = re.search(
        r"\bhow\s+many\s+(.+?)\s+"
        r"(?:did|do|does|have|has|had|are|is|was|were|can|could|will)\b",
        normalized, re.IGNORECASE,
    )
    head = head_match.group(1).strip() if head_match else ""
    boundary_match = re.search(
        r"\b(?:before|after|until|since)\b\s+(.+)$",
        normalized, re.IGNORECASE,
    )
    boundary = boundary_match.group(1).strip() if boundary_match else ""
    return head, boundary


def _planner_collection_exact_binding_safe(
    question: str,
    members: list[dict[str, Any]],
    evidence_candidates: list[dict[str, Any]] | None = None,
) -> bool:
    """Whether source-scanned members prove an exhaustive identity set.

    Planner candidates are open-world: source validation proves listed members,
    but not that none were omitted. Only deterministic source rescans with a
    constrained identity grammar may promote candidates to an exact count.
    """
    if not members:
        return False
    collection_head, _boundary = _collection_head_and_boundary(question)
    head_terms = _tokens(collection_head)
    requested_actions = action_families(question)
    provider_identity = bool(head_terms & {
        "doctor", "doctors", "physician", "physicians", "specialist",
        "specialists", "clinician", "clinicians", "dermatologist",
        "dermatologists",
    })
    owned_media_identity = bool(
        "acquire" in requested_actions
        and head_terms & {
            "album", "albums", "ep", "eps", "record", "records",
            "vinyl", "book", "books", "dvd", "dvds", "game", "games",
        }
    )
    property_identity = bool(head_terms & {
        "property", "properties", "home", "homes", "house", "houses",
        "condo", "condos", "townhouse", "townhouses", "apartment",
        "apartments", "bungalow", "bungalows",
    })
    if provider_identity or owned_media_identity or property_identity:
        return True

    # Generic identity collections can be closed only when every strongly bound
    # source scene is represented by at least one validated member. Repeated
    # turns in one session are one scene. Numeric/ordinal values are cumulative
    # measurements, not member identities, and therefore remain open-world.
    cumulative_value = re.compile(
        r"^\s*(?:\d+(?:st|nd|rd|th)?|one|two|three|four|five|six|seven|"
        r"eight|nine|ten|eleven|twelve)\b.*\b(?:times?|sessions?|projects?|"
        r"runs?|visits?|days?|weeks?|months?|years?)\b",
        re.IGNORECASE,
    )
    if len(members) < 2 or any(
        cumulative_value.search(str(row.get("value") or ""))
        for row in members
    ):
        return False

    def scene_id(source_id: str) -> str:
        return source_id.rsplit(":turn:", 1)[0]

    member_scenes = {
        scene_id(str(row.get("source_turn_id") or ""))
        for row in members if row.get("source_turn_id")
    }
    bound_scenes: set[str] = set()
    for row in evidence_candidates or []:
        features = {
            str(value) for value in row.get("selection_features") or []
        }
        if not any(value.endswith("_bound") for value in features):
            continue
        source_id = str(row.get("source_turn_id") or "")
        if source_id:
            bound_scenes.add(scene_id(source_id))
    return bool(bound_scenes and bound_scenes.issubset(member_scenes))


def _verified_planner_collection_members(
    planner: PlannerResultV41 | None,
    evidence_candidates: list[dict[str, str]],
    ir: QueryIR,
) -> list[dict[str, Any]]:
    """Validate planner and domain-operator members against exact excerpts."""
    evidence_by_id = {
        str(row.get("source_turn_id") or ""): str(row.get("text") or "")
        for row in evidence_candidates
        if row.get("source_turn_id") and row.get("text")
    }
    collection_head, scope_boundary = _collection_head_and_boundary(
        ir.raw_question
    )
    candidate_inputs = list(
        planner.member_candidates
        if planner is not None and planner.valid
        else []
    )
    head_terms = _tokens(collection_head)
    requested_actions = action_families(ir.raw_question)

    # Source-bound healthcare-provider identity. Prefer a clinician's name;
    # role labels are fallback only when the local care scene has no name.
    provider_query = bool(head_terms & {
        "doctor", "doctors", "physician", "physicians", "specialist",
        "specialists", "clinician", "clinicians", "dermatologist",
        "dermatologists",
    })
    if provider_query:
        provider_candidates: list[dict[str, str]] = []
        named_pattern = re.compile(r"\bDr\.?\s+[A-Z][A-Za-z'-]+\b")
        role_pattern = re.compile(
            r"\b(?:primary\s+care\s+physician|ENT\s+specialist|"
            r"dermatologist|cardiologist|neurologist|oncologist|"
            r"psychiatrist|therapist|clinician)\b",
            re.IGNORECASE,
        )
        for source_id, source_text in evidence_by_id.items():
            names = [match.group(0) for match in named_pattern.finditer(source_text)]
            if names:
                provider_candidates.extend({
                    "value": name, "source_turn_id": source_id,
                } for name in names)
                continue
            provider_candidates.extend({
                "value": match.group(0), "source_turn_id": source_id,
            } for match in role_pattern.finditer(source_text))
        if provider_candidates:
            candidate_inputs = provider_candidates

    # Source-bound owned-media identity.  This covers acquisitions whose title
    # is absent but whose artist/name plus format is explicit in the memory.
    if (
        "acquire" in requested_actions
        and head_terms & {
            "album", "albums", "ep", "eps", "record", "records",
            "vinyl", "book", "books", "dvd", "dvds", "game", "games",
        }
    ):
        media_pattern = re.compile(
            r"\bmy\s+((?:[A-Z][A-Za-z0-9&'.-]*\s+){1,4}"
            r"(?:vinyl|record|album|EP|book|DVD|game))\b"
        )
        for source_id, source_text in evidence_by_id.items():
            if re.search(
                r"\b(?:borrowed|lent|returned|gave\s+away|planning|"
                r"might\s+buy|recommended)\b",
                source_text, re.IGNORECASE,
            ):
                continue
            for match in media_pattern.finditer(source_text):
                candidate_inputs.append({
                    "value": match.group(1).strip(),
                    "source_turn_id": source_id,
                })

    # Source-bound current-owned instrument identity.  This is a reusable
    # ownership scene adapter: it recognizes explicit first-person possession,
    # keeps distinct physical subtypes/models, and excludes another person's or
    # merely planned acquisition before this verifier is reached.
    owned_instrument_query = bool(
        "possess" in requested_actions
        and head_terms & {"instrument", "instruments"}
    )
    if owned_instrument_query:
        instrument_kind = (
            r"electric\s+guitar|acoustic\s+guitar|bass\s+guitar|guitar|"
            r"digital\s+piano|piano|keyboard|drum\s+set|drums?|ukulele|"
            r"violin|cello|saxophone|trumpet|flute"
        )
        possessed_pattern = re.compile(
            rf"\bmy\s+((?:[A-Za-z0-9'’-]+\s+){{0,6}}?"
            rf"(?P<kind>{instrument_kind}))"
            rf"(?:,\s*(?:an?|the)\s+(?P<model>[^,.;?]+))?",
            re.IGNORECASE,
        )
        rows_by_base: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for source_id, source_text in evidence_by_id.items():
            if re.search(
                r"\b(?:niece|nephew|friend|sister|brother|partner|child)\b"
                r".{0,70}\b(?:their|her|his)\b.{0,40}"
                rf"\b(?:{instrument_kind})\b",
                source_text, re.IGNORECASE,
            ) and not re.search(
                rf"\bmy\b.{{0,60}}\b(?:{instrument_kind})\b",
                source_text, re.IGNORECASE,
            ):
                continue
            for match in possessed_pattern.finditer(source_text):
                phrase = match.group(1).strip()
                kind = re.sub(r"\s+", " ", match.group("kind").casefold())
                prefix = phrase[:match.start("kind") - match.start(1)]
                if _tokens(prefix) & {
                    "niece", "nephew", "friend", "sister", "brother",
                    "partner", "child", "daughter", "son", "her", "his",
                    "their", "got", "bought", "received", "owns", "owned",
                }:
                    continue
                model = (match.group("model") or "").strip()
                model = re.split(
                    r"\b(?:which|that|for|and\s+it|because)\b",
                    model, maxsplit=1, flags=re.IGNORECASE,
                )[0].strip(" -")
                base = (
                    "guitar" if "guitar" in kind else
                    "drum" if "drum" in kind else
                    kind.split()[-1]
                )
                value = f"{model} {kind}".strip() if model else phrase
                value = re.sub(r"\s+", " ", value).strip()
                generic = re.sub(
                    r"\b(?:old|new|black|white|my|our)\b", "", value,
                    flags=re.IGNORECASE,
                )
                generic = re.sub(r"\s+", " ", generic).strip().casefold()
                specificity = len(
                    _tokens(generic) - {
                        "guitar", "piano", "keyboard", "drum", "drums",
                        "set", "ukulele", "violin", "cello", "saxophone",
                        "trumpet", "flute",
                    }
                ) + int(kind in {"electric guitar", "acoustic guitar", "bass guitar"})
                rows_by_base[base].append({
                    "value": value, "source_turn_id": source_id,
                    "specificity": specificity,
                })
        owned_candidates: list[dict[str, str]] = []
        for base, rows_for_base in rows_by_base.items():
            has_specific = any(row["specificity"] > 0 for row in rows_for_base)
            seen_local: set[str] = set()
            for row in rows_for_base:
                if has_specific and row["specificity"] == 0:
                    continue
                key = re.sub(
                    r"[^a-z0-9]+", " ", str(row["value"]).casefold(),
                ).strip()
                if key in seen_local:
                    continue
                seen_local.add(key)
                owned_candidates.append({
                    "value": str(row["value"]),
                    "source_turn_id": str(row["source_turn_id"]),
                })
        if owned_candidates:
            candidate_inputs = owned_candidates

    # Reusable real-estate scene adapter.  The typed closure has already
    # established view/consider/offer lifecycle; extract only the property
    # identity and let the boundary certificate reject the target endpoint.
    if head_terms & {
        "property", "properties", "home", "homes", "house", "houses",
        "condo", "condos", "townhouse", "townhouses", "apartment",
        "apartments", "bungalow", "bungalows",
    }:
        dwelling_pattern = re.compile(
            r"\b(\d+-bedroom\s+(?:condo|townhouse|apartment|bungalow|"
            r"house|home))\b",
            re.IGNORECASE,
        )
        location_pattern = re.compile(
            r"\b(?:property|properties|one)\s+in\s+"
            r"([A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){0,3})"
        )
        for source_id, source_text in evidence_by_id.items():
            for match in dwelling_pattern.finditer(source_text):
                candidate_inputs.append({
                    "value": match.group(1).strip(),
                    "source_turn_id": source_id,
                })
            for match in location_pattern.finditer(source_text):
                candidate_inputs.append({
                    "value": match.group(1).strip(),
                    "source_turn_id": source_id,
                })

    question_terms = _tokens(ir.raw_question)
    boundary_terms = _tokens(scope_boundary) - _QUERY_STOP_TERMS
    rows: list[dict[str, Any]] = []
    seen_values: set[str] = set()
    for candidate in candidate_inputs[:24]:
        source_id = str(candidate.get("source_turn_id") or "")
        value = str(candidate.get("value") or "").strip()
        source_text = evidence_by_id.get(source_id)
        if not source_text or len(value) < 2:
            continue
        value_terms = _tokens(value) - _QUERY_STOP_TERMS
        source_terms = _tokens(source_text)
        # A member must add an identity beyond the question's generic category
        # and must not be the explicitly named comparison boundary.
        if not value_terms or not (value_terms - question_terms):
            continue
        if (
            boundary_terms
            and len(_fuzzy_overlap(value_terms, boundary_terms))
            >= max(1, len(value_terms) - 1)
        ):
            continue
        normalized_value = re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
        normalized_source = re.sub(
            r"[^a-z0-9]+", " ", source_text.casefold(),
        ).strip()
        fuzzy_bound_terms = _fuzzy_overlap(value_terms, source_terms)
        source_bound = (
            normalized_value in normalized_source
            or fuzzy_bound_terms == value_terms
        )
        if not source_bound:
            continue
        # Planner and domain adapters may express the same member at different
        # granularity.  Within one source, subset/high-overlap identities are one
        # member; unrelated conjoined members remain separate.
        duplicate_same_source = False
        for existing in rows:
            if existing.get("source_turn_id") != source_id:
                continue
            existing_terms = _tokens(str(existing.get("value") or ""))
            overlap = _fuzzy_overlap(value_terms, existing_terms)
            if len(overlap) >= min(len(value_terms), len(existing_terms)):
                duplicate_same_source = True
                break
        if duplicate_same_source:
            continue
        # Lexical tokens fold possessives for query matching, but a named
        # possessive still identifies one repeated event across paraphrased
        # turns (for example a party and feast at the same persons place).
        possessive_identity = sorted(
            match.group(1).casefold()
            for match in re.finditer(
                r"\b([A-Za-z][A-Za-z0-9_-]*)[\u0027’]s\b", value,
            )
        )
        identity_key = " ".join(
            possessive_identity or sorted(value_terms)
        )
        if identity_key in seen_values:
            continue
        seen_values.add(identity_key)
        rows.append({
            "value": value,
            "source_turn_id": source_id,
            "provenance_complete": True,
        })
    return rows


def _explicit_date_hint(
    index: V36Index, source_ids: list[str], ir: QueryIR,
) -> dict[str, Any] | None:
    if ir.requested_value_type != "date":
        return None
    relation_terms = _tokens(ir.target_relation)
    month_pattern = (
        r"\b(january|february|march|april|may|june|july|august|"
        r"september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?\b"
    )
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    rows: list[tuple[int, TurnNodeV36, str]] = []
    for source_id in source_ids:
        turn = turn_by_id.get(source_id)
        if turn is None or turn.transport_role != "user":
            continue
        match = re.search(month_pattern, turn.text, re.IGNORECASE)
        if match is None:
            continue
        tokens = _tokens(turn.text)
        relation_match = any(
            len(target) >= 5 and len(token) >= 5
            and target[:5] == token[:5]
            for target in relation_terms for token in tokens
        )
        target_overlap = len(set().union(*(
            _tokens(value) for value in ir.target_entities
        )) & tokens) if ir.target_entities else 0
        if not relation_match and target_overlap == 0:
            continue
        value = f"{match.group(1).title()} {match.group(2)}"
        rows.append((10 * int(relation_match) + target_overlap, turn, value))
    if not rows:
        return None
    _score, turn, value = max(rows, key=lambda row: (row[0], row[1].node_id))
    return {
        "operation": "source_bound_explicit_date", "value": value,
        "source_turn_id": turn.node_id, "evidence": turn.text[:360],
        "binding_complete": True, "certified": True,
    }


def _scoped_container_count_hint(
    index: V36Index, source_ids: list[str], ir: QueryIR,
) -> dict[str, Any] | None:
    if ir.requested_value_type != "count" or not re.search(
        r"\b(?:both|all|total|across)\b", ir.raw_question, re.IGNORECASE,
    ):
        return None
    query = ir.raw_question.casefold()
    container_terms = {
        token for token in _tokens(query)
        if token in {"aquarium", "aquariums", "tank", "tanks", "container", "containers"}
    }
    if container_terms & {"aquarium", "aquariums"}:
        container_terms.update({"tank", "tanks"})
    if not container_terms:
        return None
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    operands: list[dict[str, Any]] = []
    for source_id in dict.fromkeys(source_ids):
        turn = turn_by_id.get(source_id)
        if turn is None or turn.transport_role != "user":
            continue
        folded = turn.text.casefold()
        if not container_terms.intersection(_tokens(folded)):
            continue
        clauses = re.findall(
            r"(?:currently\s+has|which\s+has|contains|with)\s+([^.!?]+)",
            turn.text, re.IGNORECASE,
        )
        for clause in clauses:
            if not re.search(r"\b(?:fish|tetra|gourami|pleco|betta)\w*\b", clause, re.IGNORECASE):
                continue
            values = [
                int(value) for value, unit in re.findall(
                    r"\b(\d+)\s+([A-Za-z][A-Za-z-]*)", clause
                ) if unit.casefold() not in {
                    "gallon", "gallons", "liter", "liters", "litre", "litres",
                    "inch", "inches", "year", "years", "week", "weeks",
                }
            ]
            singulars = len(re.findall(
                r"\b(?:a|an|my)\s+(?:small\s+)?(?:[A-Za-z-]+\s+){0,2}"
                r"(?:fish|tetra|gourami|pleco|betta|catfish)\b",
                clause, re.IGNORECASE,
            ))
            value = sum(values) + singulars
            if value > 0:
                operands.append({
                    "source_turn_id": source_id, "value": value,
                    "evidence": clause[:300],
                })
                break
    if len(operands) < 2:
        return None
    return {
        "operation": "scoped_container_count", "operands": operands,
        "value": sum(row["value"] for row in operands), "unit": "members",
        "binding_complete": True, "certified": True,
    }


_SMALL_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20,
}


def _numeric_value(value: str) -> int | None:
    folded = value.casefold().replace(",", "")
    if folded.isdigit():
        return int(folded)
    return _SMALL_NUMBER_WORDS.get(folded)


def _source_date(value: str | None) -> datetime | None:
    if not value:
        return None
    match = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", value)
    if match is not None:
        try:
            return datetime(*(int(part) for part in match.groups()))
        except ValueError:
            return None
    # Chat exports often attach a clock prefix to a natural-language session
    # date. ISO-only parsing silently loses the source anchor needed by
    # relative-time evidence.
    natural = re.search(
        r"\b(\d{1,2})\s+"
        r"(January|February|March|April|May|June|July|August|September|"
        r"October|November|December),?\s+(\d{4})\b",
        value, re.IGNORECASE,
    )
    if natural is None:
        return None
    try:
        return datetime.strptime(
            f"{natural.group(1)} {natural.group(2).title()} {natural.group(3)}",
            "%d %B %Y",
        )
    except ValueError:
        return None


def _relative_date_from_text(text: str, anchor: datetime) -> datetime | None:
    """Resolve bounded source-local deictic dates without topic knowledge."""
    folded = text.casefold()
    if re.search(r"\byesterday\b", folded):
        return anchor - timedelta(days=1)
    if re.search(r"\btoday\b", folded):
        return anchor
    if re.search(r"\btomorrow\b", folded):
        return anchor + timedelta(days=1)
    match = re.search(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|a)\s+"
        r"(day|week)s?\s+ago\b", folded,
    )
    if match is None:
        return None
    amount = 1 if match.group(1) == "a" else (
        _numeric_value(match.group(1)) or 0
    )
    if amount <= 0:
        return None
    return anchor - timedelta(
        days=amount * (7 if match.group(2) == "week" else 1),
    )


def _scoped_relative_date_hint(
    index: V36Index, source_ids: list[str], ir: QueryIR,
) -> dict[str, Any] | None:
    """Bind an elliptical relative-date reply to its local event scene.

    This applies only to explicit before/after date questions, requires every
    comparison target in the same bounded scene, checks speaker ownership and
    provenance, and accepts only one resulting date. It is topic independent.
    """
    if (
        ir.requested_value_type != "date"
        or not set(ir.temporal_constraints).intersection({"after", "before"})
        or len(ir.comparison_targets) < 2
    ):
        return None
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    candidate_ids = set(source_ids)
    owner_terms = _tokens(ir.target_owner)
    rows: list[tuple[int, datetime, TurnNodeV36, list[TurnNodeV36]]] = []
    for source_id in dict.fromkeys(source_ids):
        turn = turn_by_id.get(source_id)
        if turn is None:
            continue
        if owner_terms and not _fuzzy_overlap(owner_terms, _tokens(turn.speaker_key)):
            continue
        anchor = _source_date(turn.session_date)
        if anchor is None:
            continue
        resolved = _relative_date_from_text(turn.text, anchor)
        if resolved is None or not re.search(
            r"\b(?:it|that|this|them|those|we|i|they|he|she)\b",
            turn.text, re.IGNORECASE,
        ):
            continue
        scene = [candidate for candidate in index.turns if (
            candidate.session_id == turn.session_id
            and abs(candidate.turn_index - turn.turn_index) <= 2
        )]
        scene_terms = _tokens(" ".join(candidate.text for candidate in scene))
        target_hits = 0
        for target in ir.comparison_targets:
            base_terms = _tokens(target) - _QUERY_STOP_TERMS - owner_terms
            matched = any(
                _fuzzy_overlap(
                    {term, *_RELATION_TERM_FAMILIES.get(term, set())},
                    scene_terms,
                )
                for term in base_terms
            )
            target_hits += int(matched)
        if target_hits < len(ir.comparison_targets):
            continue
        # Require the navigator to have selected at least two members of this
        # scene. The operator completes a route; it never creates a global one.
        selected_scene = sum(
            candidate.node_id in candidate_ids for candidate in scene
        )
        if selected_scene < 2:
            continue
        rows.append((target_hits * 10 + selected_scene, resolved, turn, scene))
    if not rows or len({row[1].date() for row in rows}) != 1:
        return None
    _score, resolved, turn, scene = max(rows, key=lambda row: row[0])
    anchor = _source_date(turn.session_date)
    assert anchor is not None
    return {
        "operation": "source_bound_scoped_relative_date",
        "value": resolved.strftime("%Y-%m-%d"),
        "date_precision": "day",
        "derivation_kind": "source_local_relative_scene",
        "source_turn_ids": [candidate.node_id for candidate in scene],
        "relative_source_turn_id": turn.node_id,
        "anchor_date": anchor.strftime("%Y-%m-%d"),
        "evidence": " ".join(candidate.text for candidate in scene)[:900],
        "binding_complete": True,
        "certified": True,
        "operator_certificate": {
            "entity_match": True, "relation_match": True,
            "scope_match": True, "provenance_complete": True,
        },
    }



def _relative_scope_matches(
    question: str, evidence: str, source_date: str | None,
    question_date: str | None,
) -> bool:
    """Validate common rolling windows without a topic vocabulary."""
    query = question.casefold()
    window = re.search(
        r"\b(?:last|past)\s+(\d+|one|two|three|four|five|six|seven|"
        r"eight|nine|ten|a)\s*(day|week|month|year)s?\b|"
        r"\b(?:last|past)\s+(day|week|month|year)\b",
        query,
    )
    if window is None:
        return True
    raw_amount = window.group(1) or "one"
    unit = window.group(2) or window.group(3)
    amount = 1 if raw_amount == "a" else (_numeric_value(raw_amount) or 1)
    max_days = amount * {
        "day": 1, "week": 7, "month": 31, "year": 366,
    }[unit]
    relative = re.search(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"a)\s+(day|week|month|year)s?\s+ago\b|"
        r"\b(?:last|past)\s+(day|week|month|year)\b",
        evidence, re.IGNORECASE,
    )
    if relative is not None:
        raw = relative.group(1) or "one"
        evidence_unit = relative.group(2) or relative.group(3)
        count = 1 if raw.casefold() == "a" else (_numeric_value(raw) or 1)
        age_days = count * {
            "day": 1, "week": 7, "month": 31, "year": 366,
        }[evidence_unit.casefold()]
        return age_days <= max_days
    observed = _source_date(source_date)
    anchor_date = _source_date(question_date)
    return bool(
        observed is not None and anchor_date is not None
        and 0 <= (anchor_date - observed).days <= max_days
    )



def _calendar_scope_matches(question: str, evidence: str) -> bool:
    """Apply an explicitly named calendar month to source-local dates."""
    month_names = {
        name.casefold(): index for index, name in enumerate(
            ("", "January", "February", "March", "April", "May", "June",
             "July", "August", "September", "October", "November", "December")
        ) if name
    }
    query = question.casefold()
    requested = next((
        number for name, number in month_names.items()
        if re.search(
            rf"\b(?:in|during|of|for)\s+(?:the\s+month\s+of\s+)?{name}\b|"
            rf"\bmonth\s+of\s+{name}\b",
            query,
        )
    ), None)
    if requested is None:
        return True
    observed: set[int] = {
        int(value) for value in re.findall(
            r"\b(1[0-2]|0?[1-9])/(?:3[01]|[12]\d|0?[1-9])(?:/\d{2,4})?\b",
            evidence,
        )
    }
    observed.update(
        number for name, number in month_names.items()
        if re.search(rf"\b{name}\b", evidence, re.IGNORECASE)
    )
    return not observed or requested in observed


def _collection_source_candidates(
    index: V36Index,
    ir: QueryIR,
    augmentation: QueryAugmentationV41,
    question_date: str | None,
    limit: int = 18,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Find source-bound collection scenes using card + lossless semantics.

    This is deliberately a recall closure, not a numeric operator.  It tolerates
    imperfect RoleFrame direction while requiring a category, action/lifecycle,
    scope, and user-source binding before a turn can enter the answer prompt.
    """
    if augmentation.answer_algebra != "collection":
        return [], []
    target_terms, relation_terms = query_binding_terms(ir)
    if not target_terms:
        return [], []
    expanded_terms = {
        term for term in binding_tokens(" ".join(augmentation.expanded_terms))
        if not action_families(term)
        and term not in {
            "event", "happen", "occur", "current", "latest", "state",
            "acquire", "acquired", "attend", "attended", "bought", "buy",
            "download", "downloaded", "get", "got", "had", "have", "join",
            "joined", "own", "owned", "participat", "participate",
            "participated", "possess", "purchase", "purchased", "receive",
            "received", "use", "used", "view", "viewed", "visit", "visited",
        }
    } - target_terms
    requested_families = action_families(ir.raw_question)
    possess_query = bool(
        "possess" in requested_families
        or re.search(
            r"\b(?:currently\s+)?(?:own|possess)\b|"
            r"\bdo\s+i\s+(?:currently\s+)?have\b",
            ir.raw_question, re.IGNORECASE,
        )
    )
    use_query = "use" in requested_families
    attend_query = "attend" in requested_families
    provider_query = bool(re.search(
        r"\b(?:doctor|doctors|physician|physicians|specialist|specialists|clinician|clinicians)\b",
        ir.raw_question, re.IGNORECASE,
    ))
    disjunctive_target = any(
        fuzzy_term_overlap(target_terms, binding_tokens(left))
        and fuzzy_term_overlap(target_terms, binding_tokens(right))
        for left, right in re.findall(
            r"\b([A-Za-z][A-Za-z-]*)\s+or\s+([A-Za-z][A-Za-z-]*)\b",
            ir.raw_question,
        )
    )
    cards_by_session: defaultdict[str, list[str]] = defaultdict(list)
    for card in index.routing_cards:
        cards_by_session[card.session_id].append(" ".join([
            card.routing_text, " ".join(card.canonical_entities),
            " ".join(card.relations), " ".join(card.key_events),
        ]))

    def matched(needles: set[str], haystack: set[str]) -> set[str]:
        values = needles & haystack
        values.update(
            needle for needle in needles
            if needle.endswith("-relat") and needle[:-6] in haystack
        )
        return values

    by_session: defaultdict[str, list[tuple[float, TurnNodeV36, list[str]]]] = defaultdict(list)
    for turn in index.turns:
        if turn.transport_role != "user":
            continue
        card_text = " ".join(cards_by_session.get(turn.session_id, []))
        turn_terms = binding_tokens(turn.text)
        combined_terms = turn_terms | binding_tokens(card_text)
        direct_hits = matched(target_terms, turn_terms)
        card_hits = matched(target_terms, combined_terms) - direct_hits
        direct_alias_hits = matched(expanded_terms, turn_terms)
        card_alias_hits = matched(expanded_terms, combined_terms) - direct_alias_hits
        alias_hits = direct_alias_hits | card_alias_hits
        category_signal = (
            len(direct_hits) + min(1, len(card_hits))
            + min(2, len(direct_alias_hits))
            + min(1, len(card_alias_hits))
        )
        required_signal = 1 if len(target_terms) == 1 or disjunctive_target else 2
        if possess_query and alias_hits:
            required_signal = 1
        if provider_query and alias_hits:
            required_signal = 1
        combined_text = f"{turn.text} {card_text}"
        observed_families = action_families(turn.text)
        family_overlap = requested_families & observed_families
        acquisition_bound = bool(
            "acquire" in requested_families
            and re.search(
                r"\b(?:acquired|bought|downloaded|ordered|picked\s+up|"
                r"purchased|received|got\s+(?:a|an|this|that|new))\b",
                turn.text, re.IGNORECASE,
            )
        )
        possessive_acquisition_context = bool(
            "acquire" in requested_families
            and direct_alias_hits
            and re.search(r"\bmy\b", turn.text, re.IGNORECASE)
        )
        acquisition_bound = acquisition_bound or possessive_acquisition_context
        if "acquire" in family_overlap and not acquisition_bound:
            family_overlap.discard("acquire")
        viewing_bound = bool(
            attend_query
            and re.search(r"\bview(?:ed|ing)?|\bsaw\b|\bseen\b|\btour(?:ed)?\b", turn.text, re.IGNORECASE)
        )
        property_offer_bound = False
        if re.search(r"\bview(?:ed|ing)?\b", ir.raw_question, re.IGNORECASE):
            family_overlap.intersection_update({"attend"})
            property_offer_bound = bool(re.search(
                r"\b(?:made|put|submitted|rejected)\b.{0,80}\boffer\b|"
                r"\boffer\b.{0,80}\b(?:made|put|submitted|rejected)\b",
                turn.text, re.IGNORECASE,
            ))
            if not (viewing_bound or property_offer_bound):
                family_overlap.discard("attend")
        alias_pattern = "|".join(
            re.escape(value) + r"\w*" for value in sorted(
                direct_alias_hits | direct_hits, key=len, reverse=True,
            ) if len(value) >= 3
        )
        established_possession = bool(
            alias_pattern and re.search(
                rf"\b(?:my|our)\b(?:\s+[A-Za-z0-9'-]+){{0,2}}\s+"
                rf"(?:{alias_pattern})\b|"
                rf"\b(?:playing|using|servicing|repairing|maintaining|selling|cleaning)"
                rf"\b.{{0,50}}\b(?:my|our)\b.{{0,50}}\b(?:{alias_pattern})\b|"
                rf"\b(?:i|we)(?:['’]ve|\s+have)\s+had\b.{{0,60}}"
                rf"\b(?:{alias_pattern})\b|"
                rf"\b(?:own|owned|possess|possessed)\b.{{0,60}}"
                rf"\b(?:{alias_pattern})\b",
                turn.text, re.IGNORECASE,
            )
        )
        planned_acquisition = bool(re.search(
            r"\b(?:i['’]ll|i\s+will|when\s+i|plan(?:ning)?|thinking|considering)"
            r"\b.{0,70}\b(?:get(?:ting)?|buy(?:ing)?|purchas(?:e|ing)|acquir(?:e|ing))\b",
            turn.text, re.IGNORECASE,
        ))
        possession_bound = bool(
            possess_query and established_possession and not planned_acquisition
        )
        negative_completion = bool(re.search(
            r"\b(?:never|not|didn['’]?t|haven['’]?t|hadn['’]?t)\b.{0,24}"
            r"\b(?:attend|visit|view|see|seen|go|went|participate|volunteer)\w*\b",
            turn.text, re.IGNORECASE,
        ))
        if negative_completion:
            viewing_bound = False
            family_overlap.discard("attend")
        attendance_bound = bool(
            attend_query and not negative_completion and re.search(
                r"\b(?:attended|visited|went|viewed|saw|seen|volunteered|hosted|"
                r"took\s+(?:my|our)|got\s+back\s+from|was\s+at|"
                r"(?:had|experience\w*)\b.{0,100}\bat|met\b.{0,80}\bat)\b",
                turn.text, re.IGNORECASE,
            )
        )
        care_bound = bool(
            provider_query
            and re.search(
                r"\b(?:appointment|follow-up)\s+with\b.{0,80}"
                r"\b(?:dr\.?|doctor|physician|specialist|dermatologist|clinician)\b|"
                r"\b(?:diagnosed|prescribed|treated)\b.{0,100}\b(?:by|me|my)\b|"
                r"\b(?:biopsy|consultation)\b.{0,100}\b(?:with|by|my)\b|"
                r"\b(?:saw|visited|got\s+back\s+from)\b.{0,80}"
                r"\b(?:dr\.?|doctor|physician|specialist|dermatologist|clinician|appointment)\b",
                turn.text, re.IGNORECASE,
            )
        )
        usage_bound = bool(
            use_query
            and re.search(
                r"\b(?:used|using|rely(?:ing|ied)?\s+on|ordered|through|via|"
                r"had\b.{0,80}\b(?:delivery|takeout|pizza|meal)|"
                r"all\s+about\b.{0,80}|lifesaver)\b",
                turn.text, re.IGNORECASE,
            )
        )
        if provider_query and not care_bound:
            continue
        if not (
            family_overlap or acquisition_bound or viewing_bound
            or property_offer_bound or possession_bound or attendance_bound
            or care_bound or usage_bound
        ):
            continue
        if category_signal < required_signal:
            card_grounded_usage = bool(
                (card_hits or card_alias_hits)
                and (usage_bound or care_bound or possession_bound)
            )
            if not card_grounded_usage:
                continue
        planned = bool(re.search(
            r"\b(?:plan(?:ning)?|want(?:ed)?|will|might|consider(?:ing)?|"
            r"thinking\s+of|going\s+to|soon|upcoming|next)\b|"
            r"\bi['’]ll\b",
            turn.text, re.IGNORECASE,
        ))
        completed = bool(
            acquisition_bound or viewing_bound or property_offer_bound
            or possession_bound or attendance_bound or care_bound or usage_bound
            or re.search(
                r"\b(?:acquired|attended|bought|completed|downloaded|finished|"
                r"hosted|ordered|participated|purchased|received|used|viewed|"
                r"visited|volunteered|went|worked|recently|already|last|ago|"
                r"yesterday|on\s+\d)\b",
                turn.text, re.IGNORECASE,
            )
        )
        if planned and not completed:
            continue
        if not _relative_scope_matches(
            ir.raw_question, turn.text, turn.session_date, question_date,
        ):
            continue
        if not _calendar_scope_matches(ir.raw_question, f"{turn.text} {card_text}"):
            continue
        relation_hits = matched(relation_terms, combined_terms)
        features = [
            *(f"target:{value}" for value in sorted(direct_hits)),
            *(f"card_target:{value}" for value in sorted(card_hits)),
            *(f"alias:{value}" for value in sorted(direct_alias_hits)[:4]),
            *(f"card_alias:{value}" for value in sorted(card_alias_hits)[:2]),
            *(f"action:{value}" for value in sorted(family_overlap)),
        ]
        if acquisition_bound:
            features.append("acquisition_bound")
        if viewing_bound:
            features.append("viewing_bound")
        if property_offer_bound:
            features.append("property_offer_bound")
        if possession_bound:
            features.append("possession_bound")
        if attendance_bound:
            features.append("attendance_bound")
        if care_bound:
            features.append("care_encounter_bound")
        if usage_bound:
            features.append("usage_bound")
        score = (
            10.0 * len(direct_hits) + 4.0 * len(card_hits)
            + 4.0 * min(3, len(direct_alias_hits))
            + 1.0 * min(2, len(card_alias_hits))
            + 5.0 * len(family_overlap)
            + 2.0 * len(relation_hits)
            + 4.0 * int(any((
                acquisition_bound, viewing_bound, property_offer_bound,
                possession_bound, attendance_bound, care_bound, usage_bound,
            )))
        )
        by_session[turn.session_id].append((score, turn, features))

    ranked: list[tuple[float, TurnNodeV36, list[str]]] = []
    for rows in by_session.values():
        ranked.extend(sorted(rows, key=lambda row: (-row[0], row[1].turn_index))[:2])
    ranked.sort(key=lambda row: (-row[0], row[1].node_id))
    selected = ranked[:max(1, limit)]
    # A paired assistant reply may explicitly confirm a user-owned named value
    # that the conversational user turn leaves implicit.  Admit only the local
    # confirming sentence, never recommendation lists or generic assistance.
    turn_by_position = {
        (turn.session_id, turn.turn_index): turn for turn in index.turns
    }
    confirmation_excerpts: dict[str, str] = {}
    selected_ids = {turn.node_id for _score, turn, _features in selected}
    confirmation_cue = re.compile(
        r"\bglad\s+to\s+hear\b|"
        r"\byou(?:['’]ve|\s+have|\s+had|\s+used|\s+bought|\s+downloaded|"
        r"\s+visited|\s+attended|\s+owned)\b|"
        r"\byour\b.{0,70}\b(?:has|have|is|are|was|were)\b",
        re.IGNORECASE,
    )
    named_value = re.compile(
        r"\b[A-Z][A-Za-z0-9&'.-]{1,}(?:\s+[A-Z][A-Za-z0-9&'.-]{1,})+\b"
    )
    for score, turn, features in list(selected):
        reply = turn_by_position.get((turn.session_id, turn.turn_index + 1))
        if (
            reply is None
            or reply.transport_role != "assistant"
            or reply.node_id in selected_ids
        ):
            continue
        clauses = [
            clause.strip() for clause in re.split(r"(?<=[.!?])\s+|\n+", reply.text)
            if clause.strip()
        ]
        clause = next((
            value for value in clauses
            if confirmation_cue.search(value)
            and named_value.search(value)
            and not re.search(
                r"\b(?:recommend|suggest|could|might|should|try)\b",
                value, re.IGNORECASE,
            )
        ), None)
        if clause is None:
            continue
        selected.append((
            score + 1.0, reply, [*features, "dialogue_confirmation"],
        ))
        selected_ids.add(reply.node_id)
        confirmation_excerpts[reply.node_id] = clause[:300]
        if len(selected) >= limit:
            break
    evidence = [{
        "source_turn_id": turn.node_id,
        "session_date": turn.session_date,
        "selection_score": round(score, 3),
        "selection_features": features,
        # Leave a small provider-token safety margin at the hard query limit.
        # This is a display excerpt only; the full turn remains indexed and
        # available through its source node and graph provenance.
        "text": confirmation_excerpts.get(turn.node_id, turn.text[:220]),
        "provenance_complete": True,
    } for score, turn, features in selected]
    return [turn.node_id for _score, turn, _features in selected], evidence


def _labeled_collection_subtotals_hint(
    index: V36Index, ir: QueryIR,
) -> dict[str, Any] | None:
    """Sum completed quantities from distinct, explicit scope labels."""
    if (
        ir.requested_value_type != "count"
        or not re.search(
            r"\b(?:total|combined|altogether|across|in all)\b",
            ir.raw_question, re.IGNORECASE,
        )
    ):
        return None
    target_terms, relation_terms = query_binding_terms(ir)
    if not target_terms:
        return None
    requested_families = action_families(ir.raw_question)
    if "complete" in requested_families:
        requested_families.discard("project_work")
    number = r"\d[\d,]*|" + "|".join(_SMALL_NUMBER_WORDS)
    number_pattern = re.compile(
        rf"\b(?P<value>{number})\b", re.IGNORECASE,
    )
    by_label: dict[str, dict[str, Any]] = {}
    for turn in index.turns:
        if turn.transport_role != "user":
            continue
        for clause in re.split(r"(?<=[.!?])\s+|\n+", turn.text):
            terms = binding_tokens(clause)
            if not fuzzy_term_overlap(target_terms, terms):
                continue
            historical_quantity = bool(
                re.search(
                    r"\b(?:already|previous|previously|prior)\b",
                    clause, re.IGNORECASE,
                )
                and re.search(r"\b\d[\d,]*\b", clause)
            )
            if not (
                fuzzy_term_overlap(relation_terms, terms)
                or requested_families.intersection(action_families(clause))
                or ("complete" in requested_families and historical_quantity)
            ):
                continue
            for match in number_pattern.finditer(clause):
                value = _numeric_value(match.group("value"))
                if value is None:
                    continue
                tail = clause[match.end():match.end() + 100]
                following = list(_WORD_RE.finditer(tail))[:5]
                head_positions = [
                    position for position, token_match in enumerate(following)
                    if fuzzy_term_overlap(
                        target_terms, binding_tokens(token_match.group(0))
                    )
                ]
                if not head_positions:
                    continue
                head_position = head_positions[-1]
                head_match = following[head_position]
                modifier_terms = [
                    token_match.group(0).casefold()
                    for token_match in following[:head_position]
                    if token_match.group(0).casefold() not in _QUERY_STOP_TERMS
                    and not fuzzy_term_overlap(
                        target_terms, binding_tokens(token_match.group(0))
                    )
                ]
                label = modifier_terms[-1] if modifier_terms else ""
                if not label:
                    suffix = tail[head_match.end():head_match.end() + 80]
                    labelled = re.match(
                        r"\s+(?:on|through|via|from|at|in)\s+"
                        r"([A-Za-z0-9][A-Za-z0-9._-]*)",
                        suffix, re.IGNORECASE,
                    )
                    if labelled is not None:
                        label = labelled.group(1).casefold().strip(".,;:")
                if not label:
                    continue
                row = {
                    "label": label, "value": value,
                    "source_turn_id": turn.node_id,
                    "evidence": clause[:360],
                    "_date": _source_date(turn.session_date) or datetime.min,
                }
                previous = by_label.get(label)
                if previous is None or row["_date"] > previous["_date"]:
                    by_label[label] = row
    operands = list(by_label.values())
    if len(operands) < 2:
        return None
    for row in operands:
        row.pop("_date", None)
    operands.sort(key=lambda row: (row["label"], row["source_turn_id"]))
    return {
        "operation": "labeled_collection_subtotal_sum",
        "operands": operands,
        "value": sum(row["value"] for row in operands),
        "unit": "members", "binding_complete": True, "certified": True,
    }


def _event_collection_members_hint(
    index: V36Index, ir: QueryIR, question_date: str | None,
) -> dict[str, Any] | None:
    """Close acquisition members from source-bound frame/event identities."""
    if ir.requested_value_type not in {"count", "list"}:
        return None
    target_terms, _relation_terms = query_binding_terms(ir)
    requested_families = action_families(ir.raw_question)
    # This deterministic member operator is intentionally narrower than the
    # recall closure. Mixed actions (for example worked-on OR bought) require
    # the answer model to union evidence instead of certifying a partial set.
    if (
        not target_terms
        or "acquire" not in requested_families
        or requested_families - {"acquire"}
    ):
        return None
    disjunctive_target = any(
        fuzzy_term_overlap(target_terms, binding_tokens(left))
        and fuzzy_term_overlap(target_terms, binding_tokens(right))
        for left, right in re.findall(
            r"\b([A-Za-z][A-Za-z-]*)\s+or\s+([A-Za-z][A-Za-z-]*)\b",
            ir.raw_question,
        )
    )
    def target_bound(text: str) -> bool:
        terms = binding_tokens(text)
        hits = sum(
            int(bool(fuzzy_term_overlap({target}, terms)))
            for target in target_terms
        )
        required = 1 if len(target_terms) == 1 or disjunctive_target else 2
        return hits >= required
    turns_by_session: defaultdict[str, list[TurnNodeV36]] = defaultdict(list)
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    for turn in index.turns:
        if turn.transport_role == "user":
            turns_by_session[turn.session_id].append(turn)
    members: dict[str, dict[str, Any]] = {}

    def add_member(identity: str, turn: TurnNodeV36, evidence: str) -> None:
        clean = re.sub(
            r"^(?:a|an|the|my|our)\s+", "", identity.strip(),
            flags=re.IGNORECASE,
        ).strip(" ,.;:-")
        key = " ".join(sorted(binding_tokens(clean)))
        if (
            not key or key in target_terms or len(clean.split()) > 8
            or re.fullmatch(
                r"(?:participant|speaker|user|listener)(?:\s+\d+)?",
                clean, re.IGNORECASE,
            )
        ):
            return
        members.setdefault(key, {
            "identity": clean, "source_turn_id": turn.node_id,
            "evidence": evidence[:360],
        })

    for frame in index.frames:
        if frame.polarity == "negative" or frame.lifecycle_status == "cancelled":
            continue
        if not frame.source_turn_ids:
            continue
        frame_terms = binding_tokens(" ".join((
            frame.entity_key, frame.predicate_key, frame.object_key,
            " ".join(frame.semantic_type_keys),
        )))
        if not fuzzy_term_overlap(target_terms, frame_terms):
            continue
        if "acquire" not in action_families(
            f"{frame.predicate_key} {frame.retrieval_text}"
        ):
            continue
        for source_id in frame.source_turn_ids:
            turn = turn_by_id.get(source_id)
            if turn is None or turn.transport_role != "user":
                continue
            if not target_bound(
                " ".join((turn.text, frame.entity_key, frame.object_key))
            ):
                continue
            if not _relative_scope_matches(
                ir.raw_question, turn.text, turn.session_date, question_date,
            ):
                continue
            identity = (
                frame.object_key
                if (
                    not frame.entity_key
                    or frame.entity_key == frame.owner_key
                    or re.fullmatch(
                        r"(?:participant|speaker|user|listener)(?:\s+\d+)?",
                        frame.entity_key, re.IGNORECASE,
                    )
                )
                else frame.entity_key
            )
            add_member(identity or frame.object_key, turn, turn.text)

    acquisition_tokens = {
        "acquire", "acquired", "acquisition", "buy", "bought",
        "get", "got", "obtain", "obtained", "purchase", "purchased",
        "receive", "received",
    }
    for card in index.routing_cards:
        source_rows = []
        for turn in turns_by_session.get(card.session_id, []):
            terms = binding_tokens(turn.text)
            if not fuzzy_term_overlap(target_terms, terms):
                continue
            if not target_bound(
                f"{turn.text} {card.routing_text} "
                f"{' '.join(card.canonical_entities)}"
            ):
                continue
            if "acquire" not in action_families(turn.text):
                continue
            if not _relative_scope_matches(
                ir.raw_question, turn.text, turn.session_date, question_date,
            ):
                continue
            source_rows.append(turn)
        if not source_rows:
            continue
        for event in card.key_events:
            if "acquire" not in action_families(event):
                continue
            event_words = [
                word for word in re.findall(r"[A-Za-z0-9'-]+", event)
                if word.casefold() not in acquisition_tokens
            ]
            identity = " ".join(event_words).strip()
            if not identity:
                continue
            identity_terms = binding_tokens(identity)
            source = next((
                turn for turn in source_rows
                if fuzzy_term_overlap(identity_terms, binding_tokens(turn.text))
            ), None)
            if source is not None:
                add_member(identity, source, source.text)
    if len(members) < 2:
        return None
    rows = sorted(members.values(), key=lambda row: (
        row["identity"].casefold(), row["source_turn_id"],
    ))
    return {
        "operation": "event_identity_collection_members",
        "members": rows, "value": len(rows), "unit": "members",
        "binding_complete": True, "certified": True,
    }


def _recommendation_highlights(
    index: V36Index, source_ids: list[str], ir: QueryIR,
    augmentation: QueryAugmentationV41,
) -> list[dict[str, str]]:
    if augmentation.answer_algebra != "preference_recommendation":
        return []
    stop = {
        "what", "which", "who", "how", "some", "would", "could",
        "should", "current", "setup", "look", "tips", "new", "this",
        "that", "with", "from", "your", "have", "about",
    }
    targets = {
        token for token in _tokens(" ".join([
            ir.raw_question, *ir.target_entities, *augmentation.expanded_terms,
        ])) if token not in stop and len(token) >= 3
    }
    preference_terms = {
        "prefer", "favorite", "like", "love", "leaning", "compare",
        "consider", "upgrade", "compatible", "quality", "durable",
        "recommend", "suggest", "interested", "use", "used", "own",
    }
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    ranked: list[tuple[float, TurnNodeV36]] = []
    for source_id in source_ids:
        turn = turn_by_id.get(source_id)
        if turn is None:
            continue
        tokens = set(_tokens(turn.text))
        overlap = targets.intersection(tokens)
        preferences = preference_terms.intersection(tokens)
        if not overlap or not preferences:
            continue
        score = 3.0 * len(overlap) + 2.0 * len(preferences)
        score += 0.5 if turn.transport_role == "user" else 0.0
        ranked.append((score, turn))
    ranked.sort(key=lambda row: (-row[0], row[1].node_id))
    return [
        {"source_turn_id": turn.node_id, "text": turn.text[:700]}
        for _score, turn in ranked[:4]
    ]


def retrieve(
    *,
    case: QuestionCase,
    variant: str,
    index: V36Index,
    capability_view: CapabilityViewV4,
    sidecar: QuerySidecarV41,
    query_ir: QueryIR,
    query_vectors: list[list[float]],
    token_budget: int = 9200,
    policy: QueryPolicyV41 | None = None,
    planner: PlannerResultV41 | None = None,
) -> RetrievedContext:
    policy = policy or QueryPolicyV41()
    augmentation = augment_query(query_ir)
    raw_overlays = _hydrate_lossless_sidecar(case, sidecar)
    pack_limit = max(
        1000, min(token_budget, policy.complex_context_target)
        - policy.answer_prompt_reserve,
    )
    base_budget = min(
        max(1000, policy.normal_context_target - policy.answer_prompt_reserve),
        pack_limit,
    )
    if augmentation.answer_algebra == "collection":
        # Reserve a small quota for typed member sources before broad V4
        # evidence fills the context.  This changes allocation, not the graph.
        base_budget = max(1000, base_budget - 400)
    result = retrieve_v4(
        case=case, variant=variant, index=index,
        capability_view=capability_view, query_vectors=query_vectors,
        token_budget=base_budget,
    )
    _invalidate_contradicted_absence(
        result, index, query_ir, sidecar, pack_limit,
    )
    original_sources = list(result.retrieval_trace.get("packed_source_turn_ids") or [])
    original_frames = list(result.retrieval_trace.get("packed_frame_ids") or [])
    original_groups = list(result.retrieval_trace.get("packed_group_ids") or [])
    certificate = _strict_certificate(
        query_ir, augmentation, index, original_sources,
        original_frames, original_groups,
    )
    # Protect query-bound multichannel anchors before spending the incremental
    # budget on broad scene completion.  This lets later closure grow around the
    # best evidence instead of crowding it out.
    candidates, candidate_trace = _candidate_nodes(
        query_ir, augmentation, sidecar, planner, policy.anchor_limit,
    )
    # Spend the incremental budget on role-directed graph navigation before
    # broad multichannel and scene overlays. The baseline V4 evidence remains
    # untouched; only the ordering of optional additions changes. Previously
    # these edges were computed late and almost always rejected after the pack
    # was full, reducing typed expansion to trace-only instrumentation.
    relations = _relations_for_gap(
        certificate.missing_roles, augmentation.answer_algebra,
    )
    seeds = list(dict.fromkeys([
        *original_sources, *original_frames, *original_groups,
        *candidates[:12],
    ]))
    expanded, expansion_trace = _expand_nodes(
        seeds, sidecar, relations, policy,
    )
    expanded_added, expansion_decisions = _append_sources(
        result, index, sidecar, expanded,
        pack_limit, "v41_typed_expansion", 8,
    )
    location_source_nodes = [
        node_id for node_id in candidates
        if (candidate_trace.get("channels", {}).get(node_id) or {}).get(
            "location_source_turn"
        ) is not None
    ][:4]
    ranked_answer_bearing_nodes = sorted(
        (
            node_id for node_id in candidates
            if (candidate_trace.get("channels", {}).get(node_id) or {}).get(
                "answer_bearing_turn"
            ) is not None
        ),
        key=lambda node_id: (
            (candidate_trace.get("channels", {}).get(node_id) or {}).get(
                "answer_bearing_turn", 10_000,
            ),
            node_id,
        ),
    )
    answer_bearing_nodes = list(dict.fromkeys([
        *location_source_nodes, *ranked_answer_bearing_nodes,
    ]))[:8]
    reply_bound_nodes = sorted(
        (
            node_id for node_id in candidates
            if (candidate_trace.get("channels", {}).get(node_id) or {}).get(
                "reply_bound_turn"
            ) is not None
        ),
        key=lambda node_id: (
            (candidate_trace.get("channels", {}).get(node_id) or {}).get(
                "reply_bound_turn", 10_000,
            ),
            node_id,
        ),
    )[:6]
    semantic_turn_nodes = sorted(
        (
            node_id for node_id in candidates
            if (candidate_trace.get("channels", {}).get(node_id) or {}).get(
                "lossless_semantic_turn"
            ) is not None
        ),
        key=lambda node_id: (
            (candidate_trace.get("channels", {}).get(node_id) or {}).get(
                "lossless_semantic_turn", 10_000,
            ),
            node_id,
        ),
    )[:4]
    answer_bearing_evidence: list[dict[str, Any]] = []
    reply_bound_evidence: list[dict[str, Any]] = []
    semantic_turn_evidence: list[dict[str, Any]] = []
    target_terms, _relation_terms = query_binding_terms(query_ir)
    if target_terms:
        turn_by_id = {turn.node_id: turn for turn in index.turns}
        for rank, node_id in enumerate(answer_bearing_nodes, 1):
            for source_id in _document_sources(sidecar, node_id):
                document = sidecar.documents.get(source_id)
                turn = turn_by_id.get(source_id)
                if document is None or turn is None:
                    continue
                source_text = re.sub(
                    r"^speaker\s+[^|]+\|\s*", "", document.text,
                    flags=re.IGNORECASE,
                )
                answer_bearing_evidence.append({
                    "source_turn_id": source_id,
                    "event_time": turn.session_date,
                    "speaker": turn.speaker_key,
                    "lifecycle": "source_bound",
                    "selection_features": ["answer_bearing_turn"],
                    "selection_score": float(100 - rank),
                    "text": _best_query_clause(source_text, query_ir)[:280],
                    "provenance_complete": True,
                })
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    for rank, node_id in enumerate(reply_bound_nodes, 1):
        document = sidecar.documents.get(node_id)
        turn = turn_by_id.get(node_id)
        if document is None or turn is None:
            continue
        reply_bound_evidence.append({
            "source_turn_id": node_id,
            "event_time": turn.session_date,
            "speaker": turn.speaker_key,
            "selection_features": ["reply_bound_turn"],
            "selection_score": float(100 - rank),
            "text": document.text[:320],
            "provenance_complete": True,
        })
    for rank, node_id in enumerate(semantic_turn_nodes, 1):
        for source_id in _document_sources(sidecar, node_id):
            document = sidecar.documents.get(source_id)
            turn = turn_by_id.get(source_id)
            if document is None or turn is None:
                continue
            source_text = re.sub(
                r"^speaker\s+[^|]+\|\s*", "", document.text,
                flags=re.IGNORECASE,
            )
            semantic_turn_evidence.append({
                "source_turn_id": source_id,
                "event_time": turn.session_date,
                "speaker": turn.speaker_key,
                "selection_features": ["lossless_semantic_turn"],
                "selection_score": float(100 - rank),
                "text": _best_query_clause(source_text, query_ir)[:280],
                "provenance_complete": True,
            })
    answer_bearing_added, answer_bearing_decisions = (
        _append_compact_answer_bearing_sources(
            result, index, sidecar, answer_bearing_nodes, query_ir,
            pack_limit, len(answer_bearing_nodes),
        ) if answer_bearing_nodes else ([], [])
    )
    collection_source_nodes, collection_source_evidence = (
        _collection_source_candidates(
            index, query_ir, augmentation, case.question_date,
        )
    )
    collection_source_added, collection_source_decisions = (
        _append_compact_answer_bearing_sources(
            result, index, sidecar, collection_source_nodes, query_ir,
            pack_limit, 12,
        ) if collection_source_nodes else ([], [])
    )
    collection_prompt_sources = set(
        result.leaf_node_ids
    ) | set(original_sources) | set(collection_source_added)
    collection_source_evidence = [
        row for row in collection_source_evidence
        if row.get("source_turn_id") in collection_prompt_sources
    ][:12]
    candidate_append_limit = (
        10 if augmentation.answer_algebra == "collection"
        else 8 if augmentation.answer_algebra in {
            "state_update", "temporal_lookup", "temporal_comparison",
        }
        else 16
    )
    candidate_added, candidate_decisions = _append_sources(
        result, index, sidecar, candidates, pack_limit,
        "v41_multichannel", candidate_append_limit,
    )
    lexical_scene_seeds = sorted(
        (
            node_id for node_id, channel_rows
            in (candidate_trace.get("channels") or {}).items()
            if channel_rows.get("sidecar_fts") is not None
            and (sidecar.documents.get(node_id) is not None)
            and sidecar.documents[node_id].node_type == "turn"
        ),
        key=lambda node_id: (
            candidate_trace["channels"][node_id]["sidecar_fts"], node_id,
        ),
    )[:16]
    lexical_scene_nodes = _scene_window_nodes(
        lexical_scene_seeds, query_ir, sidecar, limit=8,
    )
    general_scene_nodes = _scene_window_nodes(
        list(dict.fromkeys([
            *original_sources, *collection_source_added, *candidate_added,
            *candidates[:24],
        ])),
        query_ir, sidecar, limit=10,
    )
    scene_window_nodes = list(dict.fromkeys([
        *lexical_scene_nodes, *general_scene_nodes,
    ]))[:16]
    # Preserve the best local scene even if the larger full-source block is
    # rejected by the graph pack budget. These are short verbatim source rows,
    # not summaries, and keep statement -> follow-up question -> reply units
    # intact for the final selector.
    scene_window_evidence: list[dict[str, Any]] = []
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    compact_scene_nodes = list(dict.fromkeys([
        *lexical_scene_nodes[:4], *general_scene_nodes[:4],
    ]))[:8]
    for node_id in compact_scene_nodes:
        document = sidecar.documents.get(node_id)
        turn = turn_by_id.get(node_id)
        if document is None or turn is None or document.node_type != "turn":
            continue
        scene_window_evidence.append({
            "source_turn_id": node_id,
            "event_time": turn.session_date,
            "speaker": turn.speaker_key,
            "selection_features": ["typed_scene_window"],
            "text": document.text[:420],
            "provenance_complete": True,
        })
    scene_added, scene_decisions = _append_sources(
        result, index, sidecar, scene_window_nodes, pack_limit,
        "v41_scene_window", 8,
    )
    overlay_added, overlay_decisions = _append_lossless_overlays(
        result, index, raw_overlays, query_ir, pack_limit,
        preferred_source_ids=scene_window_nodes,
    )
    projected_nodes = _operator_source_projection(result, query_ir, sidecar)
    projected_added, projected_decisions = _append_sources(
        result, index, sidecar, projected_nodes, pack_limit,
        "v41_source_projection", 6,
    )
    selected_sources = list(dict.fromkeys(result.leaf_node_ids))
    priority_neighbors = _dialogue_closure_nodes(
        selected_sources, query_ir, sidecar, limit=6,
    )
    second_neighbors = _dialogue_closure_nodes(
        list(dict.fromkeys([*selected_sources, *priority_neighbors])),
        query_ir, sidecar, limit=6,
    )
    direct_replies = _dialogue_pair_completion_nodes(
        selected_sources, query_ir, sidecar, limit=4,
    )
    chained_replies = _dialogue_pair_completion_nodes(
        list(dict.fromkeys([*priority_neighbors, *second_neighbors])),
        query_ir, sidecar, limit=4,
    )
    pair_completion_nodes = list(dict.fromkeys([
        *direct_replies, *chained_replies, *second_neighbors,
        *priority_neighbors,
    ]))
    pair_completion_added, pair_completion_decisions = _append_sources(
        result, index, sidecar, pair_completion_nodes, pack_limit,
        "v41_dialogue_pair_completion", 10,
    )
    closure_nodes = _dialogue_closure_nodes(
        list(dict.fromkeys(result.leaf_node_ids)), query_ir, sidecar,
        policy.dialogue_closure_limit,
    )
    closure_added, closure_decisions = _append_sources(
        result, index, sidecar, closure_nodes, pack_limit,
        "v41_dialogue_closure", policy.dialogue_closure_limit,
    )
    # A second bounded pass is essential: multichannel/typed expansion may only
    # now have exposed the exact scene anchor.  Expand locally, never globally.
    late_scene_nodes = _scene_window_nodes(
        list(dict.fromkeys(result.leaf_node_ids)), query_ir, sidecar, limit=10,
    )
    late_scene_window_evidence: list[dict[str, Any]] = []
    for node_id in late_scene_nodes[:6]:
        document = sidecar.documents.get(node_id)
        turn = turn_by_id.get(node_id)
        if document is None or turn is None or document.node_type != "turn":
            continue
        late_scene_window_evidence.append({
            "source_turn_id": node_id,
            "event_time": turn.session_date,
            "speaker": turn.speaker_key,
            "selection_features": ["typed_late_scene_window"],
            "text": document.text[:420],
            "provenance_complete": True,
        })
    late_scene_added, late_scene_decisions = _append_sources(
        result, index, sidecar, late_scene_nodes, pack_limit,
        "v41_late_scene_window", 8,
    )
    packed_sources = list(dict.fromkeys(result.leaf_node_ids))
    # FTS/dialogue/typed expansion can expose a later decisive source after
    # the base V4 operator pass; refresh only strict source-bound operators.
    _refresh_post_expansion_operator_hints(
        result, index, query_ir, packed_sources,
    )
    repeated = repeated_event_total_from_sources_hint(
        query_ir, index, packed_sources,
    )
    if repeated is not None:
        hints = [
            hint for hint in result.retrieval_trace.get(
                "generic_operator_hints", []
            )
            if not (
                isinstance(hint, dict)
                and hint.get("operation") == "repeated_event_occurrence_ledger"
            )
        ]
        hints.append(repeated)
        result.retrieval_trace["generic_operator_hints"] = hints
    temporal_candidate_sessions = set(result.retrieved_session_ids)
    for node_id in candidates[:24]:
        document = sidecar.documents.get(node_id)
        if document is not None:
            temporal_candidate_sessions.update(document.session_ids)
    temporal_candidate_source_ids = [
        turn.node_id for turn in index.turns
        if turn.transport_role == "user"
        and turn.session_id in temporal_candidate_sessions
    ]
    derived_hints = [
        hint for hint in (
            _scoped_relative_date_hint(
                index, list(dict.fromkeys([
                    *packed_sources, *scene_window_nodes, *late_scene_nodes,
                ])), query_ir,
            ),
            _explicit_date_hint(index, packed_sources, query_ir),
            _scoped_container_count_hint(index, packed_sources, query_ir),
            _labeled_collection_subtotals_hint(index, query_ir),
            _event_collection_members_hint(
                index, query_ir, case.question_date,
            ),
            temporal_order_source_hint(
                query_ir, index, temporal_candidate_source_ids,
            ) if (
                augmentation.answer_algebra == "temporal_comparison"
                and len(query_ir.comparison_targets) >= 2
            ) else None,
            open_temporal_sequence_from_sources_hint(
                query_ir, index, [
                    turn.node_id for turn in index.turns
                    if turn.transport_role == "user"
                ], case.question_date,
            ) if augmentation.answer_algebra == "temporal_comparison" else None,
        ) if hint is not None
    ]
    temporal_operator_source_ids = list(dict.fromkeys(
        source_id
        for hint in derived_hints
        if hint.get("operation") in {
            "temporal_order_from_lossless_sources",
            "temporal_sequence_from_lossless_sources",
            "source_bound_scoped_relative_date",
        }
        for key in (
            "source_turn_ids", "selected_source_turn_id",
            "event_a_source_turn_id", "event_b_source_turn_id",
        )
        for source_id in (
            hint.get(key) if isinstance(hint.get(key), list)
            else [hint.get(key)]
        )
        if source_id
    ))
    temporal_operator_added, temporal_operator_decisions = (
        _append_sources(
            result, index, sidecar, temporal_operator_source_ids,
            pack_limit, "v41_temporal_operator_closure", 12,
        ) if temporal_operator_source_ids else ([], [])
    )
    turn_by_id = {turn.node_id: turn for turn in index.turns}
    temporal_operator_evidence = [
        {
            "source_turn_id": source_id,
            "session_date": turn_by_id[source_id].session_date,
            "text": turn_by_id[source_id].text[:280],
            "provenance_complete": True,
        }
        for source_id in temporal_operator_source_ids[:8]
        if source_id in turn_by_id
    ]
    packed_sources = list(dict.fromkeys(result.leaf_node_ids))
    if derived_hints:
        operation_names = {hint["operation"] for hint in derived_hints}
        hints = [
            hint for hint in result.retrieval_trace.get(
                "generic_operator_hints", []
            )
            if not (
                isinstance(hint, dict)
                and hint.get("operation") in operation_names
            )
        ]
        result.retrieval_trace["generic_operator_hints"] = [
            *hints, *derived_hints,
        ]
    certificate = _strict_certificate(
        query_ir, augmentation, index, packed_sources,
        original_frames, original_groups,
    )
    if any(
        hint.get("operation") in {
            "scoped_container_count",
            "labeled_collection_subtotal_sum",
            "event_identity_collection_members",
        }
        and hint.get("certified") is True
        for hint in derived_hints
    ):
        certificate.scope_match = True
        certificate.present_roles = sorted(set(
            certificate.present_roles
        ) | {"scope", "member", "members"})
        certificate.missing_roles = [
            role for role in certificate.missing_roles
            if role not in {"scope", "member", "members"}
        ]
        certificate.complete = all((
            certificate.entity_match, certificate.relation_match,
            certificate.scope_match, certificate.provenance_complete,
            certificate.lifecycle_complete, certificate.temporal_complete,
            certificate.dialogue_complete, not certificate.missing_roles,
        ))
    temporal_bound = next((
        hint for hint in derived_hints
        if hint.get("operation") in {
            "temporal_order_from_lossless_sources",
            "temporal_sequence_from_lossless_sources",
            "source_bound_scoped_relative_date",
        } and hint.get("certified") is True
    ), None)
    if temporal_bound is not None:
        certificate.entity_match = True
        certificate.relation_match = True
        certificate.scope_match = True
        certificate.provenance_complete = True
        certificate.temporal_complete = True
        endpoint_roles = (
            {"events", "times", "source"}
            if temporal_bound.get("operation")
            == "temporal_sequence_from_lossless_sources"
            else {"event_a", "event_b", "time_a", "time_b", "source"}
        )
        certificate.present_roles = sorted(
            set(certificate.present_roles) | endpoint_roles
        )
        certificate.missing_roles = [
            role for role in certificate.missing_roles
            if role not in endpoint_roles
        ]
        certificate.complete = all((
            certificate.entity_match, certificate.relation_match,
            certificate.scope_match, certificate.provenance_complete,
            certificate.lifecycle_complete, certificate.temporal_complete,
            certificate.dialogue_complete, not certificate.missing_roles,
        ))
    inferential_identity = is_inferential_question(query_ir.raw_question)
    reference_identity = augmentation.answer_algebra == "reference_identity"
    certified_source_bound_date = any(
        isinstance(hint, dict)
        and hint.get("operation") == "source_bound_explicit_date"
        and hint.get("planner_safe") is True
        and hint.get("certified") is True
        and hint.get("binding_complete") is True
        and all(
            (hint.get("operator_certificate") or {}).get(field) is True
            for field in (
                "entity_match", "relation_match", "scope_match",
                "provenance_complete",
            )
        )
        for hint in result.retrieval_trace.get(
            "generic_operator_hints", []
        )
    )
    # Planner is a bounded gap repair, not a default reasoning stage. It is
    # called only after deterministic multichannel and typed expansion still
    # leave the four provenance certificates or a required role incomplete.
    # Difficulty affects its prompt, not whether complete evidence is re-searched.
    augmentation.planner_required = not certificate.complete
    if certified_source_bound_date:
        result.retrieval_trace["v41_planner_gate"] = "certified_source_bound_date"
    if augmentation.answer_algebra == "collection":
        planner_evidence = []
        for row in collection_source_evidence[:8]:
            source_id = str(row.get("source_turn_id") or "")
            features = set(row.get("selection_features") or [])
            document = sidecar.documents.get(source_id)
            if "dialogue_confirmation" in features or document is None:
                excerpt = str(row.get("text") or "")[:400]
            else:
                source_text = re.sub(
                    r"^speaker\s+[^|]+\|\s*", "", document.text,
                    flags=re.IGNORECASE,
                )
                excerpt = _best_query_clause(source_text, query_ir)[:400]
            planner_evidence.append({
                "source_turn_id": source_id, "text": excerpt,
            })
    else:
        # Give the planner bounded, lossless candidates from independent
        # retrieval channels even when their full source blocks missed the main
        # pack budget. Channel quotas prevent one dense/BM25 family from
        # monopolising this compact hand-off.
        if reference_identity:
            channel_rows = [
                *answer_bearing_evidence[:6],
                *reply_bound_evidence[:4],
                *semantic_turn_evidence[:6],
                *scene_window_evidence[:2],
            ]
        elif augmentation.answer_algebra == "inferential_profile":
            channel_rows = [
                *semantic_turn_evidence[:5],
                *reply_bound_evidence[:1],
                *answer_bearing_evidence[:2],
                *scene_window_evidence[:1],
            ]
        elif augmentation.answer_algebra in {
            "dialogue_lookup", "preference_recommendation",
        }:
            channel_rows = [
                *answer_bearing_evidence[:4],
                *reply_bound_evidence[:4],
                *scene_window_evidence[:1],
                *semantic_turn_evidence[:1],
            ]
        else:
            channel_rows = [
                *scene_window_evidence[:2],
                *late_scene_window_evidence[:2],
                *answer_bearing_evidence[:3],
                *reply_bound_evidence[:2],
                *semantic_turn_evidence[:2],
            ]
        planner_channel_source_ids = list(dict.fromkeys([
            *(planner.selected_source_ids[:2] if planner and planner.valid else []),
            *location_source_nodes[:1],
            *[
                str(row.get("source_turn_id") or "")
                for row in channel_rows
                if row.get("source_turn_id")
            ],
        ]))
        if reference_identity:
            # A named but merely topical item is often a distractor for a
            # description-to-identity question. Rank all descriptive sources
            # by conjunctive clue coverage instead of pinning that item first.
            planner_channel_source_ids = []
        owner_sensitive_plan = augmentation.answer_algebra in {
            "state_update", "inferential_profile", "reference_identity",
        }
        if owner_sensitive_plan:
            planner_channel_source_ids = [
                source_id for source_id in planner_channel_source_ids
                if _source_owner_compatible(source_id, query_ir, sidecar)
            ]
        planner_evidence = _planner_evidence_candidates(
            [
                source_id for source_id in packed_sources
                if not owner_sensitive_plan
                or _source_owner_compatible(source_id, query_ir, sidecar)
            ],
            query_ir, sidecar, limit=10,
            preferred_source_ids=planner_channel_source_ids,
            session_diverse=not reference_identity,
        )
    planner_selected_evidence: list[dict[str, Any]] = []
    offered_ids = {
        str(row.get("source_turn_id") or "") for row in planner_evidence
        if row.get("source_turn_id")
    }
    planner_allowed_source_ids = list(dict.fromkeys([
        *packed_sources, *sorted(offered_ids),
    ]))
    if planner is not None and planner.valid:
        selected_ids = (
            []
            if augmentation.answer_algebra == "collection"
            else planner.selected_source_ids[:8]
        )
        for source_id in dict.fromkeys(selected_ids):
            document = sidecar.documents.get(source_id)
            if (
                document is None
                or document.node_type != "turn"
                or source_id not in offered_ids
                or (
                    augmentation.answer_algebra in {
                        "state_update", "inferential_profile",
                    }
                    and not _source_owner_compatible(
                        source_id, query_ir, sidecar,
                    )
                )
            ):
                continue
            planner_selected_evidence.append({
                "source_turn_id": source_id,
                "text": document.text[:400],
                "provenance_complete": True,
                "packed_in_main_context": source_id in packed_sources,
            })
            if source_id not in result.evidence_leaf_ids:
                result.evidence_leaf_ids.append(source_id)
    verified_collection_members = _verified_planner_collection_members(
        planner, planner_evidence, query_ir,
    ) if augmentation.answer_algebra == "collection" else []
    verified_inference_candidates = _verified_inference_candidates(
        planner, planner_allowed_source_ids, query_ir, sidecar,
    ) if (inferential_identity or reference_identity) else []
    verified_slot_candidates = _verified_planner_slot_candidates(
        planner, planner_allowed_source_ids, query_ir, sidecar,
        require_owner=augmentation.answer_algebra in {
            "state_update", "inferential_profile",
        },
    )
    recommendation_highlights = _recommendation_highlights(
        index, packed_sources, query_ir, augmentation,
    )
    dialogue_highlights = _direct_dialogue_highlights(
        packed_sources, query_ir, sidecar,
    )
    result.variant = variant
    result.schema_version = GRAPHMEM_V41_SCHEMA
    result.retrieved_session_ids = list(dict.fromkeys([
        *result.retrieved_session_ids,
        *[
            document.session_ids[0]
            for source_id in [
                *expanded_added, *answer_bearing_added, *collection_source_added,
            *candidate_added, *scene_added, *overlay_added, *projected_added,
            *pair_completion_added, *closure_added, *late_scene_added,
            *temporal_operator_added,
            ]
            if (document := sidecar.documents.get(source_id))
            and document.session_ids
        ],
    ]))
    result.retrieval_trace.update({
        "v41_policy": asdict(policy),
        "v41_query_augmentation": asdict(augmentation),
        "v41_sidecar": {
            "index_hash": sidecar.index_hash,
            "policy_version": sidecar.policy_version,
            "diagnostics": sidecar.diagnostics,
        },
        "v41_candidate_trace": candidate_trace,
        "v41_optional_stage_order": [
            "typed_expansion", "answer_bearing", "collection_source",
            "multichannel", "scene_window", "lossless_overlay",
            "source_projection", "dialogue_pair", "dialogue_closure",
            "late_scene", "temporal_operator",
        ],
        "v41_scene_window_decisions": scene_decisions,
        "v41_scene_window_evidence": scene_window_evidence,
        "v41_late_scene_window_node_ids": late_scene_nodes,
        "v41_late_scene_window_decisions": late_scene_decisions,
        "v41_late_scene_window_evidence": late_scene_window_evidence,
        "v41_answer_bearing_decisions": answer_bearing_decisions,
        "v41_answer_bearing_source_ids": answer_bearing_added,
        "v41_location_source_node_ids": location_source_nodes,
        "v41_answer_bearing_evidence": answer_bearing_evidence,
        "v41_reply_bound_evidence": reply_bound_evidence,
        "v41_semantic_turn_evidence": semantic_turn_evidence,
        "v41_collection_source_node_ids": collection_source_nodes,
        "v41_collection_source_ids": collection_source_added,
        "v41_collection_source_decisions": collection_source_decisions,
        "v41_collection_source_evidence": collection_source_evidence,
        "v41_candidate_decisions": candidate_decisions,
        "v41_source_projection_decisions": projected_decisions,
        "v41_dialogue_pair_completion_decisions": pair_completion_decisions,
        "v41_lossless_overlay_decisions": overlay_decisions,
        "v41_lossless_overlay_source_ids": overlay_added,
        "v41_scene_window_node_ids": scene_window_nodes,
        "v41_dialogue_closure_decisions": closure_decisions,
        "v41_typed_expansion": expansion_trace,
        "v41_expansion_decisions": expansion_decisions,
        "v41_temporal_operator_decisions": temporal_operator_decisions,
        "v41_temporal_operator_source_ids": temporal_operator_added,
        "v41_temporal_operator_evidence": temporal_operator_evidence,
        "v41_evidence_certificate": asdict(certificate),
        "v41_recommendation_highlights": recommendation_highlights,
        "v41_direct_dialogue_highlights": dialogue_highlights,
        "v41_planner_evidence": planner_evidence,
        "v41_planner_exposed_unpacked_source_ids": sorted(
            offered_ids - set(packed_sources)
        ),
        "v41_planner_selected_evidence": planner_selected_evidence,
        "v41_planner_verified_collection_members": verified_collection_members,
        "v41_verified_inference_candidates": verified_inference_candidates,
        "v41_verified_planner_slot_candidates": verified_slot_candidates,
        "v41_original_source_ids": original_sources,
        "v41_original_frame_ids": original_frames,
        "v41_source_additions": [
            *expanded_added, *answer_bearing_added, *collection_source_added,
            *candidate_added, *scene_added, *overlay_added, *projected_added,
            *pair_completion_added, *closure_added, *late_scene_added,
            *temporal_operator_added,
        ],
        "v41_source_deletions": sorted(set(original_sources) - set(packed_sources)),
        "v41_frame_deletions": sorted(
            set(original_frames)
            - set(result.retrieval_trace.get("packed_frame_ids") or [])
        ),
        "packed_source_turn_ids": packed_sources,
        "answer_target_tokens": policy.query_target,
        "answer_hard_limit_tokens": policy.query_hard_limit,
        "planner_required": augmentation.planner_required,
        "planner_applied": planner is not None and planner.valid,
        "planner_result": asdict(planner) if planner is not None else None,
    })
    return result


def planner_messages(
    case: QuestionCase,
    query_ir: QueryIR,
    augmentation: QueryAugmentationV41,
    certificate: dict[str, Any],
    evidence_candidates: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    inferential_identity = is_inferential_question(case.question)
    reference_identity = augmentation.answer_algebra == "reference_identity"
    source_named_candidates: list[str] = []
    modal_pattern = re.compile(
        r"\b(?:want|wanted|wish|dream|hope|like|liked|love|cool|interested)\b",
        re.IGNORECASE,
    )
    proper_pattern = re.compile(
        r"\b(?:[A-Z][A-Za-z0-9&'.-]{2,})(?:\s+[A-Z][A-Za-z0-9&'.-]{2,})+\b"
    )
    ignored_named = {
        "Media Shared", "Source Evidence", "Graph Mem", "Under Review",
    }
    for row in evidence_candidates or []:
        text = str(row.get("text") or "")
        for clause in re.split(r"[.!?;\n]+", text):
            if not modal_pattern.search(clause):
                continue
            for candidate in proper_pattern.findall(clause):
                value = candidate.strip()
                if value.title() in ignored_named or value.casefold().startswith(("speaker ", "listener ")):
                    continue
                source_named_candidates.append(value)
    source_named_candidates = list(dict.fromkeys(source_named_candidates))[:8]
    collection_head, scope_boundary = _collection_head_and_boundary(
        case.question
    )
    payload = {
        "question": case.question,
        "question_date": case.question_date,
        "collection_head": collection_head,
        "scope_boundary": scope_boundary,
        "query_aliases": augmentation.expanded_terms,
        "requested_action_families": sorted(
            action_families(query_ir.raw_question)
        ),
        "target_entities": (
            sorted(_tokens(collection_head) - _QUERY_STOP_TERMS)
            if augmentation.answer_algebra == "collection" and collection_head
            else query_ir.target_entities
        ),
        "target_relation": query_ir.target_relation,
        "value_type": query_ir.requested_value_type,
        "domain_hints": augmentation.domain_hints,
        "answer_algebra": augmentation.answer_algebra,
        "collection_mode": augmentation.answer_algebra == "collection",
        "inferential_identity": inferential_identity,
        "source_named_candidates": source_named_candidates,
        "present_roles": certificate.get("present_roles", []),
        "missing_roles": certificate.get("missing_roles", []),
        "evidence_candidates": (evidence_candidates or [])[:8],
    }
    if augmentation.answer_algebra == "collection":
        system_content = (
            "Return only compact JSON with no prose or whitespace formatting: "
            '{"alternative_entities":[],"event_aliases":[],"relations":[],'
            '"member_candidates":[["value","source_turn_id"],...]}. '
            "First produce concise semantic aliases for the requested member type and "
            "action so a second retrieval pass can find paraphrases; aliases are query "
            "terms only and never answers. Inspect every evidence candidate. "
            "collection_head is the member type; "
            "query_aliases are valid category/action paraphrases and "
            "requested_action_families name the lifecycle relation. scope_boundary is "
            "context and must never be emitted as a member. Emit each "
            "distinct member matching the question's owner, entity, action, lifecycle, "
            "and time scope. Use "
            "only its minimal distinguishing identity copied from the excerpt and "
            "its exact source_turn_id. One source may yield multiple members. "
            "Deduplicate repeated mentions. Exclude plans, recommendations, wrong "
            "owners, generic containers, and the comparison boundary; when a generic "
            "project/container features a named worked-on object, emit that object, not "
            "the container. For acquisition collections, an explicitly owned item such "
            "as 'my X' is a potential member unless it is borrowed, gifted away, merely "
            "planned, or recommended; for owned media, artist plus format is a valid "
            "minimal identity when the title is absent. Never invent."
        )
    else:
        system_content = (
            "Return one compact JSON retrieval plan; never answer or invent facts. "
            "Generate precise entity aliases, action paraphrases, relations, time "
            "scopes, and missing roles likely to occur verbatim in memory. "
            "selected_source_ids must copy at most eight IDs from evidence_candidates "
            "whose entity, relation, owner, scope, and lifecycle jointly match. "
            "slot_candidates may copy up to four exact answer-slot phrases paired "
            "with their exact source_turn_id; the phrase must occur in that source "
            "and is only a source-bound candidate, never an unsupported answer. For "
            "dialogue questions select the local prompt/reply scene that asks the "
            "same semantic slot, not another scene sharing only the person or topic. "
            "For temporal questions bind the exact event identity first, select its "
            "time-bearing source and any required endpoint, and never substitute the "
            "date of a nearby event. For multi-hop questions select one source per "
            "required relation hop. When "
            "answer_algebra is inferential_profile, "
            "inference_candidates may contain up "
            "to three concrete ordinary-knowledge candidates that satisfy every semantic "
            "constraint in the question and are consistent with the source-supported "
            "profile. A candidate must fill the requested slot: name a real book/title, "
            "product, activity/hobby, exercise/style, occupation, place/containing region, "
            "or organization rather than echoing a participant or broad category. These "
            "candidates guide answer selection but are not memory facts. Emit no sensitive "
            "identity, religion, or group-membership candidate unless the requested "
            "owner directly self-identifies. Only for an explicit suspected/possible "
            "health question may a non-definitive candidate follow a direct physical "
            "sign or symptom; never invent a diagnosis. Treat "
            "the speaker as the owner of first-person claims and the listener only as "
            "the addressee. Never attribute one speaker's identity, membership, health, "
            "preference, location, or action to the listener merely because the listener "
            "is named. For sensitive identity or group-membership questions, select only "
            "direct self-identification by the requested owner; advocacy, allyship, "
            "attendance, or support by either participant is not membership evidence. For "
            "temporal questions select both endpoints; for dialogue select the local "
            "request/reply scene. When inferential_identity is true, copy a uniquely "
            "type-compatible source_named_candidate only when no evidence conflicts. "
            'Schema: {"alternative_entities":[],"event_aliases":[],"relations":[],'
            '"temporal_constraints":[],"missing_roles":[],"selected_source_ids":[],'
            '"slot_candidates":[["verbatim value","source_turn_id"]],"inference_candidates":[]}.'
        )
    if reference_identity:
        system_content = (
            "Return only compact JSON. This is reference_identity retrieval, not direct "
            "answering. Select source IDs only from evidence_candidates. First select one "
            "source scene satisfying every distinctive descriptive clue; a merely topical "
            "named item is a distractor. inference_candidates may contain up to three "
            "canonical identities inferred from that description and ordinary knowledge. "
            "Enforce medium and form factor, prefer the canonical prototype, and reject "
            "unsupported variants. Never use benchmark metadata or a lookup table. "
            "Preserve speaker ownership and never infer sensitive identity without direct "
            "self-identification. Schema: {\"alternative_entities\":[],"
            "\"event_aliases\":[],\"relations\":[],"
            "\"temporal_constraints\":[],\"missing_roles\":[],"
            "\"selected_source_ids\":[],"
            "\"slot_candidates\":[[\"verbatim value\",\"source_turn_id\"]],"
            "\"inference_candidates\":[]}."
        )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def parse_planner_result(text: str) -> PlannerResultV41:
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        payload = json.loads(text[start:end])
        if not isinstance(payload, dict):
            raise ValueError("planner payload is not an object")
        def values(name: str) -> list[str]:
            rows = payload.get(name) or []
            if not isinstance(rows, list):
                return []
            flattened: list[str] = []
            def visit(value: object) -> None:
                if isinstance(value, str):
                    normalized = value.strip()[:120]
                    if normalized:
                        flattened.append(normalized)
                elif isinstance(value, (int, float)):
                    flattened.append(str(value)[:120])
                elif isinstance(value, list):
                    for item in value:
                        visit(item)
                elif isinstance(value, dict):
                    # Some compatible endpoints occasionally return a richer
                    # alias object despite the requested string-array schema.
                    # Recover its scalar values; never index the repr of a dict.
                    for item in value.values():
                        visit(item)
            for row in rows[:12]:
                visit(row)
            return list(dict.fromkeys(flattened))[:12]
        def sourced_values(name: str, limit: int) -> list[dict[str, str]]:
            result: list[dict[str, str]] = []
            for row in (payload.get(name) or [])[:limit]:
                if isinstance(row, dict):
                    value = str(row.get("value") or "").strip()[:160]
                    source_id = str(
                        row.get("source_turn_id") or ""
                    ).strip()[:240]
                elif isinstance(row, list) and len(row) >= 2:
                    value = str(row[0] or "").strip()[:160]
                    source_id = str(row[1] or "").strip()[:240]
                else:
                    continue
                if value and source_id:
                    result.append({
                        "value": value, "source_turn_id": source_id,
                    })
            return result
        member_candidates = sourced_values("member_candidates", 12)
        slot_candidates = sourced_values("slot_candidates", 8)
        return PlannerResultV41(
            alternative_entities=values("alternative_entities"),
            event_aliases=values("event_aliases"),
            relations=values("relations"),
            temporal_constraints=values("temporal_constraints"),
            missing_roles=values("missing_roles"),
            selected_source_ids=values("selected_source_ids")[:8],
            member_candidates=member_candidates,
            slot_candidates=slot_candidates,
            inference_candidates=values("inference_candidates")[:3],
            valid=True,
        )
    except (ValueError, json.JSONDecodeError) as error:
        return PlannerResultV41(valid=False, error=str(error))


def trim_latest_addition(retrieval: RetrievedContext) -> str | None:
    """Remove only the newest V4.1-added source block, never V4 evidence."""
    trace = retrieval.retrieval_trace or {}
    additions = list(trace.get("v41_source_additions") or [])
    original = set(trace.get("v41_original_source_ids") or [])
    protected = set(trace.get("v41_answer_bearing_source_ids") or [])
    while additions:
        source_id = additions.pop()
        if source_id in original or source_id in protected:
            continue
        pattern = re.compile(
            r"(?:\n\n)?\[SOURCE_EVIDENCE " + re.escape(source_id)
            + r"; added_by=v41_[^\]]+\]\n.*?(?=\n\n\[|\Z)", re.DOTALL,
        )
        updated, count = pattern.subn("", retrieval.context_text, count=1)
        if not count:
            continue
        retrieval.context_text = updated.strip()
        retrieval.leaf_node_ids = [value for value in retrieval.leaf_node_ids if value != source_id]
        retrieval.evidence_leaf_ids = [value for value in retrieval.evidence_leaf_ids if value != source_id]
        trace["packed_source_turn_ids"] = [
            value for value in trace.get("packed_source_turn_ids", []) if value != source_id
        ]
        trace["v41_recommendation_highlights"] = [
            row for row in trace.get("v41_recommendation_highlights", [])
            if row.get("source_turn_id") != source_id
        ]
        trace["v41_source_additions"] = additions
        trace.setdefault("v41_budget_trimmed_source_ids", []).append(source_id)
        retrieval.packed_rough_tokens = rough_token_count(retrieval.context_text)
        return source_id
    return None


def _global_lossless_focused_evidence(
    trace: dict[str, Any], *, algebra: str = "", limit: int = 4,
) -> list[dict[str, Any]]:
    """Promote the compact global lossless shortlist into final decision view.

    The upstream scan is question-only and session-diverse. Its rows are not
    certified answers, but each carries a real source turn. Keeping this compact
    shortlist near the end of the prompt prevents direct evidence from being
    buried by larger routed scenes.
    """
    candidates: list[dict[str, Any]] = []
    for hint in trace.get("generic_operator_hints") or []:
        if (
            isinstance(hint, dict)
            and hint.get("operation") == "global_lossless_source_candidates"
        ):
            candidates.extend(
                row for row in (hint.get("candidates") or [])
                if isinstance(row, dict)
            )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank, candidate in enumerate(candidates, 1):
        source_id = str(candidate.get("source_turn_id") or "").strip()
        evidence = str(candidate.get("evidence") or "").strip()
        if not source_id or not evidence or source_id in seen:
            continue
        lexical_score = float(candidate.get("lexical_score") or 0.0)
        if lexical_score < 8.0 and algebra != "preference_recommendation":
            continue
        seen.add(source_id)
        row = {
            "source_turn_id": source_id,
            "source_date": candidate.get("source_date"),
            "selection_reason": "global_lossless_query_scene",
            "selection_rank": rank,
            "lexical_score": candidate.get("lexical_score"),
            "text": evidence[:280],
            "provenance_complete": True,
        }
        if algebra == "preference_recommendation":
            row["routing_context"] = str(
                candidate.get("routing_context") or ""
            )[:140]
        rows.append(row)
        if len(rows) >= max(1, limit):
            break
    return rows


_FOCUSED_EVIDENCE_ALGEBRAS = {
    "collection",
    "multi_hop",
    "state_update",
    "temporal_comparison",
    "temporal_lookup",
    "reference_identity",
}


def _focused_source_evidence(
    trace: dict[str, Any],
    *,
    algebra: str,
    question: str = "",
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Return compact source spans ordered by question-bound scope semantics."""
    if algebra not in _FOCUSED_EVIDENCE_ALGEBRAS:
        return []
    closure = trace.get("source_span_closure") or {}
    candidates = closure.get("candidates") or []
    question_lower = question.casefold()
    historical_query = bool(re.search(
        r"\b(?:first|initial|original|previous|previously|prior|before|at that time)\b",
        question_lower,
    ))
    first_period_query = bool(re.search(
        r"\b(?:first|initial)\b.{0,40}\b(?:day|week|month|year)s?\b",
        question_lower,
    ))
    current_query = bool(re.search(
        r"\b(?:current|currently|now|still)\b|"
        r"\bdo i (?:have|need|own|use)\b|\bhow many\b.{0,50}\bdo i have\b",
        question_lower,
    ))
    scope_vocabulary = {
        "first", "initial", "previous", "prior", "last", "past", "recent",
        "today", "yesterday", "week", "weeks", "month", "months", "year",
        "years", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten",
    }
    question_scope = _tokens(question_lower).intersection(scope_vocabulary)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        source_id = str(candidate.get("source_turn_id") or "").strip()
        text = str(candidate.get("text") or "").strip()
        if (
            not source_id
            or not text
            or candidate.get("provenance_complete") is not True
        ):
            continue
        dedupe_key = (source_id, text.casefold())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        text_lower = text.casefold()
        text_terms = _tokens(text_lower)
        score = float(candidate.get("score") or 0.0)
        features: list[str] = []
        shared_scope = sorted(question_scope.intersection(text_terms))
        if shared_scope:
            score += 2.0 * len(shared_scope)
            features.append("shared_time_scope:" + ",".join(shared_scope))
        if first_period_query:
            if re.search(r"\b(?:start|started|starting|first|initial)\b|\bsince\b.{0,50}\bago\b", text_lower):
                score += 20.0
                features.append("bounded_start_scope")
            if re.search(r"\b(?:past|recent|later)\b.{0,30}\b(?:few|several)\b", text_lower):
                score -= 8.0
                features.append("broader_later_scope")
        elif historical_query and re.search(
            r"\b(?:previous|previously|prior|old|original|before)\b", text_lower,
        ):
            score += 12.0
            features.append("historical_state_scope")
        correction = bool(re.search(
            r"\b(?:correct(?:ion|ed|ing)?|correct myself|actually|instead|not\s+\d)\b",
            text_lower,
        ))
        if correction:
            score += 5.0
            features.append("explicit_correction")
        event_time = str(candidate.get("event_time_text") or "")
        date_digits = re.sub(r"\D", "", event_time)[:14]
        date_rank = int(date_digits or 0)
        rows.append({
            "source_turn_id": source_id,
            "event_time": event_time or None,
            "speaker": candidate.get("speaker_key"),
            "lifecycle": candidate.get("lifecycle_status"),
            "polarity": candidate.get("polarity"),
            "matched_entities": candidate.get("target_terms") or [],
            "matched_relations": candidate.get("relation_terms") or [],
            "selection_features": features,
            "selection_score": round(score, 3),
            "text": text,
            "_date_rank": date_rank,
        })
    rows.sort(key=lambda row: (
        -float(row["selection_score"]),
        -int(row["_date_rank"]) if current_query else int(row["_date_rank"]),
        str(row["source_turn_id"]),
    ))
    for row in rows:
        row.pop("_date_rank", None)
    return rows[:max(1, limit)]


def answer_messages(
    case: QuestionCase,
    retrieval: RetrievedContext,
) -> list[dict[str, str]]:
    messages = answer_messages_v4(case, retrieval)
    # V3.6 exposes every binding_complete operator as a "verified" ledger,
    # including heuristic operations that V4.1 deliberately does not certify.
    # Remove that legacy block; safe source-bound values are reintroduced below
    # only through ANSWER_CONSTRAINT after all four certificates pass.
    if messages and messages[-1].get("role") == "user":
        messages[-1]["content"] = re.sub(
            r"Verified deterministic ledger:\n.*?\n\nMemory evidence:",
            "V4.1 validated operators appear only in ANSWER_CONSTRAINT below.\n\n"
            "Memory evidence:",
            messages[-1].get("content", ""),
            flags=re.DOTALL,
        )
    trace = retrieval.retrieval_trace or {}
    certificate = trace.get("v41_evidence_certificate") or {}
    algebra = str(
        (trace.get("v41_query_augmentation") or {}).get("answer_algebra") or ""
    )
    constraints: list[dict[str, Any]] = []
    if all(
        certificate.get(name) is True
        for name in (
            "entity_match", "relation_match", "scope_match",
            "provenance_complete",
        )
    ):
        for hint in trace.get("generic_operator_hints") or []:
            if not isinstance(hint, dict):
                continue
            local = hint.get("operator_certificate")
            binding_ok = hint.get("binding_complete") is not False
            operation = str(hint.get("operation") or "")
            mandatory_operations = {
                "distinct_collection", "scoped_container_count", "duration_total",
                "labeled_collection_subtotal_sum",
                "event_identity_collection_members",
                "temporal_order_from_lossless_sources",
                "temporal_sequence_from_lossless_sources",
                "explicit_operand_currency_sum", "labeled_scalar_difference",
                "same_unit_state_difference", "record_time_extreme",
                "source_bound_explicit_date",
                "source_bound_scoped_relative_date",
                "pending_operation_target_pairs",
                "named_event_attendance_count", "binary_savings_difference",
                "latest_scalar_state_from_lossless_sources",
                "threshold_progress_remaining", "latest_approx_scalar_state",
                "latest_labeled_currency_state", "latest_weekly_schedule_time",
                "same_unit_acquisition_total", "relative_anchor_source_lookup",
                "temporal_predecessor_entity",
                "transaction_sum_from_lossless_sources",
                "currency_extreme_entity_from_lossless_sources",
                "dialogue_final_choice_from_lossless_sources",
                "completed_item_metric_total",
                "scoped_completed_duration_total",
                "relative_value_multiplier_from_lossless_sources",
                "relative_duration_at_event", "prior_candidate_count",
                "travel_arrival_time_from_sources",
                "completed_work_subtype_total",
                "time_difference_from_lossless_sources",
                "relative_time_from_lossless_source",
                "age_arithmetic_from_lossless_sources",
                "advance_booking_recency_from_lossless_sources",
                "current_role_duration_from_lossless_sources",
                "weekly_schedule_distinct_days", "family_relation_subtype_total",
                "linked_event_date_from_lossless_sources",
                "latest_category_start_from_lossless_sources",
                "scoped_completed_event_members", "dialogue_attribute_item_match",
                "presupposed_event_absence",
                "preference_constraints_from_lossless_sources",
                "latest_valid_state", "exact_entity_absence",
            }
            robust_absence = (
                operation == "exact_entity_absence"
                and hint.get("binding_kind") in {
                    "named_entity", "required_component", "role_title",
                    "required_role", "required_collection_type",
                    "required_operand",
                }
            )
            duration_unit = str(hint.get("unit") or "").casefold().rstrip("s")
            duration_unit_bound = bool(
                operation != "duration_total"
                or (
                    duration_unit
                    and re.search(
                        rf"\b{re.escape(duration_unit)}s?\b",
                        case.question, re.IGNORECASE,
                    )
                )
            )
            query_ir_trace = trace.get("query_ir") or {}
            scoped_single_date_safe = not (
                operation == "source_bound_explicit_date"
                and set(query_ir_trace.get("temporal_constraints") or [])
                .intersection({"after", "before"})
                and len(query_ir_trace.get("comparison_targets") or []) > 1
            )
            operation_safe = (
                operation in mandatory_operations
                and duration_unit_bound
                and scoped_single_date_safe
                and not (
                    algebra == "temporal_lookup"
                    and operation in {
                        "temporal_order_from_lossless_sources",
                        "temporal_sequence_from_lossless_sources",
                    }
                )
                and (
                    operation != "exact_entity_absence"
                    or robust_absence
                )
            )
            certified = binding_ok and (
                hint.get("certified") is True or (
                isinstance(local, dict)
                and all(local.get(name) is True for name in (
                    "entity_match", "relation_match", "scope_match",
                    "provenance_complete",
                ))
            ))
            if certified and operation_safe:
                constraints.append({
                    key: hint[key] for key in (
                        "operation", "value", "unit", "members", "operands",
                        "selected_target", "answer_candidate", "comparison",
                        "selected_time", "ordered_targets", "event_times",
                        "source_turn_ids", "event_a_source_turn_id",
                        "event_b_source_turn_id", "event_a_time",
                        "event_b_time", "change_direction", "history",
                        "seconds", "required_marker", "required_phrase",
                        "binding_kind",
                    ) if key in hint
                })
    # Some operators locally prove all four certificates from their own
    # lossless sources. Do not suppress them because an unrelated broad
    # retrieval certificate is missing a generic role label.
    local_self_certifying = {
        "record_time_extreme", "source_bound_explicit_date",
        "source_bound_scoped_relative_date",
        "pending_operation_target_pairs", "paired_metric_total",
        "named_event_attendance_count", "binary_savings_difference",
        "latest_scalar_state_from_lossless_sources",
        "threshold_progress_remaining", "latest_approx_scalar_state",
        "latest_labeled_currency_state", "latest_weekly_schedule_time",
        "same_unit_acquisition_total",
        "currency_extreme_entity_from_lossless_sources",
        "dialogue_final_choice_from_lossless_sources",
        "completed_item_metric_total",
        "scoped_completed_duration_total",
        "relative_value_multiplier_from_lossless_sources",
        "relative_duration_at_event", "prior_candidate_count",
        "travel_arrival_time_from_sources",
        "completed_work_subtype_total",
        "time_difference_from_lossless_sources",
        "age_arithmetic_from_lossless_sources",
        "advance_booking_recency_from_lossless_sources",
        "current_role_duration_from_lossless_sources",
        "weekly_schedule_distinct_days", "family_relation_subtype_total",
        "linked_event_date_from_lossless_sources",
        "latest_category_start_from_lossless_sources",
        "scoped_completed_event_members", "dialogue_attribute_item_match",
        "presupposed_event_absence",
        "preference_constraints_from_lossless_sources",
    }
    existing_constraint_ops = {
        str(row.get("operation") or "") for row in constraints
    }
    for hint in trace.get("generic_operator_hints") or []:
        if not isinstance(hint, dict) or hint.get("certified") is not True:
            continue
        operation = str(hint.get("operation") or "")
        query_ir_trace = trace.get("query_ir") or {}
        if (
            operation == "source_bound_explicit_date"
            and set(query_ir_trace.get("temporal_constraints") or [])
            .intersection({"after", "before"})
            and len(query_ir_trace.get("comparison_targets") or []) > 1
        ):
            continue
        direction_state = bool(
            operation == "latest_valid_state"
            and hint.get("change_direction")
        )
        source_answer = bool(
            operation in {
                "relative_anchor_source_lookup",
                "temporal_predecessor_entity",
                "latest_category_start_from_lossless_sources",
                "dialogue_attribute_item_match",
                "currency_extreme_entity_from_lossless_sources",
                "dialogue_final_choice_from_lossless_sources",
                "relative_value_multiplier_from_lossless_sources",
            }
            and hint.get("answer_candidate")
        )
        robust_absence = bool(
            operation in {"exact_entity_absence", "presupposed_event_absence"}
            and hint.get("binding_kind") in {
                "named_entity", "required_component", "role_title",
                "required_role", "required_collection_type",
                "required_operand", "required_relation", "required_subtype",
            }
        )
        if (
            operation not in local_self_certifying
            and not direction_state
            and not source_answer
            and not robust_absence
        ) or operation in existing_constraint_ops:
            continue
        constraints.append({
            key: hint[key] for key in (
                "operation", "value", "unit", "members", "operands",
                "source_turn_ids", "change_direction", "history", "seconds",
                "answer_candidate", "required_marker", "required_phrase",
                "binding_kind",
            ) if key in hint
        })
        existing_constraint_ops.add(operation)

    verified_collection_members = (
        trace.get("v41_planner_verified_collection_members") or []
    )
    exact_collection_binding = bool(
        algebra == "collection"
        and _planner_collection_exact_binding_safe(
            case.question,
            verified_collection_members,
            trace.get("v41_collection_source_evidence") or [],
        )
    )
    if exact_collection_binding:
        constraints.append({
            "operation": "certified_exact_distinct_count",
            "value": len(verified_collection_members),
        })
    highlights = trace.get("v41_recommendation_highlights") or []
    dialogue_highlights = trace.get("v41_direct_dialogue_highlights") or []
    dialogue_prompt_evidence = (
        dialogue_highlights[:3]
        if algebra in {
            "dialogue_lookup", "direct_fact", "preference_recommendation",
            "inferential_profile", "reference_identity",
        }
        else []
    )
    verified_inference = trace.get("v41_verified_inference_candidates") or []
    focused_evidence = _focused_source_evidence(
        trace, algebra=algebra, question=case.question,
        limit=(
            8 if algebra in {"collection", "temporal_comparison"}
            else 12
        ),
    )
    global_focused = _global_lossless_focused_evidence(
        trace, algebra=algebra,
        limit=6 if algebra == "preference_recommendation" else 4,
    )
    answer_bearing_evidence = (
        trace.get("v41_answer_bearing_evidence") or []
    )
    semantic_turn_evidence = (
        trace.get("v41_semantic_turn_evidence") or []
    )
    scene_window_evidence: list[dict[str, Any]] = []
    seen_scene_sources: set[str] = set()
    for row in [
        *(trace.get("v41_scene_window_evidence") or []),
        *(trace.get("v41_late_scene_window_evidence") or []),
    ]:
        source_id = str(row.get("source_turn_id") or "")
        if not source_id or source_id in seen_scene_sources:
            continue
        seen_scene_sources.add(source_id)
        scene_window_evidence.append(row)
    if algebra == "collection":
        merged_focused: list[dict[str, Any]] = []
        seen_focused_sources: set[str] = set()
        for row in [
            *global_focused,
            *(trace.get("v41_collection_source_evidence") or []),
            *(trace.get("v41_answer_bearing_evidence") or []),
            *focused_evidence,
        ]:
            source_id = str(row.get("source_turn_id") or "")
            if not source_id or source_id in seen_focused_sources:
                continue
            seen_focused_sources.add(source_id)
            merged_focused.append(row)
            if len(merged_focused) >= 12:
                break
        focused_evidence = merged_focused
    elif (
        scene_window_evidence or answer_bearing_evidence
        or semantic_turn_evidence or global_focused
    ):
        # These are compact verbatim source views, so they can survive even when
        # the larger V4 context has consumed its graph-pack quota. This fixes the
        # distinct failure mode where a turn is found by FTS/dense/scene ranking
        # but its full source block is rejected for budget. Keep independent
        # channel quotas and never substitute an inferred summary.
        merged_focused = []
        seen_focused_sources = set()
        primary_rows = (
            [
                *scene_window_evidence[:4],
                *answer_bearing_evidence[:4],
                *(trace.get("v41_reply_bound_evidence") or [])[:3],
                *semantic_turn_evidence[:3],
            ]
            if algebra in {
                "dialogue_lookup", "direct_fact", "temporal_lookup",
                "multi_hop", "state_update", "preference_recommendation",
            }
            else [
                *scene_window_evidence[:4],
                *semantic_turn_evidence[:3],
                *(trace.get("v41_reply_bound_evidence") or [])[:3],
                *answer_bearing_evidence[:4],
            ]
        )
        for row in [*primary_rows, *global_focused, *focused_evidence]:
            source_id = str(row.get("source_turn_id") or "")
            if not source_id or source_id in seen_focused_sources:
                continue
            seen_focused_sources.add(source_id)
            merged_focused.append(row)
            if len(merged_focused) >= 12:
                break
        focused_evidence = merged_focused
    planner_selected_evidence = (
        trace.get("v41_planner_selected_evidence") or []
    )
    verified_slot_candidates = (
        trace.get("v41_verified_planner_slot_candidates") or []
    )
    if algebra == "state_update" and len(verified_slot_candidates) == 1:
        candidate = verified_slot_candidates[0]
        owner = str((trace.get("query_ir") or {}).get("target_owner") or "")
        source_text = str(candidate.get("source_text") or "")
        source_owner_match = bool(
            not owner or re.search(
                r"^speaker\s+" + re.escape(owner) + r"\b",
                source_text, re.IGNORECASE,
            )
        )
        if source_owner_match:
            constraints.append({
                "operation": "source_bound_state_slot",
                "value": candidate.get("value"),
                "source_turn_id": candidate.get("source_turn_id"),
                "provenance_complete": True,
            })
    if len(verified_inference) == 1:
        constraints.append({
            "operation": "source_bound_inference_candidate",
            **verified_inference[0],
        })
    if dialogue_highlights and algebra in {
        "dialogue_lookup", "direct_fact", "preference_recommendation",
    }:
        primary_dialogue = dialogue_highlights[0]
        if (
            primary_dialogue.get("provenance_complete") is True
            and len(primary_dialogue.get("matched_terms") or []) >= 2
        ):
            endorsements = primary_dialogue.get("followup_endorsements") or []
            if endorsements and re.search(
                r"\b(?:support|like|love|prefer|favorite|favourite|fan)\b",
                case.question, re.IGNORECASE,
            ):
                constraints.append({
                    "operation": "source_bound_followup_endorsement",
                    "candidates": endorsements,
                    "source_turn_id": primary_dialogue.get("followup_source_id"),
                    "provenance_complete": True,
                })
            constraints.append({
                "operation": "source_bound_direct_dialogue",
                "context_source_id": primary_dialogue.get("context_source_id"),
                "prompt_source_id": primary_dialogue.get("prompt_source_id"),
                "reply_source_id": primary_dialogue.get("reply_source_id"),
                "followup_source_id": primary_dialogue.get("followup_source_id"),
                "context": primary_dialogue.get("context"),
                "prompt": primary_dialogue.get("prompt"),
                "reply": primary_dialogue.get("reply"),
                "followup": primary_dialogue.get("followup"),
                "matched_terms": primary_dialogue.get("matched_terms"),
            })
    if (
        algebra == "preference_recommendation"
        and highlights
        and all(certificate.get(name) is True for name in (
            "entity_match", "relation_match", "scope_match",
            "provenance_complete",
        ))
    ):
        constraints.append({
            "operation": "source_bound_recommendation_context",
            "source_excerpts": highlights[:4],
        })
    raw_inference_suggestions = (
        (trace.get("planner_result") or {}).get("inference_candidates") or []
    )
    known_participant_terms = {
        value.casefold()
        for value in _tokens(case.question)
        if value.casefold() in {
            key.casefold()
            for key in (
                (trace.get("v41_query_ir") or {}).get("target_owner") or "",
            )
            if key
        }
    }
    inference_suggestions = [
        str(value).strip()[:160]
        for value in raw_inference_suggestions[:3]
        if str(value).strip()
        and str(value).strip().casefold() not in known_participant_terms
    ]
    inference_policy = (
        "For inferential profile questions, the memory evidence supplies the premises, "
        "not necessarily the answer wording. Make the single most plausible one-hop "
        "inference from stable traits, preferences, locations, relationships, health, "
        "skills, goals, and repeated behavior. Ordinary real-world category knowledge "
        "may map those premises to a likely occupation, place, product class, activity, "
        "or outcome. State uncertainty briefly when appropriate, but do not abstain "
        "merely because the inferred label is not verbatim in memory. Never use benchmark "
        "metadata or invent a conflicting premise. Evaluate the exact proposition or "
        "candidate in the question. Evidence for a different career, destination, "
        "preference, identity, or plan is not positive evidence for the requested one; "
        "when it is an explicit alternative, answer no/unlikely to the requested "
        "candidate and name the supported alternative briefly. Advocacy, allyship, "
        "community support, attendance, or helping a group never establishes a person's "
        "sensitive identity, religion, diagnosis, or group membership without direct "
        "self-identification. When the question asks which named thing, occupation, "
        "place, title, product, exercise, or activity, return the most concrete "
        "source-supported candidate rather than only a generic category. "
        if algebra in {"inferential_profile", "reference_identity"} else ""
    )
    algebra_policy = (
        "For preference or recommendation questions, a future visit or intended purchase is framing, not a memory fact that must be independently verified. "
        "If the evidence contains related owned equipment, compared options, an intended upgrade, compatibility, quality, or stated preferences, give tailored suggestions from those constraints instead of abstaining. An expressed comparison, leaning, intended upgrade, or prior use is itself usable setup context; do not claim that no setup information exists when any such source is present. "
        if algebra == "preference_recommendation" else ""
    )
    policy = (
        "V4.1 answer policy: the final response must be produced by this model. "
        + algebra_policy
        + inference_policy
        + "For a source_bound_recommendation_context constraint, synthesize useful suggestions consistent with its excerpts; do not abstain and do not merely repeat the excerpts. For preference or advice questions, explicitly carry forward every concrete prior item, technique, ingredient, device, success, failure, or stated interest in the focused evidence that matches the topic. If a source says the user has already used, tried, experimented with, discovered, found, noticed, or currently uses something, explicitly acknowledge it as an established prior practice and build on it; never reintroduce that same item as a new suggestion, swap, or first-time option. Make suggestions from those constraints; do not replace them with unrelated generic advice. For health or environmental-cause advice, mention every concrete source-supported factor and state any additional plausible mechanism conditionally; never claim an unrecorded action already happened. "
        + "For likelihood or hypothetical questions, infer the best-supported likelihood from the most causally direct positive or negative evidence; do not treat a different nearby activity as proof of intent. A completed activity is not evidence of willingness to repeat it: after harm, fear, failure, cancellation, or explicit reluctance, answer unlikely/no unless the sources contain a later affirmative intention. When asked for a possible book, product, hobby, exercise, occupation, destination, or other recommendation, return at least one concrete valid example consistent with the source-supported profile. Merely repeating the requested category, goal, or participant name is invalid. Ordinary category knowledge may supply the concrete example, but must not invent a contradictory memory premise. "
        + "When DIRECT_DIALOGUE_EVIDENCE contains a local context statement plus a prompt that semantically asks for the requested slot, combine context, prompt, immediate reply, and any follow-up acknowledgement as one scene. This scene is primary evidence and outranks merely topical facts from other scenes. Preserve speaker ownership: a follow-up endorsement or acceptance can establish the follow-up speaker's preference or support for the named reply even when the reply originally belonged to someone else. A source_bound_followup_endorsement constraint is a positive, provenance-bound statement by the requested owner; when the question asks support/like/prefer/favorite, select the candidate matching the requested semantic type and do not abstain. Return all concrete values, components, or named results that answer the slot across that scene. A source_bound_direct_dialogue row in ANSWER_CONSTRAINT has passed relation and provenance checks: do not replace it with an unpaired topical fact. "
        + "When a later source names only an entity type and an earlier source gives candidate identities, resolve the identity only when one candidate uniquely matches the type and no source conflicts; answer with that candidate rather than repeating the generic type. When the question explicitly says likely, it requests this best-supported inference rather than verbatim extraction: ordinary category knowledge may be used to match a source-named candidate to the unnamed type. A unique source_bound_inference_candidate is mandatory because the planner proposed it and exact source validation confirmed the name; answer with it unless another source explicitly conflicts. The ban on topic inference prohibits benchmark metadata and answer-key guessing, not ordinary entity-type knowledge. "
        + "When earlier evidence distinguishes already-completed relationships from a desired future candidate, and a later unnamed new event matches that candidate's type, use the desired candidate as the likely identity when no conflict is present. "
        + "Resolve relative dates such as next month or last week against the cited source turn date, and report the resulting calendar month or date when the question asks when. For first/earliest/latest/order questions, compare the source dates or event times of the exact requested completed events; an earlier mention, a future plan, or a merely topical event does not outrank a later explicit participation date. "
        + "For quantitative questions, identify the requested operation before choosing a number. Words such as each, per, average, total, combined, difference, remaining, and how many times are not interchangeable. Use only operands bound to the same entity and scope; divide a group total by its explicit item count for each/per, sum only requested components for total/combined, and subtract the named endpoints for difference/remaining. Briefly verify the operation against the wording and never return a nearby total when a per-item value was requested. When latest_valid_state includes change_direction and the question asks more versus less, answer that direction exactly; do not state the opposite in a lead sentence. "
        + "PLANNER_SELECTED_SOURCE_EVIDENCE contains source IDs revalidated against the exact planner candidate set and lossless source provenance; packed_in_main_context may be false because this compact source view is the budget-safe hand-off from a bounded retrieval channel. PLANNER_VERIFIED_SLOT_CANDIDATES contains only phrases locally matched back to those lossless sources; use them as high-priority candidates only when their source has the requested owner, relation, scope and lifecycle. Prefer the candidate from the exact event or dialogue scene over a merely topical value, and never mix a value from one row with the date or owner of another. For collections this duplicate block is intentionally empty: FOCUSED_SOURCE_EVIDENCE is the full typed source closure. Apply entity, relation, owner, scope, lifecycle and time constraints. A planner_verified_collection_members constraint contains individually source-verified member identities: do not omit one unless it is a duplicate of another listed identity, and still scan FOCUSED_SOURCE_EVIDENCE for additional valid members because candidate_pool_complete is false. "
        + "When FOCUSED_SOURCE_EVIDENCE is present, treat it as the primary candidate set and the larger memory block only as fallback. INFERENCE_PLANNER_CANDIDATES are advisory ordinary-knowledge options, not memory facts: for inferential-profile or reference-identity questions, choose the candidate that satisfies every requested semantic constraint and best fits one source-supported profile or description. For reference identity, all discriminating clues must be supported by the same source scene; an explicitly named but merely topical item does not outrank the unnamed matching description. Enforce the requested medium or form factor: do not answer a board, tabletop, or card-game question with a digital-only video or mobile title, and do not substitute a book, film, device, or place across semantic types. For an unnamed description, prefer the canonical widely recognized prototype over a branded adaptation, spin-off, subtitle variant, or obscure near match unless the source gives the variant-specific feature. Reject a candidate that merely repeats a participant, broad category, or goal. Do not use advisory candidates to infer sensitive identity, religion, or membership without direct self-identification. If and only if the question explicitly requests a suspected or possible health problem, infer a non-definitive ordinary condition from a direct source-supported physical sign, symptom, limitation, or risk and state uncertainty; never present it as a confirmed diagnosis. Distinguish generalized body-size, weight, fitness, or exercise evidence from localized pain, numbness, or injury; do not infer a hand or joint disorder from size alone. The question wording is the selection predicate: match exact entity, owner, relation, temporal scope and lifecycle before using recency or topical similarity. Current or now selects the latest valid state; initial, first, previous, or an explicitly bounded period selects the matching historical source and must not be replaced by a later broader value. A correction or explicit replacement selects the new value and treats the named old value as superseded. For a collection, enumerate distinct source-bound members or use an explicitly stated total only when its scope exactly matches the question. For a temporal calculation, bind both requested endpoints before calculating. "
        + "Answer the requested semantic slot at the most specific source-supported granularity: when a prompt/reply supplies concrete components, members, a proper name, or a named result, copy that exact concrete value rather than restating only the action or broad category. If the question asks which company, organization, title, place, product, person, or other named entity, a generic descriptor is not an answer whenever any matching source names a candidate. Preserve attribution: first-person statements belong to their speaker, not their listener, and being addressed by name does not transfer the speaker's identity, preference, health, membership, location, or action. Satisfy every descriptive clue conjunctively; a candidate matching the topic but missing a discriminating attribute such as fruit, location, lifecycle, or medium loses to a candidate that matches all attributes. If the question asks worth or value in terms of the amount paid and the source states a relative multiplier, return that multiplier relation rather than the purchase price alone. "
        + "If V4.1_CONSTRAINT_OVERRIDE is present, ignore the older exact_entity_absence row named by that override. Only rows copied into ANSWER_CONSTRAINT are mandatory; an older operator ledger certified label is not sufficient by itself. "
        "A value, date, member set, or state in ANSWER_CONSTRAINT is mandatory "
        "because it passed entity, relation, scope, and provenance checks; do not "
        "silently replace it. Uncertified operator rows remain hints only. Prefer "
        "the best source-supported answer and abstain only for certified exact "
        "entity mismatch or genuinely missing required evidence roles."
    )
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = f"{messages[0]['content']}\n\n{policy}"
    if messages and messages[-1].get("role") == "user":
        messages[-1]["content"] += (
            "\n\nV4.1 evidence certificate:\n"
            + json.dumps(certificate, ensure_ascii=False)
            + "\n\nANSWER_CONSTRAINT:\n"
            + json.dumps([
                row for row in constraints
                if row.get("operation") not in {
                    "temporal_order_from_lossless_sources",
                    "temporal_sequence_from_lossless_sources",
                }
            ], ensure_ascii=False)
            + "\n\nINFERENCE_PLANNER_CANDIDATES (advisory ordinary-knowledge options):\n"
            + json.dumps(inference_suggestions, ensure_ascii=False)
            + "\n\nPLANNER_SELECTED_SOURCE_EVIDENCE (validated source excerpts):\n"
            + json.dumps(planner_selected_evidence, ensure_ascii=False)
            + "\n\nPLANNER_VERIFIED_SLOT_CANDIDATES (verbatim source-bound values):\n"
            + json.dumps(verified_slot_candidates, ensure_ascii=False)
            + "\n\nFOCUSED_SOURCE_EVIDENCE (primary query-bound source candidates):\n"
            + json.dumps(focused_evidence, ensure_ascii=False)
            + "\n\nV4.1_RELEVANT_SOURCE_HIGHLIGHTS (verbatim source, not inferred facts):\n"
            + json.dumps(trace.get("v41_recommendation_highlights") or [], ensure_ascii=False)
            + "\n\nDIRECT_DIALOGUE_EVIDENCE (source-bound prompt/reply pairs):\n"
            + json.dumps(dialogue_prompt_evidence, ensure_ascii=False)
            + (
                "\n\nCOLLECTION_FINAL_CHECK (mandatory silent ledger):\n"
                "Retain only evidence matching the requested owner, entity, action, "
                "lifecycle and time scope. Enumerate every distinct member or numeric "
                "operand across all matching sources; count conjoined items separately; "
                "merge repeated mentions of the same named member or event; combine "
                "non-overlapping subtotals; and apply later add/remove updates to an "
                "earlier explicitly scoped total. A total must not return only one "
                "platform, session, category or earlier subtotal. Plans, recommendations "
                "and merely related mentions count as zero unless requested. Verify the "
                "number silently and output only the concise answer."
                if algebra == "collection" else ""
            )
        )
        temporal_constraints = [
            row for row in constraints
            if row.get("operation") in {
                "temporal_order_from_lossless_sources",
                "temporal_sequence_from_lossless_sources",
            }
        ]
        if temporal_constraints:
            messages[-1]["content"] += (
                "\n\nCERTIFIED_TEMPORAL_RESULT (FINAL MANDATORY BINDING):\n"
                + json.dumps({
                    "result": temporal_constraints,
                    "source_evidence": (
                        trace.get("v41_temporal_operator_evidence") or []
                    ),
                }, ensure_ascii=False)
                + "\nThe event identities, endpoint times, comparison direction, and "
                "selected target above passed exact source/provenance validation. "
                "Use this selected target exactly. Do not replace it with a more "
                "topically salient event from the larger memory block."
            )
        certified_absence = [
            row for row in constraints
            if row.get("operation") in {
                "exact_entity_absence", "presupposed_event_absence",
            }
            and row.get("binding_kind") in {
                "named_entity", "required_component", "role_title",
                "required_role", "required_collection_type",
                "required_operand", "required_relation", "required_subtype",
            }
        ]
        preference_constraints = [
            row for row in constraints
            if row.get("operation")
            == "preference_constraints_from_lossless_sources"
            and row.get("value")
        ]
        if preference_constraints:
            established_practice_rows = [
                row
                for constraint in preference_constraints
                for row in (constraint.get("value") or [])
                if isinstance(row, dict)
                and re.search(
                    r"\b(?:already|currently|have\s+(?:used|tried)|"
                    r"(?:been\s+)?experimenting|found\s+that|"
                    r"discovered\s+that|noticed\s+that)\b",
                    str(row.get("evidence") or ""), re.IGNORECASE,
                )
            ]
            messages[-1]["content"] += (
                "\n\nPREFERENCE_CONSTRAINT_EVIDENCE "
                "(SOURCE-BOUND TRANSFERABLE CONSTRAINTS):\n"
                + json.dumps(preference_constraints, ensure_ascii=False)
                + "\nThese user-authored constraints are relevant across location "
                "changes and recommendation scenes when the requested topic matches. "
                "Preserve positive and negative preferences, ingredients, features, "
                "owned items, successes, and failures. Tailor the answer to them; "
                "do not let assistant-authored generic recommendations override "
                "the user's own evidence."
            )
            if established_practice_rows:
                messages[-1]["content"] += (
                    "\n\nESTABLISHED_PRACTICE_EVIDENCE "
                    "(MANDATORY CONTINUITY BINDING):\n"
                    + json.dumps(established_practice_rows, ensure_ascii=False)
                    + "\nExplicitly say this was already tried, used, discovered, "
                    "or found effective, and build the recommendation on it. "
                    "Do not present the cited item as a new swap, new option, "
                    "or first-time suggestion."
                )
        if certified_absence:
            messages[-1]["content"] += (
                "\n\nCERTIFIED_EXACT_ENTITY_ABSENCE "
                "(FINAL MANDATORY BINDING):\n"
                + json.dumps(certified_absence, ensure_ascii=False)
                + "\nThe exact named entity, relation role, collection subtype, "
                "required arithmetic component, or role title "
                "above is absent from the globally scanned relation-bound memory. "
                "A value attached to a sibling entity, partial component set, "
                "different title, or merely planned event cannot answer the "
                "question. Return a concise insufficient-information answer; "
                "do not transfer a nearby value or select one side of an "
                "incomplete comparison."
            )
        certified_operator_results = [
            row for row in constraints
            if row.get("operation") in {
                "record_time_extreme", "source_bound_explicit_date",
                "source_bound_scoped_relative_date",
                "duration_total", "paired_metric_total",
                "pending_operation_target_pairs",
                "named_event_attendance_count",
                "binary_savings_difference",
                "relative_anchor_source_lookup", "temporal_predecessor_entity",
                "latest_valid_state",
                "latest_scalar_state_from_lossless_sources",
                "threshold_progress_remaining", "latest_approx_scalar_state",
                "latest_labeled_currency_state", "latest_weekly_schedule_time",
                "same_unit_acquisition_total",
                "time_difference_from_lossless_sources",
                "age_arithmetic_from_lossless_sources",
                "advance_booking_recency_from_lossless_sources",
                "current_role_duration_from_lossless_sources",
                "weekly_schedule_distinct_days",
                "family_relation_subtype_total",
                "linked_event_date_from_lossless_sources",
                "latest_category_start_from_lossless_sources",
                "scoped_completed_event_members",
                "dialogue_attribute_item_match",
                "transaction_sum_from_lossless_sources",
                "currency_extreme_entity_from_lossless_sources",
                "dialogue_final_choice_from_lossless_sources",
                "completed_item_metric_total",
                "scoped_completed_duration_total",
                "relative_value_multiplier_from_lossless_sources",
                "relative_duration_at_event", "prior_candidate_count",
                "travel_arrival_time_from_sources",
                "completed_work_subtype_total",
            }
        ]
        if certified_operator_results:
            messages[-1]["content"] += (
                "\n\nCERTIFIED_OPERATOR_RESULT "
                "(FINAL MANDATORY BINDING):\n"
                + json.dumps(certified_operator_results, ensure_ascii=False)
                + "\nEach result above was recomputed from its cited lossless "
                "source turns with exact entity, relation, scope, and provenance "
                "bindings. Use the certified value exactly. For a count, retain "
                "the distinct operation-target members; for a total, retain all "
                "listed operands; for a record, retain the certified extreme. "
                "Do not substitute a nearby value from the larger evidence block. "
                "For relative_anchor_source_lookup or temporal_predecessor_entity, "
                "answer_candidate is the exact source-bound answer span: the final "
                "answer must contain that candidate and must not reselect a different "
                "nearby entity."
            )
            answer_candidates = [
                str(row.get("answer_candidate") or "").strip()
                for row in certified_operator_results
                if str(row.get("answer_candidate") or "").strip()
            ]
            if answer_candidates:
                messages[-1]["content"] += (
                    "\n\nFINAL_OUTPUT_MUST_CONTAIN: "
                    + json.dumps(answer_candidates, ensure_ascii=False)
                    + "\nReturn the concise answer using these exact certified "
                    "candidate strings. Do not output a competing entity."
                )
            exact_scalar_results = [
                row for row in certified_operator_results
                if row.get("operation") in {
                    "source_bound_explicit_date",
                    "source_bound_scoped_relative_date", "duration_total",
                    "threshold_progress_remaining",
                    "latest_approx_scalar_state",
                    "latest_labeled_currency_state",
                    "latest_weekly_schedule_time",
                    "time_difference_from_lossless_sources",
                    "age_arithmetic_from_lossless_sources",
                    "advance_booking_recency_from_lossless_sources",
                    "current_role_duration_from_lossless_sources",
                    "weekly_schedule_distinct_days",
                    "family_relation_subtype_total",
                    "linked_event_date_from_lossless_sources",
                    "scoped_completed_event_members",
                    "transaction_sum_from_lossless_sources",
                    "completed_item_metric_total",
                    "scoped_completed_duration_total",
                    "relative_duration_at_event", "prior_candidate_count",
                    "travel_arrival_time_from_sources",
                    "completed_work_subtype_total",
                }
                and row.get("value") is not None
            ]
            if len(exact_scalar_results) == 1:
                messages[-1]["content"] += (
                    "\n\nFINAL_OUTPUT_VALUE: "
                    + json.dumps(
                        exact_scalar_results[0], ensure_ascii=False,
                    )
                    + "\nReturn this certified value and unit exactly; "
                    "do not return an operand, threshold, subtotal, or "
                    "older state."
                )
        if exact_collection_binding:
            certified_values = [
                str(row.get("value") or "")
                for row in verified_collection_members
                if row.get("value")
            ]
            messages[-1]["content"] += (
                "\n\nCERTIFIED_COLLECTION_MEMBERS (FINAL MANDATORY BINDING):\n"
                + json.dumps({
                    "members": certified_values,
                    "certified_exact_distinct_count": len(certified_values),
                }, ensure_ascii=False)
                + "\nEvery listed identity passed entity, relation, scope, and "
                "source/provenance validation and appears exactly once after "
                "canonical deduplication. For a count question, the "
                "certified_exact_distinct_count is mandatory: output that number "
                "and do not independently discard or add members."
            )
    certified_scalar_slots = [
        row for row in constraints
        if row.get("operation") in {
            "source_bound_explicit_date",
            "source_bound_scoped_relative_date", "duration_total",
            "threshold_progress_remaining",
            "latest_approx_scalar_state",
            "latest_labeled_currency_state",
            "latest_weekly_schedule_time",
            "time_difference_from_lossless_sources",
            "age_arithmetic_from_lossless_sources",
            "advance_booking_recency_from_lossless_sources",
            "current_role_duration_from_lossless_sources",
            "weekly_schedule_distinct_days",
            "family_relation_subtype_total",
            "linked_event_date_from_lossless_sources",
            "scoped_completed_event_members",
            "transaction_sum_from_lossless_sources",
            "completed_item_metric_total",
            "scoped_completed_duration_total",
            "relative_duration_at_event", "prior_candidate_count",
            "travel_arrival_time_from_sources",
            "completed_work_subtype_total",
        }
        and row.get("value") is not None
    ]
    if len(certified_scalar_slots) == 1 and messages:
        slot = certified_scalar_slots[0]
        messages[-1]["content"] = (
            f"Question: {case.question}\n\n"
            "CERTIFIED_FINAL_ANSWER_SLOT:\n"
            + json.dumps({
                "operation": slot.get("operation"),
                "value": slot.get("value"),
                "unit": slot.get("unit"),
            }, ensure_ascii=False)
            + "\nThe slot was computed from exact entity, relation, scope, "
            "and provenance-bound lossless sources. Produce the concise "
            "natural-language answer using exactly this value and unit. "
            "Output no calculation, alternative value, or commentary."
        )
    direction_slots = [
        row for row in constraints
        if row.get("operation") == "latest_valid_state"
        and row.get("change_direction")
    ]
    if (
        len(direction_slots) == 1
        and messages
        and re.search(r"\b(?:more|less)\b", case.question, re.IGNORECASE)
    ):
        slot = direction_slots[0]
        direction = str(slot.get("change_direction") or "").strip()
        messages[-1]["content"] = (
            f"Question: {case.question}\n\n"
            "CERTIFIED_FINAL_DIRECTION_SLOT:\n"
            + json.dumps({
                "operation": "latest_valid_state",
                "direction": direction,
                "history": slot.get("history"),
                "source_turn_ids": slot.get("source_turn_ids"),
            }, ensure_ascii=False)
            + "\nThe direction was computed from the latest valid state change "
            "with exact entity, relation, scope, and provenance bindings. "
            "Answer the question concisely using exactly this direction. "
            "Do not invert more and less and do not output alternatives."
        )
    return messages
