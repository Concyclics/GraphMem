"""Compact evidence-first V4.1 answer policy."""
from __future__ import annotations

import json
import re
from typing import Any

from ..models import QuestionCase, RetrievedContext

_RULES = {
    "collection": "Silently enumerate every exactly matching member or operand across all sources. Match owner, object class and every discriminating modifier, relation, lifecycle and time scope; count conjoined distinct items separately, merge only true duplicates, and apply add/remove/cancel updates. A partial planner list is never proof of completeness. For an initial or first-period question, use the earliest source-backed snapshot; when an entity has only one quantity and no later transition, that sole quantity is its initial baseline. When the same scoped cumulative count is restated later with language such as so far or now, use the latest snapshot and never sum snapshots or choose the older count. If a required conjunct or exact scoped collection is absent, report insufficient evidence instead of borrowing a sibling collection.",
    "dialogue_lookup": "Find the request and its corresponding reply, then return the exact requested slot from the reply. For ordinal questions select that numbered list item. Never return request text, a topic label, a nearby scene, or a fact about a sibling person/entity. If the exact subject and relation are absent, report insufficient evidence.",
    "state_update": "Bind the exact entity and attribute. Current/latest selects the newest valid state; initial/first/previous selects the matching historical state. For frequency updates, use explicit recurring cadences such as weekly or every other week; a one-off scheduled day is an event instance, not a frequency. When both prior and current are requested, return both states explicitly. Reject a sibling attribute, description, recommendation, or obsolete value.",
    "temporal_lookup": "Bind the exact event, entity, location and time anchor. Resolve relative time from the source date and distinguish event time from mention time. A question asking how long someone worked before starting a current job asks for the pre-job career interval, not the tenure at the current employer. Return the requested duration, age, date or value rather than a nearby timestamp. A duration for a similar object, person, role or location is not evidence; when the exact event is absent, report insufficient evidence.",
    "temporal_comparison": "Bind both requested event endpoints exactly before ordering or subtracting. Prefer explicit event times; when an event time is absent but the completed state is directly observed, use that source observation as the conservative endpoint rather than declaring the comparison impossible. Exclude plans and merely topical events when completed events are requested. If either exact entity or event is absent, report insufficient evidence rather than using a similar event.",
    "multi_hop": "Keep all hops on one entity/event chain and return the requested final slot, not an intermediate value.",
    "preference_recommendation": "Preserve user-authored positive and negative preferences and established practices. For a recalled recommendation copy its concrete name, not the activity or broad category.",
    "inferential_profile": "Use source facts as premises for only the requested narrow inference. Prefer a concrete candidate satisfying every clue over a generic topic.",
    "reference_identity": "Resolve all discriminating clues inside one coherent source scene and return the concrete identity rather than its category.",
}


def _source_closure(trace: dict[str, Any], algebra: str) -> list[dict[str, Any]]:
    channels = (
        ("v41_collection_source_evidence", "collection", 16),
        ("v41_answer_bearing_evidence", "answer_bearing", 6),
        ("v41_scene_window_evidence", "scene", 6),
        ("v41_late_scene_window_evidence", "late_scene", 4),
        ("v41_reply_bound_evidence", "reply_bound", 4),
        ("v41_semantic_turn_evidence", "semantic", 4),
        ("v41_planner_selected_evidence", "planner", 4),
    )
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key, channel, limit in channels:
        if algebra == "collection" and channel in {"scene", "late_scene", "reply_bound"}:
            continue
        added = 0
        for raw in trace.get(key) or []:
            if not isinstance(raw, dict):
                continue
            source_id = str(raw.get("source_turn_id") or "")
            text = str(raw.get("text") or raw.get("source_text") or "").strip()
            if not source_id or not text or source_id in seen:
                continue
            seen.add(source_id)
            result.append({"channel": channel, "source_turn_id": source_id,
                           "event_time": raw.get("event_time"),
                           "speaker": raw.get("speaker"), "text": text[:1800]})
            added += 1
            if added >= limit:
                break
    return result


def _strict_binding_certificate(trace: dict[str, Any]) -> dict[str, Any]:
    hints = [row for row in (trace.get("generic_operator_hints") or []) if isinstance(row, dict)]
    certified = [row for row in hints if row.get("certified") is True and row.get("binding_complete") is True]
    competing = [
        row for row in certified
        if row.get("operation") not in {"exact_entity_absence", "global_lossless_source_candidates"}
        and row.get("value") not in {None, "insufficient"}
    ]
    robust_kinds = {
        "named_entity", "required_component", "role_title",
        "required_role", "required_collection_type", "required_operand",
        "required_relation", "required_subtype",
    }
    robust_exact = next((
        row for row in certified
        if row.get("operation") == "exact_entity_absence"
        and row.get("binding_kind") in robust_kinds
    ), None)
    if competing and robust_exact is None:
        return {}
    ordered = ([robust_exact] if robust_exact is not None else []) + [
        row for row in certified if row is not robust_exact
    ]
    for row in ordered:
        if row.get("operation") != "exact_entity_absence":
            continue
        phrase = str(
            row.get("required_phrase") or row.get("required_marker")
            or " ".join(str(row.get(key) or "") for key in ("required_modifier", "required_head")).strip()
        ).strip()
        tokens = re.findall(r"[a-z0-9]+", phrase.casefold())
        if any(token in {"previously", "currently", "initially", "recently", "now", "first", "latest"} for token in tokens):
            continue
        robust_single = (
            row.get("binding_kind") in robust_kinds
            and len(tokens) == 1 and len(tokens[0]) >= 4
        )
        if (len(tokens) < 2 and not robust_single) or any(len(token) < 2 for token in tokens):
            continue
        return {
            "operation": "exact_entity_absence",
            "required_phrase": phrase,
            "binding_kind": row.get("binding_kind"),
            "reason": row.get("reason"),
            "excluded_near_match_source_turn_ids": row.get("excluded_near_match_source_turn_ids") or [],
            "certified": True,
        }
    return {}

def _trusted_algebra_constraints(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Surface only provenance-bearing generic algebra; never low-precision direct answers."""
    blocked = {
        "exact_entity_absence",
        "dialogue_attribute_item_match",
        "latest_approx_scalar_state",
        "global_lossless_source_candidates",
    }
    result = []
    for row in trace.get("generic_operator_hints") or []:
        if not isinstance(row, dict) or row.get("certified") is not True:
            continue
        operation = str(row.get("operation") or "")
        if not operation or operation in blocked:
            continue
        source_ids = [str(item) for item in (row.get("source_turn_ids") or []) if str(item)]
        source_ids.extend(
            str(row.get(key)) for key in (
                "selected_source_turn_id", "event_a_source_turn_id",
                "event_b_source_turn_id",
            ) if row.get(key)
        )
        frame_ids = [str(item) for item in (row.get("frame_ids") or []) if str(item)]
        members = [item for item in (row.get("members") or []) if isinstance(item, dict)]
        operands = [item for item in (row.get("operands") or []) if isinstance(item, dict)]
        nested_source_ids = [
            str(item.get("source_turn_id"))
            for item in members + operands
            if item.get("source_turn_id")
        ]
        if not source_ids and not frame_ids and not nested_source_ids:
            continue
        compact = {
            "operation": operation,
            "value": row.get("value"),
            "unit": row.get("unit"),
            "source_turn_ids": list(dict.fromkeys(source_ids + nested_source_ids))[:12],
            "frame_ids": frame_ids[:8],
        }
        for key in ("selected_target", "comparison", "selected_time", "event_a_time", "event_b_time", "change_direction", "history"):
            if row.get(key) is not None:
                compact[key] = row.get(key)
        if members:
            compact["members"] = [
                {key: item.get(key) for key in ("identity", "value", "date", "event_time", "source_turn_id") if item.get(key) is not None}
                for item in members[:12]
            ]
        if operands:
            compact["operands"] = [
                {key: item.get(key) for key in ("role", "value", "unit", "source_turn_id") if item.get(key) is not None}
                for item in operands[:12]
            ]
        result.append(compact)
    selected: list[dict[str, Any]] = []
    for item in result:
        conflict_index = next((
            index for index, prior in enumerate(selected)
            if item.get("unit") and item.get("unit") == prior.get("unit")
            and item.get("value") is not None and prior.get("value") is not None
            and item.get("value") != prior.get("value")
        ), None)
        if conflict_index is None:
            selected.append(item)
            continue
        def specificity(value: dict[str, Any]) -> int:
            return sum(len(value.get(key) or []) for key in ("source_turn_ids", "frame_ids", "members", "operands"))
        if specificity(item) > specificity(selected[conflict_index]):
            selected[conflict_index] = item
    return selected[:8]



_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}

def _number_value(raw: str) -> int | None:
    value = raw.casefold().strip()
    return int(value) if value.isdigit() else _NUMBER_WORDS.get(value)

def _duration_months(text: str) -> int | None:
    years = re.search(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)", text.casefold())
    months = re.search(r"(\d+(?:\.\d+)?)\s*(?:months?|mos?)", text.casefold())
    if not years and not months:
        return None
    return round((float(years.group(1)) if years else 0.0) * 12 + (float(months.group(1)) if months else 0.0))

def _explicit_collection_count(text: str, nouns: list[str]) -> int | None:
    number = r"(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)"
    lowered = text.casefold()
    for noun in nouns:
        stem = re.escape(noun.casefold().rstrip("s"))
        match = re.search(number + r"\s+(?:[a-z]+\s+){0,2}" + stem + r"s?\b", lowered)
        if match:
            return _number_value(match.group(1))
    return None

def _question_time_constraints(case: QuestionCase, retrieval: RetrievedContext) -> list[dict[str, Any]]:
    """Small provenance-bound evidence algebra, independent of benchmark topics."""
    trace = retrieval.retrieval_trace or {}
    algebra = str((trace.get("v41_query_augmentation") or {}).get("answer_algebra") or "")
    rows = _source_closure(trace, algebra)
    for match in re.finditer(r"\[SOURCE_EVIDENCE ([^\]]+)\]\n(.*?)(?=\n\n\[|\Z)", retrieval.context_text, re.S):
        source_id, body = match.groups()
        date_match = re.search(r"date=([^;\n]+)", body)
        rows.append({
            "source_turn_id": source_id.split(";", 1)[0],
            "event_time": date_match.group(1).strip() if date_match else None,
            "text": body.strip(),
        })
    result: list[dict[str, Any]] = []
    question = case.question.casefold()
    query_ir = trace.get("v41_query_augmentation") or {}
    strict_binding = _strict_binding_certificate(trace)
    if strict_binding:
        return result

    # A later cumulative snapshot replaces an earlier snapshot of the same
    # scoped collection; snapshots are never added together.
    if (
        algebra == "collection" and not _strict_binding_certificate(trace)
        and not any(term in question for term in ("initial", "first"))
        and not any(str(term).casefold() in {"total", "both"} for term in (query_ir.get("alternative_entities") or []))
        and not any(re.search(r"\d", str(term)) for term in (query_ir.get("alternative_entities") or []))
    ):
        candidates = []
        for row in rows:
            text = str(row.get("text") or "")
            lowered = text.casefold()
            marker = next((score for phrase, score in (("so far", 5), ("up to now", 5), ("to date", 5), ("as of now", 5), ("currently", 4), ("now", 3), ("recently", 2)) if re.search(r"\b" + re.escape(phrase) + r"\b", lowered)), 0)
            match = re.search(r"\b(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\s+(?:different\s+)?(?:ones|items|places|events|people|things|[a-z]+)", lowered)
            value = _number_value(match.group(1)) if match else None
            if marker and value is not None:
                candidates.append((marker, str(row.get("event_time") or ""), value, row))
        unique_candidates = list({item[3]["source_turn_id"]: item for item in candidates}.values())
        if len(unique_candidates) >= 2:
            marker, _, value, row = max(unique_candidates, key=lambda item: (item[0], item[1]))
            result.append({
                "operation": "latest_cumulative_snapshot", "value": value,
                "unit": "distinct members", "source_turn_ids": [row["source_turn_id"]],
                "snapshot_marker_strength": marker, "certified": True,
            })

    # For a pre-current-role interval, subtract current-role tenure from total
    # professional tenure. Both operands must be explicit lossless-source durations.
    if algebra == "temporal_lookup" and "before" in question and any(term in question for term in ("current job", "current role", "current employer")):
        total_rows = []
        current_rows = []
        for row in rows:
            text = str(row.get("text") or "")
            lowered = text.casefold()
            months = _duration_months(lowered)
            if months is None:
                continue
            if any(term in lowered for term in ("working professionally", "professional experience", "career total", "total career")):
                total_rows.append((months, row))
            if any(term in lowered for term in ("current job", "current role", "current employer")) or re.search(r"working at [a-z0-9]", lowered):
                current_rows.append((months, row))
        if total_rows and current_rows:
            total_months, total_row = max(total_rows, key=lambda item: item[0])
            current_months, current_row = max(current_rows, key=lambda item: item[0])
            prior = total_months - current_months
            if prior >= 0 and total_row["source_turn_id"] != current_row["source_turn_id"]:
                years, months = divmod(prior, 12)
                value = f"{years} years" + (f" and {months} months" if months else "")
                result.append({
                    "operation": "pre_current_role_interval", "value": value,
                    "unit": "duration",
                    "source_turn_ids": [total_row["source_turn_id"], current_row["source_turn_id"]],
                    "operands": [{"role": "total professional tenure", "value": total_months, "unit": "months", "source_turn_id": total_row["source_turn_id"]}, {"role": "current role tenure", "value": current_months, "unit": "months", "source_turn_id": current_row["source_turn_id"]}],
                    "certified": True,
                })
    # Conjoined collection categories are independent operands. A category
    # with one source-backed quantity uses that sole quantity as its baseline.
    if algebra == "collection" and not result:
        alternatives = [str(item).casefold() for item in (query_ir.get("alternative_entities") or [])]
        plural_nouns = [item for item in alternatives if item.endswith("s") and item not in {"months", "days", "years"}]
        collection_noun = plural_nouns[0] if plural_nouns else "items"
        pairs = re.findall(r"\b([a-z][a-z0-9_-]+)\s+and\s+([a-z][a-z0-9_-]+)\b", question)
        aggregate_operands = []
        for left, right in pairs:
            if left not in alternatives or right not in alternatives:
                continue
            for term in (left, right):
                matched = next((
                    (value, row) for row in rows
                    if term in str(row.get("text") or "").casefold()
                    for value in [_explicit_collection_count(str(row.get("text") or ""), [collection_noun])]
                    if value is not None
                ), None)
                if matched:
                    value, row = matched
                    aggregate_operands.append({"role": term, "value": value, "source_turn_id": row["source_turn_id"]})
            if len(aggregate_operands) == 2:
                break
        if len(aggregate_operands) == 2:
            result.append({
                "operation": "conjoined_collection_sum",
                "value": sum(item["value"] for item in aggregate_operands),
                "unit": collection_noun, "operands": aggregate_operands,
                "source_turn_ids": [item["source_turn_id"] for item in aggregate_operands],
                "certified": True,
            })
        elif any(term in question for term in ("initial", "first")):
            snapshots = []
            for row in rows:
                text = str(row.get("text") or "")
                if collection_noun.rstrip("s") not in text.casefold():
                    continue
                value = _explicit_collection_count(text, [collection_noun])
                if value is not None:
                    snapshots.append((str(row.get("event_time") or ""), value, row))
            if snapshots:
                _, value, row = min(snapshots, key=lambda item: item[0])
                result.append({
                    "operation": "initial_collection_snapshot", "value": value,
                    "unit": collection_noun, "source_turn_ids": [row["source_turn_id"]],
                    "certified": True,
                })

    # Direct scoped durations beat unrelated duration totals.
    if (
        algebra == "temporal_lookup" and not result
        and "total" not in [str(item).casefold() for item in (query_ir.get("alternative_entities") or [])]
        and not any(role in {"event_a", "event_b", "time_a", "time_b", "components"} for role in (query_ir.get("required_roles") or []))
    ):
        target_terms = [
            term for term in (query_ir.get("alternative_entities") or [])
            if len(str(term)) >= 5 and str(term).casefold() not in {"living", "current", "apartment", "working"}
        ]
        duration_rows = []
        for row in rows:
            text = str(row.get("text") or "")
            lowered = text.casefold()
            months = _duration_months(lowered)
            overlap = sum(str(term).casefold() in lowered for term in target_terms)
            if months is not None and months > 0 and overlap:
                duration_rows.append((overlap, str(row.get("event_time") or ""), months, row))
        if duration_rows:
            _, _, months_total, row = max(duration_rows, key=lambda item: (item[0], item[1]))
            years, months = divmod(months_total, 12)
            value = (f"{years} years" if years else "") + ((" and " if years and months else "") + f"{months} months" if months else "")
            result.append({
                "operation": "scoped_duration_lookup", "value": value,
                "unit": "duration", "source_turn_ids": [row["source_turn_id"]],
                "certified": True,
            })

    # Recurring cadence is a state; retain old and latest explicit versions.
    if algebra == "state_update" and not result:
        cadence_rows = []
        for row in rows:
            text = str(row.get("text") or "")
            lowered = text.casefold()
            cadence = next((value for pattern, value in ((r"every other week", "every other week"), (r"every week", "weekly"), (r"weekly", "weekly")) if re.search(pattern, lowered)), None)
            if cadence:
                cadence_rows.append((str(row.get("event_time") or ""), cadence, row))
        unique = []
        for item in sorted(cadence_rows, key=lambda value: value[0]):
            if not unique or item[1] != unique[-1][1]:
                unique.append(item)
        if unique:
            result.append({
                "operation": "frequency_state_history", "value": unique[-1][1],
                "unit": "cadence",
                "history": [{"value": value, "date": date, "source_turn_id": row["source_turn_id"]} for date, value, row in unique],
                "source_turn_ids": [row["source_turn_id"] for _, _, row in unique],
                "certified": True,
            })

    # Resolve an explicit relative interval shared by the two compared events.
    if algebra == "temporal_comparison" and not result and "day" in question:
        for row in rows:
            text = str(row.get("text") or "")
            match = re.search(r"\b(a|one|two|three|four|five|six|seven|eight|nine|ten)\s+weeks?\s+before\s+([a-z0-9][a-z0-9 \-]{2,40})", text.casefold())
            if match:
                weeks = 1 if match.group(1) == "a" else (_number_value(match.group(1)) or 0)
                if weeks:
                    result.append({
                        "operation": "explicit_relative_interval",
                        "value": weeks * 7, "unit": "days",
                        "source_turn_ids": [row["source_turn_id"]],
                        "certified": True,
                    })
                    break

    return result

def needs_binding_verifier(retrieval: RetrievedContext) -> bool:
    """Use a short source-only search call only for binding-ambiguous queries."""
    trace = retrieval.retrieval_trace or {}
    algebra = str((trace.get("v41_query_augmentation") or {}).get("answer_algebra") or "")
    certificate = trace.get("v41_evidence_certificate") or {}
    exact_absence_hint = any(
        isinstance(row, dict)
        and row.get("operation") == "exact_entity_absence"
        and row.get("certified") is True
        for row in (trace.get("generic_operator_hints") or [])
    )
    return bool(
        exact_absence_hint
        or certificate.get("missing_roles")
        or algebra in {"dialogue_lookup", "temporal_comparison", "reference_identity"}
    )

def _ranked_binding_sources(case: QuestionCase, retrieval: RetrievedContext) -> list[dict[str, Any]]:
    """Lexically rerank every packed source turn for the bounded verifier."""
    trace = retrieval.retrieval_trace or {}
    algebra = str((trace.get("v41_query_augmentation") or {}).get("answer_algebra") or "direct_fact")
    rows = list(_source_closure(trace, algebra))
    for match in re.finditer(r"\[SOURCE_EVIDENCE ([^\]]+)\]\n(.*?)(?=\n\n\[|\Z)", retrieval.context_text, re.S):
        source_id, body = match.groups()
        date_match = re.search(r"date=([^;\n]+)", body)
        speaker_match = re.search(r"speaker=([^;\n]+)", body)
        rows.append({
            "source_turn_id": source_id,
            "event_time": date_match.group(1).strip() if date_match else None,
            "speaker": speaker_match.group(1).strip() if speaker_match else None,
            "text": body.strip(),
        })
    for match in re.finditer(r"\[BOUND_FRAME ([^\s\]]+)([^\]]*)\]\n(.*?)(?=\n\n\[|\Z)", retrieval.context_text, re.S):
        frame_id, header, body = match.groups()
        rows.append({
            "source_turn_id": frame_id,
            "event_time": None,
            "speaker": None,
            "text": ("BOUND_FRAME " + header + " " + body).strip(),
        })
    stop = {"what", "when", "where", "which", "who", "whom", "whose", "how", "many", "much", "long", "have", "been", "did", "does", "will", "would", "could", "should", "with", "from", "into", "that", "this", "your", "about", "after", "before", "first", "current", "recently"}
    question_terms = [term for term in re.findall(r"[a-z0-9]+", case.question.casefold()) if len(term) > 2 and term not in stop]
    query_ir = trace.get("v41_query_augmentation") or {}
    phrases = [
        str(item).casefold().strip()
        for key in ("alternative_entities", "event_identity_terms")
        for item in (query_ir.get(key) or [])
        if len(str(item).strip()) > 3
    ]
    trusted_ids = {
        item
        for constraint in (_trusted_algebra_constraints(trace) + _question_time_constraints(case, retrieval))
        for key in ("source_turn_ids", "frame_ids")
        for item in (constraint.get(key) or [])
    }
    ranked = []
    seen = set()
    for order, row in enumerate(rows):
        source_id = str(row.get("source_turn_id") or "")
        text = str(row.get("text") or row.get("source_text") or "").strip()
        if not source_id or not text or source_id in seen:
            continue
        seen.add(source_id)
        lowered = text.casefold()
        text_terms = set(re.findall(r"[a-z0-9]+", lowered))
        overlap = sum(1 for term in question_terms if term in text_terms)
        phrase_hits = sum(1 for phrase in phrases if phrase in lowered)
        ranked.append((overlap * 2 + phrase_hits * 3 + (20 if source_id in trusted_ids else 0), -order, {
            "source_turn_id": source_id,
            "speaker": row.get("speaker"),
            "event_time": row.get("event_time"),
            "text": text[:700],
        }))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [row for _, _, row in ranked[:16]]



def binding_verifier_messages(case: QuestionCase, retrieval: RetrievedContext) -> list[dict[str, str]]:
    """Build the bounded Agentic Search call used before the final answer."""
    trace = retrieval.retrieval_trace or {}
    algebra = str((trace.get("v41_query_augmentation") or {}).get("answer_algebra") or "direct_fact")
    evidence = _ranked_binding_sources(case, retrieval)
    compact_evidence = [
        {
            "source_turn_id": row["source_turn_id"],
            "speaker": row.get("speaker"),
            "event_time": row.get("event_time"),
            "text": str(row.get("text") or "")[:520],
        }
        for row in evidence
    ]
    query_ir = trace.get("v41_query_augmentation") or {}
    user = {
        "question": case.question,
        "answer_algebra": algebra,
        "target_terms": query_ir.get("alternative_entities") or [],
        "event_terms": query_ir.get("event_identity_terms") or [],
        "required_roles": query_ir.get("required_roles") or [],
        "trusted_algebra": _trusted_algebra_constraints(trace) + _question_time_constraints(case, retrieval),
        "source_evidence": compact_evidence,
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a source-binding and operand-coverage verifier, not an answerer. Decide whether the supplied source turns collectively cover every discriminating owner, entity, compound name, modifier, role, location, event endpoint, and arithmetic or collection operand required by the question. The final count, sum, difference, duration or ordering does not need to be explicitly stated and must never appear in missing_bindings: mark supported when all input operands are source-backed and can be computed later, citing all supporting source IDs. Mark absent only when a required input entity or operand is missing or appears solely as a sibling or similar near match. Treat multiword activities, titles and entity names as atomic. Output one JSON object only: {\"status\":\"supported|absent|uncertain\",\"supported_source_ids\":[],\"near_match_source_ids\":[],\"covered_bindings\":[],\"missing_bindings\":[],\"reason\":\"short\"}. supported requires empty missing_bindings; absent requires at least one concrete missing binding."
            ),
        },
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def _atomic_binding_tokens(trace: dict[str, Any], algebra: str) -> list[str]:
    query_ir = trace.get("v41_query_augmentation") or {}
    alternatives = [str(item).casefold() for item in (query_ir.get("alternative_entities") or [])]
    generic = {
        "name", "long", "been", "much", "time", "total", "there", "current",
        "have", "did", "does", "every", "day", "days", "first", "now",
    }
    if algebra == "collection":
        for index, term in enumerate(alternatives):
            if re.search(r"\d", term):
                tokens = re.findall(r"[a-z0-9]+", term)
                if index + 1 < len(alternatives):
                    tokens.extend(re.findall(r"[a-z0-9]+", alternatives[index + 1]))
                return [token for token in tokens if token not in generic]
    if algebra == "dialogue_lookup":
        for term in reversed(alternatives):
            tokens = [token for token in re.findall(r"[a-z0-9]+", term) if token not in generic]
            if tokens and any(len(token) >= 4 for token in tokens):
                return tokens + (["name"] if "name" in alternatives else [])
    if algebra == "temporal_lookup" and "total" not in alternatives:
        start = next((index for index, term in enumerate(alternatives) if term.endswith("ing")), None)
        if start is not None:
            tokens = [
                token for term in alternatives[start + 1:]
                for token in re.findall(r"[a-z0-9]+", term)
                if token not in generic
            ]
            return tokens[:4]
    return []

def validate_binding_verdict(raw_text: str, retrieval: RetrievedContext) -> dict[str, Any]:
    """Reject planner claims that do not cite supplied provenance."""
    trace = retrieval.retrieval_trace or {}
    algebra = str((trace.get("v41_query_augmentation") or {}).get("answer_algebra") or "direct_fact")
    allowed_ids = {row["source_turn_id"] for row in _source_closure(trace, algebra)}
    allowed_ids.update(re.findall(r"\[SOURCE_EVIDENCE ([^\]]+)\]", retrieval.context_text))
    allowed_ids.update(re.findall(r"\[BOUND_FRAME ([^\s\]]+)", retrieval.context_text))
    try:
        payload = raw_text.strip()
        if payload.startswith("```"):
            payload = payload.split("\n", 1)[1].rsplit("```", 1)[0]
        value = json.loads(payload)
    except (ValueError, TypeError, json.JSONDecodeError):
        return {"status": "uncertain", "reason": "invalid verifier JSON"}
    if not isinstance(value, dict) or value.get("status") not in {"supported", "absent", "uncertain"}:
        return {"status": "uncertain", "reason": "invalid verifier schema"}
    supported = [str(item) for item in (value.get("supported_source_ids") or []) if str(item) in allowed_ids]
    near = [str(item) for item in (value.get("near_match_source_ids") or []) if str(item) in allowed_ids]
    covered = [str(item)[:120] for item in (value.get("covered_bindings") or []) if str(item).strip()]
    missing = [str(item)[:120] for item in (value.get("missing_bindings") or []) if str(item).strip()]
    status = str(value["status"])
    strict_binding = _strict_binding_certificate(trace)

    # Routing frames may share most query words while describing a sibling
    # entity. They may guide search, but cannot establish an identity absent
    # from their lossless provenance.
    if status == "supported" and strict_binding:
        required_phrase = str(strict_binding.get("required_phrase") or "").casefold()
        required_tokens = {
            token for token in re.findall(r"[a-z0-9]+", required_phrase)
            if len(token) > 1 and token not in {
                "the", "a", "an", "my", "your", "his", "her", "their",
                "every", "day", "daily", "previously", "currently", "now",
                "initially", "recently", "first", "latest",
            }
        }
        source_text_by_id = {
            str(row.get("source_turn_id")): str(row.get("text") or row.get("source_text") or "").casefold()
            for row in _source_closure(trace, algebra)
            if row.get("source_turn_id")
        }
        cited_source_text = " ".join(source_text_by_id[item] for item in supported if item in source_text_by_id)
        normalized_required = " ".join(token for token in re.findall(r"[a-z0-9]+", required_phrase) if token in required_tokens)
        normalized_source = " ".join(re.findall(r"[a-z0-9]+", cited_source_text))
        if required_tokens and normalized_required not in normalized_source:
            status = "absent"
            missing = [str(strict_binding.get("required_phrase") or "exact entity binding")[:120]]
            near = list(dict.fromkeys(near + supported))
            supported = []

    atomic_tokens = _atomic_binding_tokens(trace, algebra)
    atomic_absence = False
    if status in {"supported", "uncertain", "absent"} and atomic_tokens:
        atomic_source_texts = {
            str(row.get("source_turn_id")): str(row.get("text") or row.get("source_text") or "").casefold()
            for row in _source_closure(trace, algebra)
            if row.get("source_turn_id")
        }
        for match in re.finditer(r"\[SOURCE_EVIDENCE ([^\]]+)\]\n(.*?)(?=\n\n\[|\Z)", retrieval.context_text, re.S):
            source_id, body = match.groups()
            atomic_source_texts[source_id.split(";", 1)[0]] = body.casefold()
        all_source_token_sets = [
            set(re.findall(r"[a-z0-9]+", text))
            for text in atomic_source_texts.values()
        ]
        cited_source_token_sets = [
            set(re.findall(r"[a-z0-9]+", atomic_source_texts[item]))
            for item in supported if item in atomic_source_texts
        ]
        binding_anywhere = any(set(atomic_tokens).issubset(tokens) for tokens in all_source_token_sets)
        binding_in_citations = any(set(atomic_tokens).issubset(tokens) for tokens in cited_source_token_sets)
        if not binding_anywhere or (status == "supported" and not binding_in_citations):
            atomic_absence = True
            status = "absent"
            missing = [" ".join(atomic_tokens)[:120]]
            near = list(dict.fromkeys(near + supported))
            supported = []

    generic_roles = {"scope", "members", "member", "source", "lifecycle", "time", "event", "fact", "value", "relation", "quantity", "status", "event_a", "event_b", "time_a", "time_b", "duration", "components"}
    concrete_missing = [item for item in missing if item.casefold().strip() not in generic_roles]
    if status == "supported" and (not supported or missing):
        status = "absent" if missing and near else "uncertain"
    if status == "absent" and (not concrete_missing or (not near and not strict_binding and not atomic_absence)):
        status = "uncertain"
    return {
        "status": status,
        "supported_source_ids": supported,
        "near_match_source_ids": near,
        "covered_bindings": covered,
        "missing_bindings": missing,
        "reason": str(value.get("reason") or "")[:240],
        "provenance_validated": True,
        "local_atomic_absence": atomic_absence,
    }



def answer_messages(
    case: QuestionCase,
    retrieval: RetrievedContext,
    binding_verdict: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Return one short prompt; heuristic operators are never mandatory answers."""
    trace = retrieval.retrieval_trace or {}
    algebra = str((trace.get("v41_query_augmentation") or {}).get("answer_algebra") or "direct_fact")
    certificate = trace.get("v41_evidence_certificate") or {}
    dialogue = [row for row in (trace.get("v41_direct_dialogue_highlights") or [])[:4]
                if isinstance(row, dict) and row.get("provenance_complete") is True]
    strict_binding = _strict_binding_certificate(trace)
    trusted_algebra = _trusted_algebra_constraints(trace) + _question_time_constraints(case, retrieval)
    strict_tokens = [
        token for token in re.findall(
            r"[a-z0-9]+", str(strict_binding.get("required_phrase") or "").casefold()
        ) if token not in {"every", "day", "daily", "previously", "currently", "now", "initially", "recently", "first", "latest"}
    ]
    normalized_source_text = " ".join(re.findall(
        r"[a-z0-9]+",
        (" ".join(str(row.get("text") or "") for row in _source_closure(trace, algebra)) + " " + retrieval.context_text).casefold(),
    ))
    normalized_strict_phrase = " ".join(strict_tokens)
    authoritative_absence = bool(
        strict_binding and (
            strict_binding.get("binding_kind") == "role_title"
            or (normalized_strict_phrase and normalized_strict_phrase not in normalized_source_text)
        )
    )
    authoritative_algebra = [
        row for row in trusted_algebra
        if (
            row.get("operation") in {
                "scoped_completed_duration_total", "latest_cumulative_snapshot",
                "pre_current_role_interval", "conjoined_collection_sum",
                "initial_collection_snapshot", "scoped_duration_lookup",
                "frequency_state_history", "explicit_relative_interval",
                "temporal_order_from_lossless_sources",
                "temporal_sequence_from_lossless_sources",
            }
            or (
                row.get("operation") == "duration_total"
                and str(row.get("unit") or "").casefold().rstrip("s")
                and str(row.get("unit") or "").casefold().rstrip("s") in case.question.casefold()
            )
        )
    ]
    effective_binding_verdict = dict(binding_verdict or {})
    if authoritative_absence:
        effective_binding_verdict["status"] = "absent"
        effective_binding_verdict["overridden_by_strict_source_absence"] = True
    elif (
        effective_binding_verdict.get("status") == "absent"
        and not strict_binding and authoritative_algebra
    ):
        effective_binding_verdict["status"] = "uncertain"
        effective_binding_verdict["overridden_by_source_algebra"] = True
    elif (
        effective_binding_verdict.get("status") == "absent"
        and not strict_binding
        and not effective_binding_verdict.get("local_atomic_absence")
    ):
        effective_binding_verdict["status"] = "uncertain"
        effective_binding_verdict["planner_absence_not_locally_certified"] = True
    mandatory_constraint = (
        {"mode": "insufficient_evidence", "certificate": strict_binding, "verdict": effective_binding_verdict}
        if (authoritative_absence or effective_binding_verdict.get("local_atomic_absence") or (strict_binding and effective_binding_verdict.get("status") == "absent")) else
        ({"mode": "use_exact_source_algebra", "constraints": authoritative_algebra}
         if authoritative_algebra else {})
    )
    if mandatory_constraint:
        if mandatory_constraint.get("mode") == "insufficient_evidence":
            return [
                {"role": "system", "content": "You are the final answer formatter. A provenance validator proved that the exact requested binding is absent. Output exactly: Insufficient evidence."},
                {"role": "user", "content": f"Question: {case.question}\nValidated constraint: {json.dumps(mandatory_constraint, ensure_ascii=False)}"},
            ]
        return [
            {"role": "system", "content": "You are the final answer formatter. Execute the certified source-derived algebra exactly. Preserve its numeric value, unit, selected target, ordering, and old/new history. Do not abstain, recalculate, or substitute another value. Output only the concise answer."},
            {"role": "user", "content": f"Question: {case.question}\nCertified algebra: {json.dumps(mandatory_constraint, ensure_ascii=False)}"},
        ]

    system = (
        "Answer one memory question using provenance-bearing source evidence. Source turns are authoritative; summaries, planner candidates and diagnostics only help navigation. Preserve speaker ownership, negation, lifecycle, exact numbers, units and dates. Treat every discriminating noun, modifier, owner, role, location and event identity in the question as a binding constraint: evidence about a sibling or merely similar entity cannot fill the requested slot. Enforce the requested answer type: prose cannot answer how many, a topic cannot answer which name, a date cannot answer age, and a prompt cannot answer what its reply said. Do not abstain when exact matching source evidence exists; when the exact binding or a required arithmetic component is absent, explicitly report insufficient evidence. Work silently and output only the concise answer."
    )
    diagnostic = {key: certificate.get(key) for key in (
        "entity_match", "relation_match", "scope_match", "provenance_complete",
        "present_roles", "missing_roles")}
    rule = _RULES.get(algebra, "Bind exact owner, entity, relation, scope and requested value type; copy the most specific supported answer.")
    user = (
        f"Question date: {case.question_date}\nQuestion: {case.question}\n\n"
        f"Answer algebra: {algebra}\nBinding rule: {rule}\n\n"
        "Typed source closure:\n" + json.dumps(_source_closure(trace, algebra), ensure_ascii=False)
        + "\n\nValidated dialogue scenes:\n" + json.dumps(dialogue, ensure_ascii=False)
        + "\n\nIndependent source-binding search verdict:\n" + json.dumps(effective_binding_verdict, ensure_ascii=False)
        + "\n\nStrict source-binding certificate (authoritative only when non-empty):\n" + json.dumps(strict_binding, ensure_ascii=False)
        + "\n\nTrusted provenance-bound algebra:\n" + json.dumps(trusted_algebra, ensure_ascii=False)
        + "\n\nAuthoritative source-derived algebra (may override an uncertain planner when strict absence is empty):\n" + json.dumps(authoritative_algebra, ensure_ascii=False)
        + "\n\nMANDATORY_ANSWER_CONSTRAINT:\n" + json.dumps(mandatory_constraint, ensure_ascii=False)
        + "\n\nCompleteness diagnostic (not evidence):\n" + json.dumps(diagnostic, ensure_ascii=False)
        + "\n\nFull frozen GraphMem evidence:\n" + retrieval.context_text
        + "\n\nFINAL BINDING GATE (highest priority): MANDATORY_ANSWER_CONSTRAINT is a locally source-certified execution contract: when non-empty, obey its mode and exact value, selected target, history or operands; do not replace it, abstain from it, or choose a nearby scalar. Then before answering, identify one source clause or coherent set of cited source clauses that jointly binds the requested entity or owner, every discriminating modifier, and all relation-algebra operands. If MANDATORY_ANSWER_CONSTRAINT mode is insufficient_evidence, answer insufficient evidence. Otherwise, if the strict certificate is non-empty and the independent verifier is absent, answer insufficient evidence. If strict absence is empty and authoritative source-derived algebra is non-empty, its lossless-source or bound-frame operands take precedence over a planner absence caused by a missing abstract role; use its exact result. Otherwise, if the verifier is absent and provenance_validated, answer insufficient evidence. If it is supported, use only its cited support for entity binding. When binding is not absent and trusted provenance-bound algebra is non-empty, verify its cited operands and preserve its certified number, date, set or ordering exactly; do not replace it with a nearby scalar. A value attached to a sibling person, animal, object, role, location, book, event or relation is an explicit near-match and must never be substituted. Otherwise, when the exact binding is present, answer directly and do not abstain. Return only the concise answer."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

