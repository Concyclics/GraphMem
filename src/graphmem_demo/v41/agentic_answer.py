"""Token-bounded Agentic Search answer policy for frozen GraphMem V4.1."""
from __future__ import annotations

import json
import re
from typing import Any

from ..models import QuestionCase, RetrievedContext

_ALGEBRA = {
    "collection": "enumerate exact scoped members/operands; deduplicate identities; apply add/remove/cancel; compute the requested count/list/aggregate",
    "dialogue_lookup": "bind the request to its corresponding reply and extract the exact requested slot from the reply",
    "state_update": "bind one owner/entity/attribute/context and select the requested historical or latest valid lifecycle state",
    "temporal_lookup": "bind the exact event and resolve its date, age, duration, or relative time from source dates",
    "temporal_comparison": "bind both exact event endpoints, normalize times, then order or subtract in the requested direction",
    "preference_recommendation": "preserve owner attribution, positive/negative preferences, established practices, and requested recommendation scope",
    "multi_hop": "keep all hops on one coherent entity/event chain and return the final requested slot",
    "inferential_profile": "make one narrow ordinary inference from direct source premises without inventing sensitive attributes",
    "reference_identity": "resolve all discriminating clues inside one coherent source scene and return the concrete identity",
}

_STOP = {"what", "when", "where", "which", "who", "whom", "whose", "how", "many", "much", "long", "did", "does", "have", "has", "been", "were", "was", "will", "would", "could", "should", "with", "from", "into", "that", "this", "then", "than", "about", "after", "before"}


def _terms(text: str) -> set[str]:
    return {x for x in re.findall(r"[a-z0-9]+", text.casefold()) if len(x) > 2 and x not in _STOP}


def _candidate_sources(case: QuestionCase, retrieval: RetrievedContext, limit: int = 14) -> list[dict[str, Any]]:
    trace = retrieval.retrieval_trace or {}
    rows: list[dict[str, Any]] = []
    for key in (
        "v41_answer_bearing_evidence", "v41_reply_bound_evidence",
        "v41_scene_window_evidence", "v41_late_scene_window_evidence",
        "v41_collection_source_evidence", "v41_semantic_turn_evidence",
        "v41_planner_selected_evidence", "v41_global_lossless_focused_evidence",
    ):
        for raw in trace.get(key) or []:
            if not isinstance(raw, dict):
                continue
            sid = str(raw.get("source_turn_id") or "")
            text = str(raw.get("text") or raw.get("source_text") or "").strip()
            if sid and text:
                rows.append({"source_turn_id": sid, "text": text, "event_time": raw.get("event_time"), "speaker": raw.get("speaker")})
    for match in re.finditer(r"\[SOURCE_EVIDENCE ([^\]]+)\]\n(.*?)(?=\n\n\[|\Z)", retrieval.context_text, re.S):
        sid, body = match.groups()
        sid = sid.split(";", 1)[0]
        date = re.search(r"date=([^;\n]+)", body)
        speaker = re.search(r"speaker=([^;\n]+)", body)
        rows.append({"source_turn_id": sid, "text": body.strip(), "event_time": date.group(1).strip() if date else None, "speaker": speaker.group(1).strip() if speaker else None})
    qterms = _terms(case.question)
    ir = trace.get("v41_query_augmentation") or {}
    target_phrases = [str(x).casefold() for key in ("alternative_entities", "event_identity_terms", "target_entities") for x in (ir.get(key) or []) if len(str(x)) > 2]
    trusted_ids = {str(x) for h in (trace.get("generic_operator_hints") or []) if isinstance(h, dict) and h.get("certified") is True for key in ("source_turn_ids",) for x in (h.get(key) or [])}
    ranked = []
    seen = set()
    for order, row in enumerate(rows):
        sid = row["source_turn_id"]
        if sid in seen:
            continue
        seen.add(sid)
        text = row["text"]
        tterms = _terms(text)
        overlap = len(qterms & tterms)
        phrases = sum(p in text.casefold() for p in target_phrases)
        score = overlap * 3 + phrases * 5 + (12 if sid in trusted_ids else 0)
        ranked.append((score, -order, {**row, "text": text[:900]}))
    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [row for _, _, row in ranked[:limit]]


def _operator_ledger(trace: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for hint in trace.get("generic_operator_hints") or []:
        if not isinstance(hint, dict) or hint.get("certified") is not True:
            continue
        source_ids = [str(x) for x in (hint.get("source_turn_ids") or []) if x]
        source_ids += [str(hint[k]) for k in ("selected_source_turn_id", "event_a_source_turn_id", "event_b_source_turn_id") if hint.get(k)]
        if not source_ids and hint.get("operation") != "exact_entity_absence":
            continue
        row = {k: hint[k] for k in ("operation", "value", "unit", "members", "operands", "answer_candidate", "selected_target", "comparison", "selected_time", "event_a_time", "event_b_time", "change_direction", "history", "required_phrase", "binding_kind") if k in hint}
        row["source_turn_ids"] = list(dict.fromkeys(source_ids))[:12]
        result.append(row)
        if len(result) >= 8:
            break
    return result


def planner_messages(case: QuestionCase, retrieval: RetrievedContext) -> list[dict[str, str]]:
    trace = retrieval.retrieval_trace or {}
    ir = trace.get("v41_query_augmentation") or {}
    algebra = str(ir.get("answer_algebra") or "direct_fact")
    payload = {
        "question_date": case.question_date,
        "question": case.question,
        "answer_algebra": algebra,
        "required_operation": _ALGEBRA.get(algebra, "bind exact owner/entity/relation/scope and requested value type"),
        "query_ir": {k: ir.get(k) for k in ("target_entities", "target_relation", "target_owner", "requested_value_type", "temporal_constraints", "required_roles", "alternative_entities", "event_identity_terms", "scope_boundary") if ir.get(k) not in (None, "", [], {})},
        "source_backed_operator_ledger": _operator_ledger(trace),
        "ranked_lossless_sources": _candidate_sources(case, retrieval),
    }
    return [
        {"role": "system", "content": "You are the bounded search stage of a memory system. Locate the exact source evidence and compute a candidate answer, but do not write the user-facing response. Source text is authoritative. Bind every discriminating owner, entity, modifier, relation, lifecycle, scope, and temporal endpoint. A completeness diagnostic is not proof of absence. For lists and arithmetic, enumerate operands first. For dialogue, pair request with reply. Return one JSON object only: {\"status\":\"answerable|insufficient|uncertain\",\"candidate_answer\":\"\",\"support_source_ids\":[],\"near_match_source_ids\":[],\"bound_roles\":{},\"operation\":\"\",\"confidence\":0.0}. Use insufficient only when a required exact input is absent after checking all supplied sources."},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def validate_plan(raw: str, retrieval: RetrievedContext, case: QuestionCase) -> dict[str, Any]:
    sources = _candidate_sources(case, retrieval)
    allowed = {row["source_turn_id"] for row in sources}
    try:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        value = json.loads(text)
    except Exception:
        return {"status": "uncertain", "candidate_answer": "", "support_source_ids": [], "valid": False}
    if not isinstance(value, dict) or value.get("status") not in {"answerable", "insufficient", "uncertain"}:
        return {"status": "uncertain", "candidate_answer": "", "support_source_ids": [], "valid": False}
    support = [str(x) for x in (value.get("support_source_ids") or []) if str(x) in allowed]
    status = str(value.get("status"))
    candidate = str(value.get("candidate_answer") or "").strip()[:600]
    if status == "answerable" and (not support or not candidate):
        status = "uncertain"
    return {"status": status, "candidate_answer": candidate, "support_source_ids": support[:8], "near_match_source_ids": [str(x) for x in (value.get("near_match_source_ids") or []) if str(x) in allowed][:8], "bound_roles": value.get("bound_roles") if isinstance(value.get("bound_roles"), dict) else {}, "operation": str(value.get("operation") or "")[:200], "confidence": value.get("confidence"), "valid": True}


def answer_messages(case: QuestionCase, retrieval: RetrievedContext, plan: dict[str, Any]) -> list[dict[str, str]]:
    trace = retrieval.retrieval_trace or {}
    ir = trace.get("v41_query_augmentation") or {}
    algebra = str(ir.get("answer_algebra") or "direct_fact")
    all_sources = _candidate_sources(case, retrieval)
    by_id = {row["source_turn_id"]: row for row in all_sources}
    selected = [by_id[sid] for sid in plan.get("support_source_ids") or [] if sid in by_id]
    used = {row["source_turn_id"] for row in selected}
    selected += [row for row in all_sources if row["source_turn_id"] not in used][:max(0, 8 - len(selected))]
    payload = {
        "question_date": case.question_date,
        "question": case.question,
        "answer_algebra": algebra,
        "required_operation": _ALGEBRA.get(algebra, "bind exact owner/entity/relation/scope and requested value type"),
        "validated_search_plan": plan,
        "source_backed_operator_ledger_advisory": _operator_ledger(trace),
        "selected_lossless_sources": selected,
    }
    return [
        {"role": "system", "content": "Answer the memory question from the selected lossless sources. The search candidate is advisory: verify it against cited text, correct it when necessary, and never copy an unsupported value. Do not abstain merely because the search status is uncertain or a diagnostic role is missing; answer whenever the exact evidence or all computable operands are present. Preserve owner, relation direction, polarity, lifecycle, exact names, numbers, units, and dates. For a true exact-entity absence or missing required operand, say insufficient evidence briefly. Perform reasoning silently and output only the concise final answer."},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
