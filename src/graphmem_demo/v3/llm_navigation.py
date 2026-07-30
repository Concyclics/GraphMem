from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Iterable

from graphmem_demo.clients import rough_token_count
from graphmem_demo.v3.structured_navigation import QueryIR, structured_navigation_summary


_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_PROPER_NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:['’][A-Za-z]+)?\b")
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "before", "did", "do",
    "does", "for", "from", "had", "has", "have", "how", "i", "in", "is",
    "it", "me", "my", "of", "on", "or", "the", "to", "was", "were",
    "what", "when", "where", "which", "who", "with",
}
_QUESTION_WORDS = {
    "Answer", "Are", "Did", "Does", "How", "Is", "No", "The", "Was",
    "Were", "What", "When", "Where", "Which", "Who", "Why", "Yes",
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
    "Sunday",
}


@dataclass(frozen=True)
class NavigationPlan:
    selected_ids: tuple[str, ...]
    operation: str
    missing_slots: tuple[str, ...]
    needed_relations: tuple[str, ...]
    resolved_slots: tuple[tuple[str, str], ...] = ()
    candidate_answer: str = ""
    confidence: float = 0.0
    parse_error: bool = False


def _tokens(value: str) -> set[str]:
    return {
        token.casefold()
        for token in _WORD_RE.findall(value)
        if len(token) > 1 and token.casefold() not in _STOP
    }


def _binding_tokens(value: str) -> set[str]:
    """Normalize morphology while preserving relation prefixes such as re-."""
    values: set[str] = set()
    for raw in _WORD_RE.findall(value):
        token = re.sub(r"[-_'’]", "", raw.casefold())
        if len(token) <= 1 or token in _STOP:
            continue
        values.add(token)
        if token.endswith("ies") and len(token) > 4:
            values.add(token[:-3] + "y")
        elif token.endswith("ing") and len(token) > 5:
            values.add(token[:-3])
        elif token.endswith("ed") and len(token) > 4:
            values.add(token[:-2])
        elif token.endswith("s") and len(token) > 3:
            values.add(token[:-1])
    return values


_AGGREGATE_OPERATION_TOKENS = frozenset(
    {"aggregate", "collect", "count", "duration", "list", "sum", "total", "unique"}
)


def is_aggregate_navigation_operation(operation: str) -> bool:
    """Identify set-wide operations from the navigator's free-form verb phrase."""

    return bool(_tokens(operation.replace("_", " ")) & _AGGREGATE_OPERATION_TOKENS)


def _node_session_id(node_id: str) -> str:
    suffix = node_id.split(":", 1)[1] if ":" in node_id else node_id
    for marker in (":turn:", ":claim:", ":event:"):
        if marker in suffix:
            return suffix.split(marker, 1)[0]
    return ""


def session_diverse_recovery_seeds(
    evidence_ledger: Iterable[dict[str, Any]],
    plan: NavigationPlan,
    *,
    max_extra: int = 8,
) -> tuple[str, ...]:
    """Add one bounded seed per routed session for set-wide graph closure.

    The LLM still chooses the primary evidence. For an aggregate operation, a
    coarse frontier may span several sessions while the plan selects only one
    or two nodes. Adding one already-retrieved node from distinct sessions lets
    typed graph edges continue the search without a second LLM call.
    """

    selected = list(dict.fromkeys(str(value) for value in plan.selected_ids if value))
    if max_extra <= 0 or not is_aggregate_navigation_operation(plan.operation):
        return tuple(selected)
    selected_set = set(selected)
    seen_sessions = {
        session for session in map(_node_session_id, selected) if session
    }
    added = 0
    for row in evidence_ledger:
        node_id = str(row.get("node_id") or "")
        if not node_id or node_id in selected_set:
            continue
        node_type = str(row.get("node_type") or "")
        if node_type not in {"turn", "claim", "event"}:
            continue
        session_id = str(row.get("session_id") or "") or _node_session_id(node_id)
        if not session_id or session_id in seen_sessions:
            continue
        selected.append(node_id)
        selected_set.add(node_id)
        seen_sessions.add(session_id)
        added += 1
        if added >= max_extra:
            break
    return tuple(selected)


def ir_guided_recovery_seeds(
    question: str,
    evidence_ledger: Iterable[dict[str, Any]],
    plan: NavigationPlan,
    query_ir: QueryIR,
    *,
    max_extra: int = 12,
) -> tuple[str, ...]:
    """Choose bounded recovery seeds by query algebra and ranked evidence."""

    selected = list(dict.fromkeys(str(value) for value in plan.selected_ids if value))
    selected_set = set(selected)
    if max_extra <= 0:
        return tuple(selected)

    ranked = _candidate_rows(
        question, evidence_ledger, max_candidates=max(48, max_extra * 6)
    )
    seen_sessions = {
        session for session in map(_node_session_id, selected) if session
    }
    added = 0
    allowed_types = {"turn", "claim", "event", "event_frame", "operand"}

    for row in ranked:
        node_id = str(row.get("node_id") or "")
        if not node_id or node_id in selected_set:
            continue
        if str(row.get("node_type") or "") not in allowed_types:
            continue
        row_tokens = _binding_tokens(str(row.get("text") or ""))
        binding_terms = _binding_tokens(
            " ".join([*query_ir.content_terms, *query_ir.subjects])
        )
        binding_hits = len(binding_terms & row_tokens)
        subject_bound = bool(set(query_ir.subjects) & row_tokens)
        if binding_hits < (1 if subject_bound else 2):
            continue
        session_id = str(row.get("session_id") or "") or _node_session_id(node_id)
        if not session_id:
            continue
        if query_ir.set_wide and session_id in seen_sessions:
            continue
        if not query_ir.set_wide and added >= min(4, max_extra):
            break
        selected.append(node_id)
        selected_set.add(node_id)
        seen_sessions.add(session_id)
        added += 1
        if added >= max_extra:
            break
    return tuple(selected)


def _years(value: str) -> set[str]:
    return set(_YEAR_RE.findall(value))


def _question_entities(question: str) -> set[str]:
    return {
        value.casefold()
        for value in _PROPER_NAME_RE.findall(question)
        if value not in _QUESTION_WORDS
    }


def _sentences(value: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|\n+", value)
        if part.strip()
    ]


def _asks_for_scalar(question: str) -> bool:
    normalized = " ".join(question.casefold().split())
    return bool(
        re.search(
            r"\b(how (?:much|many|long|old)|what (?:amount|price|cost|time|date|"
            r"number|percentage|percent|budget)|when)\b",
            normalized,
        )
    )


def focused_snippet(text: str, question: str, *, max_chars: int) -> str:
    """Keep query-bearing spans from a lossless node without topic rules."""
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    query_terms = _binding_tokens(question)
    sentences = _sentences(text)
    if not sentences:
        return compact[:max_chars]
    scalar_intent = _asks_for_scalar(question)

    def rank(index: int) -> tuple[float, float, int, int]:
        sentence = sentences[index]
        terms = _binding_tokens(sentence)
        overlap = len(query_terms & terms)
        density = overlap / max(1, len(terms))
        has_scalar = bool(
            re.search(r"(?:[$€£¥]\s*)?\d[\d,]*(?:\.\d+)?%?", sentence)
        )
        structured = bool(
            len(sentence) <= 240
            and re.search(r"(?:^|\s)[*#-]*\s*[^:|]{2,80}\s*[:|]", sentence)
        )
        # Scalar-bearing key/value rows are often the only answer-bearing part
        # of a long table or plan. Rank them by query form and lexical binding,
        # without relying on a benchmark topic or named entity.
        relevance = (
            overlap * 10.0
            + density * 4.0
            + (4.0 if scalar_intent and has_scalar else 0.0)
            + (2.0 if structured and overlap else 0.0)
            # In long plans, tables, inventories, and reports the answer is
            # often a compact ``label: scalar`` row. Give that structural
            # pattern priority over repeated prose about the same subject.
            # This depends only on query form and lexical binding.
            + (
                60.0
                if scalar_intent and has_scalar and structured and overlap >= 2
                else 0.0
            )
        )
        return relevance, density, -len(sentence), -index

    ranked_indices = sorted(range(len(sentences)), key=rank, reverse=True)
    chosen: set[int] = set()

    def rendered(indices: set[int]) -> str:
        return " ".join(sentences[index] for index in sorted(indices))

    # Reserve space for the strongest answer-bearing spans before adding
    # adjacent context, so a long introduction cannot evict a short table row.
    for index in ranked_indices[:8]:
        candidate = {*chosen, index}
        if len(rendered(candidate)) <= max_chars:
            chosen = candidate
    for index in ranked_indices[:6]:
        for neighbor in (index - 1, index + 1):
            if 0 <= neighbor < len(sentences):
                candidate = {*chosen, neighbor}
                if len(rendered(candidate)) <= max_chars:
                    chosen = candidate
    if not chosen:
        best = sentences[ranked_indices[0]]
        return best[:max_chars]
    return rendered(chosen)


def _candidate_rows(
    question: str,
    evidence_ledger: Iterable[dict[str, Any]],
    *,
    max_candidates: int,
    diversify_sessions: bool = True,
    include_adjacent_context: bool = False,
) -> list[dict[str, Any]]:
    rows = [row for row in evidence_ledger if str(row.get("node_id") or "")]
    query_terms = _binding_tokens(question)
    query_years = _years(question)
    query_entities = _question_entities(question)
    source_priority = {
        "protected_catalog": 5,
        "relation_operator_provenance": 5,
        "catalog_operator_provenance": 5,
        "scope_lossless_event": 5,
        "focused_provenance_expansion": 4,
        "routed_lossless_session": 5,
        "routed_lossless_dense_session": 6,
        "coarse_fine_projection": 3,
        "relation_focus": 3,
        "protected_graph_rescue": 2,
        "protected_direct": 1,
        "multiview_exact": 4,
        "multiview_bm25": 3,
        "multiview_dense": 2,
        "multiview_dense_view_1": 2,
        "multiview_dense_view_2": 2,
        "multiview_rrf": 2,
    }
    scored = []
    for position, row in enumerate(rows):
        text = str(row.get("text") or "")
        dated_text = f"{text} {row.get('session_date') or row.get('observed_at') or ''}"
        row_years = _years(dated_text)
        if query_years and row_years and query_years.isdisjoint(row_years):
            continue
        lexical = len(query_terms & _tokens(text))
        folded = dated_text.casefold()
        entity_hits = sum(entity in folded for entity in query_entities)
        year_hits = len(query_years & row_years)
        score = float(row.get("score") or 0.0)
        semantic = max(
            0.0,
            min(1.0, float(row.get("semantic_similarity") or 0.0)),
        )
        priority = source_priority.get(str(row.get("selection_source") or ""), 0)
        binding = (
            (entity_hits * 3)
            + (year_hits * 3)
            + lexical
            + (semantic * 6)
        )
        scored.append((binding, lexical, entity_hits, priority, score, -position, row))

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add(row: dict[str, Any]) -> None:
        node_id = str(row.get("node_id") or "")
        if node_id and node_id not in selected_ids and len(selected) < max_candidates:
            selected.append(row)
            selected_ids.add(node_id)

    def session_key(row: dict[str, Any]) -> str:
        explicit = str(row.get("session_id") or "")
        if explicit:
            return explicit
        return _node_session_id(str(row.get("node_id") or ""))

    ranked = sorted(scored, key=lambda item: item[:-1], reverse=True)

    # Reserve space for independent session routes. Previously ranked[:8]
    # filled a small fallback quota before diversity could run at all.
    relevance_slots = min(8, max(1, (max_candidates + 1) // 2))
    for item in ranked[:relevance_slots]:
        add(item[-1])
    # Preserve short dialogue context around strong turn seeds. Answer-bearing
    # replies often share few query words with the initiating turn.
    adjacency_cap = max(
        len(selected),
        max_candidates - min(8, max_candidates / 2),
    )
    by_turn: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        match = re.search(
            r":([^:]+):turn:(\d+)$", str(row.get("node_id") or "")
        )
        if match:
            by_turn[(match.group(1), int(match.group(2)))] = row
    for seed in list(selected[:10]):
        match = re.search(
            r":([^:]+):turn:(\d+)$", str(seed.get("node_id") or "")
        )
        if not match:
            continue
        session, raw_index = match.groups()
        index = int(raw_index)
        for distance in (1, 2):
            for neighbor in (index - distance, index + distance):
                row = by_turn.get((session, neighbor))
                if (
                    include_adjacent_context
                    and row is not None
                    and len(selected) < adjacency_cap
                ):
                    add(row)

    if diversify_sessions:
        seen_sessions = {session_key(row) for row in selected if session_key(row)}
        for item in ranked:
            row = item[-1]
            session = session_key(row)
            if session and session not in seen_sessions:
                add(row)
                seen_sessions.add(session)
            if len(seen_sessions) >= min(16, max_candidates) or len(selected) >= max_candidates:
                break


    # Preserve representation diversity before filling by relevance. This is
    # the coarse-to-fine navigation frontier, not a second flat top-k list.
    for kind in ("turn", "claim", "event", "event_frame", "operand", "episode", "theme"):
        candidates = [
            item for item in scored
            if str(item[-1].get("node_type")) == kind and item[0] > 0
        ]
        if candidates:
            add(max(candidates, key=lambda item: item[:-1])[-1])
    for source in source_priority:
        candidates = [
            item for item in scored
            if str(item[-1].get("selection_source") or "") == source
            and item[0] > 0
        ]
        if candidates:
            add(max(candidates, key=lambda item: item[:-1])[-1])
    for *_ranking, row in ranked:
        add(row)
    return selected


def deterministic_navigation_plan(
    *,
    question: str,
    evidence_ledger: list[dict[str, Any]],
    query_ir: QueryIR,
    max_selected: int = 20,
    include_adjacent_context: bool = False,
) -> NavigationPlan:
    """Build a bounded graph frontier without spending an LLM call.

    Query understanding is supplied by the benchmark-neutral QueryIR. The
    selected nodes are traversal seeds: graph recovery still opens provenance,
    adjacency, state, temporal, and entity relations before evidence packing.
    """

    rows = _candidate_rows(
        question,
        evidence_ledger,
        max_candidates=max(1, max_selected),
        diversify_sessions=True,
        include_adjacent_context=include_adjacent_context,
    )
    return NavigationPlan(
        selected_ids=tuple(
            str(row.get("node_id") or "")
            for row in rows
            if str(row.get("node_id") or "")
        ),
        operation=query_ir.intent,
        missing_slots=(),
        needed_relations=query_ir.allowed_relations,
        resolved_slots=(),
        candidate_answer="",
        confidence=0.0,
        parse_error=False,
    )


def navigation_messages(
    *,
    question: str,
    question_date: str | None,
    evidence_ledger: list[dict[str, Any]],
    query_ir: QueryIR | None = None,
    max_candidates: int = 36,
    max_prompt_rough_tokens: int = 3000,
) -> tuple[list[dict[str, str]], list[str]]:
    rows = _candidate_rows(
        question, evidence_ledger, max_candidates=max_candidates
    )
    cards: list[str] = []
    valid_ids: list[str] = []
    for row in rows:
        node_id = str(row["node_id"])
        valid_ids.append(node_id)
        snippet = focused_snippet(
            str(row.get("text") or ""), question, max_chars=520
        )
        cards.append(
            f"ID={node_id} | type={row.get('node_type', 'unknown')} | "
            f"route={row.get('selection_source', 'unknown')} | "
            f"observed_at={row.get('session_date') or row.get('observed_at') or 'unknown'} | "
            f"text={snippet}"
        )

    system = (
        "You are a path selector for a hierarchical memory hypergraph. Select the smallest complete "
        "evidence closure needed to answer the question; do not answer it. "
        "Query IR is a deterministic routing contract. Use its fixed intent, required slots, and "
        "allowed relations; do not replace it with a topic-specific operation. A set-wide query "
        "requires collection closure across all relevant sessions, not one representative example. "
        "A temporal query requires the event mention and its own observation-time anchor. "
        "Treat extracted claims and deterministic operator outputs as fallible proposals and verify them against "
        "lossless turn evidence. When a fact may be stated in a reply, keep the initiating turn and "
        "its short adjacent dialogue span together. Preserve entity identity, speaker ownership, polarity, modality, "
        "time order, units, and set membership. For a count, list, total, arithmetic, or ordering "
        "question, do not select representative examples: select every distinct candidate endpoint "
        "or operand present in the frontier, while excluding candidates that violate the requested "
        "speaker, entity, action, status, or time scope. For an exact-entity absence check, select "
        "both the closest match and evidence "
        "showing whether the requested entity is actually present. Bind every Query IR required "
        "slot that the frontier supports. For count/list, enumerate distinct members before deriving "
        "the result; for temporal/order, bind each event to its own date or relative-time anchor; "
        "for latest/state, bind old and new values; for absence, record the exact target and the "
        "nearest non-matching contrast. Do not provide reasoning prose. Return JSON only with keys "
        "selected_ids (at most 12 IDs from the list), operation (a short generic operation name), "
        "needed_relations, missing_slots, resolved_slots (a compact object), candidate_answer "
        "(a concise evidence-derived proposal or empty string), and confidence (0 to 1)."
    )
    ir_payload = (
        structured_navigation_summary(query_ir) if query_ir is not None else {}
    )
    prefix = (
        f"Question date: {question_date or 'unknown'}\n"
        f"Question: {question}\n"
        f"Query IR: {json.dumps(ir_payload, ensure_ascii=False)}\n\n"
        "Candidate graph frontier:\n"
    )
    kept: list[str] = []
    for card in cards:
        candidate = prefix + "\n".join([*kept, card])
        if kept and rough_token_count(system + candidate) > max_prompt_rough_tokens:
            break
        kept.append(card)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prefix + "\n".join(kept)},
    ]
    kept_ids = [card.split(" |", 1)[0].removeprefix("ID=") for card in kept]
    return messages, kept_ids


def compact_proposal_messages(
    *,
    question: str,
    question_date: str | None,
    evidence_ledger: list[dict[str, Any]],
    query_ir: QueryIR,
    operator_hint: dict[str, Any] | None = None,
    max_candidates: int = 32,
    max_prompt_rough_tokens: int = 2200,
) -> tuple[list[dict[str, str]], list[str]]:
    """Create a fallible answer proposal from this graph's compact frontier.

    The proposal is not an external candidate and is never authoritative. It
    gives the final lossless-evidence call explicit semantic slots to verify,
    while staying benchmark- and topic-neutral.
    """

    ranked_rows = _candidate_rows(
        question,
        evidence_ledger,
        max_candidates=max(64, max_candidates * 2),
        diversify_sessions=True,
        include_adjacent_context=True,
    )
    # Interleave sessions before spending the compact prompt budget. A flat
    # relevance list frequently spends every card on one session and drops the
    # earlier state, second operand, or competing event.
    session_buckets: dict[str, list[dict[str, Any]]] = {}
    unscoped: list[dict[str, Any]] = []
    for row in ranked_rows:
        session = str(row.get("session_id") or "") or _node_session_id(
            str(row.get("node_id") or "")
        )
        if session:
            session_buckets.setdefault(session, []).append(row)
        else:
            unscoped.append(row)
    rows: list[dict[str, Any]] = []
    for depth in range(3):
        for bucket in session_buckets.values():
            if depth < len(bucket):
                rows.append(bucket[depth])
                if len(rows) >= max_candidates:
                    break
        if len(rows) >= max_candidates:
            break
    for row in [*unscoped, *ranked_rows]:
        if row not in rows:
            rows.append(row)
        if len(rows) >= max_candidates:
            break
    cards: list[str] = []
    for row in rows:
        node_id = str(row.get("node_id") or "")
        if not node_id:
            continue
        snippet = focused_snippet(
            str(row.get("text") or ""), question, max_chars=520
        )
        cards.append(
            f"ID={node_id} | type={row.get('node_type', 'unknown')} | "
            f"observation_date={row.get('session_date') or row.get('observed_at') or 'unknown'} | "
            f"text={snippet}"
        )

    system = (
        "You are the semantic binding stage of a general-purpose hierarchical "
        "memory graph. Use only the supplied cards. Produce a compact, fallible "
        "proposal for a later verifier; never invent missing evidence. Bind the "
        "requested owner or speaker, entity, relation, polarity, status, time, "
        "unit, and every required operand. For count or list questions, enumerate "
        "all visible distinct members and mark the result incomplete when closure "
        "is missing. For temporal duration, bind both endpoints. For an update, "
        "bind old and new states without replacing a completed event with a plan. "
        "Do not substitute a sibling entity or a related but different action. "
        "Return JSON only with selected_ids (at most 12 supplied IDs), operation, "
        "needed_relations, missing_slots, resolved_slots, candidate_answer, and "
        "confidence. candidate_answer must be empty when required evidence is "
        "missing. Do not expose chain-of-thought. Observation dates anchor "
        "relative expressions but are not necessarily event dates. Compare "
        "fixed-numerator ratios by their denominators. A mechanically certified "
        "operator hint is still fallible, but when its complete source operands "
        "agree, use its value rather than an incomplete single-card calculation."
    )
    prefix = (
        f"Question date: {question_date or 'unknown'}\n"
        f"Question: {question}\n"
        "Query IR: "
        f"{json.dumps(structured_navigation_summary(query_ir), ensure_ascii=False)}\n\n"
        "Certified operator proposal (verify against cards): "
        f"{json.dumps(operator_hint or {}, ensure_ascii=False)}\n\n"
        "Current graph frontier:\n"
    )
    kept: list[str] = []
    for card in cards:
        candidate = prefix + "\n".join([*kept, card])
        if kept and rough_token_count(system + candidate) > max_prompt_rough_tokens:
            break
        kept.append(card)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prefix + "\n".join(kept)},
    ]
    kept_ids = [card.split(" |", 1)[0].removeprefix("ID=") for card in kept]
    return messages, kept_ids


def parse_navigation_plan(text: str, valid_ids: Iterable[str]) -> NavigationPlan:
    allowed = set(valid_ids)
    match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    try:
        payload = json.loads(match.group(0) if match else "")
    except (json.JSONDecodeError, AttributeError):
        return NavigationPlan((), "unknown", (), (), True)
    def selected_id(value: Any) -> str:
        if isinstance(value, dict):
            value = value.get("id") or value.get("ID") or value.get("node_id") or ""
        return str(value)

    selected = tuple(dict.fromkeys(
        selected_id(value) for value in payload.get("selected_ids", [])
        if selected_id(value) in allowed
    ))[:12]
    raw_slots = payload.get("resolved_slots") or {}
    resolved_slots = (
        tuple((str(key)[:80], str(value)[:240]) for key, value in raw_slots.items())[:12]
        if isinstance(raw_slots, dict)
        else ()
    )
    try:
        confidence = max(0.0, min(1.0, float(payload.get("confidence") or 0.0)))
    except (TypeError, ValueError):
        confidence = 0.0
    return NavigationPlan(
        selected_ids=selected,
        operation=str(payload.get("operation") or "unknown")[:80],
        missing_slots=tuple(str(value)[:120] for value in payload.get("missing_slots", [])[:8]),
        needed_relations=tuple(str(value)[:80] for value in payload.get("needed_relations", [])[:8]),
        parse_error=False,
        resolved_slots=resolved_slots,
        candidate_answer=str(payload.get("candidate_answer") or "")[:500],
        confidence=confidence,
    )


def selected_evidence_text(
    *,
    question: str,
    evidence_ledger: list[dict[str, Any]],
    plan: NavigationPlan,
    max_rough_tokens: int = 3000,
) -> str:
    by_id = {str(row.get("node_id")): row for row in evidence_ledger}
    selected_ids = list(plan.selected_ids)
    if not selected_ids:
        selected_ids = [
            str(row.get("node_id"))
            for row in _candidate_rows(question, evidence_ledger, max_candidates=10)
        ]
    header = (
        f"[NAVIGATION operation={plan.operation or 'unknown'} "
        f"missing_slots={json.dumps(plan.missing_slots, ensure_ascii=False)} "
        f"needed_relations={json.dumps(plan.needed_relations, ensure_ascii=False)}]"
    )
    blocks = [header]
    for node_id in selected_ids:
        row = by_id.get(node_id)
        if row is None:
            continue
        text = focused_snippet(
            str(row.get("text") or ""), question, max_chars=2600
        )
        block = (
            f"[EVIDENCE id={node_id} type={row.get('node_type', 'unknown')} "
            f"route={row.get('selection_source', 'unknown')}]\n{text}"
        )
        if len(blocks) > 1 and rough_token_count("\n\n".join([*blocks, block])) > max_rough_tokens:
            continue
        blocks.append(block)
    return "\n\n".join(blocks)

def recovered_evidence_text(
    *,
    question: str,
    evidence_ledger: list[dict[str, Any]],
    plan: NavigationPlan,
    recovery_rows: Iterable[dict[str, Any]],
    max_rough_tokens: int = 5000,
    fallback_rows: int = 12,
    evidence_profile: str = "mixed",
) -> tuple[str, list[str]]:
    """Pack a verified navigation closure without discarding strong direct evidence."""

    combined: dict[str, dict[str, Any]] = {
        str(row.get("node_id")): row
        for row in evidence_ledger
        if str(row.get("node_id") or "")
    }
    recovery_ids: list[str] = []
    for row in recovery_rows:
        node_id = str(row.get("node_id") or "")
        if not node_id:
            continue
        combined[node_id] = row
        recovery_ids.append(node_id)

    ordered_ids: list[str] = []

    def add(node_id: str) -> None:
        if node_id in combined and node_id not in ordered_ids:
            ordered_ids.append(node_id)

    adjacency_by_session: dict[str, list[str]] = {}
    for node_id in recovery_ids:
        row = combined.get(node_id) or {}
        if str(row.get("selection_source") or "") != "navigator_turn_context_recovery":
            continue
        match = re.search(r":([^:]+):turn:(\d+)$", node_id)
        if match:
            adjacency_by_session.setdefault(match.group(1), []).append(node_id)
    for values in adjacency_by_session.values():
        values.sort(key=lambda value: int(value.rsplit(":", 1)[-1]))

    for node_id in plan.selected_ids:
        add(node_id)
        match = re.search(r":([^:]+):turn:(\d+)$", node_id)
        if match:
            for adjacent_id in adjacency_by_session.get(match.group(1), []):
                add(adjacent_id)
    for node_id in recovery_ids:
        add(node_id)
    for row in _candidate_rows(
        question, evidence_ledger, max_candidates=max(1, fallback_rows)
    ):
        add(str(row.get("node_id") or ""))

    if evidence_profile == "lossless-first":
        lossless = [
            node_id
            for node_id in ordered_ids
            if str(combined[node_id].get("node_type") or "") == "turn"
        ]
        selected_routes = [
            node_id
            for node_id in plan.selected_ids
            if node_id in combined
            and str(combined[node_id].get("node_type") or "") != "turn"
        ][:4]
        if lossless:
            ordered_ids = list(dict.fromkeys([*lossless, *selected_routes]))
    elif evidence_profile != "mixed":
        raise ValueError(f"unknown evidence profile: {evidence_profile}")

    header = (
        f"[NAVIGATION operation={plan.operation or 'unknown'} "
        f"missing_slots={json.dumps(plan.missing_slots, ensure_ascii=False)} "
        f"needed_relations={json.dumps(plan.needed_relations, ensure_ascii=False)} "
        f"resolved_slots={json.dumps(dict(plan.resolved_slots), ensure_ascii=False)} "
        f"candidate_answer={json.dumps(plan.candidate_answer, ensure_ascii=False)} "
        f"planner_confidence={plan.confidence:.3f} "
        f"recovery_candidates={len(recovery_ids)}]"
    )
    blocks = [header]
    kept_ids: list[str] = []
    for node_id in ordered_ids:
        row = combined[node_id]
        raw_text = str(row.get("text") or "")
        if str(row.get("node_type") or "") in {"episode", "theme", "unknown"}:
            raw_text = re.sub(
                r"(?m)^Time:\s*",
                "Conversation observed at (not necessarily event time): ",
                raw_text,
            )
        text = focused_snippet(raw_text, question, max_chars=2200)
        relation_path = list(row.get("relation_path") or [])
        block = (
            f"[EVIDENCE id={node_id} type={row.get('node_type', 'unknown')} "
            f"route={row.get('selection_source', 'unknown')} "
            f"observed_at={row.get('session_date') or row.get('observed_at') or 'unknown'} "
            f"path={json.dumps(relation_path, ensure_ascii=False)}]\n{text}"
        )
        if (
            len(blocks) > 1
            and rough_token_count("\n\n".join([*blocks, block])) > max_rough_tokens
        ):
            continue
        blocks.append(block)
        kept_ids.append(node_id)
    return "\n\n".join(blocks), kept_ids



def navigated_answer_messages(
    *,
    question: str,
    question_date: str | None,
    evidence_text: str,
    plan: NavigationPlan,
) -> list[dict[str, str]]:
    system = (
        "Answer a memory question from the complete selected evidence closure. Scan every evidence "
        "block before deciding; a lower-ranked specific fact beats a higher-ranked generic match. "
        "Give only the concise answer, with a short audit list only when a count, sum, or ordering "
        "needs it. The navigation "
        "operation, resolved slots, certified operator result, and candidate answer are auditable "
        "proposals, not independent facts. Verify them against cited lossless evidence. When a "
        "high-confidence candidate is supported by every bound operand, preserve its exact value "
        "instead of performing a second inconsistent calculation. When it conflicts with source "
        "turns, correct it from those turns. Keep "
        "speaker and entity ownership exact; do not substitute a sibling entity that merely shares "
        "words. If evidence mentions only a near-match but never the exact requested entity-relation "
        "tuple, answer that the requested information was not mentioned rather than copying the near-match. "
        "Distinguish completed facts from plans, suggestions, examples, and negations. Resolve "
        "relative time against the evidence date. A missing_slots entry records what the first-pass "
        "frontier lacked; it is not a conclusion. Recovered or fallback evidence may fill it, so check "
        "the full closure before abstaining. "
        "Prefer lossless source turns over conflicting extracted claims. For counts or totals, "
        "deduplicate repeated mentions of one state/event and use cumulative updates as states rather "
        "than additional occurrences. The observed_at field is a conversation timestamp, not an "
        "event date unless the source text says the event occurred then. If several memories share "
        "a person or topic, bind the answer to every requested entity, relation, and date constraint; "
        "do not answer from a merely similar event. For temporal questions, compare competing event "
        "instances, bind each relative phrase to its own observation date, and choose the instance "
        "satisfying all event and time constraints. For open-ended questions such as whether someone "
        "would do something or has a trait, make the shortest direct causal inference supported by "
        "the memories and qualify uncertainty with 'likely' or 'somewhat'; an explicit statement in "
        "the exact question wording is not required. Ordinary background knowledge may connect "
        "directly implied facts, but must not invent a name, place, number, or date absent from "
        "evidence. When an answer is expressed relatively and its "
        "observation anchor is available, return the resolved calendar date, month, season, or year "
        "instead of leaving 'last week', 'last summer', or 'last year' unresolved. If relevant evidence "
        "exists, return the most specific supported answer rather than a generic summary or reflexive "
        "abstention. Say memory is insufficient only when the full closure contains no evidence that "
        "can answer or directly imply the requested fact."
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"Question date: {question_date or 'unknown'}\n"
                f"Question: {question}\n"
                f"Navigation operation: {plan.operation}\n\n"
                f"Selected evidence closure:\n{evidence_text}"
            ),
        },
    ]


def verification_messages(
    *,
    question: str,
    question_date: str | None,
    draft_answer: str,
    evidence_text: str,
    max_rough_tokens: int = 1500,
) -> list[dict[str, str]]:
    """Build a bounded, evidence-only verifier prompt for a draft answer."""
    max_chars = max(800, max_rough_tokens * 4 - 1200)
    evidence = focused_snippet(evidence_text, question, max_chars=max_chars)
    system = (
        "Verify a draft memory answer against the supplied evidence. Return only the final concise "
        "answer, without analysis. Keep the draft unchanged when it is directly supported. Otherwise "
        "correct only what the evidence establishes. Bind every requested person, object, relation, "
        "polarity, status, and date; reject facts from a merely similar event. observed_at is a "
        "conversation timestamp, not an event date unless the text says the event happened then. "
        "Resolve anchored relative time to a calendar value. If the required fact is absent, answer "
        "that memory is insufficient."
    )
    user = (
        f"Question date: {question_date or 'unknown'}\n"
        f"Question: {question}\n"
        f"Draft answer: {draft_answer}\n\n"
        f"Verification evidence:\n{evidence}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
