from __future__ import annotations

from collections import defaultdict
import math
import re
from typing import Any, Iterable

from .structured_navigation import QueryIR


_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")
_STOP = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "can",
    "did", "do", "does", "for", "from", "had", "has", "have", "how",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "the", "to",
    "was", "were", "what", "when", "where", "which", "who", "with",
})


def _tokens(value: str) -> set[str]:
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


def _session_id_from_reference(value: str) -> str:
    suffix = value.split(":", 1)[1] if ":" in value else value
    for marker in (":turn:", ":claim:", ":event:"):
        if marker in suffix:
            return suffix.split(marker, 1)[0]
    return ""


def scope_rows_to_sessions(
    rows: Iterable[dict[str, Any]],
    selected_session_ids: Iterable[str],
) -> list[dict[str, Any]]:
    """Materialize a true local subgraph after coarse session routing."""

    selected = {str(value) for value in selected_session_ids if str(value)}
    scoped: list[dict[str, Any]] = []
    for row in rows:
        explicit = str(row.get("session_id") or "")
        if explicit:
            if explicit in selected:
                scoped.append(row)
            continue
        references = {
            session_id
            for session_id in (
                _session_id_from_reference(str(value))
                for value in row.get("source_turn_ids") or ()
            )
            if session_id
        }
        if not references:
            node_session = _session_id_from_reference(
                str(row.get("node_id") or "")
            )
            if node_session:
                references.add(node_session)
        # Cross-session theme nodes are intentionally not admitted merely
        # because they touch one selected session. Typed graph recovery can
        # reopen an external session later when an allowed relation warrants it.
        if references and references.issubset(selected):
            scoped.append(row)
    return scoped


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def dense_reranked_lossless_session_rows(
    *,
    graph_store: Any,
    embedder: Any,
    question_id: str,
    question: str,
    retrieved_session_ids: Iterable[str],
    query_ir: QueryIR,
    max_sessions: int = 8,
    semantic_seeds_per_session: int = 3,
    max_turns_per_session: int = 6,
) -> list[dict[str, Any]]:
    """Dense rerank only inside coarse-routed sessions, then keep adjacency."""

    wanted = list(dict.fromkeys(
        str(value) for value in retrieved_session_ids if str(value)
    ))[:max_sessions]
    all_rows = routed_lossless_session_rows(
        graph_store=graph_store,
        question_id=question_id,
        retrieved_session_ids=wanted,
        query_ir=query_ir,
        max_sessions=max_sessions,
        max_turns_per_session=10000,
        seed_turns_per_session=10000,
    )
    if not all_rows:
        return []
    query_text = (
        "Instruct: Retrieve conversation turns containing facts, states, "
        "operands, or personal constraints needed to answer the memory "
        f"question.\nQuery: {question}"
    )
    passage_texts = [str(row.get("text") or "")[:4000] for row in all_rows]
    vectors = embedder.embed(
        [query_text, *passage_texts],
        question_id=question_id,
        variant="v3_5_session_dense_rerank",
    )
    query_vector = vectors[0]
    similarities = [_cosine(query_vector, vector) for vector in vectors[1:]]
    by_session: dict[str, list[tuple[int, dict[str, Any], float]]] = defaultdict(list)
    for row, similarity in zip(all_rows, similarities):
        session_id = str(row.get("session_id") or "")
        if not session_id:
            continue
        by_session[session_id].append(
            (int(row.get("turn_index") or 0), row, similarity)
        )

    selected: list[dict[str, Any]] = []
    for route_rank, session_id in enumerate(wanted):
        turns = sorted(by_session.get(session_id, ()), key=lambda item: item[0])
        if not turns:
            continue
        position_by_turn = {
            turn_index: position
            for position, (turn_index, _row, _similarity) in enumerate(turns)
        }
        seed_positions = [
            position_by_turn[turn_index]
            for turn_index, _row, _similarity in sorted(
                turns,
                key=lambda item: (item[2], float(item[1].get("score") or 0.0)),
                reverse=True,
            )[:semantic_seeds_per_session]
        ]
        chosen: list[int] = []
        for position in seed_positions:
            turn_index = turns[position][0]
            for candidate_turn in (turn_index, turn_index - 1, turn_index + 1):
                candidate = position_by_turn.get(candidate_turn)
                if (
                    candidate is not None
                    and candidate not in chosen
                    and len(chosen) < max_turns_per_session
                ):
                    chosen.append(candidate)
        for position in sorted(chosen):
            _turn_index, row, similarity = turns[position]
            enriched = dict(row)
            enriched["selection_source"] = "routed_lossless_dense_session"
            enriched["score"] = round(
                similarity * 10.0 + 3.0 / (route_rank + 1),
                6,
            )
            enriched["semantic_similarity"] = round(similarity, 6)
            selected.append(enriched)
    return selected


def routed_lossless_session_rows(
    *,
    graph_store: Any,
    question_id: str,
    retrieved_session_ids: Iterable[str],
    query_ir: QueryIR,
    max_sessions: int = 16,
    max_turns_per_session: int = 3,
    seed_turns_per_session: int = 2,
) -> list[dict[str, Any]]:
    """Open a bounded lossless window inside every coarse-routed session.

    The coarse session order comes from retrieval, not gold metadata. Within a
    session, lexical binding chooses turn seeds and then retains adjacent
    dialogue turns so an answer-bearing reply is not lost merely because it
    shares few words with the question.
    """

    nodes = graph_store.nodes_for(question_id)
    wanted = list(dict.fromkeys(
        str(value) for value in retrieved_session_ids if str(value)
    ))[:max_sessions]
    by_session: dict[str, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    for suffix, row in nodes.items():
        if str(row.get("node_type") or "") != "turn":
            continue
        session_id = str(row.get("session_id") or "")
        if session_id not in wanted:
            continue
        try:
            turn_index = int(row.get("turn_index"))
        except (TypeError, ValueError):
            match = re.search(r":turn:(\d+)$", suffix)
            if match is None:
                continue
            turn_index = int(match.group(1))
        by_session[session_id].append((turn_index, suffix, row))

    query_tokens = set(query_ir.content_terms)
    query_tokens.update(query_ir.subjects)
    query_tokens.update(_tokens(query_ir.answer_slot or ""))
    rows: list[dict[str, Any]] = []
    for route_rank, session_id in enumerate(wanted):
        turns = sorted(by_session.get(session_id, ()))
        if not turns:
            continue
        scored: list[tuple[float, int]] = []
        for position, (_turn_index, _suffix, row) in enumerate(turns):
            text = str(row.get("retrieval_text") or row.get("text") or "")
            overlap = len(query_tokens & _tokens(text))
            scalar_bonus = 1.0 if (
                query_ir.answer_form in {"number", "date", "duration", "frequency"}
                and re.search(r"(?:[$€£¥]\s*)?\d[\d,]*(?:\.\d+)?%?", text)
            ) else 0.0
            scored.append((overlap * 3.0 + scalar_bonus, position))
        best_positions = [
            position
            for _score, position in sorted(scored, reverse=True)[
                :max(1, seed_turns_per_session)
            ]
        ]
        chosen: list[int] = []
        for position in best_positions:
            for candidate in (position, position + 1, position - 1):
                if (
                    0 <= candidate < len(turns)
                    and candidate not in chosen
                    and len(chosen) < max_turns_per_session
                ):
                    chosen.append(candidate)
        for position in sorted(chosen):
            turn_index, suffix, row = turns[position]
            canonical = str(row.get("node_id") or f"{question_id}:{suffix}")
            canonical_suffix = canonical.split(":", 1)[1] if ":" in canonical else canonical
            rows.append({
                "node_id": f"{question_id}:{canonical_suffix}",
                "node_type": "turn",
                "selection_source": "routed_lossless_session",
                "score": round(3.0 / (route_rank + 1) + scored[position][0], 6),
                "text": str(row.get("retrieval_text") or row.get("text") or ""),
                "source_turn_ids": [],
                "session_id": session_id,
                "session_date": row.get("session_date"),
                "turn_index": turn_index,
                "relation_path": ["coarse_session", "next_turn"],
            })
    return rows
