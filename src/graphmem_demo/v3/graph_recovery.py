from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable


_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")
_GROUPED_CASE_RE = re.compile(r"^(.+)_\d+$")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_PROPER_NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:['’][A-Za-z]+)?\b")
_TURN_SUFFIX_RE = re.compile(r"^([^:]+):turn:(\d+)$")
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "can",
    "did", "do", "does", "for", "from", "had", "has", "have", "how",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "the", "to",
    "was", "were", "what", "when", "where", "which", "who", "with",
}
_NODE_FILES = (
    "nodes.jsonl",
    "operands.jsonl",
    "event_frames.jsonl",
    "episodes.jsonl",
    "themes.jsonl",
)
_EDGE_FILES = ("hyperedges.jsonl", "edges.jsonl")
_RELATION_PRIOR = {
    "supports": 1.0,
    "source": 1.0,
    "event_entity_member": 1.0,
    "event_frame_member": 0.98,
    "operand_projection": 0.96,
    "reference_binding": 0.95,
    "supersedes": 0.94,
    "contradicts": 0.92,
    "same_event": 0.90,
    "same_entity": 0.88,
    "same_predicate": 0.84,
    "participates_in": 0.84,
    "before": 0.82,
    "after": 0.82,
    "next_turn": 0.78,
    "episode_member": 0.74,
    "theme_member": 0.62,
    "semantic_neighbor": 0.52,
}
_WIDE_RELATIONS = frozenset(
    {"theme_member", "semantic_neighbor", "same_entity", "same_predicate"}
)
_ALWAYS_LOCAL_RELATIONS = frozenset(
    {
        "supports", "source", "event_entity_member", "event_frame_member",
        "operand_projection", "reference_binding", "supersedes", "contradicts",
        "same_event", "participates_in", "before", "after", "next_turn",
        "episode_member",
    }
)
_QUESTION_WORDS = frozenset(
    {
        "Answer", "Are", "Did", "Does", "How", "Is", "No", "The", "Was",
        "Were", "What", "When", "Where", "Which", "Who", "Why", "Yes",
        "January", "February", "March", "April", "May", "June", "July",
        "August", "September", "October", "November", "December",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
        "Sunday",
    }
)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _node_id(row: dict[str, Any]) -> str:
    return str(
        row.get("node_id")
        or row.get("operand_id")
        or row.get("frame_id")
        or ""
    )


def _scope(question_id: str) -> str:
    match = _GROUPED_CASE_RE.match(question_id)
    return match.group(1) if match else question_id


def _suffix(node_id: str) -> str:
    return node_id.split(":", 1)[1] if ":" in node_id else node_id


def _alias(node_id: str, question_id: str) -> str:
    return f"{question_id}:{_suffix(node_id)}" if ":" in node_id else node_id


def _tokens(value: str) -> set[str]:
    return {
        token.casefold()
        for token in _WORD_RE.findall(value)
        if len(token) > 1 and token.casefold() not in _STOP
    }


def _node_text(row: dict[str, Any]) -> str:
    return str(row.get("retrieval_text") or row.get("text") or "")


def _years(value: str) -> set[str]:
    return set(_YEAR_RE.findall(value))


def _question_entities(question: str) -> set[str]:
    """Extract only explicit proper-name anchors, without topic vocabularies."""
    return {
        value.casefold()
        for value in _PROPER_NAME_RE.findall(question)
        if value not in _QUESTION_WORDS
    }


def _relation_tokens(value: str) -> set[str]:
    return _tokens(value.replace("_", " "))


def _relation_is_requested(relation: str, requested_tokens: set[str]) -> bool:
    relation_parts = _relation_tokens(relation)
    return bool(relation_parts and relation_parts <= requested_tokens)


def _allow_relation(
    relation: str,
    *,
    depth: int,
    current_type: str,
    requested_tokens: set[str],
    operation_tokens: set[str],
) -> bool:
    """Keep local provenance edges open and gate cross-topic graph shortcuts."""
    if relation in _ALWAYS_LOCAL_RELATIONS:
        return True
    if _relation_is_requested(relation, requested_tokens):
        return True
    if relation == "theme_member":
        # A selected theme may be opened once. Do not use a theme reached from
        # an episode as a bridge into unrelated episodes.
        return depth == 0 and current_type == "theme"
    if relation in {"same_entity", "same_predicate"}:
        return bool(
            operation_tokens
            & {"compare", "history", "latest", "list", "state", "timeline"}
        )
    # semantic_neighbor and unknown relation types require an explicit request.
    return False


def _constraint_adjustment(
    text: str,
    *,
    query_years: set[str],
    query_entities: set[str],
) -> float | None:
    """Reject explicit time conflicts and softly prefer exact entity bindings."""
    target_years = _years(text)
    if query_years and target_years and query_years.isdisjoint(target_years):
        return None
    if not query_entities:
        return 0.0
    folded = text.casefold()
    hits = sum(entity in folded for entity in query_entities)
    return min(1.2, hits * 0.6) if hits else -1.1


@dataclass(frozen=True)
class RecoveryResult:
    rows: tuple[dict[str, Any], ...]
    graph_rows: int
    lexical_rows: int
    source_rows: int
    searched_nodes: int
    adjacency_rows: int = 0
    relation_filtered_edges: int = 0
    constraint_filtered_candidates: int = 0


class PersistedGraphStore:
    """Read-only graph assets keyed by conversation scope, with per-question aliases."""

    def __init__(self, run_dir: Path) -> None:
        self._nodes: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        self._edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._adjacency: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for filename in _NODE_FILES:
            for row in _read_jsonl(run_dir / filename):
                node_id = _node_id(row)
                question_id = str(row.get("question_id") or "")
                if node_id and question_id:
                    scope = _scope(question_id)
                    suffix = _suffix(node_id)
                    previous = self._nodes[scope].get(suffix)
                    # nodes.jsonl carries the canonical node type and source
                    # provenance.  Auxiliary projections may repeat the same
                    # node without those fields; never let the lossy duplicate
                    # erase richer graph semantics during reload.
                    if previous is None or self._row_richness(row) > self._row_richness(previous):
                        self._nodes[scope][suffix] = row
        for filename in _EDGE_FILES:
            for row in _read_jsonl(run_dir / filename):
                question_id = str(row.get("question_id") or "")
                incidences = list(row.get("incidences") or [])
                if not question_id or not incidences:
                    continue
                scope = _scope(question_id)
                edge_index = len(self._edges[scope])
                self._edges[scope].append(row)
                for incidence in incidences:
                    node_id = str(incidence.get("node_id") or "")
                    if node_id:
                        self._adjacency[scope][_suffix(node_id)].append(edge_index)

    @staticmethod
    def _row_richness(row: dict[str, Any]) -> tuple[int, int, int, int]:
        return (
            int(bool(row.get("node_type"))),
            int(bool(row.get("retrieval_text"))),
            int(bool(row.get("source_turn_ids"))),
            len(str(row.get("retrieval_text") or row.get("text") or "")),
        )

    @property
    def node_count(self) -> int:
        return sum(len(rows) for rows in self._nodes.values())

    @property
    def edge_count(self) -> int:
        return sum(len(rows) for rows in self._edges.values())

    def has_scope(self, question_id: str) -> bool:
        scope = _scope(question_id)
        return bool(self._nodes.get(scope)) and bool(self._edges.get(scope))

    def nodes_for(self, question_id: str) -> dict[str, dict[str, Any]]:
        return self._nodes.get(_scope(question_id), {})

    def evidence_rows_for_ids(
        self,
        question_id: str,
        node_ids: Iterable[str],
        *,
        source: str = "certified_operator_provenance",
    ) -> list[dict[str, Any]]:
        """Hydrate graph nodes named by a provenance certificate."""
        rows: list[dict[str, Any]] = []
        for node_id in dict.fromkeys(str(value) for value in node_ids if value):
            row = self._aliased_row(
                question_id, _suffix(node_id), source=source, score=12.0
            )
            if row is not None:
                rows.append(row)
        return rows

    def _aliased_row(
        self,
        question_id: str,
        suffix: str,
        *,
        source: str,
        score: float,
        relation_path: list[str] | None = None,
    ) -> dict[str, Any] | None:
        row = self.nodes_for(question_id).get(suffix)
        if row is None:
            return None
        return {
            "node_id": _alias(_node_id(row), question_id),
            "node_type": str(row.get("node_type") or "unknown"),
            "selection_source": source,
            "score": round(score, 6),
            "text": _node_text(row),
            "source_turn_ids": [
                _alias(str(value), question_id)
                for value in row.get("source_turn_ids") or []
            ],
            "session_id": row.get("session_id"),
            "session_date": row.get("session_date") or row.get("observed_at"),
            "relation_path": list(relation_path or []),
        }

    def recover(
        self,
        *,
        question_id: str,
        question: str,
        selected_ids: Iterable[str],
        missing_slots: Iterable[str],
        needed_relations: Iterable[str],
        operation: str = "",
        max_graph_rows: int = 5,
        max_lexical_rows: int = 6,
        max_source_rows: int = 4,
        max_adjacency_rows: int = 4,
        max_depth: int = 2,
    ) -> RecoveryResult:
        scope = _scope(question_id)
        nodes = self._nodes.get(scope, {})
        edges = self._edges.get(scope, [])
        adjacency = self._adjacency.get(scope, {})
        selected_suffixes = {_suffix(value) for value in selected_ids}
        query_text = " ".join([question, *missing_slots, *needed_relations])
        query_tokens = _tokens(query_text)
        query_years = _years(question)
        query_entities = _question_entities(question)
        requested_tokens = set().union(
            *(_relation_tokens(value) for value in needed_relations)
        )
        operation_tokens = _tokens(operation)
        relation_filtered_edges = 0
        constraint_filtered_candidates = 0

        graph_scored: dict[str, tuple[float, list[str]]] = {}
        queue = deque((suffix, 0, []) for suffix in selected_suffixes if suffix in nodes)
        best_depth = {suffix: 0 for suffix in selected_suffixes}
        while queue:
            current, depth, path = queue.popleft()
            if depth >= max_depth:
                continue
            for edge_index in adjacency.get(current, []):
                edge = edges[edge_index]
                relation = str(edge.get("relation") or "unknown")
                current_type = str(nodes[current].get("node_type") or "unknown")
                if current_type == "unknown":
                    current_type = current.split(":", 1)[0]
                if not _allow_relation(
                    relation,
                    depth=depth,
                    current_type=current_type,
                    requested_tokens=requested_tokens,
                    operation_tokens=operation_tokens,
                ):
                    relation_filtered_edges += 1
                    continue
                confidence = float(edge.get("confidence") or 0.0)
                prior = _RELATION_PRIOR.get(relation, 0.55)
                for incidence in edge.get("incidences") or []:
                    target = _suffix(str(incidence.get("node_id") or ""))
                    if not target or target == current or target not in nodes:
                        continue
                    target_text = _node_text(nodes[target])
                    adjustment = _constraint_adjustment(
                        target_text,
                        query_years=query_years,
                        query_entities=query_entities,
                    )
                    if adjustment is None:
                        constraint_filtered_candidates += 1
                        continue
                    lexical = len(query_tokens & _tokens(target_text))
                    if relation in _WIDE_RELATIONS and lexical == 0:
                        constraint_filtered_candidates += 1
                        continue
                    requested_bonus = (
                        1.0 if _relation_is_requested(relation, requested_tokens) else 0.0
                    )
                    score = (
                        (prior * 1.6) + confidence + (lexical * 0.55)
                        + adjustment + requested_bonus - depth * 0.25
                    )
                    candidate_path = [*path, relation]
                    previous = graph_scored.get(target)
                    if target not in selected_suffixes and (
                        previous is None or score > previous[0]
                    ):
                        graph_scored[target] = (score, candidate_path)
                    next_depth = depth + 1
                    if next_depth < best_depth.get(target, max_depth + 1):
                        best_depth[target] = next_depth
                        queue.append((target, next_depth, candidate_path))

        adjacency_scored: dict[str, tuple[float, list[str]]] = {}
        for selected in selected_suffixes:
            match = _TURN_SUFFIX_RE.fullmatch(selected)
            if not match:
                continue
            session_key, raw_index = match.groups()
            turn_index = int(raw_index)
            for distance in (1, 2):
                for neighbor_index in (turn_index - distance, turn_index + distance):
                    if neighbor_index < 0:
                        continue
                    target = f"{session_key}:turn:{neighbor_index}"
                    if target in selected_suffixes or target not in nodes:
                        continue
                    target_text = _node_text(nodes[target])
                    adjustment = _constraint_adjustment(
                        target_text,
                        query_years=query_years,
                        query_entities=query_entities,
                    )
                    if adjustment is None:
                        constraint_filtered_candidates += 1
                        continue
                    lexical = len(query_tokens & _tokens(target_text))
                    score = 7.0 + lexical * 0.7 + adjustment - distance * 0.3
                    previous = adjacency_scored.get(target)
                    if previous is None or score > previous[0]:
                        adjacency_scored[target] = (score, ["next_turn"])

        adjacency_rows: list[dict[str, Any]] = []
        for suffix, (score, path) in sorted(
            adjacency_scored.items(), key=lambda item: item[1][0], reverse=True
        ):
            row = self._aliased_row(
                question_id,
                suffix,
                source="navigator_turn_context_recovery",
                score=score,
                relation_path=path,
            )
            if row is not None:
                adjacency_rows.append(row)
            if len(adjacency_rows) >= max_adjacency_rows:
                break

        graph_rows: list[dict[str, Any]] = []
        for suffix, (score, path) in sorted(
            graph_scored.items(), key=lambda item: item[1][0], reverse=True
        ):
            row = self._aliased_row(
                question_id, suffix, source="navigator_graph_recovery",
                score=score, relation_path=path,
            )
            if row is not None:
                graph_rows.append(row)
            if len(graph_rows) >= max_graph_rows:
                break

        documents = {
            suffix: _tokens(_node_text(row))
            for suffix, row in nodes.items()
            if suffix not in selected_suffixes and _node_text(row)
        }
        document_count = max(1, len(documents))
        document_frequency = {
            token: sum(token in tokens for tokens in documents.values())
            for token in query_tokens
        }
        lexical_scored: list[tuple[float, str]] = []
        for suffix, tokens in documents.items():
            overlap = query_tokens & tokens
            if not overlap:
                continue
            adjustment = _constraint_adjustment(
                _node_text(nodes[suffix]),
                query_years=query_years,
                query_entities=query_entities,
            )
            if adjustment is None:
                constraint_filtered_candidates += 1
                continue
            if query_entities and adjustment < 0 and len(overlap) < 2:
                constraint_filtered_candidates += 1
                continue
            score = sum(
                math.log((document_count + 1) / (document_frequency[token] + 1)) + 1.0
                for token in overlap
            ) / math.sqrt(max(8, len(tokens)))
            node_type = str(nodes[suffix].get("node_type") or "")
            if node_type in {"turn", "claim", "event", "event_entity"}:
                score += 0.35
            score += adjustment
            lexical_scored.append((score, suffix))
        lexical_rows: list[dict[str, Any]] = []
        already = {
            str(row["node_id"]) for row in [*adjacency_rows, *graph_rows]
        }
        for score, suffix in sorted(lexical_scored, reverse=True):
            row = self._aliased_row(
                question_id, suffix, source="navigator_lexical_recovery", score=score
            )
            if row is None or str(row["node_id"]) in already:
                continue
            lexical_rows.append(row)
            already.add(str(row["node_id"]))
            if len(lexical_rows) >= max_lexical_rows:
                break

        source_rows: list[dict[str, Any]] = []
        source_suffixes: list[str] = []
        for row in [*adjacency_rows, *graph_rows, *lexical_rows]:
            source_suffixes.extend(_suffix(value) for value in row.get("source_turn_ids") or [])
        for suffix in dict.fromkeys(source_suffixes):
            aliased = self._aliased_row(
                question_id, suffix, source="navigator_source_recovery", score=2.5
            )
            if aliased is None or str(aliased["node_id"]) in already:
                continue
            source_rows.append(aliased)
            already.add(str(aliased["node_id"]))
            if len(source_rows) >= max_source_rows:
                break

        # Lossless source turns lead the recovered closure; extracted nodes
        # remain useful routing evidence but must not crowd out provenance.
        rows = tuple([*adjacency_rows, *source_rows, *graph_rows, *lexical_rows])
        return RecoveryResult(
            rows=rows,
            graph_rows=len(graph_rows),
            lexical_rows=len(lexical_rows),
            source_rows=len(source_rows),
            searched_nodes=len(nodes),
            adjacency_rows=len(adjacency_rows),
            relation_filtered_edges=relation_filtered_edges,
            constraint_filtered_candidates=constraint_filtered_candidates,
        )
