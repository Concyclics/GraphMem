from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from ..v36.schema import V36Index
from .schema import (
    QuerySidecarV41, SidecarDocumentV41, V41_POLICY_VERSION,
)


_RELATIONS = {
    "source", "next_turn", "dialogue_pair", "reference", "same_event",
    "state_transition", "collection_member", "temporal_endpoint", "contrast",
}


def _key(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9][a-z0-9'_-]*", value.casefold()))


def _values(*items: object) -> list[str]:
    result: list[str] = []
    for item in items:
        if isinstance(item, str) and item.strip():
            result.append(_key(item))
        elif isinstance(item, (list, tuple, set)):
            result.extend(_values(*item))
    return [value for value in dict.fromkeys(result) if value]


def index_hash(index: V36Index) -> str:
    payload = {
        "schema": index.schema_version,
        "turns": [(node.node_id, node.text) for node in index.turns],
        "frames": [
            (node.frame_id, node.retrieval_text, node.source_turn_ids)
            for node in index.frames
        ],
        "groups": [
            (node.group_id, node.member_frame_ids, node.source_turn_ids)
            for node in index.evidence_groups
        ],
        "cards": [
            (node.card_id, node.routing_text, node.frame_ids, node.turn_ids)
            for node in index.routing_cards
        ],
        "edges": [
            (edge.src, edge.dst, edge.relation, edge.directed)
            for edge in index.edges if edge.relation in _RELATIONS
        ],
    }
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_sidecar(index: V36Index) -> QuerySidecarV41:
    documents: dict[str, SidecarDocumentV41] = {}
    inverted: defaultdict[str, defaultdict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    adjacency: defaultdict[str, defaultdict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )

    def add(document: SidecarDocumentV41) -> None:
        documents[document.node_id] = document
        for field, values in document.fields.items():
            for value in values:
                inverted[field][_key(value)].append(document.node_id)

    for turn in index.turns:
        add(SidecarDocumentV41(
            node_id=turn.node_id, node_type="turn",
            session_ids=[turn.session_id], source_turn_ids=[turn.node_id],
            text=turn.retrieval_text or turn.text,
            fields={
                "speaker": _values(turn.speaker_key, turn.speaker),
                "owner": _values(turn.speaker_key),
                "date": _values(turn.session_date or ""),
                "semantic_type": _values(turn.transport_role),
            },
        ))
    for frame in index.frames:
        temporal = frame.temporal
        add(SidecarDocumentV41(
            node_id=frame.frame_id, node_type="frame",
            session_ids=list(frame.session_ids),
            source_turn_ids=list(frame.source_turn_ids),
            text=frame.retrieval_text,
            fields={
                "entity": _values(frame.entity_key),
                "owner": _values(frame.owner_key),
                "predicate": _values(frame.predicate_key),
                "object": _values(frame.object_key),
                "event_identity": _values(frame.event_identity_key),
                "date": _values(
                    temporal.event_time or "", temporal.observed_at or "",
                    temporal.start or "", temporal.end or "",
                ),
                "status": _values(frame.lifecycle_status),
                "polarity": _values(frame.polarity),
                "semantic_type": _values(
                    frame.frame_kind, frame.semantic_type_keys,
                ),
            },
        ))
    for group in index.evidence_groups:
        add(SidecarDocumentV41(
            node_id=group.group_id, node_type="group",
            session_ids=list(group.session_ids),
            source_turn_ids=list(group.source_turn_ids),
            text=group.retrieval_text,
            fields={"semantic_type": _values(group.group_kind, group.required_roles)},
        ))
    for card in index.routing_cards:
        add(SidecarDocumentV41(
            node_id=card.card_id, node_type="card",
            session_ids=[card.session_id],
            source_turn_ids=list(card.turn_ids),
            text=card.routing_text,
            fields={
                "entity": _values(card.canonical_entities),
                "speaker": _values(card.speaker_keys),
                "predicate": _values(card.relations),
                "date": _values(card.time_range),
                "semantic_type": _values("routing_card"),
            },
        ))

    for edge in index.edges:
        if edge.relation not in _RELATIONS:
            continue
        adjacency[edge.src][edge.relation].append(edge.dst)
        # Sidecar navigation is intentionally bidirectional. Direction remains
        # available in the immutable graph and is not changed here.
        adjacency[edge.dst][edge.relation].append(edge.src)

    frozen_inverted = {
        field: {value: list(dict.fromkeys(ids)) for value, ids in rows.items()}
        for field, rows in inverted.items()
    }
    frozen_adjacency = {
        node_id: {
            relation: list(dict.fromkeys(ids))
            for relation, ids in relations.items()
        }
        for node_id, relations in adjacency.items()
    }
    return QuerySidecarV41(
        index_hash=index_hash(index), policy_version=V41_POLICY_VERSION,
        documents=documents, inverted=frozen_inverted,
        adjacency=frozen_adjacency,
        diagnostics={
            "document_count": len(documents),
            "inverted_value_count": sum(
                len(values) for values in frozen_inverted.values()
            ),
            "adjacency_node_count": len(frozen_adjacency),
            "relation_counts": {
                relation: sum(
                    len(rows.get(relation, []))
                    for rows in frozen_adjacency.values()
                ) // 2
                for relation in sorted(_RELATIONS)
            },
        },
    )


def persist_sidecar(path: Path, sidecar: QuerySidecarV41) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
            node_id UNINDEXED, node_type UNINDEXED, text
        );
        CREATE TABLE IF NOT EXISTS documents(
            node_id TEXT PRIMARY KEY, node_type TEXT NOT NULL,
            session_ids TEXT NOT NULL, source_turn_ids TEXT NOT NULL,
            fields TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS inverted(
            field TEXT NOT NULL, value TEXT NOT NULL, node_id TEXT NOT NULL,
            PRIMARY KEY(field, value, node_id)
        );
        CREATE INDEX IF NOT EXISTS inverted_lookup ON inverted(field, value);
        CREATE TABLE IF NOT EXISTS adjacency(
            node_id TEXT NOT NULL, relation TEXT NOT NULL, neighbor_id TEXT NOT NULL,
            PRIMARY KEY(node_id, relation, neighbor_id)
        );
        CREATE INDEX IF NOT EXISTS adjacency_lookup ON adjacency(node_id, relation);
        """)
        connection.execute("DELETE FROM meta")
        connection.execute("DELETE FROM documents_fts")
        connection.execute("DELETE FROM documents")
        connection.execute("DELETE FROM inverted")
        connection.execute("DELETE FROM adjacency")
        connection.executemany("INSERT INTO meta VALUES(?, ?)", [
            ("index_hash", sidecar.index_hash),
            ("policy_version", sidecar.policy_version),
            ("schema_version", sidecar.schema_version),
        ])
        for document in sidecar.documents.values():
            connection.execute(
                "INSERT INTO documents_fts VALUES(?, ?, ?)",
                (document.node_id, document.node_type, document.text),
            )
            connection.execute(
                "INSERT INTO documents VALUES(?, ?, ?, ?, ?)",
                (
                    document.node_id, document.node_type,
                    json.dumps(document.session_ids),
                    json.dumps(document.source_turn_ids),
                    json.dumps(document.fields, ensure_ascii=False),
                ),
            )
        connection.executemany(
            "INSERT INTO inverted VALUES(?, ?, ?)",
            (
                (field, value, node_id)
                for field, rows in sidecar.inverted.items()
                for value, node_ids in rows.items()
                for node_id in node_ids
            ),
        )
        connection.executemany(
            "INSERT INTO adjacency VALUES(?, ?, ?)",
            (
                (node_id, relation, neighbor)
                for node_id, rows in sidecar.adjacency.items()
                for relation, neighbors in rows.items()
                for neighbor in neighbors
            ),
        )
        connection.commit()
    finally:
        connection.close()


def sidecar_matches(path: Path, index: V36Index) -> bool:
    if not path.exists():
        return False
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            values = dict(connection.execute("SELECT key, value FROM meta"))
        finally:
            connection.close()
    except (sqlite3.Error, ValueError):
        return False
    return (
        values.get("index_hash") == index_hash(index)
        and values.get("policy_version") == V41_POLICY_VERSION
    )


def lexical_rank(
    sidecar: QuerySidecarV41, terms: Iterable[str], limit: int = 120,
) -> list[tuple[str, float]]:
    query_terms = [_key(term) for term in terms if _key(term)]
    scores: defaultdict[str, float] = defaultdict(float)
    for document in sidecar.documents.values():
        text = _key(document.text)
        document_tokens = set(text.split())
        document_score = 0.0
        for term in query_terms:
            pieces = term.split()
            if term in text:
                document_score += 2.0 + 0.25 * len(pieces)
            document_score += 0.35 * len(set(pieces) & document_tokens)
        # Zero-overlap documents must not become candidates merely because a
        # defaultdict entry was touched.  Their former node-ID tie ordering
        # acted like random retrieval noise on large memories.
        if document_score > 0.0:
            scores[document.node_id] = document_score
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]


def inverted_rank(
    sidecar: QuerySidecarV41,
    lookups: dict[str, Iterable[str]],
    limit: int = 120,
) -> list[tuple[str, float]]:
    scores: defaultdict[str, float] = defaultdict(float)
    for field, values in lookups.items():
        rows = sidecar.inverted.get(field, {})
        for raw in values:
            value = _key(raw)
            if not value:
                continue
            for indexed, node_ids in rows.items():
                if value == indexed:
                    weight = 4.0
                elif value in indexed or indexed in value:
                    weight = 1.5
                else:
                    continue
                for node_id in node_ids:
                    scores[node_id] += weight
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
