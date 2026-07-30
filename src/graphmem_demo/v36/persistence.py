from __future__ import annotations

import json
import hashlib
import sqlite3
import threading
from pathlib import Path

import numpy as np

from .schema import V36Index


_SQLITE_LOCK = threading.Lock()


def persist_retrieval_index(path: Path, index: V36Index) -> None:
    """Persist searchable text and structured filters without duplicating graph nodes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _SQLITE_LOCK:
        connection = sqlite3.connect(path, timeout=60)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS routing_fts USING fts5(
                    node_id UNINDEXED, question_id UNINDEXED,
                    session_id UNINDEXED, routing_text
                )
            """)
            connection.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    node_id UNINDEXED, question_id UNINDEXED,
                    node_type UNINDEXED, session_ids UNINDEXED, retrieval_text
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS structured_index(
                    question_id TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    field_value TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    PRIMARY KEY(question_id, field_name, field_value, node_id)
                )
            """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS structured_lookup
                ON structured_index(question_id, field_name, field_value)
            """)
            question_id = index.turns[0].question_id if index.turns else ""
            connection.execute(
                "DELETE FROM routing_fts WHERE question_id = ?", (question_id,)
            )
            connection.execute(
                "DELETE FROM memory_fts WHERE question_id = ?", (question_id,)
            )
            connection.execute(
                "DELETE FROM structured_index WHERE question_id = ?",
                (question_id,),
            )
            connection.executemany(
                "INSERT INTO routing_fts VALUES(?,?,?,?)",
                [
                    (
                        card.card_id, card.question_id, card.session_id,
                        card.routing_text,
                    )
                    for card in index.routing_cards
                ],
            )
            rows = []
            rows.extend(
                (
                    frame.frame_id, frame.question_id, "role_frame",
                    json.dumps(frame.session_ids), frame.retrieval_text,
                )
                for frame in index.frames
            )
            rows.extend(
                (
                    group.group_id, group.question_id, "evidence_group",
                    json.dumps(group.session_ids), group.retrieval_text,
                )
                for group in index.evidence_groups
            )
            rows.extend(
                (
                    turn.node_id, turn.question_id, "turn",
                    json.dumps([turn.session_id]), turn.retrieval_text,
                )
                for turn in index.turns
            )
            connection.executemany(
                "INSERT INTO memory_fts VALUES(?,?,?,?,?)", rows
            )
            structured = [
                (question_id, field_name, field_value, node_id)
                for field_name, values in index.inverted_indexes.items()
                for field_value, node_ids in values.items()
                for node_id in node_ids
            ]
            connection.executemany(
                "INSERT OR REPLACE INTO structured_index VALUES(?,?,?,?)",
                structured,
            )
            connection.commit()
        finally:
            connection.close()


def persist_vector_matrix(directory: Path, index: V36Index) -> tuple[Path, Path]:
    """Write one non-duplicated, memory-mappable vector row per searchable node."""
    nodes = [*index.routing_cards, *index.frames, *index.evidence_groups, *index.turns]
    rows = [(node.node_id, node.embedding) for node in nodes if node.embedding is not None]
    question_id = index.turns[0].question_id if index.turns else "empty"
    key = hashlib.sha256(question_id.encode("utf-8")).hexdigest()[:20]
    directory.mkdir(parents=True, exist_ok=True)
    matrix_path = directory / f"{key}.npy"
    ids_path = directory / f"{key}.ids.json"
    if not rows:
        np.save(matrix_path, np.empty((0, 0), dtype=np.float32))
        ids_path.write_text("[]", encoding="utf-8")
        return matrix_path, ids_path
    dimensions = {len(vector or []) for _node_id, vector in rows}
    if len(dimensions) != 1:
        raise ValueError(f"inconsistent V3.6 embedding dimensions: {dimensions}")
    matrix = np.asarray([vector for _node_id, vector in rows], dtype=np.float32)
    np.save(matrix_path, matrix, allow_pickle=False)
    ids_path.write_text(json.dumps([node_id for node_id, _vector in rows], ensure_ascii=True), encoding="utf-8")
    np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    return matrix_path, ids_path
