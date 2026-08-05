from __future__ import annotations

from typing import Any

import numpy as np


def dense_rank_many(
    query_vectors: list[list[float]],
    nodes: list[Any],
) -> tuple[list[list[str]], dict[str, float]]:
    """Rank several query views with one normalized matrix multiplication."""
    if not query_vectors:
        return [], {}
    rows: list[tuple[str, list[float]]] = []
    for node in nodes:
        node_id = getattr(node, "node_id", getattr(node, "edge_id", ""))
        embedding = getattr(node, "embedding", None)
        if node_id and embedding:
            rows.append((node_id, embedding))
    if not rows:
        return [[] for _ in query_vectors], {}

    dimension = len(rows[0][1])
    rows = [row for row in rows if len(row[1]) == dimension]
    queries = [vector for vector in query_vectors if len(vector) == dimension]
    if not rows or len(queries) != len(query_vectors):
        return [[] for _ in query_vectors], {}

    matrix = np.asarray([embedding for _node_id, embedding in rows], dtype=np.float32)
    query_matrix = np.asarray(queries, dtype=np.float32)
    matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    query_norms = np.linalg.norm(query_matrix, axis=1, keepdims=True)
    matrix /= np.maximum(matrix_norms, np.finfo(np.float32).eps)
    query_matrix /= np.maximum(query_norms, np.finfo(np.float32).eps)
    scores = matrix @ query_matrix.T

    node_ids = [node_id for node_id, _embedding in rows]
    rankings: list[list[str]] = []
    for column in range(scores.shape[1]):
        order = np.argsort(-scores[:, column], kind="stable")
        rankings.append([
            node_ids[int(index)]
            for index in order
            if float(scores[int(index), column]) > 0.0
        ])
    primary_scores = {
        node_id: float(scores[index, 0])
        for index, node_id in enumerate(node_ids)
    }
    return rankings, primary_scores
