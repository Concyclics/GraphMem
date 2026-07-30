from __future__ import annotations

from dataclasses import dataclass

import pytest

from graphmem_demo.clients import cosine_similarity
from graphmem_demo.v3.dense import dense_rank_many


@dataclass
class _Node:
    node_id: str
    embedding: list[float]


def test_dense_rank_many_matches_scalar_cosine_order() -> None:
    nodes = [
        _Node("x", [1.0, 0.0]),
        _Node("y", [0.5, 0.5]),
        _Node("z", [0.0, 1.0]),
    ]
    queries = [[1.0, 0.0], [0.0, 1.0]]
    rankings, primary = dense_rank_many(queries, nodes)
    assert rankings[0] == ["x", "y"]
    assert rankings[1] == ["z", "y"]
    assert primary["y"] == pytest.approx(
        cosine_similarity(queries[0], nodes[1].embedding), abs=1e-6
    )


def test_dense_rank_many_handles_missing_embeddings() -> None:
    rankings, primary = dense_rank_many([[1.0, 0.0]], [])
    assert rankings == [[]]
    assert primary == {}
