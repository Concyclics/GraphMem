from graphmem_demo.v3.session_navigation import (
    dense_reranked_lossless_session_rows,
    routed_lossless_session_rows,
    scope_rows_to_sessions,
)
from graphmem_demo.v3.structured_navigation import build_query_ir


class _Store:
    def nodes_for(self, _question_id: str):
        return {
            "session_1:turn:0": {
                "node_id": "q:session_1:turn:0",
                "node_type": "turn",
                "session_id": "session_1",
                "turn_index": 0,
                "retrieval_text": "Alex asked about the destination.",
            },
            "session_1:turn:1": {
                "node_id": "q:session_1:turn:1",
                "node_type": "turn",
                "session_id": "session_1",
                "turn_index": 1,
                "retrieval_text": "The destination was Lisbon.",
            },
            "session_2:turn:0": {
                "node_id": "q:session_2:turn:0",
                "node_type": "turn",
                "session_id": "session_2",
                "turn_index": 0,
                "retrieval_text": "Unrelated memory.",
            },
        }


def test_coarse_sessions_open_query_turn_and_adjacent_reply() -> None:
    rows = routed_lossless_session_rows(
        graph_store=_Store(),
        question_id="q",
        retrieved_session_ids=["session_1", "session_2"],
        query_ir=build_query_ir("Where was Alex's destination?"),
        max_turns_per_session=2,
    )
    ids = [row["node_id"] for row in rows]
    assert "q:session_1:turn:0" in ids
    assert "q:session_1:turn:1" in ids
    assert all(row["selection_source"] == "routed_lossless_session" for row in rows)


def test_session_navigation_never_uses_unrouted_session() -> None:
    rows = routed_lossless_session_rows(
        graph_store=_Store(),
        question_id="q",
        retrieved_session_ids=["session_1"],
        query_ir=build_query_ir("What was the destination?"),
    )
    assert {row["session_id"] for row in rows} == {"session_1"}


def test_session_scope_keeps_only_nodes_wholly_owned_by_selected_sessions() -> None:
    rows = [
        {"node_id": "q:s1:turn:0", "session_id": "s1"},
        {"node_id": "q:s2:turn:0", "session_id": "s2"},
        {"node_id": "q:operand:1", "source_turn_ids": ["q:s1:turn:0"]},
        {
            "node_id": "q:theme:0",
            "source_turn_ids": ["q:s1:turn:0", "q:s2:turn:0"],
        },
    ]
    scoped = scope_rows_to_sessions(rows, ["s1"])
    assert [row["node_id"] for row in scoped] == [
        "q:s1:turn:0",
        "q:operand:1",
    ]


def test_dense_rerank_is_bounded_inside_coarse_selected_session() -> None:
    class Store:
        def nodes_for(self, _question_id: str):
            return {
                f"answer_alpha:turn:{index}": {
                    "node_id": f"q:answer_alpha:turn:{index}",
                    "node_type": "turn",
                    "session_id": "answer_alpha",
                    "turn_index": index,
                    "retrieval_text": text,
                }
                for index, text in enumerate((
                    "Generic dinner discussion.",
                    "I harvested basil and tomatoes from my garden.",
                    "Unrelated closing exchange.",
                ))
            }

    class Embedder:
        def embed(self, texts, **_kwargs):
            return [
                [1.0, 0.0]
                if index == 0 or "harvested" in text
                else [0.0, 1.0]
                for index, text in enumerate(texts)
            ]

    rows = dense_reranked_lossless_session_rows(
        graph_store=Store(),
        embedder=Embedder(),
        question_id="q",
        question="What should I make with homegrown ingredients?",
        retrieved_session_ids=["answer_alpha"],
        query_ir=build_query_ir(
            "What should I make with homegrown ingredients?"
        ),
        semantic_seeds_per_session=1,
        max_turns_per_session=1,
    )
    assert [row["node_id"] for row in rows] == ["q:answer_alpha:turn:1"]
    assert rows[0]["selection_source"] == "routed_lossless_dense_session"
