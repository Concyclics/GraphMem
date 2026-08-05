from graphmem_demo.clients import rough_token_count
from graphmem_demo.v3.compact_packing import pack_context, render_block
from graphmem_demo.v3.retrieval import build_query_frame
from graphmem_demo.v3.schema import TurnNode


def _turn(node_id: str, text: str) -> TurnNode:
    return TurnNode(
        node_id=node_id,
        question_id="q",
        session_id="s",
        session_date="2024-01-01",
        turn_index=0,
        speaker="Alex",
        speaker_key="alex",
        listener="",
        transport_role="user",
        text=text,
        retrieval_text=text,
    )


def test_focused_source_turn_survives_tight_pack_budget() -> None:
    frame = build_query_frame("What did Alex plan to make for the family?")
    source = _turn("s:turn:source", "I plan to make the recipe for my family.")
    distractor = _turn("s:turn:distractor", "Alex discussed a recipe yesterday.")
    budget = rough_token_count(render_block(frame, "turn", source))
    kept, context, trace = pack_context(
        frame,
        [
            ("turn", distractor, 1.0, "protected_direct"),
            ("turn", source, 0.1, "focused_provenance_expansion"),
        ],
        budget,
    )
    assert [node.node_id for _kind, node, _score, _source in kept] == [source.node_id]
    assert "for my family" in context
    assert next(row for row in trace if row["node_id"] == source.node_id)["decision"] == "keep"


def test_direct_evidence_precedes_relation_expansion_under_tight_budget() -> None:
    frame = build_query_frame("Where did Alex go during August 2023?")
    direct = _turn("s:turn:direct", "Alex went camping during August 2023.")
    expanded = _turn("s:turn:expanded", "Alex discussed a different trip during August 2023.")
    budget = rough_token_count(render_block(frame, "turn", direct))
    kept, _context, trace = pack_context(
        frame,
        [
            ("turn", expanded, 2.0, "relation_focus"),
            ("turn", direct, 0.1, "protected_direct"),
        ],
        budget,
    )
    assert [node.node_id for _kind, node, _score, _source in kept] == [direct.node_id]
    by_id = {row["node_id"]: row for row in trace}
    assert by_id[direct.node_id]["pack_priority"] > by_id[expanded.node_id]["pack_priority"]
