from __future__ import annotations

import re

from graphmem_demo.v3.retrieval import build_query_frame
from graphmem_demo.v3.schema import TurnNode
from graphmem_demo.v3.state_temporal_operators import relative_age_hint


def _turn(node_id: str, text: str) -> TurnNode:
    return TurnNode(
        node_id=node_id,
        question_id="q",
        session_id="s",
        session_date="2026-01-10",
        turn_index=0,
        speaker="A",
        speaker_key="a",
        listener="B",
        transport_role="user",
        text=text,
        retrieval_text=text,
    )


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def test_relative_expression_binds_lossless_source_by_event_terms() -> None:
    frame = build_query_frame("How long ago was the compressor overhaul?")
    distractor = _turn("noise", "I started training four years ago.")
    support = _turn(
        "support",
        "The commemorative plate was made for the compressor overhaul ten years ago.",
    )
    result = relative_age_hint(
        frame,
        [
            ("turn", distractor, 0.9, "test"),
            ("turn", support, 0.7, "test"),
        ],
        "2026-01-10",
        tokenize=_tokens,
        node_text=lambda node: node.text,
    )
    assert result is not None
    assert result["operation"] == "relative_age_from_evidence_expression"
    assert result["value"] == "10 years ago"
    assert result["supporting_node_ids"] == ["support"]


def test_relative_expression_respects_requested_unit() -> None:
    frame = build_query_frame("How many months ago was the calibration?")
    wrong_unit = _turn("years", "The calibration was two years ago.")
    right_unit = _turn("months", "The calibration was six months ago.")
    result = relative_age_hint(
        frame,
        [
            ("turn", wrong_unit, 0.9, "test"),
            ("turn", right_unit, 0.7, "test"),
        ],
        "2026-01-10",
        tokenize=_tokens,
        node_text=lambda node: node.text,
    )
    assert result is not None
    assert result["value"] == "6 months ago"
    assert result["supporting_node_ids"] == ["months"]
