from __future__ import annotations

import json

from graphmem_demo.v3.build import parse_session_extraction
from graphmem_demo.v3.schema import TurnNode
from graphmem_demo.v3.source_resolution import resolve_source_ids


def _turn(index: int) -> TurnNode:
    return TurnNode(
        f"q:s:turn:{index}", "q", "s", "2026-01-01", index,
        "A", "a", "B", "user", f"message {index}", f"A message {index}",
    )


def test_source_resolver_accepts_only_unique_local_turn_references() -> None:
    valid = {_turn(index).node_id for index in range(3)}
    assert resolve_source_ids(["1", "turn_2", "q:s:turn:0"], valid) == [
        "q:s:turn:1", "q:s:turn:2", "q:s:turn:0"
    ]
    assert resolve_source_ids(["99", "other-session:turn:1"], valid) == []


def test_parser_grounds_compact_numeric_source_indices() -> None:
    turns = [_turn(0), _turn(1)]
    payload = {
        "claims": [[
            "A", "completed", "a repair", "event", "positive", "asserted",
            "complete", "repair", "2026-01-01", ["1"], None, None, 0.9,
        ]],
        "events": [[
            "repair", "complete", "2026-01-01", ["A"], [0], ["turn:1"], 0.9
        ]],
        "episodes": [],
    }
    claims, events, _episodes, error = parse_session_extraction(
        json.dumps(payload),
        question_id="q",
        session_id="s",
        session_date="2026-01-01",
        turns=turns,
    )
    assert error is None
    assert claims[0].source_turn_ids == ["q:s:turn:1"]
    assert events[0].source_turn_ids == ["q:s:turn:1"]
