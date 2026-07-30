from graphmem_demo.v3.session_llm_navigation import (
    parse_session_navigation_result,
    parse_session_selection,
    session_navigation_messages,
)
from graphmem_demo.v3.structured_navigation import build_query_ir


def test_session_selector_only_accepts_listed_session_ids() -> None:
    selected, missing, parse_error = parse_session_selection(
        '{"selected_session_ids":["session_2","invented"],"missing_slots":["end"]}',
        ["session_1", "session_2"],
    )
    assert selected == ("session_2",)
    assert missing == ("end",)
    assert parse_error is False


def test_session_cards_are_bounded_and_use_query_ir() -> None:
    messages, valid = session_navigation_messages(
        question="How many places did Alex visit?",
        question_date="2026-07-28",
        query_ir=build_query_ir("How many places did Alex visit?"),
        session_rows=[
            {
                "session_id": "session_1",
                "session_date": "2026-07-20",
                "text": "Alex visited Lisbon.",
            },
            {
                "session_id": "session_2",
                "session_date": "2026-07-21",
                "text": "Alex visited Porto.",
            },
        ],
    )
    assert valid == ["session_1", "session_2"]
    assert "collection_closure" in messages[1]["content"]
    assert "gold" not in messages[1]["content"].casefold()


def test_session_selector_parses_auditable_slots_and_candidate() -> None:
    parsed = parse_session_navigation_result(
        '{"selected_session_ids":["session_2"],"missing_slots":[],'
        '"resolved_slots":{"old":"250","new":"350"},'
        '"candidate_answer":"100","confidence":0.91}',
        ["session_1", "session_2"],
    )
    assert parsed.selected_session_ids == ("session_2",)
    assert dict(parsed.resolved_slots) == {"old": "250", "new": "350"}
    assert parsed.candidate_answer == "100"
    assert parsed.confidence == 0.91
    assert parsed.parse_error is False
