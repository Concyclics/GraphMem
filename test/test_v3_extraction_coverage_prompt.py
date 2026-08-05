from graphmem_demo.models import QuestionCase
from graphmem_demo.v3.build import build_turn_nodes, session_extraction_messages


def test_extraction_prompt_prioritizes_source_turn_coverage_over_long_lists() -> None:
    case = QuestionCase(
        question_id="q",
        question="unused",
        question_type="multi-session",
        question_date="2026-01-02",
        answer="unused",
        haystack_session_ids=["s"],
        haystack_dates=["2026-01-01"],
        haystack_sessions=[[{"role": "user", "content": "I own a transit pass."}]],
        answer_session_ids=[],
    )
    turns = build_turn_nodes(case)
    prompt = session_extraction_messages("s", "2026-01-01", turns)[0]["content"]
    assert "cover every source turn" in prompt
    assert "at most three representative claims" in prompt
    assert "must never crowd out later participant-authored memory" in prompt
    assert "complete lossless turn" in prompt
