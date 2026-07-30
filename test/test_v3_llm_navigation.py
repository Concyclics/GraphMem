import json

from graphmem_demo.clients import rough_token_count
from graphmem_demo.v3.llm_navigation import (
    NavigationPlan,
    compact_proposal_messages,
    deterministic_navigation_plan,
    focused_snippet,
    is_aggregate_navigation_operation,
    navigation_messages,
    parse_navigation_plan,
    session_diverse_recovery_seeds,
    selected_evidence_text,
    verification_messages,
)
from graphmem_demo.v3.structured_navigation import build_query_ir


def _row(node_id: str, kind: str, source: str, text: str, score: float = 0.5):
    return {
        "node_id": node_id,
        "node_type": kind,
        "selection_source": source,
        "text": text,
        "score": score,
    }


def test_navigation_frontier_is_bounded_and_does_not_use_gold_metadata() -> None:
    ledger = [
        _row(
            f"n:{index}",
            "turn" if index % 2 else "operand",
            "focused_provenance_expansion" if index % 3 else "protected_catalog",
            f"The package event number {index} has a long descriptive value " * 20,
        )
        for index in range(80)
    ]
    messages, ids = navigation_messages(
        question="How long did the package take to arrive?",
        question_date="2026-07-27",
        evidence_ledger=ledger,
        max_prompt_rough_tokens=700,
    )
    rendered = "\n".join(message["content"] for message in messages)
    assert rough_token_count(rendered) <= 700
    assert ids
    assert "gold" not in rendered.casefold()
    assert "question type" not in rendered.casefold()


def test_compact_proposal_is_bounded_and_uses_only_current_graph_cards() -> None:
    ledger = [
        {
            **_row(
                f"q:s{index % 5}:turn:{index}",
                "turn",
                "routed_lossless_session",
                f"Taylor changed item {index} from inactive to active. " * 12,
            ),
            "session_id": f"s{index % 5}",
            "session_date": f"2026-07-{(index % 20) + 1:02d}",
        }
        for index in range(50)
    ]
    messages, ids = compact_proposal_messages(
        question="Which items did Taylor change to active?",
        question_date="2026-07-27",
        evidence_ledger=ledger,
        query_ir=build_query_ir("Which items did Taylor change to active?"),
        max_prompt_rough_tokens=900,
    )
    rendered = "\n".join(message["content"] for message in messages)
    assert ids
    assert rough_token_count(rendered) <= 900
    assert "candidate_answer" in rendered
    assert "owner or speaker" in rendered
    assert "gold" not in rendered.casefold()
    assert "benchmark" not in rendered.casefold()


def test_navigation_plan_rejects_invented_node_ids() -> None:
    plan = parse_navigation_plan(
        '{"selected_ids":["valid", "invented", "valid"],'
        '"operation":"duration", "needed_relations":["before"],'
        '"missing_slots":[]}',
        ["valid"],
    )
    assert plan.selected_ids == ("valid",)
    assert plan.operation == "duration"
    assert not plan.parse_error


def test_navigation_plan_accepts_structured_id_objects() -> None:
    plan = parse_navigation_plan(
        '{"selected_ids":[{"ID":"valid"},{"node_id":"also"}],'
        '"operation":"lookup","needed_relations":[],"missing_slots":[]}',
        ["valid", "also"],
    )
    assert plan.selected_ids == ("valid", "also")
    assert not plan.parse_error


def test_navigation_plan_keeps_twelve_distinct_aggregate_operands() -> None:
    valid = [f"n:{index}" for index in range(14)]
    payload = json.dumps({
        "selected_ids": valid,
        "operation": "sum distinct operands",
        "needed_relations": [],
        "missing_slots": [],
    })
    plan = parse_navigation_plan(payload, valid)
    assert plan.selected_ids == tuple(valid[:12])


def test_aggregate_navigation_adds_bounded_session_diverse_seeds() -> None:
    plan = NavigationPlan(("q:session_1:turn:0",), "count unique entities", (), ())
    ledger = [
        _row("q:session_1:turn:1", "turn", "protected_direct", "one"),
        _row("q:session_2:turn:0", "turn", "protected_direct", "two"),
        _row("q:session_2:turn:1", "turn", "protected_direct", "duplicate"),
        _row("q:session_3:event:0", "event", "protected_direct", "three"),
    ]
    seeds = session_diverse_recovery_seeds(ledger, plan, max_extra=2)
    assert seeds == (
        "q:session_1:turn:0",
        "q:session_2:turn:0",
        "q:session_3:event:0",
    )
    assert is_aggregate_navigation_operation("aggregate_weekly_duration")
    assert not is_aggregate_navigation_operation("single fact lookup")


def test_adjacent_turn_closure_accepts_arbitrary_session_ids() -> None:
    question = "What destination did Alex choose?"
    ledger = [
        {
            **_row(
                "q:answer_alpha:turn:0",
                "turn",
                "routed_lossless_session",
                "Alex asked about choosing a destination.",
            ),
            "session_id": "answer_alpha",
        },
        {
            **_row(
                "q:answer_alpha:turn:1",
                "turn",
                "routed_lossless_session",
                "Lisbon.",
            ),
            "session_id": "answer_alpha",
        },
    ]
    plan = deterministic_navigation_plan(
        question=question,
        evidence_ledger=ledger,
        query_ir=build_query_ir(question),
        max_selected=2,
        include_adjacent_context=True,
    )
    assert plan.selected_ids == (
        "q:answer_alpha:turn:0",
        "q:answer_alpha:turn:1",
    )


def test_focused_snippet_keeps_query_bearing_middle_span() -> None:
    text = (
        "Unrelated introduction. " * 80
        + "The remote shutter release arrived exactly five days after ordering. "
        + "Unrelated ending. " * 80
    )
    snippet = focused_snippet(
        text, "How many days did the remote shutter release take to arrive?",
        max_chars=500,
    )
    assert "five days" in snippet
    assert len(snippet) <= 500


def test_focused_snippet_preserves_scalar_table_row_in_long_turn() -> None:
    text = "\n".join(
        [
            "Campaign Plan",
            "Objective:",
            *[f"* General campaign tactic {index} with supporting details." for index in range(30)],
            "Budget:",
            "* Influencer marketing: $2,000",
            *[f"* Measurement method {index} with supporting details." for index in range(30)],
        ]
    )
    snippet = focused_snippet(
        text,
        "How much was allocated for influencer marketing?",
        max_chars=520,
    )
    assert "Influencer marketing: $2,000" in snippet
    assert len(snippet) <= 520


def test_scalar_key_value_beats_repeated_subject_prose() -> None:
    text = "\n".join(
        [
            "Operational Plan",
            *[
                f"* The distribution campaign description {index} discusses "
                "regional distribution campaign goals and partners."
                for index in range(24)
            ],
            "Budget:",
            "* Regional distribution: $4,250",
            *[
                f"* The distribution campaign measurement {index} discusses "
                "regional distribution campaign progress."
                for index in range(24)
            ],
        ]
    )
    snippet = focused_snippet(
        text,
        "How much was allocated for regional distribution in the campaign?",
        max_chars=520,
    )
    assert "Regional distribution: $4,250" in snippet


def test_selected_evidence_only_expands_navigator_closure() -> None:
    ledger = [
        _row("keep", "turn", "focused_provenance_expansion", "Relevant exact fact."),
        _row("drop", "turn", "protected_direct", "Distractor fact."),
    ]
    text = selected_evidence_text(
        question="What is the exact fact?",
        evidence_ledger=ledger,
        plan=NavigationPlan(("keep",), "lookup", (), ("source",)),
        max_rough_tokens=200,
    )
    assert "Relevant exact fact" in text
    assert "Distractor fact" not in text


def test_navigation_frontier_filters_explicit_year_conflicts_and_shows_date() -> None:
    ledger = [
        {
            **_row("right", "turn", "focused_provenance_expansion", "Nate had a career setback."),
            "session_date": "14 September 2022",
        },
        {
            **_row("wrong", "turn", "focused_provenance_expansion", "Nate had a career success."),
            "session_date": "14 September 2023",
        },
    ]
    messages, ids = navigation_messages(
        question="Was September 2022 good career-wise for Nate?",
        question_date="2024-01-01",
        evidence_ledger=ledger,
    )
    rendered = "\n".join(message["content"] for message in messages)
    assert "right" in ids
    assert "wrong" not in ids
    assert "observed_at=14 September 2022" in rendered


def test_verification_prompt_is_bounded_and_contains_no_gold_metadata() -> None:
    messages = verification_messages(
        question="Where did Ada find the missing parcel?",
        question_date="2026-07-27",
        draft_answer="At the station.",
        evidence_text=("Ada discussed an unrelated station. " * 1000)
        + "Ada found the missing parcel at the depot.",
        max_rough_tokens=500,
    )
    rendered = "\n".join(message["content"] for message in messages)
    assert rough_token_count(rendered) <= 700
    assert "gold" not in rendered.casefold()
