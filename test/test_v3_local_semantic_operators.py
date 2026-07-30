from __future__ import annotations

from graphmem_demo.v3.answer_hints import structured_section_hint
from graphmem_demo.v3.catalog_schema import OperandRecordV3
from graphmem_demo.v3.retrieval import build_query_frame
from graphmem_demo.v3.schema import TurnNode
from graphmem_demo.v3.semantic_operators import (
    final_choice_hint, frequency_state_comparison_hint, ordered_event_hint,
)


def _operand(
    operand_id: str,
    predicate: str,
    obj: str,
    day: str,
    *,
    semantic: float,
    event_types: list[str] | None = None,
) -> OperandRecordV3:
    item = OperandRecordV3(
        operand_id=operand_id,
        question_id="q",
        subject_key="participant",
        predicate_key=predicate,
        object_key=obj.casefold(),
        object_text=obj,
        event_time=day,
        observed_at=day,
        source_turn_ids=[f"turn:{operand_id}"],
        event_type_keys=list(event_types or []),
        retrieval_text=f"participant | {predicate} | {obj}",
    )
    item.embedding = [semantic]
    return item


def test_final_choice_prefers_accepted_state_over_initial_proposal() -> None:
    frame = build_query_frame("What did we finally decide to name it?")
    proposed = _operand("p", "proposed name", "Initial option", "2026-01-01", semantic=0.9)
    accepted = _operand("a", "liked name", "Accepted option", "2026-01-01", semantic=0.8)
    hint = final_choice_hint(
        frame,
        [proposed, accepted],
        semantic_similarity=lambda item: item.embedding[0],
    )
    assert hint is not None
    assert hint["value"] == "Accepted option"


def test_ordered_event_hint_orders_semantic_local_collection() -> None:
    frame = build_query_frame(
        "What is the order of the three athletic events, earliest to latest?"
    )
    rows = [
        _operand("late", "participated", "team tournament", "2026-03-03", semantic=0.8),
        _operand("early", "completed", "distance race", "2026-03-01", semantic=0.9),
        _operand("middle", "completed", "cycling trial", "2026-03-02", semantic=0.85),
        _operand("noise", "purchased", "office chair", "2026-03-04", semantic=0.1),
    ]
    hint = ordered_event_hint(
        frame,
        rows,
        semantic_similarity=lambda item: item.embedding[0],
        query_overlap=lambda _frame, _text: 0.0,
    )
    assert hint is not None
    assert [row["value"] for row in hint["values"]] == [
        "distance race", "cycling trial", "team tournament",
    ]


def test_open_ordered_collection_deduplicates_retrospective_aliases() -> None:
    frame = build_query_frame(
        "What is the order of the field inspections I completed in the past two months?"
    )
    rows = [
        _operand("early-open", "completed", "northern reservoir inspection", "2026-03-01", semantic=0.9),
        _operand("early-repeat", "completed", "northern reservoir survey", "2026-03-12", semantic=0.85),
        _operand("middle-open", "completed", "western bridge inspection", "2026-03-05", semantic=0.88),
        _operand("late-open", "completed", "southern tunnel inspection", "2026-03-09", semantic=0.86),
        _operand("noise-open", "completed", "tax return", "2026-03-10", semantic=0.05),
    ]
    hint = ordered_event_hint(
        frame, rows,
        semantic_similarity=lambda item: item.embedding[0],
        query_overlap=lambda _frame, text: float("inspection" in text),
    )
    assert hint is not None
    assert [row["value"] for row in hint["values"]] == [
        "northern reservoir inspection", "western bridge inspection",
        "southern tunnel inspection",
    ]


def test_open_ordered_collection_rejects_projected_type_mismatch() -> None:
    frame = build_query_frame(
        "What is the order of the concerts and musical events I attended "
        "in the past month?"
    )
    rows = [
        _operand(
            "concert", "attended", "symphony concert", "2026-03-01",
            semantic=0.8, event_types=["musical event", "concert"],
        ),
        _operand(
            "jazz", "attended", "jazz night", "2026-03-02",
            semantic=0.6, event_types=["musical event", "jazz performance"],
        ),
        _operand(
            "meeting", "attended", "planning meeting", "2026-03-03",
            semantic=0.99, event_types=["business event", "meeting"],
        ),
    ]
    hint = ordered_event_hint(
        frame, rows,
        semantic_similarity=lambda item: item.embedding[0],
        object_semantic_similarity=lambda item: item.embedding[0],
        query_overlap=lambda _frame, _text: 0.0,
    )
    assert hint is not None
    assert [row["value"] for row in hint["values"]] == [
        "symphony concert", "jazz night",
    ]


def test_structured_section_hint_reads_requested_ordinal_artifact() -> None:
    frame = build_query_frame(
        "What was the value for the chorus in the second result?"
    )
    turns = []
    for index, value in enumerate(("A B C", "D E F")):
        turns.append(TurnNode(
            node_id=f"t{index}",
            question_id="q",
            session_id="s",
            session_date="2026-01-01",
            turn_index=index,
            speaker="B",
            speaker_key="b",
            listener="A",
            transport_role="assistant",
            text=f"Result {index + 1}\nChorus:\n{value}\nExplanation",
            retrieval_text=f"result chorus {value}",
        ))
    hint = structured_section_hint(
        frame,
        [("turn", turn, 1.0, "test") for turn in turns],
        query_overlap=lambda _frame, _text: 1.0,
    )
    assert hint is not None
    assert hint["value"] == "D E F"


def test_final_choice_accepts_liked_explicit_name_projection() -> None:
    frame = build_query_frame("What did we finally decide to name it?")
    proposed = _operand("p2", "proposed", "name First label", "2026-01-01", semantic=0.9)
    accepted = _operand("a2", "liked", "name \x27Stable label\x27", "2026-01-01", semantic=0.8)
    hint = final_choice_hint(
        frame,
        [proposed, accepted],
        semantic_similarity=lambda item: item.embedding[0],
    )
    assert hint is not None
    assert hint["value"] == "Stable label"


def test_frequency_state_comparison_uses_dated_rates_for_same_activity() -> None:
    frame = build_query_frame("Do I practice more frequently than I did previously?")
    older = _operand("old-rate", "practice days", "Tuesday, Thursday", "2026-01-01", semantic=0.8)
    newer = _operand("new-rate", "practice routine", "three times a week", "2026-02-01", semantic=0.8)
    noise = _operand("noise-rate", "swim routine", "seven times a week", "2026-03-01", semantic=0.2)
    hint = frequency_state_comparison_hint(
        frame, [older, newer, noise],
        query_overlap=lambda _frame, text: float("practice" in text),
    )
    assert hint is not None
    assert hint["operation"] == "frequency_state_comparison"
    assert hint["value"] == "yes"
    assert hint["previous_rate_per_week"] == 2
    assert hint["current_rate_per_week"] == 3


def test_frequency_comparison_closes_typed_state_over_user_source() -> None:
    frame = build_query_frame(
        "Do I use the studio more frequently than I did previously?"
    )
    older = _operand(
        "old-source-rate", "training days", "Tuesday, Thursday, Saturday",
        "2026-01-01", semantic=0.8,
    )
    newer = _operand(
        "new-source-rate", "exercise routine", "four times a week",
        "2026-02-01", semantic=0.8,
    )
    assistant_noise = _operand(
        "assistant-rate", "studio recommendation", "seven times a week",
        "2026-03-01", semantic=0.9,
    )
    turns = [
        TurnNode(
            node_id="turn:old-source-rate", question_id="q", session_id="s1",
            session_date="2026-01-01", turn_index=0, speaker="participant_1",
            speaker_key="participant 1", listener="", transport_role="user",
            text="I use the studio on Tuesdays, Thursdays, and Saturdays.",
            retrieval_text="studio schedule",
        ),
        TurnNode(
            node_id="turn:new-source-rate", question_id="q", session_id="s2",
            session_date="2026-02-01", turn_index=0, speaker="participant_1",
            speaker_key="participant 1", listener="", transport_role="user",
            text="My studio routine is now four times a week.",
            retrieval_text="studio routine",
        ),
        TurnNode(
            node_id="turn:assistant-rate", question_id="q", session_id="s3",
            session_date="2026-03-01", turn_index=0, speaker="participant_2",
            speaker_key="participant 2", listener="", transport_role="assistant",
            text="I recommend using the studio seven times a week.",
            retrieval_text="studio recommendation",
        ),
    ]
    hint = frequency_state_comparison_hint(
        frame, [older, newer, assistant_noise], turns=turns,
        query_overlap=lambda _frame, text: float("studio" in text.casefold()),
    )
    assert hint is not None
    assert hint["value"] == "yes"
    assert hint["previous_rate_per_week"] == 3
    assert hint["current_rate_per_week"] == 4
    assert hint["operand_ids"] == ["old-source-rate", "new-source-rate"]
    assert hint["completion_basis"] == "typed_state_plus_lossless_source_comparison"
