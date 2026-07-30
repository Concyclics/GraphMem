from __future__ import annotations

from types import SimpleNamespace

from graphmem_demo.v3.catalog_arithmetic import event_occurrence_count
from graphmem_demo.v3.catalog_schema import OperandRecordV3
from graphmem_demo.v3.ordinal_event import ordinal_event_hint
from graphmem_demo.v3.quantified_relation import all_subjects_relation_hint
from graphmem_demo.v3.query_planning import answer_slot_phrase, query_views
from graphmem_demo.v3.relation_focus import relation_focus_turn_ids
from graphmem_demo.v3.media_relation import media_attribute_hint
from graphmem_demo.v3.retrieval import build_query_frame
from scripts.convert_locomo10 import convert_sample


def _operand(
    node_id: str,
    subject: str,
    predicate: str,
    value: str,
    source: str,
    *,
    modality: str = "asserted",
    observed_at: str = "2023-01-01",
) -> OperandRecordV3:
    return OperandRecordV3(
        operand_id=node_id,
        question_id="q",
        subject_key=subject,
        predicate_key=predicate,
        object_key=value.casefold(),
        object_text=value,
        polarity="positive",
        modality=modality,
        observed_at=observed_at,
        event_time=observed_at,
        source_turn_ids=[source],
        session_ids=[source.split(":")[0]],
        confidence=0.9,
        retrieval_text=f"{subject} {predicate} {value}",
    )


def test_all_quantifier_requires_independent_positive_proof_per_subject() -> None:
    frame = build_query_frame(
        "Did Alex and Morgan both participate in robotics competitions?"
    )
    operands = [
        _operand("o1", "alex", "won", "local robotics competition", "s1:t1"),
        _operand(
            "o2", "morgan", "competed", "regional robotics competition", "s2:t1"
        ),
    ]
    hint = all_subjects_relation_hint(frame, operands)
    assert hint is not None
    assert hint["value"] == "yes"
    assert set(hint["proofs"]) == {"alex", "morgan"}
    assert len(hint["source_turn_ids"]) == 2


def test_all_quantifier_does_not_treat_missing_evidence_as_false() -> None:
    frame = build_query_frame(
        "Did Alex and Morgan both participate in robotics competitions?"
    )
    operands = [
        _operand("o1", "alex", "won", "local robotics competition", "s1:t1")
    ]
    assert all_subjects_relation_hint(frame, operands) is None


def test_plan_count_is_not_overridden_by_completed_event_counter() -> None:
    frame = build_query_frame(
        "How many times did Alex and Morgan plan to hike together?"
    )
    operands = [
        _operand("o1", "alex", "went", "hiking with Morgan", "s1:t1"),
        _operand("o2", "alex", "planned", "hike with Morgan", "s2:t1"),
    ]
    assert event_occurrence_count(
        frame,
        operands,
        event_frames=[],
        query_overlap=lambda _frame, _text: 1.0,
        turns=[],
    ) is None


def test_query_views_include_topic_agnostic_answer_type_slot() -> None:
    frame = build_query_frame("What kitchen appliance did I buy ten days ago?")
    assert answer_slot_phrase(frame.raw_question) == "kitchen appliance"
    assert query_views(frame)[-1] == "kitchen appliance"


def test_natural_language_date_is_normalized_in_query_frame() -> None:
    frame = build_query_frame("Where was Alex on July 12, 2022?")
    assert "2022-07-12" in frame.explicit_dates
    assert frame.participant_terms == ["alex"]


def test_relation_focus_preserves_adjacent_dialogue_answer() -> None:
    frame = build_query_frame(
        "What game was the second tournament that Alex won based on?"
    )
    turns = [
        SimpleNamespace(
            node_id="s1:t1",
            session_id="s1",
            session_date="2023-01-01",
            turn_index=1,
            speaker="Alex",
            speaker_key="alex",
            text="I won my second tournament last week.",
        ),
        SimpleNamespace(
            node_id="s1:t2",
            session_id="s1",
            session_date="2023-01-01",
            turn_index=2,
            speaker="Morgan",
            speaker_key="morgan",
            text="What game was it?",
        ),
        SimpleNamespace(
            node_id="s1:t3",
            session_id="s1",
            session_date="2023-01-01",
            turn_index=3,
            speaker="Alex",
            speaker_key="alex",
            text="I usually play another game, but this time it was Star Arena.",
        ),
    ]
    focused = relation_focus_turn_ids(frame, turns)
    assert {"s1:t1", "s1:t2", "s1:t3"}.issubset(focused)


def test_ordinal_generic_projection_degrades_to_dialogue_window() -> None:
    frame = build_query_frame(
        "What game was the second tournament that Alex won based on?"
    )
    operands = [
        _operand(
            "o1", "alex", "won", "first tournament", "s1:t1",
            observed_at="2023-01-01",
        ),
        _operand(
            "o2", "alex", "won", "second tournament", "s2:t1",
            observed_at="2023-02-01",
        ),
    ]
    turns = [
        SimpleNamespace(
            node_id="s2:t1", session_id="s2", turn_index=1,
            text="I won my second tournament.",
        ),
        SimpleNamespace(
            node_id="s2:t2", session_id="s2", turn_index=2,
            text="Which game was it?",
        ),
        SimpleNamespace(
            node_id="s2:t3", session_id="s2", turn_index=3,
            text="Usually I play something else, but this time it was Star Arena.",
        ),
    ]
    hint = ordinal_event_hint(frame, operands, turns)
    assert hint is not None
    assert hint["complete"] is False
    assert hint["attribute_resolution"] == "dialogue_window_required"
    assert "s2:t3" in hint["source_turn_ids"]
    assert "Star Arena" in hint["evidence"]


def test_month_year_is_normalized_without_becoming_participant() -> None:
    frame = build_query_frame("What event did Morgan attend in June 2023?")
    assert "2023-06" in frame.explicit_dates
    assert frame.participant_terms == ["morgan"]


def test_relation_focus_matches_natural_session_date_to_iso_query_date() -> None:
    frame = build_query_frame("What did Alex share on 19 August, 2023?")
    turns = [
        SimpleNamespace(
            node_id="s1:t1", session_id="s1",
            session_date="6:17 pm on 19 August, 2023", turn_index=1,
            speaker="Alex", speaker_key="alex", text="I shared a photo.",
        ),
        SimpleNamespace(
            node_id="s2:t1", session_id="s2",
            session_date="6:17 pm on 18 August, 2023", turn_index=1,
            speaker="Alex", speaker_key="alex", text="I shared a different photo.",
        ),
    ]
    focused = relation_focus_turn_ids(frame, turns, anchor_limit=1, neighbor_radius=0)
    assert focused == ["s1:t1"]


def test_convert_locomo_preserves_image_caption_as_source_evidence() -> None:
    sample = {
        "sample_id": "conv-x",
        "conversation": {
            "speaker_a": "Alex", "speaker_b": "Morgan",
            "session_1_date_time": "1 pm on 1 January, 2024",
            "session_1": [{
                "speaker": "Alex", "text": "Look at this.", "dia_id": "D1:1",
                "blip_caption": "a red bicycle beside a tree",
                "img_url": ["https://example.invalid/image.jpg"],
            }],
        },
        "qa": [{
            "question": "What did Alex share?", "answer": "a bicycle",
            "evidence": ["D1:1"], "category": 4,
        }],
    }
    row = convert_sample(sample, 0)[0]
    turn = row["haystack_sessions"][0][0]
    assert turn["content"] == (
        "Look at this.\n[Media shared by Alex; caption: a red bicycle beside a tree]"
    )
    assert turn["media_captions"] == ["a red bicycle beside a tree"]
    assert turn["media_urls"] == ["https://example.invalid/image.jpg"]


def test_media_attribute_binds_enumerated_caption_by_speaker_and_date() -> None:
    frame = build_query_frame("What objects did Alex share a photo of on 19 August, 2023?")
    turns = [
        SimpleNamespace(
            node_id="s1:t1", session_date="6 pm on 19 August, 2023",
            text="[Media shared by Alex; caption: a photo of a compass, map, and lantern]",
        ),
        SimpleNamespace(
            node_id="s1:t2", session_date="6 pm on 19 August, 2023",
            text="[Media shared by Alex; caption: a photo of a table with supplies]",
        ),
        SimpleNamespace(
            node_id="s2:t1", session_date="6 pm on 20 August, 2023",
            text="[Media shared by Alex; caption: a photo of a rope, tent, and backpack]",
        ),
    ]
    hint = media_attribute_hint(frame, turns)
    assert hint is not None
    assert hint["complete"] is True
    assert hint["value"] == "a compass, map, and lantern"
    assert hint["source_turn_ids"][0] == "s1:t1"


def test_media_attribute_refuses_two_equally_enumerated_captions() -> None:
    frame = build_query_frame("What objects did Alex share a photo of?")
    turns = [
        SimpleNamespace(
            node_id="s1:t1", session_date="2023-01-01",
            text="[Media shared by Alex; caption: a photo of a compass, map, and lantern]",
        ),
        SimpleNamespace(
            node_id="s1:t2", session_date="2023-01-01",
            text="[Media shared by Alex; caption: a photo of a rope, tent, and backpack]",
        ),
    ]
    hint = media_attribute_hint(frame, turns)
    assert hint is not None
    assert hint["complete"] is False


def test_qualified_count_checks_source_cause_and_targeted_missing_turns() -> None:
    frame = build_query_frame(
        "How many workshops did I miss because of travel delays?"
    )
    operands = [
        _operand("o1", "alex", "missed", "ceramics workshop", "s1:t1"),
        _operand("o2", "alex", "missed", "writing workshop", "s2:t1"),
        _operand("o3", "alex", "missed", "music workshop", "s3:t1"),
    ]
    turns = [
        SimpleNamespace(
            node_id="s1:t1", speaker_key="alex",
            text="Travel delays made me miss the ceramics workshop.",
        ),
        SimpleNamespace(
            node_id="s2:t1", speaker_key="alex",
            text="I missed the writing workshop because of travel delays.",
        ),
        SimpleNamespace(
            node_id="s3:t1", speaker_key="alex",
            text="I missed the music workshop because I was ill.",
        ),
        SimpleNamespace(
            node_id="s4:t1", speaker_key="alex",
            text="I went through the checklist because of travel delays.",
        ),
    ]
    hint = event_occurrence_count(
        frame, operands, event_frames=[],
        query_overlap=lambda _frame, _text: 1.0, turns=turns,
    )
    assert hint is not None
    assert hint["value"] == 2
    assert hint["complete"] is True
    assert hint["missing_source_turn_ids"] == []


def test_relation_focus_respects_month_precision_when_available() -> None:
    frame = build_query_frame("Where did Alex go during August 2023?")
    turns = [
        SimpleNamespace(
            node_id="aug:t1", session_id="aug",
            session_date="11 am on 4 August, 2023", turn_index=1,
            speaker="Alex", speaker_key="alex", text="I went camping outdoors.",
        ),
        SimpleNamespace(
            node_id="nov:t1", session_id="nov",
            session_date="11 am on 4 November, 2023", turn_index=1,
            speaker="Alex", speaker_key="alex", text="I went camping outdoors.",
        ),
    ]
    focused = relation_focus_turn_ids(frame, turns, anchor_limit=1, neighbor_radius=0)
    assert focused == ["aug:t1"]
