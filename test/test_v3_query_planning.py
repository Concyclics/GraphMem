from __future__ import annotations

from graphmem_demo.v3.query_planning import query_views
from graphmem_demo.v3.retrieval import build_query_frame


def test_query_views_are_bounded_and_topic_derived() -> None:
    frame = build_query_frame(
        "What is the order of the three workshops I attended, earliest to latest?"
    )
    views = query_views(frame)
    assert len(views) == 3
    assert views[0] == frame.raw_question
    assert "workshop" in views[1]
    assert "completed attended joined events dates chronological" in views[2]


def test_query_views_do_not_add_benchmark_topic_names() -> None:
    frame = build_query_frame("How many ceramic pieces did I receive?")
    joined = " ".join(query_views(frame)).casefold()
    assert "ceramic" in joined
    assert "participant acquired owned received" in joined


def test_count_query_adds_the_grammatical_target_as_a_dense_view() -> None:
    frame = build_query_frame(
        "How many pieces of equipment did I buy, assemble, or repair?"
    )
    assert query_views(frame)[-1] == "pieces of equipment"


def test_which_calendar_unit_requests_a_date_not_a_collection() -> None:
    frame = build_query_frame("Which week did Alex attend the conference?")
    assert frame.requested_operation == "date"
    assert frame.answer_form == "date"


def test_frequency_question_uses_recurrence_operation() -> None:
    frame = build_query_frame("How often does Alex inspect the pump?")
    assert frame.requested_operation == "recurrence"
    assert frame.answer_form == "frequency"
    assert "temporal_scope" in frame.hypotheses
    assert "collection_scope" in frame.hypotheses


def test_relational_before_question_uses_ordering_operation() -> None:
    frame = build_query_frame("Which station was Alex at before traveling to Northport?")
    assert frame.requested_operation == "ordering"
    assert frame.answer_form == "entity"
    assert "before" in frame.temporal_terms


def test_what_did_do_before_remains_an_event_ordering_question() -> None:
    frame = build_query_frame("What did Alex do before visiting Northport?")
    assert frame.requested_operation == "ordering"


def test_first_person_object_before_anchor_is_relational_ordering() -> None:
    frame = build_query_frame(
        "What new household device did I purchase before getting the countertop oven?"
    )
    assert frame.requested_operation == "ordering"
    assert frame.answer_form == "entity"


def test_before_after_modifiers_do_not_override_the_requested_answer_slot() -> None:
    expected = {
        "What did Evan share after their hiking trip?": "lookup",
        "What activity does Deborah do after a morning jog?": "lookup",
        "When did Melanie hike after the road trip?": "date",
        "What is the puppy's name two weeks before August 11, 2023?": "lookup",
        "What are some problems Alex faced before adopting the dog?": "list",
        "How does Evan spend time after the wedding?": "lookup",
    }
    for question, operation in expected.items():
        frame = build_query_frame(question)
        assert frame.requested_operation == operation
        assert "temporal_scope" in frame.hypotheses


def test_possessive_participant_is_canonicalized_to_the_speaker_key() -> None:
    frame = build_query_frame(
        "What book did Melanie read from Caroline's suggestion?"
    )
    assert frame.participant_terms == ["melanie", "caroline"]


def test_subordinate_when_scopes_a_lookup_without_changing_its_answer_slot() -> None:
    frame = build_query_frame(
        "What does Nate want to do when he goes over to Joanna's place?"
    )
    assert frame.requested_operation == "lookup"
    assert frame.answer_form == "span"
