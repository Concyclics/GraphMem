from graphmem_demo.v3.scalar_comparison import scalar_comparison_hint
from graphmem_demo.v3.schema import QueryFrame, TurnNode


def _frame(question: str) -> QueryFrame:
    return QueryFrame(
        raw_question=question,
        content_terms=[],
        participant_terms=[],
        temporal_terms=[],
        explicit_dates=[],
        requested_operation="lookup",
        answer_form="number",
        hypotheses=[],
    )


def _turn(node_id: str, speaker: str, text: str) -> TurnNode:
    return TurnNode(
        node_id=node_id,
        question_id="q",
        session_id="s",
        session_date=None,
        turn_index=0,
        speaker=speaker,
        speaker_key=speaker,
        listener="",
        transport_role="user",
        text=text,
        retrieval_text=text,
    )


def test_entity_bound_age_difference_uses_two_lossless_sources() -> None:
    turns = [
        _turn("t1", "Mira", "We celebrated Mira\x27s aunt\x27s 68th birthday."),
        _turn("t2", "Mira", "Do you think 29 is young or old?"),
    ]
    hint = scalar_comparison_hint(
        _frame("How many years older is Mira\x27s aunt than Mira?"), turns
    )
    assert hint is not None
    assert hint["value"] == 39
    assert hint["source_turn_ids"] == ["t1", "t2"]


def test_first_person_deixis_survives_canonical_stopword_removal() -> None:
    turns = [
        _turn("t1", "participant 1", "My aunt\x27s 68th birthday was joyful."),
        _turn("t2", "participant 1", "Is 29 considered young or old?"),
    ]
    hint = scalar_comparison_hint(
        _frame("How many years older is my aunt than me?"), turns
    )
    assert hint is not None
    assert hint["value"] == 39


def test_unrelated_number_without_age_cue_is_not_an_operand() -> None:
    turns = [
        _turn("t1", "Mira", "Mira\x27s aunt\x27s 68th birthday is next week."),
        _turn("t2", "Mira", "I bought 29 paper lanterns."),
    ]
    assert scalar_comparison_hint(
        _frame("How many years older is Mira\x27s aunt than Mira?"), turns
    ) is None


def test_non_age_unit_is_not_claimed() -> None:
    turns = [
        _turn("t1", "Mira", "Mira\x27s aunt\x27s 68th birthday is next week."),
        _turn("t2", "Mira", "Is 29 considered young or old?"),
    ]
    assert scalar_comparison_hint(
        _frame("How many months older is Mira\x27s aunt than Mira?"), turns
    ) is None
