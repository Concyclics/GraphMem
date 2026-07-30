from graphmem_demo.v3.compact_packing import _focused_text
from graphmem_demo.v3.retrieval import build_query_frame


def test_what_year_is_classified_as_date_lookup() -> None:
    frame = build_query_frame(
        "What year did the installation of the ventilation system begin?"
    )
    assert frame.requested_operation == "date"
    assert frame.answer_form == "date"


def test_focused_text_prioritizes_rare_predicate_with_answer_value() -> None:
    repeated_header = (
        "North Harbor renovation report discusses the construction process. "
    )
    text = (
        repeated_header * 8
        + "The installation of the ventilation system began in 2018, "
        + "and its maintenance contract was signed in 2019. "
        + repeated_header * 8
    )
    frame = build_query_frame(
        "What year did the installation of the ventilation system begin?"
    )
    focused = _focused_text(text, frame, 420)
    assert "began in 2018" in focused
    assert focused.index("began in 2018") < 120
