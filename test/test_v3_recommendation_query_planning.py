from graphmem_demo.v3.retrieval import build_query_frame


def test_can_you_suggest_is_a_recommendation_request() -> None:
    frame = build_query_frame(
        "Can you suggest some activities I can do during my commute to work?"
    )
    assert frame.requested_operation == "recommendation"
    assert frame.answer_form == "recommendation"


def test_prior_suggestion_and_recommendation_questions_are_memory_lookups() -> None:
    questions = (
        "What book did Melanie read from Caroline's suggestion?",
        "What book did Caroline recommend to Melanie?",
        "What kind of healthy food suggestions has Evan given to Sam?",
        "What did John suggest James practice before playing together?",
    )
    for question in questions:
        frame = build_query_frame(question)
        assert frame.requested_operation == "lookup"
        assert frame.answer_form == "span"
    temporal_lookup = build_query_frame(questions[-1])
    assert "temporal_scope" in temporal_lookup.hypotheses


def test_open_ended_advice_forms_remain_recommendation_requests() -> None:
    questions = (
        "Any tips for rearranging my room?",
        "What would be a good hobby for Tim to pick up?",
        "Which tool should Alex use to learn drawing?",
        "Where can Sam learn advanced woodworking?",
    )
    for question in questions:
        frame = build_query_frame(question)
        assert frame.requested_operation == "recommendation"
        assert frame.answer_form == "recommendation"
