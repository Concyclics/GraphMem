from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from graphmem.answer import (
    AnswerConfig, AnswerStage, PreparedAnswer, PROMPT_HASH, build_aggregation_ledger,
    build_answer_messages, compose, is_preference_synthesis_query,
    prompt_contract, render_evidence, render_turn, resolve_evidence_order,
)
from graphmem.domain import (
    AlgebraResult, AnswerMember, CandidateScore, Conversation, EvidenceMember, EvidenceUnit, NavigationResult,
    QueryBudget, QueryOperator, Session, SourceTurn, StateResult, TemporalEndpoint, TemporalKey,
    stable_id,
)
from graphmem.tokenization import HeuristicTokenCounter
from graphmem.storage import SQLiteGraphStore
from graphmem.answer.stage import (
    _aggregation_source_reserve_ids,
    _preference_focus_index,
    _query_focus_index,
)
from graphmem.answer.aggregation import selective_operand_worksheet_route


def _turn(index: int, text: str, session: str = "s1", speaker: str = "user") -> SourceTurn:
    return SourceTurn(stable_id("turn", "m", session, index), "m", session, index, speaker, "",
                      "user", "2023-05-01", text, hashlib.sha256(text.encode()).hexdigest())


COUNTER = HeuristicTokenCounter()


# --- rendering ----------------------------------------------------------------

def test_full_turn_rendering_carries_session_speaker_and_date() -> None:
    text = render_turn(_turn(0, "I adopted a beagle named Rex."), AnswerConfig())

    assert text == "[s1 @ 2023-05-01] user: I adopted a beagle named Rex."


def test_relative_memory_time_is_anchored_to_the_source_not_question_date() -> None:
    turn = _turn(0, "I joined the gym last week.")
    turn = replace(turn, timestamp="2023-06-16")

    text = render_turn(turn, AnswerConfig(normalize_relative_time=True))

    assert '[source-time "last week" => 2023-06-05..2023-06-11; anchor=2023-06-16]' in text


def test_source_time_annotation_can_be_disabled_for_a_frozen_ablation() -> None:
    turn = replace(_turn(0, "I joined the gym last week."), timestamp="2023-06-16")

    text = render_turn(turn, AnswerConfig(normalize_relative_time=False))

    assert "source-time" not in text


def test_span_window_zero_renders_only_the_cited_span() -> None:
    turn = _turn(0, "Small talk here. I adopted a beagle named Rex. More small talk.")
    spans = (EvidenceMember(turn.turn_id, 17, 46, "source"),)

    text = render_turn(turn, AnswerConfig(span_window=0), spans)

    assert "I adopted a beagle named Rex." in text
    assert "Small talk" not in text


def test_span_window_widens_by_characters_and_merges_overlaps() -> None:
    turn = _turn(0, "aaaa bbbb cccc dddd eeee")
    spans = (EvidenceMember(turn.turn_id, 5, 9, "source"),
             EvidenceMember(turn.turn_id, 10, 14, "source"))

    text = render_turn(turn, AnswerConfig(span_window=2), spans)

    # The two widened spans touch, so they must merge into one contiguous quote
    # rather than being rendered as two fragments joined by an ellipsis.
    assert "..." not in text
    assert "bbbb cccc" in text


def test_a_turn_with_no_span_falls_back_to_the_whole_turn() -> None:
    """Until the projection derives spans, span mode must not silently blank evidence."""
    turn = _turn(0, "I adopted a beagle named Rex.")

    assert "beagle" in render_turn(turn, AnswerConfig(span_window=0), ())


def test_rendering_is_ordered_by_session_then_turn_not_by_input_order() -> None:
    turns = [_turn(1, "second", "s2"), _turn(0, "first", "s1"), _turn(0, "third", "s2")]

    rendered = render_evidence(turns, config=AnswerConfig(), counter=COUNTER,
                               max_tokens=1000, session_order={"s1": 0, "s2": 1})

    assert [row.split(": ", 1)[1] for row in rendered.text.splitlines()] == \
        ["first", "third", "second"]


def test_relevance_rendering_preserves_packer_rank() -> None:
    turns = [_turn(1, "strongest", "s2"), _turn(0, "second", "s1"),
             _turn(0, "weakest", "s2")]

    rendered = render_evidence(
        turns, config=AnswerConfig(evidence_order="relevance"), counter=COUNTER,
        max_tokens=1000, session_order={"s1": 0, "s2": 1})

    assert [row.split(": ", 1)[1] for row in rendered.text.splitlines()] == \
        ["strongest", "second", "weakest"]


def test_invalid_evidence_order_is_rejected() -> None:
    with pytest.raises(ValueError, match="evidence_order"):
        AnswerConfig(evidence_order="random")


@pytest.mark.parametrize("question,operator,expected", [
    ("What did Alice and Bob both enjoy?", "intersection_distinct", "relevance"),
    ("When did Alice move to Kyoto?", "lookup", "chronological"),
    ("What changed after Alice moved?", "lookup", "chronological"),
    ("Where does Alice live?", "latest_state", "chronological"),
])
def test_adaptive_evidence_order_is_query_directed(
        question: str, operator: str, expected: str) -> None:
    assert resolve_evidence_order("adaptive", question, operator) == expected


def test_rendering_drops_optional_turns_before_mandatory_ones() -> None:
    turns = [_turn(index, f"turn number {index} with some filler words") for index in range(6)]
    mandatory = [turns[5].turn_id]

    rendered = render_evidence(turns, config=AnswerConfig(), counter=COUNTER,
                               max_tokens=25, mandatory_turn_ids=mandatory)

    assert turns[5].turn_id in rendered.turn_ids
    assert rendered.truncated and not rendered.mandatory_dropped


def test_dropping_a_mandatory_turn_is_reported_not_hidden() -> None:
    turns = [_turn(index, "a fairly long mandatory turn with plenty of words in it")
             for index in range(4)]

    rendered = render_evidence(turns, config=AnswerConfig(), counter=COUNTER, max_tokens=12,
                               mandatory_turn_ids=[turn.turn_id for turn in turns])

    assert rendered.mandatory_dropped


def test_rendering_is_deterministic() -> None:
    turns = [_turn(index, f"content {index}") for index in range(5)]
    kwargs = dict(config=AnswerConfig(), counter=COUNTER, max_tokens=1000)

    assert render_evidence(turns, **kwargs) == render_evidence(list(reversed(turns)), **kwargs)


# --- closed-form composition --------------------------------------------------

def _algebra(**kwargs) -> AlgebraResult:
    base = dict(operator=QueryOperator.LOOKUP, bindings=(), output_binding_ids=())
    return AlgebraResult(**{**base, **kwargs})


def test_a_lookup_has_no_closed_form() -> None:
    assert compose(_algebra(answer_kind="lookup")) is None


def test_a_closed_scope_count_answers_exactly() -> None:
    members = tuple(AnswerMember(f"k{i}", f"v{i}", f"v{i}") for i in range(3))

    draft = compose(_algebra(answer_kind="count", members=members, count=3, scope_complete=True))

    assert draft is not None and draft.text == "3" and draft.certified


def test_an_open_scope_count_proposes_nothing_at_all() -> None:
    """Measured: an uncertified count is noise, not a floor.

    "How many antique items did I inherit" counted 15 members that were
    unrelated facts, because operand predicate candidates are retrieved from the
    graph rather than parsed from the question.  Proposing "at least 15" to the
    answer prompt is worse than proposing nothing.
    """
    members = tuple(AnswerMember(f"k{i}", f"v{i}", f"v{i}") for i in range(3))

    assert compose(_algebra(answer_kind="count", members=members, count=3,
                            scope_complete=False)) is None


def test_absence_is_not_claimed_from_an_unclosed_scope() -> None:
    assert compose(_algebra(answer_kind="existence", members=(), scope_complete=False)) is None


def test_absence_is_claimed_once_the_scope_is_closed() -> None:
    draft = compose(_algebra(answer_kind="existence", members=(), scope_complete=True))

    assert draft is not None and draft.text == "no"


def test_date_difference_needs_two_resolved_endpoints() -> None:
    one = TemporalEndpoint("start", TemporalKey(start="2023-01-01", kind="point"), "b1")
    unresolved = TemporalEndpoint("end", TemporalKey(raw_text="later"), "b2")

    assert compose(_algebra(answer_kind="date_difference",
                            temporal_endpoints=(one, unresolved))) is None


def test_date_difference_counts_whole_days() -> None:
    left = TemporalEndpoint("start", TemporalKey(start="2023-01-01", kind="point"), "b1")
    right = TemporalEndpoint("end", TemporalKey(start="2023-03-02", kind="point"), "b2")

    draft = compose(_algebra(answer_kind="date_difference", temporal_endpoints=(left, right),
                             scope_complete=True))

    assert draft is not None and draft.text == "60 days"
    assert draft.witness_binding_ids == ("b1", "b2")


def test_latest_state_answers_with_the_current_value() -> None:
    state = StateResult(owner_id="o", predicate="lives_in", current_value="Kyoto",
                        current_binding_id="b9", prior_binding_ids=("b1",), superseded=True)

    draft = compose(_algebra(answer_kind="state", state_result=state, scope_complete=True))

    assert draft is not None and draft.text == "Kyoto" and draft.witness_binding_ids == ("b9",)


def test_a_degraded_result_is_never_certified() -> None:
    """A degradation blocks certification, and an uncertified count is withheld."""
    members = (AnswerMember("k", "v", "v"),)

    assert compose(_algebra(answer_kind="count", members=members, count=1,
                            scope_complete=True, degradations=("scope_truncated",))) is None
    # A list still renders its members: they are named, not inferred from a
    # scope claim, so a partial list is useful where a partial count is not.
    listed = compose(_algebra(answer_kind="list", members=members,
                              scope_complete=True, degradations=("scope_truncated",)))
    assert listed is not None and not listed.certified


# --- prompt -------------------------------------------------------------------

def test_the_prompt_hash_is_frozen() -> None:
    """A prompt edit invalidates every arm scored before it; make that loud."""
    assert PROMPT_HASH == hashlib.sha256(
        ("graphmem-v5.57-answer-unit-rate-v1" + __import__(
            "graphmem.answer.prompts", fromlist=["x"]).ANSWER_SYSTEM_PROMPT).encode()).hexdigest()


def test_source_time_prompt_is_an_explicit_separate_contract() -> None:
    messages = build_answer_messages(
        question="when?", question_date="2023-06-20", evidence_text="",
        normalize_relative_time=True)

    assert "[source-time ...]" in messages[0]["content"]


def test_contextual_question_date_omits_global_anchor_without_query_cue() -> None:
    from graphmem.answer import question_needs_global_date

    assert question_needs_global_date("How many trips did I take this year?")
    assert question_needs_global_date("How many days ago did I visit?")
    assert question_needs_global_date("What did Alice plan for next month?")
    assert question_needs_global_date("Where was I living last year?")
    assert not question_needs_global_date("When did Alice go camping?")
    messages = build_answer_messages(
        question="When did Alice go camping?", question_date="2023-10-22",
        evidence_text="next month", include_question_date=False)
    assert "Question date:" not in messages[1]["content"]


def test_question_recency_footer_repeats_question_after_evidence() -> None:
    messages = build_answer_messages(
        question="What degree did I earn?", question_date="2023-10-22",
        evidence_text="MEMORY_SENTINEL", question_recency_footer=True)
    user = messages[1]["content"]
    assert user.index("MEMORY_SENTINEL") < user.rindex(
        "Answer the original Question now:")
    assert "source-time" in user


def test_precision_grounded_prompt_is_opt_in_and_separately_hashed() -> None:
    baseline = build_answer_messages(
        question="Did Alice surf?", question_date=None, evidence_text="")
    grounded = build_answer_messages(
        question="Did Alice surf?", question_date=None, evidence_text="",
        precision_grounding=True)

    assert "smallest set of memories" not in baseline[0]["content"]
    assert "smallest set of memories" in grounded[0]["content"]
    assert "Exact wording" in grounded[0]["content"]


def test_topological_layout_prompt_explains_that_chain_labels_are_hints() -> None:
    messages = build_answer_messages(
        question="When did Alice move?", question_date=None, evidence_text="",
        topological_layout=True)

    assert "[CHAIN k step=d]" in messages[0]["content"]
    assert "navigation hints, not facts" in messages[0]["content"]


def test_compact_topological_contract_preserves_semantics_and_saves_text() -> None:
    verbose = build_answer_messages(
        question="When did Alice move?", question_date=None, evidence_text="",
        topological_layout=True)
    compact = build_answer_messages(
        question="When did Alice move?", question_date=None, evidence_text="",
        topological_layout=True, compact_topological_contract=True)

    assert "[CHAIN] follows one QueryIR path" in compact[0]["content"]
    assert "navigation hints only" in compact[0]["content"]
    assert len(compact[0]["content"]) < len(verbose[0]["content"])
    assert prompt_contract(False, False, True)[2] != prompt_contract(
        False, False, True, False, False, False, False, False, True)[2]


def test_compact_labels_and_focus_index_are_explicit_prompt_contracts() -> None:
    messages = build_answer_messages(
        question="When did Alice move?", question_date=None,
        evidence_text="[C1.0] source", topological_layout=True,
        compact_topological_labels=True,
        query_focus_index="Query focus:\n[F1] exact source")

    assert "[Ck.d] is QueryIR chain" in messages[0]["content"]
    assert "reading aid, not extra evidence" in messages[0]["content"]
    assert messages[1]["content"].index("[C1.0]") < (
        messages[1]["content"].index("Query focus:"))
    assert prompt_contract(False, False, True)[2] != prompt_contract(
        False, False, True, False, False, False, False, False, False,
        True, True)[2]


def test_query_focus_reads_requested_ordinal_from_full_packed_turn() -> None:
    text = (
        "Work from home jobs for seniors: 1. Tutor. 2. Bookkeeper. "
        "3. Consultant. 4. Translator. 5. Customer support. "
        "6. Virtual assistant. 7. Transcriptionist. 8. Survey taker.")
    turn = replace(_turn(0, text, speaker="assistant"), role="assistant")
    candidate = CandidateScore(
        turn.turn_id, turn.session_id, 1.0, 1.0, 1.0, 0.0,
        0.0, 0.0, 40, 5.0, ("exact", "bm25", "dense"))

    focus, ids = _query_focus_index(
        "Can you remind me what was the 7th job in the list you provided?",
        {turn.turn_id: turn}, (turn.turn_id,), (candidate,),
        limit=2, excerpt_chars=160)

    assert focus is not None
    assert "7. Transcriptionist" in focus
    assert ids == (turn.turn_id,)


def test_query_focus_uses_question_tail_to_recover_clipped_relation_value() -> None:
    text = (
        "Yes, here are DIY home decor projects using recycled materials. "
        "1. Wine Cork Board - arrange and glue the corks into a board. "
        "2. Newspaper Flower Vase - roll newspaper into a vase, then seal "
        "the vase with Mod Podge or another sealant to make it water-resistant. "
        "3. Bottle Cap Coasters - glue caps onto cork bases.")
    turn = replace(_turn(0, text, speaker="assistant"), role="assistant")
    candidate = CandidateScore(
        turn.turn_id, turn.session_id, 1.0, 1.0, 1.0, 0.0,
        0.0, 0.0, 40, 5.0, ("exact", "bm25", "dense"))

    focus, ids = _query_focus_index(
        "I'm going back to our previous conversation about DIY home decor "
        "projects using recycled materials. Can you remind me what sealant "
        "you recommended for the newspaper flower vase?",
        {turn.turn_id: turn}, (turn.turn_id,), (candidate,),
        limit=2, excerpt_chars=180)

    assert focus is not None
    assert "Mod Podge" in focus
    assert "sealant" in focus
    assert ids == (turn.turn_id,)


def test_query_focus_uses_adjacent_session_context_to_disambiguate_ordinals() -> None:
    question_turn = _turn(0, "Brainstorm ideas for work from home jobs for seniors")
    answer_turn = replace(
        _turn(1, "1. Tutor 2. Bookkeeper 3. Consultant 4. Translator "
              "5. Customer support 6. Virtual assistant 7. Transcriptionist "
              "8. Survey taker", speaker="assistant"), role="assistant")
    noise_question = _turn(0, "How should I present to a class?", session="noise")
    noise_answer = replace(
        _turn(1, "1. Start early 2. Add slides 3. Rehearse 4. Speak clearly "
              "5. Add examples 6. Pause 7. Encourage Questions 8. Conclude",
              session="noise", speaker="assistant"), role="assistant")
    turns = {row.turn_id: row for row in (
        question_turn, answer_turn, noise_question, noise_answer)}
    candidates = tuple(
        CandidateScore(
            row.turn_id, row.session_id,
            2.0 if row is noise_answer else 0.2,
            2.0 if row is noise_answer else 0.2,
            2.0 if row is noise_answer else 0.2,
            0.0, 0.0, 0.0, 40,
            20.0 if row is noise_answer else 1.0,
            ("exact", "bm25", "dense"))
        for row in (noise_answer, answer_turn))

    focus, ids = _query_focus_index(
        "I think we discussed work from home jobs for seniors earlier. Can "
        "you remind me what was the 7th job in the list you provided?",
        turns, (noise_answer.turn_id, answer_turn.turn_id), candidates,
        limit=2, excerpt_chars=160)

    assert focus is not None
    assert focus.index("7. Transcriptionist") < focus.index("7. Encourage Questions")
    assert ids[0] == answer_turn.turn_id


def test_query_focus_is_disabled_for_named_multi_party_transcript() -> None:
    turn = replace(
        _turn(0, "I bought the camera in June.", speaker="Caroline"),
        role="user")
    candidate = CandidateScore(
        turn.turn_id, turn.session_id, 1.0, 1.0, 1.0, 0.0,
        0.0, 0.0, 10, 5.0, ("exact", "bm25", "dense"))

    focus, ids = _query_focus_index(
        "When did Caroline buy the camera?", {turn.turn_id: turn},
        (turn.turn_id,), (candidate,))

    assert focus is None
    assert ids == ()


def test_query_focus_stage_routes_only_non_temporal_lookup(tmp_path) -> None:
    store = _store(tmp_path, [
        "A long answer introduced several craft projects before saying that "
        "the newspaper vase should be sealed with Mod Podge.",
    ])
    result = replace(
        _result([turn.turn_id for turn in store.turns("m")]),
        trace={"ast_operator": "lookup"})
    stage = _stage(
        store, _FakeClient(), answer_config=AnswerConfig(
            query_focus_index_enabled=True,
            query_focus_index_limit=2,
            query_focus_excerpt_chars=160))

    lookup = stage.prepare(
        "q1", "Can you remind me what sealant you recommended for the "
        "newspaper vase?", result, QueryBudget())
    temporal = stage.prepare(
        "q2", "Can you remind me when you recommended the newspaper vase?",
        result, QueryBudget())

    assert lookup.trace["query_focus_index"]
    assert "Mod Podge" in lookup.messages[1]["content"]
    assert not temporal.trace["query_focus_index"]
    store.close()


def test_default_focused_prompt_scope_preserves_specialized_contracts(
        tmp_path) -> None:
    store = _store(tmp_path, [
        "I spent $35 on bicycle tires.",
        "I paid $150 to repair the bicycle.",
        "I prefer quiet mystery films.",
    ])
    result = _result([turn.turn_id for turn in store.turns("m")])
    common = dict(evidence_order="topological", normalize_relative_time=True)
    rewrite = dict(
        question_date_mode="query_relative", question_recency_footer=True,
        compact_topological_contract=True, focused_prompt_scope="default")

    for question, specialized in (
        ("How much did I spend on the bicycle in total?",
         {"aggregation_ledger_enabled": True}),
        ("Can you recommend a movie?",
         {"preference_synthesis_enabled": True}),
    ):
        baseline = _stage(
            store, _FakeClient(),
            answer_config=AnswerConfig(**common, **specialized)).prepare(
                "baseline", question, result, QueryBudget(),
                question_date="2023-10-22")
        routed = _stage(
            store, _FakeClient(),
            answer_config=AnswerConfig(
                **common, **specialized, **rewrite)).prepare(
                    "routed", question, result, QueryBudget(),
                    question_date="2023-10-22")
        assert routed.messages == baseline.messages
        assert routed.prompt_hash == baseline.prompt_hash
        assert not routed.trace["focused_prompt_applied"]

    default = _stage(
        store, _FakeClient(),
        answer_config=AnswerConfig(**common, **rewrite)).prepare(
            "default", "What did I spend on bicycle tires?", result,
            QueryBudget(), question_date="2023-10-22")
    assert default.trace["focused_prompt_applied"]
    assert default.trace["compact_topological_contract"]
    assert "Answer the original Question now:" in default.messages[1]["content"]
    store.close()


def test_exact_grounding_footer_is_after_evidence_and_skips_preferences() -> None:
    grounded = build_answer_messages(
        question="Did Alice surf?", question_date=None,
        evidence_text="MEMORY_SENTINEL", exact_grounding_footer=True)
    preference = build_answer_messages(
        question="What should Alice try?", question_date=None,
        evidence_text="MEMORY_SENTINEL", preference_synthesis=True,
        exact_grounding_footer=True)

    user = grounded[1]["content"]
    assert user.index("MEMORY_SENTINEL") < user.index("Final check:")
    assert "exact entity and relation" in user
    assert "Final check:" not in preference[1]["content"]


def test_a_candidate_answer_is_labelled_a_proposal_not_evidence() -> None:
    messages = build_answer_messages(question="how many?", question_date="2023-05-01",
                                     evidence_text="[s1] user: three cats", candidate_answer="3")

    user = messages[1]["content"]
    assert "Candidate answer (unverified proposal): 3" in user
    assert "fallible mechanical proposal" in messages[0]["content"]


def test_scalar_delta_questions_state_the_arithmetic_contract() -> None:
    messages = build_answer_messages(question="How many more points do I need to earn?",
                                     question_date=None, evidence_text="")

    assert "scalar_delta" in messages[1]["content"]


def test_aggregation_ledger_is_opt_in_post_evidence_and_separately_hashed() -> None:
    messages = build_answer_messages(
        question="How much did I spend in total?", question_date=None,
        evidence_text="source evidence", aggregation_ledger="Operation: sum")

    assert messages[1]["content"].index("source evidence") < messages[1]["content"].index("Operation: sum")
    assert "not a certified answer" in messages[0]["content"]
    assert prompt_contract(False, False, False, True)[2] != prompt_contract()[2]


def test_preference_synthesis_is_wording_routed_and_separately_hashed() -> None:
    assert is_preference_synthesis_query(
        "I've got some free time tonight, any documentary recommendations?")
    assert is_preference_synthesis_query(
        "Do you have any helpful tips for getting around Tokyo?")
    assert not is_preference_synthesis_query(
        "What documentary did I watch last Thursday?")

    baseline = build_answer_messages(
        question="Can you recommend a movie?", question_date=None,
        evidence_text="I enjoy mysteries.", precision_grounding=True)
    routed = build_answer_messages(
        question="Can you recommend a movie?", question_date=None,
        evidence_text="I enjoy mysteries.", precision_grounding=True,
        preference_synthesis=True)
    assert baseline[1] != routed[1]
    assert "Grounded recommendation check" in routed[1]["content"]
    assert "Answer this recommendation request now" in routed[1]["content"]
    assert baseline[0] != routed[0]
    assert "may synthesize a new recommendation" in routed[0]["content"]
    assert prompt_contract(False, True, False, False, True)[2] != (
        prompt_contract(False, True, False, False, False)[2])


def test_aggregation_ledger_indexes_money_operands_without_claiming_closure() -> None:
    turns = {
        turn.turn_id: turn for turn in (
            _turn(0, "I spent $35 on my bicycle tires."),
            _turn(1, "We talked about a movie from 2021."),
            _turn(2, "I paid $150 to repair the bicycle."),
        )
    }
    ledger = build_aggregation_ledger(
        "How much did I spend on the bicycle in total?", turns, tuple(turns))

    assert ledger is not None and ledger.operation == "sum"
    assert "$35" in ledger.text and "$150" in ledger.text
    assert "Certified deterministic result: unavailable" in ledger.text
    assert not ledger.result_certified


def test_aggregation_execution_card_keeps_candidates_trace_only() -> None:
    turns = {
        turn.turn_id: turn for turn in (
            _turn(0, "I spent $35 on my bicycle tires."),
            _turn(1, "I paid $150 to repair the bicycle."),
        )
    }
    card = build_aggregation_ledger(
        "How much did I spend on the bicycle in total?", turns, tuple(turns),
        execution_card=True)

    assert card is not None and card.operation == "sum"
    assert set(card.candidate_turn_ids) == set(turns)
    assert "compact execution card" in card.text
    assert "Question: How much did I spend" in card.text
    assert "Candidate 1:" not in card.text
    assert "Certified deterministic result: unavailable" not in card.text
    assert card.worksheet_lines
    assert card.worksheet_turn_ids


def test_v5_54_compact_card_keeps_a_bounded_operand_worksheet(tmp_path) -> None:
    store = _store(tmp_path, [
        "I earned $150 selling potted plants at the Saturday market.",
        "I earned $75 selling herb bundles at the Sunday market.",
        "An unrelated article mentioned a $900 budget.",
    ])
    result = _result([turn.turn_id for turn in store.turns("m")])
    prepared = _stage(
        store, _FakeClient(), answer_config=AnswerConfig.v5_54(
            aggregation_operand_worksheet_enabled=True)).prepare(
            "q1", "How much more did I earn at the Saturday market than "
            "the Sunday market?",
            result, QueryBudget())

    user = prepared.messages[-1]["content"]
    assert "Operand worksheet" in user
    assert "$150" in user and "$75" in user
    assert prepared.trace["aggregation_worksheet_rows"] >= 1
    assert not prepared.trace["aggregation_ledger"]["execution_card"]
    assert prepared.trace["readout_policy_evidence_set_frozen"]
    store.close()


def test_selective_operand_workspace_requires_complete_alternatives() -> None:
    turns = {
        turn.turn_id: turn for turn in (
            _turn(0, "The train fare was $10."),
            _turn(1, "The taxi fare was $60."),
        )
    }
    train = build_aggregation_ledger(
        "How much will I save by taking the train instead of a taxi?",
        turns, tuple(turns))
    bus = build_aggregation_ledger(
        "How much will I save by taking the bus instead of a taxi?",
        turns, tuple(turns))

    assert train is not None and selective_operand_worksheet_route(
        "How much will I save by taking the train instead of a taxi?", train,
    ) == "complete_alternative"
    assert bus is not None and selective_operand_worksheet_route(
        "How much will I save by taking the bus instead of a taxi?", bus,
    ) is None


def test_selective_operand_workspace_accepts_complete_money_sum() -> None:
    turns = {
        turn.turn_id: turn for turn in (
            _turn(0, "I raised $5,000 at a charity bike event."),
            _turn(1, "I raised $250 at a charity walk."),
            _turn(2, "I raised $600 at a charity yoga event."),
        )
    }
    question = "How much money did I raise through all charity events in total?"
    ledger = build_aggregation_ledger(question, turns, tuple(turns))

    assert ledger is not None
    assert selective_operand_worksheet_route(
        question, ledger) == "complete_money_sum"


def test_selective_operand_workspace_accepts_one_direct_count() -> None:
    turn = _turn(
        0, "I attended five sessions of the bereavement support group.")
    question = "How many sessions of the bereavement support group did I attend?"
    ledger = build_aggregation_ledger(question, {turn.turn_id: turn}, (turn.turn_id,))

    assert ledger is not None
    assert selective_operand_worksheet_route(question, ledger) == "direct_count"


def test_non_aggregation_question_has_no_ledger() -> None:
    turn = _turn(0, "My dog is Rex.")
    assert build_aggregation_ledger(
        "What is my dog called?", {turn.turn_id: turn}, (turn.turn_id,)) is None


def test_count_in_a_week_is_not_misclassified_as_duration_sum() -> None:
    from graphmem.answer import aggregation_operation

    assert aggregation_operation(
        "How many fitness classes do I attend in a typical week?") == "count_distinct"
    assert aggregation_operation("How many days did I travel in total?") == "sum"
    assert aggregation_operation(
        "How many days did it take me to finish the book?") == "date_difference"
    assert aggregation_operation(
        "How long had I been bird watching when I attended the workshop?") == "date_difference"
    assert aggregation_operation(
        "How many weeks in total did I spend reading three books?") == "sum"
    assert aggregation_operation(
        "How many online courses have I completed in total?") == "count_distinct"
    assert aggregation_operation(
        "What is the total number of plants I bought?") == "count_distinct"
    assert aggregation_operation(
        "How much will I save by taking the train instead of a taxi?") == "difference"


def test_historical_window_and_savings_procedures_are_explicit() -> None:
    first = _turn(0, "That is 15 autographed baseballs in three months.")
    later = _turn(1, "I have added 20 autographed baseballs in the past few months.")
    rows = {row.turn_id: row for row in (first, later)}

    historical = build_aggregation_ledger(
        "How many autographed baseballs did I add in the first three months?",
        rows, tuple(rows))
    savings = build_aggregation_ledger(
        "How much will I save by taking the train instead of a taxi?",
        rows, tuple(rows))

    assert historical is not None
    assert "never a later cumulative or current total" in historical.text
    assert savings is not None and savings.operation == "difference"
    assert "explicitly rejected option" in savings.text


def test_money_ledger_reserves_terse_direct_currency_operands() -> None:
    relevant = _turn(0, "The bike helmet cost $120.")
    noise = tuple(_turn(index + 1, f"Money saving total expense article {index}.")
                  for index in range(30))
    turns = {turn.turn_id: turn for turn in (relevant, *noise)}

    ledger = build_aggregation_ledger(
        "How much money did I spend on bike expenses in total?", turns,
        tuple(turns), limit=8)

    assert ledger is not None
    assert relevant.turn_id in ledger.candidate_turn_ids


def test_named_multi_party_speaker_is_not_treated_as_assistant_context() -> None:
    named = replace(
        _turn(0, "I adopted two dogs."),
        speaker="Maria", role="assistant", timestamp="2023-08-01")
    rows = {named.turn_id: named}

    ledger = build_aggregation_ledger(
        "How many dogs did Maria adopt?", rows, (named.turn_id,))

    assert ledger is not None
    assert "status=source_speaker_statement" in ledger.text


def test_aggregation_source_reserve_is_role_safe_across_dataset_shapes() -> None:
    direct = _turn(0, "I bought the coffee table.")
    assistant = replace(
        _turn(1, "Here is a long list of unrelated furniture advice."),
        speaker="assistant", role="assistant")
    generic = {turn.turn_id: turn for turn in (direct, assistant)}

    assert _aggregation_source_reserve_ids(
        generic, (assistant.turn_id, direct.turn_id)) == (direct.turn_id,)

    # LoCoMo represents both named people through user/assistant transport
    # roles.  Reserving only the user side would introduce a speaker bias.
    maria = replace(direct, speaker="Maria", role="user")
    caroline = replace(assistant, speaker="Caroline", role="assistant")
    named = {turn.turn_id: turn for turn in (maria, caroline)}
    assert not _aggregation_source_reserve_ids(
        named, (maria.turn_id, caroline.turn_id))


def test_aggregation_feature_keeps_non_aggregation_prompt_byte_identical(tmp_path) -> None:
    store = _store(tmp_path, ["My dog is named Rex."])
    result = _result([turn.turn_id for turn in store.turns("m")])
    baseline = _stage(store, _FakeClient()).prepare(
        "q1", "What is my dog called?", result, QueryBudget())
    routed = _stage(
        store, _FakeClient(),
        answer_config=AnswerConfig(aggregation_ledger_enabled=True)).prepare(
            "q1", "What is my dog called?", result, QueryBudget())

    assert routed.messages == baseline.messages
    assert routed.prompt_hash == baseline.prompt_hash
    assert routed.prompt_payload_hash == baseline.prompt_payload_hash
    store.close()


def test_ledger_keeps_age_and_schedule_source_turns() -> None:
    grandparent = _turn(0, "My grandma is 75 and my grandpa is 78.")
    classes = _turn(1, "I attend Zumba classes on Tuesdays and Thursdays.")
    rows = {turn.turn_id: turn for turn in (grandparent, classes)}

    ages = build_aggregation_ledger(
        "What is the average age of my grandparents?", rows, tuple(rows), limit=1)
    schedule = build_aggregation_ledger(
        "How many classes do I attend each week?", rows, tuple(rows), limit=2)

    assert ages is not None and grandparent.turn_id in ages.candidate_turn_ids
    assert schedule is not None
    assert classes.turn_id in schedule.candidate_turn_ids
    assert "Tuesdays and Thursdays" in schedule.text


def test_money_ledger_leaves_duplicate_resolution_to_the_answer_model() -> None:
    first = _turn(0, "I installed new bike lights for $40.")
    repeated = _turn(1, "The new bike lights I installed cost $40.")
    helmet = _turn(2, "My bicycle helmet cost $120.")
    rows = {turn.turn_id: turn for turn in (first, repeated, helmet)}

    ledger = build_aggregation_ledger(
        "How much did I spend on bike expenses in total?", rows, tuple(rows))

    assert ledger is not None
    assert ledger.text.count("$40") >= 2
    assert not ledger.result_certified
    assert "possible_duplicate_group" not in ledger.text


def test_ledger_certifies_closed_age_and_weekly_class_arithmetic() -> None:
    age_turns = (
        _turn(0, "I just turned 32."),
        _turn(1, "My mom is 55 and my dad is 58."),
        _turn(2, "My grandma is 75 and my grandpa is 78."),
    )
    ages = {turn.turn_id: turn for turn in age_turns}
    age_ledger = build_aggregation_ledger(
        "What is the average age of me, my parents, and my grandparents?",
        ages, tuple(ages))
    assert age_ledger is not None
    assert age_ledger.result_certified
    assert age_ledger.deterministic_result == "59.6 years"

    money_turns = (
        _turn(0, "I replaced the bike chain and it cost $25. I installed bike lights for $40."),
        _turn(1, "The bike lights I installed were $40."),
        _turn(2, "I bought my bicycle helmet for $120."),
    )
    money = {turn.turn_id: turn for turn in money_turns}
    money_ledger = build_aggregation_ledger(
        "How much money did I spend on bike expenses in total?", money, tuple(money))
    assert money_ledger is not None
    assert not money_ledger.result_certified
    assert not money_ledger.deterministic_result

    class_turns = (
        _turn(0, "I attend Hip Hop Abs on Saturdays."),
        _turn(1, "I take Zumba classes on Tuesdays and Thursdays."),
        _turn(2, "I take a BodyPump class on Mondays."),
        _turn(3, "I attend yoga classes on Sundays."),
    )
    classes = {turn.turn_id: turn for turn in class_turns}
    class_ledger = build_aggregation_ledger(
        "How many fitness classes do I attend in a typical week?",
        classes, tuple(classes))
    assert class_ledger is not None
    assert class_ledger.result_certified
    assert class_ledger.deterministic_result == "5"


def test_certified_aggregation_bypasses_the_answer_model(tmp_path) -> None:
    store = _store(tmp_path, [
        "I just turned 32.",
        "My mom is 55 and my dad is 58.",
        "My grandma is 75 and my grandpa is 78.",
    ])
    client = _FakeClient("wrong")
    stage = _stage(
        store, client,
        answer_config=AnswerConfig(aggregation_ledger_enabled=True))
    result = _result([turn.turn_id for turn in store.turns("m")])

    answer = stage.answer(
        "q1", "What is the average age of me, my parents, and my grandparents?",
        result, QueryBudget())

    assert answer.prediction == "59.6 years"
    assert answer.finish_reason == "deterministic"
    assert not client.requests
    assert answer.trace["deterministic_bypass_source"] == "aggregation_ledger"
    store.close()


# --- stage --------------------------------------------------------------------

class _FakeClient:
    """Records requests and returns a fixed completion."""

    def __init__(self, text: str = "Rex", finish_reason: str = "stop") -> None:
        self.text, self.finish_reason, self.requests = text, finish_reason, []
        self.chat = type("_Chat", (), {"completions": self})()

    def create(self, **request):
        self.requests.append(request)
        message = type("_M", (), {"content": self.text, "reasoning_content": None})()
        choice = type("_C", (), {
            "message": message, "finish_reason": self.finish_reason})()
        usage = type("_U", (), {"prompt_tokens": 100, "completion_tokens": 3})()
        return type("_R", (), {"choices": [choice], "usage": usage, "model": "test"})()


class _TransientFakeClient(_FakeClient):
    def __init__(self, failures: int) -> None:
        super().__init__("Rex")
        self.failures = failures

    def create(self, **request):
        if self.failures:
            self.failures -= 1
            error = RuntimeError("Service temporarily unavailable")
            error.status_code = 503
            raise error
        return super().create(**request)


def _store(tmp_path, texts: list[str]) -> SQLiteGraphStore:
    store = SQLiteGraphStore(tmp_path / "a.sqlite")
    turns = [_turn(index, text) for index, text in enumerate(texts)]
    store.ingest_conversation(Conversation("m", "test", "m", "h"),
                              [Session("s1", "m", 0, "2023-05-01", "sh")], turns)
    return store


def _result(turn_ids: list[str], units: tuple[EvidenceUnit, ...] = ()) -> NavigationResult:
    return NavigationResult(
        question_id="q1", memory_id="m", graph_artifact_id="g",
        retrieved_session_ids=("s1",), retrieved_turn_ids=tuple(turn_ids), proof=(),
        visited_nodes=0, visited_edges=0, frontier_peak=0, evidence_tokens=0,
        budget_exhausted=False, packed_turn_ids=tuple(turn_ids), proof_units=units)


def _stage(store, client, **kwargs) -> AnswerStage:
    from graphmem.config import GraphMemV5Config
    return AnswerStage(store, GraphMemV5Config(), "dataset", client=client,
                       require_exact_tokenizer=False, **kwargs)


def test_topological_layout_keeps_a_root_to_leaf_chain_contiguous(tmp_path) -> None:
    store = _store(tmp_path, ["unbound noise", "root evidence", "leaf evidence"])
    turns = store.turns("m")
    units = (
        EvidenceUnit("leaf", ("need",), ("b2",), (turns[2].turn_id,),
                     ("edge-1", "edge-2"), 0, True, ("operand",)),
        EvidenceUnit("root", ("need",), ("b1",), (turns[1].turn_id,),
                     ("edge-1",), 0, True, ("operand",)),
    )
    stage = _stage(
        store, _FakeClient(),
        answer_config=AnswerConfig(evidence_order="topological"))
    result = _result([turn.turn_id for turn in turns], units)

    prepared = stage.prepare("q1", "When did Alice move?", result, QueryBudget())
    user = prepared.messages[1]["content"]

    assert user.index("[CHAIN 1 step=1]") < user.index("[CHAIN 1 step=2]")
    assert user.index("[CHAIN 1 step=2]") < user.index("[AUX 1")
    assert prepared.trace["evidence_chain_turns"] == 2
    assert prepared.trace["evidence_graph_turns"] == 0
    assert prepared.trace["evidence_auxiliary_turns"] == 1
    store.close()


def test_topological_plain_reorders_without_exposing_graph_labels(tmp_path) -> None:
    store = _store(tmp_path, ["unbound noise", "root evidence", "leaf evidence"])
    turns = store.turns("m")
    units = (
        EvidenceUnit("leaf", ("need",), ("b2",), (turns[2].turn_id,),
                     ("edge-1", "edge-2"), 0, True, ("operand",)),
        EvidenceUnit("root", ("need",), ("b1",), (turns[1].turn_id,),
                     ("edge-1",), 0, True, ("operand",)),
    )
    stage = _stage(
        store, _FakeClient(),
        answer_config=AnswerConfig(evidence_order="topological_plain"))
    result = _result([turn.turn_id for turn in turns], units)

    prepared = stage.prepare("q1", "Which event happened first?", result, QueryBudget())
    user = prepared.messages[1]["content"]
    system = prepared.messages[0]["content"]

    assert user.index("root evidence") < user.index("leaf evidence")
    assert "[CHAIN" not in user and "[GRAPH" not in user and "[AUX" not in user
    assert "graph-derived blocks" not in system
    assert prepared.trace["evidence_layout"] == "topological_plain"
    assert prepared.trace["evidence_chain_turns"] == 2
    store.close()


def test_topological_recency_places_strongest_block_last(tmp_path) -> None:
    # Source chronology deliberately puts the strong chain first; the recency
    # layout must override chronology and move its whole block to the end.
    store = _store(tmp_path, ["strong root", "strong leaf", "weak auxiliary"])
    turns = store.turns("m")
    units = (
        EvidenceUnit("leaf", ("need",), ("b2",), (turns[1].turn_id,),
                     ("edge-1", "edge-2"), 0, True, ("operand",)),
        EvidenceUnit("root", ("need",), ("b1",), (turns[0].turn_id,),
                     ("edge-1",), 0, True, ("operand",)),
    )
    stage = _stage(
        store, _FakeClient(),
        answer_config=AnswerConfig(evidence_order="topological_recency"))
    result = replace(
        _result([turn.turn_id for turn in turns], units),
        candidate_scores=tuple(
            CandidateScore(turn.turn_id, turn.session_id, 0, 0, 0, 0,
                           0, 0, 1, float(3 - index), ())
            for index, turn in enumerate(turns)))
    prepared = stage.prepare(
        "q1", "When did Alice move?", result, QueryBudget())
    user = prepared.messages[1]["content"]

    assert user.index("weak auxiliary") < user.index("strong root")
    assert user.index("strong root") < user.index("strong leaf")
    assert prepared.trace["evidence_layout"] == "topological_recency"
    store.close()


def test_topological_recency_budget_drops_weak_prefix_not_strong_tail() -> None:
    turns = [
        _turn(0, "weak evidence " * 20, "weak-session"),
        _turn(0, "strong answer " * 20, "strong-session"),
    ]
    single_cost = COUNTER.count(render_turn(
        turns[0], AnswerConfig(evidence_order="topological_recency"))) + 1
    rendered = render_evidence(
        turns,
        config=AnswerConfig(evidence_order="topological_recency"),
        counter=COUNTER,
        max_tokens=single_cost,
    )
    assert rendered.turn_ids == (turns[1].turn_id,)
    assert rendered.dropped_turn_ids == (turns[0].turn_id,)


def test_v5_54_answer_config_freezes_the_measured_contract() -> None:
    config = AnswerConfig.v5_54()

    assert config.readout_policy == "v5_54"
    assert config.evidence_order == "topological"
    assert config.normalize_relative_time
    assert config.aggregation_ledger_enabled
    assert config.aggregation_ledger_limit == 32
    assert config.aggregation_source_reserve_enabled
    assert config.preference_synthesis_enabled
    assert config.question_date_mode == "query_relative"
    assert config.question_recency_footer
    assert config.compact_topological_contract
    assert config.focused_prompt_scope == "default"
    assert not config.candidate_answer_injection
    assert config.max_output_tokens == 2000


def test_v5_63_answer_config_enables_only_selective_accuracy_routes() -> None:
    config = AnswerConfig.v5_63()

    assert config.readout_policy == "v5_54"
    assert not config.query_focus_index_enabled
    assert config.temporal_query_focus_enabled
    assert config.preference_focus_strategy == "domain_idf"
    assert config.aggregation_operand_worksheet_enabled
    assert config.aggregation_operand_worksheet_selective
    assert config.query_focus_excerpt_chars == 480
    assert config.max_output_tokens == 2000


def test_domain_preference_focus_uses_the_query_domain_not_dense_rank() -> None:
    unrelated = _turn(
        0, "I like photography and own a camera with several lenses.")
    exact = _turn(
        1, "I love stand-up comedy specials and watch them on Netflix.")
    turns = {row.turn_id: row for row in (unrelated, exact)}
    scores = (
        CandidateScore(unrelated.turn_id, unrelated.session_id, 0, 0, 9, 0,
                       0, 0, 1, 9, ()),
        CandidateScore(exact.turn_id, exact.session_id, 0, 0, 0.1, 0,
                       0, 0, 1, 0.1, ()),
    )

    focus, turn_ids = _preference_focus_index(
        "Can you recommend a show or movie for me to watch tonight?",
        turns, (unrelated.turn_id, exact.turn_id), scores,
        strategy="domain_idf")

    assert focus is not None and "stand-up comedy" in focus
    assert turn_ids == (exact.turn_id,)


def test_v5_63_routes_explicit_date_difference_to_query_focus(tmp_path) -> None:
    store = _store(tmp_path, [
        "I replaced the spark plugs on February 14, 2023.",
        "I attended Turbocharged Tuesdays on March 15, 2023.",
        "An unrelated event happened in January 2023.",
    ])
    result = replace(
        _result([turn.turn_id for turn in store.turns("m")]),
        trace={"query_operator": "lookup"})
    prepared = _stage(
        store, _FakeClient(), answer_config=AnswerConfig.v5_63()).prepare(
            "q", "How many days passed between replacing the spark plugs and "
            "attending Turbocharged Tuesdays?", result,
            QueryBudget(max_evidence_turns=64, max_evidence_tokens=12000))

    assert prepared.trace["query_focus_index"]
    assert "Query focus (verbatim excerpts" in prepared.messages[-1]["content"]
    store.close()


def test_v5_63_renders_only_a_complete_selective_money_workspace(tmp_path) -> None:
    store = _store(tmp_path, [
        "I raised $5,000 at a charity bike event.",
        "I raised $250 at a charity walk.",
        "I raised $600 at a charity yoga event.",
    ])
    result = _result([turn.turn_id for turn in store.turns("m")])
    prepared = _stage(
        store, _FakeClient(), answer_config=AnswerConfig.v5_63()).prepare(
            "q", "How much money did I raise through all charity events in total?",
            result, QueryBudget(max_evidence_turns=64, max_evidence_tokens=12000))

    ledger = prepared.trace["aggregation_ledger"]
    assert ledger["worksheet_route"] == "complete_money_sum"
    assert "Operand worksheet" in prepared.messages[-1]["content"]
    assert "selective_operand_worksheet:complete_money_sum" in (
        prepared.trace["readout_policy_route"])
    assert prepared.trace["readout_policy_token_delta"] <= 500
    store.close()


def test_v5_63_keeps_nonselected_aggregation_system_contract_frozen(
        tmp_path) -> None:
    store = _store(tmp_path, [
        "My commute takes 25 minutes each way.",
        "I usually leave home at 8 AM.",
    ])
    result = _result([turn.turn_id for turn in store.turns("m")])
    question = "How long is my daily commute to work?"
    budget = QueryBudget(max_evidence_turns=64, max_evidence_tokens=12000)
    baseline = _stage(
        store, _FakeClient(), answer_config=AnswerConfig.v5_54()).prepare(
            "baseline", question, result, budget)
    selective = _stage(
        store, _FakeClient(), answer_config=AnswerConfig.v5_63()).prepare(
            "selective", question, result, budget)

    assert not selective.trace["aggregation_ledger"]["worksheet_enabled"]
    assert baseline.messages[0]["content"] == selective.messages[0]["content"]
    assert "bounded reading index" not in selective.messages[0]["content"]
    store.close()


def test_v5_54_policy_is_in_core_and_freezes_the_evidence_set(tmp_path) -> None:
    store = _store(tmp_path, [
        "I joined the running club in April 2023.",
        "I won the spring race in May 2023.",
        "The unrelated book club met in June 2023.",
    ])
    result = _result([turn.turn_id for turn in store.turns("m")])
    winner = AnswerConfig.v5_54()
    budget = QueryBudget(max_evidence_turns=64, max_evidence_tokens=12000)
    baseline = _stage(
        store, _FakeClient(),
        answer_config=replace(winner, readout_policy="legacy")).prepare(
            "base", "When did I win the spring race?", result, budget,
            question_date="2024-01-01")
    prepared = _stage(
        store, _FakeClient(), answer_config=winner).prepare(
            "winner", "When did I win the spring race?", result, budget,
            question_date="2024-01-01")

    assert prepared.trace["readout_policy"] == "v5_54"
    assert "anonymous_typed" in prepared.trace["readout_policy_route"]
    assert prepared.trace["typed_readout_kind"] == "temporal"
    assert set(prepared.evidence_turn_ids) == set(baseline.evidence_turn_ids)
    assert prepared.packing_prompt_tokens <= baseline.packing_prompt_tokens
    assert "Candidate answer" not in prepared.messages[-1]["content"]
    store.close()


def test_v5_54_modal_route_replaces_strict_lookup_with_grounded_inference(
        tmp_path) -> None:
    store = _store(tmp_path, [
        "I hike every weekend and enjoy difficult mountain trails.",
        "I prefer outdoor activities to staying indoors.",
    ])
    result = _result([turn.turn_id for turn in store.turns("m")])
    prepared = _stage(
        store, _FakeClient(), answer_config=AnswerConfig.v5_54()).prepare(
            "q", "Which sport would I likely enjoy?", result,
            QueryBudget(max_evidence_turns=64, max_evidence_tokens=12000))

    assert prepared.trace["inference_synthesis"]
    assert "inference" in prepared.trace["readout_policy_route"]
    assert "Inference Question:" in prepared.messages[-1]["content"]
    assert "infer from stated facts and ordinary knowledge" in (
        prepared.messages[0]["content"])
    store.close()


def test_the_stage_makes_exactly_one_call_and_returns_the_prediction(tmp_path) -> None:
    store = _store(tmp_path, ["I adopted a beagle named Rex."])
    client = _FakeClient("Rex")
    stage = _stage(store, client)
    result = _result([turn.turn_id for turn in store.turns("m")])

    answer = stage.answer("q1", "What is the dog called?", result, QueryBudget())

    assert answer.prediction == "Rex"
    assert len(client.requests) == 1
    assert client.requests[0]["temperature"] == 0
    assert client.requests[0]["seed"] == 0
    assert "max_tokens" not in client.requests[0]
    assert client.requests[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert answer.finish_reason == "stop"
    assert answer.api_prompt_tokens == 100
    assert answer.api_total_tokens == 103
    store.close()


def test_the_stage_retries_transient_transport_failure_in_place(
        tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("graphmem.answer.stage.time.sleep", lambda _seconds: None)
    store = _store(tmp_path, ["I adopted a beagle named Rex."])
    client = _TransientFakeClient(failures=2)
    stage = _stage(store, client)
    result = _result([turn.turn_id for turn in store.turns("m")])

    answer = stage.answer("q1", "What is the dog called?", result, QueryBudget())

    assert answer.prediction == "Rex"
    retry_count = store._read_one(
        "SELECT retry_count FROM llm_calls WHERE stage='answer'")[0]
    assert retry_count == 2
    store.close()


def test_prepared_answer_round_trip_replays_identical_prompt(tmp_path) -> None:
    store = _store(tmp_path, ["I adopted a beagle named Rex."])
    client = _FakeClient("Rex")
    stage = _stage(store, client)
    result = _result([turn.turn_id for turn in store.turns("m")])

    prepared = stage.prepare("q1", "What is the dog called?", result, QueryBudget())
    restored = PreparedAnswer.from_record(prepared.to_record())
    answer = stage.complete(restored)

    assert restored.prompt_payload_hash == prepared.prompt_payload_hash
    assert restored.messages == prepared.messages
    assert answer.prompt_payload_hash == prepared.prompt_payload_hash
    assert answer.prediction == "Rex"
    store.close()


def test_openai_answer_profile_uses_separate_model_and_no_output_cap(tmp_path) -> None:
    store = _store(tmp_path, ["I adopted a beagle named Rex."])
    client = _FakeClient("Rex")
    stage = _stage(
        store, client, answer_model="gpt-5.4-mini",
        answer_request_profile="openai")
    result = _result([turn.turn_id for turn in store.turns("m")])

    answer = stage.answer("q1", "What is the dog called?", result, QueryBudget())

    request = client.requests[0]
    assert request["model"] == "gpt-5.4-mini"
    assert request["reasoning_effort"] == "none"
    assert "extra_body" not in request
    assert "max_tokens" not in request
    assert "max_completion_tokens" not in request
    assert answer.answer_model == "gpt-5.4-mini"
    store.close()


def test_an_explicit_output_cap_is_sent_only_for_an_ablation(tmp_path) -> None:
    store = _store(tmp_path, ["I adopted a beagle named Rex."])
    client = _FakeClient("Rex")
    stage = _stage(
        store, client, answer_config=AnswerConfig(max_output_tokens=512))
    result = _result([turn.turn_id for turn in store.turns("m")])

    stage.answer("q1", "What is the dog called?", result, QueryBudget())

    assert client.requests[0]["max_tokens"] == 512
    store.close()


def test_output_length_finish_is_reported_as_truncation(tmp_path) -> None:
    store = _store(tmp_path, ["I adopted a beagle named Rex."])
    stage = _stage(store, _FakeClient("Rex...", finish_reason="length"))
    result = _result([turn.turn_id for turn in store.turns("m")])

    answer = stage.answer("q1", "What is the dog called?", result, QueryBudget())

    assert answer.finish_reason == "length"
    assert "answer_output_truncated" in answer.warnings
    store.close()


def test_a_repeated_question_is_served_from_cache_without_a_second_call(tmp_path) -> None:
    store = _store(tmp_path, ["I adopted a beagle named Rex."])
    client = _FakeClient("Rex")
    stage = _stage(store, client)
    result = _result([turn.turn_id for turn in store.turns("m")])

    first = stage.answer("q1", "What is the dog called?", result, QueryBudget())
    second = stage.answer("q1", "What is the dog called?", result, QueryBudget())

    assert len(client.requests) == 1
    assert second.cached and not first.cached
    assert first.prediction == second.prediction
    store.close()


def test_an_oversized_pack_is_trimmed_to_the_soft_budget(tmp_path) -> None:
    store = _store(tmp_path, [f"turn {index} " + "filler " * 60 for index in range(40)])
    stage = _stage(store, _FakeClient())
    result = _result([turn.turn_id for turn in store.turns("m")])

    answer = stage.answer("q1", "what?", result, replace(QueryBudget(), max_answer_tokens=800))

    assert answer.prompt_tokens <= 800
    assert answer.dropped_turn_ids
    store.close()


def test_a_pack_that_cannot_fit_the_hard_ceiling_fails_loudly(tmp_path) -> None:
    """Silently overspending the answer budget would invalidate the token ledger."""
    store = _store(tmp_path, ["word " * 400 for _ in range(30)])
    stage = _stage(store, _FakeClient())
    units = (EvidenceUnit("u1", (), (), tuple(turn.turn_id for turn in store.turns("m")), (),
                          token_cost=0, mandatory=True),)
    result = _result([turn.turn_id for turn in store.turns("m")], units)
    budget = replace(QueryBudget(), max_answer_tokens=50, max_answer_tokens_hard=60)

    with pytest.raises(RuntimeError, match="hard ceiling"):
        stage.answer("q1", "what?", result, budget)
    store.close()


def test_the_answer_call_is_logged_with_its_token_usage(tmp_path) -> None:
    store = _store(tmp_path, ["I adopted a beagle named Rex."])
    stage = _stage(store, _FakeClient("Rex"))
    result = _result([turn.turn_id for turn in store.turns("m")])

    stage.answer("q1", "What is the dog called?", result, QueryBudget())

    rows = [tuple(row) for row in store._connection.execute(
        "SELECT stage, cached FROM llm_calls WHERE memory_id='m'").fetchall()]
    assert rows == [("answer", 0)]
    store.close()


def test_a_certified_closed_form_is_reported_as_such(tmp_path) -> None:
    store = _store(tmp_path, ["I have three cats."])
    client = _FakeClient("3")
    stage = _stage(store, client)
    result = _result([turn.turn_id for turn in store.turns("m")])
    algebra = _algebra(answer_kind="count", count=3, scope_complete=True,
                       members=(AnswerMember("k", "v", "v"),))

    answer = stage.answer("q1", "How many cats?", result, QueryBudget(), algebra=algebra)

    assert answer.closed_form and answer.draft_text == "3"
    assert "Candidate answer" not in client.requests[0]["messages"][1]["content"]
    store.close()


def test_candidate_answer_injection_requires_explicit_opt_in(tmp_path) -> None:
    store = _store(tmp_path, ["I have three cats."])
    client = _FakeClient("3")
    stage = _stage(
        store, client,
        answer_config=AnswerConfig(candidate_answer_injection=True))
    result = _result([turn.turn_id for turn in store.turns("m")])
    algebra = _algebra(answer_kind="count", count=3, scope_complete=True,
                       members=(AnswerMember("k", "v", "v"),))

    stage.answer("q1", "How many cats?", result, QueryBudget(), algebra=algebra)

    assert "Candidate answer (unverified proposal): 3" in (
        client.requests[0]["messages"][1]["content"])
    store.close()


def test_closed_form_can_be_switched_off_for_an_ablation_arm(tmp_path) -> None:
    store = _store(tmp_path, ["I have three cats."])
    stage = _stage(store, _FakeClient("3"), answer_config=AnswerConfig(closed_form_enabled=False))
    result = _result([turn.turn_id for turn in store.turns("m")])
    algebra = _algebra(answer_kind="count", count=3, scope_complete=True)

    answer = stage.answer("q1", "How many cats?", result, QueryBudget(), algebra=algebra)

    assert not answer.closed_form and answer.draft_text == ""
    store.close()


# --- storage concurrency ------------------------------------------------------

def test_concurrent_cache_reads_and_writes_do_not_corrupt_the_connection(tmp_path) -> None:
    """One connection is shared by every build worker; reads must hold the lock.

    Before this was fixed the suite failed intermittently with
    ``sqlite3.InterfaceError: bad parameter or other API misuse`` whenever a
    16-thread extraction fan-out read the cache while a writer held
    ``BEGIN IMMEDIATE``.
    """
    from concurrent.futures import ThreadPoolExecutor

    store = SQLiteGraphStore(tmp_path / "race.sqlite")
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            for step in range(40):
                key = f"key-{index}-{step}"
                store.cache_put(key, "answer", {"q": index}, {"content": "x"},
                                {"total_tokens": 1}, "hash")
                store.cache_get(key)
                store.cache_get(f"key-{(index + 1) % 8}-{step}")
        except BaseException as error:  # noqa: BLE001 - the assertion is the report
            errors.append(error)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, range(8)))

    assert not errors, repr(errors[:3])
    store.close()
