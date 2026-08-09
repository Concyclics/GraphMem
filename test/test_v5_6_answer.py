from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from graphmem.answer import (
    AnswerConfig, AnswerStage, PROMPT_HASH, build_answer_messages, compose, render_evidence,
    render_turn, resolve_evidence_order,
)
from graphmem.domain import (
    AlgebraResult, AnswerMember, Conversation, EvidenceMember, EvidenceUnit, NavigationResult,
    QueryBudget, QueryOperator, Session, SourceTurn, StateResult, TemporalEndpoint, TemporalKey,
    stable_id,
)
from graphmem.tokenization import HeuristicTokenCounter
from graphmem.storage import SQLiteGraphStore


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
        ("graphmem-v5.6-answer-v1" + __import__(
            "graphmem.answer.prompts", fromlist=["x"]).ANSWER_SYSTEM_PROMPT).encode()).hexdigest()


def test_source_time_prompt_is_an_explicit_separate_contract() -> None:
    messages = build_answer_messages(
        question="when?", question_date="2023-06-20", evidence_text="",
        normalize_relative_time=True)

    assert "[source-time ...]" in messages[0]["content"]


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
    stage = _stage(store, _FakeClient("3"))
    result = _result([turn.turn_id for turn in store.turns("m")])
    algebra = _algebra(answer_kind="count", count=3, scope_complete=True,
                       members=(AnswerMember("k", "v", "v"),))

    answer = stage.answer("q1", "How many cats?", result, QueryBudget(), algebra=algebra)

    assert answer.closed_form and answer.draft_text == "3"
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
