from __future__ import annotations

from dataclasses import asdict

from graphmem_demo.models import QuestionCase
from graphmem_demo.v3.build import clone_index, validate_hypergraph
from graphmem_demo.v3.catalog import ensure_catalog, recurrence_days
from graphmem_demo.v3.catalog_schema import OperandRecordV3
from graphmem_demo.v3.catalog_operators import catalog_operator_hint
from graphmem_demo.v3.retrieval import build_query_frame, retrieve
from graphmem_demo.v3.schema import (
    ClaimNode,
    EventNode,
    TurnNode,
    V3Index,
    index_from_dict,
)


def _index() -> V3Index:
    turn = TurnNode(
        node_id="q:s:turn:0",
        question_id="q",
        session_id="s",
        session_date="2026-01-02",
        turn_index=0,
        speaker="Alex",
        speaker_key="alex",
        listener="Sam",
        transport_role="user",
        text="I train on Mondays, Wednesday, and Fridays.",
        retrieval_text="speaker Alex train Mondays Wednesday Fridays",
        embedding=[1.0, 0.0],
    )
    claim = ClaimNode(
        node_id="q:s:claim:0",
        question_id="q",
        session_id="s",
        subject="Alex",
        subject_key="alex",
        predicate="trains",
        predicate_key="train",
        object="Mondays, Wednesday, and Fridays",
        object_key="monday wednesday friday",
        kind="event",
        context_key="weekly training",
        observed_at="2026-01-02",
        source_turn_ids=[turn.node_id],
        retrieval_text="Alex trains Mondays Wednesday Fridays",
        embedding=[1.0, 0.0],
    )
    event = EventNode(
        node_id="q:s:event:0",
        question_id="q",
        session_id="s",
        label="weekly training",
        label_key="weekly training",
        participant_keys=["alex"],
        claim_ids=[claim.node_id],
        source_turn_ids=[turn.node_id],
        retrieval_text="Alex weekly training",
        embedding=[1.0, 0.0],
    )
    return ensure_catalog(V3Index(turns=[turn], claims=[claim], events=[event]))


def _case() -> QuestionCase:
    return QuestionCase(
        question_id="q",
        question_type="unknown",
        question="How many days per week does Alex train?",
        answer="3",
        question_date="2026-01-03",
        haystack_sessions=[],
        haystack_session_ids=[],
        haystack_dates=[],
        answer_session_ids=[],
    )


def test_recurrence_parser_is_vocabulary_independent() -> None:
    assert recurrence_days("Mon, Wednesdays, and Friday") == [
        "monday", "wednesday", "friday"
    ]
    assert recurrence_days("orchids and kilns") == []
    assert recurrence_days("I went for a check-up Monday") == []
    assert recurrence_days("I go for a check-up every Monday") == ["monday"]
    assert recurrence_days("I hurt my knee last Friday and now swim every week") == []
    assert recurrence_days("I train weekly on Friday") == ["friday"]


def test_catalog_is_grounded_roundtrippable_and_cloneable() -> None:
    index = _index()
    assert validate_hypergraph(index) == []
    assert len(index.event_frames) == 1
    assert len(index.operands) == 1
    assert index.operands[0].recurrence_count == 3
    assert {
        edge.relation for edge in index.hyperedges
    } >= {"event_frame_member", "operand_projection"}

    restored = index_from_dict(asdict(index))
    cloned = clone_index(restored, "q2")
    assert validate_hypergraph(cloned) == []
    assert cloned.operands[0].source_claim_ids == ["q2:s:claim:0"]
    assert cloned.event_frames[0].source_turn_ids == ["q2:s:turn:0"]


def test_catalog_operator_counts_recurrence_from_typed_operands() -> None:
    index = _index()
    frame = build_query_frame(_case().question)
    hint = catalog_operator_hint(
        frame,
        index.operands,
        query_overlap=lambda query, text: len(
            set(query.content_terms) & set(text.casefold().split())
        ) / max(1, len(query.content_terms)),
    )
    assert hint is not None
    assert hint["operation"] == "weekly_recurrence_count"
    assert hint["value"] == 3
    assert hint["complete"] is True


def test_weekly_recurrence_unions_same_subject_semantic_class() -> None:
    frame = build_query_frame("How many days a week do I attend exercise classes?")
    primary = OperandRecordV3(
        "q:o:0", "q", "alex", "attends", "dance classes", "dance classes",
        context_key="exercise classes", recurrence_days=["tuesday", "thursday"],
        source_turn_ids=["t0"], retrieval_text="Alex attends dance exercise classes",
    )
    second = OperandRecordV3(
        "q:o:1", "q", "alex", "started", "yoga class", "yoga class",
        recurrence_days=["wednesday"], source_turn_ids=["t1"],
        retrieval_text="Alex started yoga class Wednesday",
    )
    unrelated = OperandRecordV3(
        "q:o:2", "q", "alex", "attends", "language class", "language class",
        recurrence_days=["friday"], source_turn_ids=["t2"],
        retrieval_text="Alex attends language class Friday",
    )
    scores = {"q:o:0": 0.8, "q:o:1": 0.72, "q:o:2": 0.2}
    hint = catalog_operator_hint(
        frame, [primary, second, unrelated], query_overlap=lambda _q, text: float("class" in text),
        semantic_similarity=lambda item: scores[item.operand_id],
    )
    assert hint is not None
    assert hint["operation"] == "weekly_recurrence_count"
    assert hint["value"] == 3
    assert set(hint["recurrence_days"]) == {"tuesday", "wednesday", "thursday"}


def test_catalog_backfills_lossless_first_person_money_without_echo_duplication() -> None:
    user = TurnNode(
        node_id="q:s:turn:0", question_id="q", session_id="s",
        session_date="2026-01-02", turn_index=0, speaker="Alex",
        speaker_key="alex", listener="Sam", transport_role="user",
        text="My team managed to raise $5,000 for the project.",
        retrieval_text="Alex raised money for project",
    )
    echo = TurnNode(
        node_id="q:s:turn:1", question_id="q", session_id="s",
        session_date="2026-01-02", turn_index=1, speaker="Sam",
        speaker_key="sam", listener="Alex", transport_role="assistant",
        text="Congratulations on raising $5,000 for the project; I'm sure it helped!",
        retrieval_text="Sam congratulated Alex",
    )
    index = ensure_catalog(V3Index(turns=[user, echo]))
    money = [item for item in index.operands if item.unit == "$"]
    assert len(money) == 1
    assert money[0].quantity == 5000
    assert money[0].subject_key == "alex"
    assert money[0].source_turn_ids == [user.node_id]
    assert len(money[0].source_claim_ids) == 1
    claim = next(
        item for item in index.claims
        if item.node_id == money[0].source_claim_ids[0]
    )
    assert claim.source_turn_ids == [user.node_id]
    assert claim.quantity == 5000
    assert validate_hypergraph(index) == []
    assert any(
        user.node_id in {inc.node_id for inc in edge.incidences}
        for edge in index.hyperedges if edge.relation == "operand_projection"
    )
    ensure_catalog(index)
    assert len([item for item in index.operands if item.quantity == 5000]) == 1


def test_retrieval_uses_catalog_and_expands_back_to_sources() -> None:
    result = retrieve(
        case=_case(),
        variant="hierarchical_hypergraph_v3",
        index=_index(),
        query_vector=[1.0, 0.0],
        token_budget=1800,
    )
    selected = result.retrieval_trace["selected_node_types"]
    assert selected["operand"] >= 1
    assert "q:s:turn:0" in result.evidence_leaf_ids
    assert result.retrieval_trace["catalog_operator_hint"]["value"] == 3
    assert result.packed_rough_tokens <= 1800
