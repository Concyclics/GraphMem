from __future__ import annotations

from dataclasses import replace

from graphmem_demo.v36.build import build_turn_nodes
from graphmem_demo.v36.retrieval import _question_relative_target, build_query_ir
from graphmem_demo.v36.source_spans import (
    build_source_span_closure, query_binding_terms,
)
from test_v36_role_graph import _case
from test_v36_structural_groups import _index


def _turns():
    return build_turn_nodes(_case())


def test_collection_closure_binds_target_and_action_in_lossless_span() -> None:
    turns = _turns()
    turns = [
        replace(
            turns[0], node_id="q:plants", session_id="plants",
            speaker_key="participant 1", listener="", transport_role="user",
            text="I bought a peace lily and a succulent plant two weeks ago.",
        ),
        replace(
            turns[1], node_id="q:advice", session_id="plants",
            speaker_key="participant 2", listener="", transport_role="assistant",
            text="You could buy several other plants in the future.",
        ),
    ]
    closure = build_source_span_closure(
        build_query_ir("How many plants did I acquire in the last month?"),
        turns, {"plants"},
    )
    assert closure.selected_source_turn_ids[0] == "q:plants"
    assert "q:advice" not in closure.selected_source_turn_ids
    assert {"members", "scope", "source"} <= set(closure.present_roles)
    candidate = closure.candidates[0]
    assert candidate.action_families == ["acquire"]
    assert candidate.lifecycle_status == "completed"


def test_source_span_uses_bounded_adjacent_sentence_relation() -> None:
    turns = [
        replace(
            _turns()[0], node_id="q:bike", session_id="bikes",
            speaker_key="participant 1", transport_role="user",
            text=(
                "My commuter bike needs a new front tire. "
                "I plan to replace it before April."
            ),
        )
    ]
    closure = build_source_span_closure(
        build_query_ir("How many bikes did I service or plan to service?"),
        turns, {"bikes"},
    )
    assert closure.selected_source_turn_ids == ["q:bike"]
    assert "members" in closure.present_roles
    assert closure.candidates[0].lifecycle_status == "planned"


def test_temporal_closure_supports_each_named_endpoint_independently() -> None:
    turns = _turns()
    turns = [
        replace(
            turns[0], node_id="q:left", session_id="s1",
            text="I visited the North Museum on March 3.",
        ),
        replace(
            turns[1], node_id="q:right", session_id="s2",
            transport_role="user",
            text="I visited the South Gallery on March 8.",
        ),
    ]
    ir = build_query_ir(
        "Which happened first: visiting the North Museum or visiting the South Gallery?"
    )
    closure = build_source_span_closure(ir, turns, {"s1", "s2"})
    assert all(closure.target_support.values())
    assert {"q:left", "q:right"} <= set(closure.selected_source_turn_ids)


def test_source_span_never_crosses_routed_scope() -> None:
    turns = [
        replace(
            _turns()[0], node_id="q:routed", session_id="routed",
            text="I bought one camera.",
        ),
        replace(
            _turns()[1], node_id="q:outside", session_id="outside",
            transport_role="user", text="I bought three cameras.",
        ),
    ]
    closure = build_source_span_closure(
        build_query_ir("How many cameras did I buy?"), turns, {"routed"},
    )
    assert closure.selected_source_turn_ids == ["q:routed"]


def test_source_span_uses_frame_type_projection_but_keeps_turn_provenance() -> None:
    index = _index()
    template = index.turns[0]
    turns = [
        replace(template, node_id="q:meter", session_id="health", text="I test my blood sugar three times a day."),
        replace(template, node_id="q:tracker", session_id="health", turn_index=1, text="I wear it throughout the day to track my activity."),
    ]
    first, second = index.frames[:2]
    first.entity_key, first.predicate_key = "blood glucose meter", "uses"
    first.semantic_type_keys, first.source_turn_ids = ["health device"], ["q:meter"]
    second.entity_key, second.predicate_key = "activity tracker", "uses"
    second.semantic_type_keys, second.source_turn_ids = ["health device"], ["q:tracker"]
    closure = build_source_span_closure(
        build_query_ir("How many health-related devices do I use in a day?"),
        turns, {"health"}, frames=[first, second],
    )
    assert {"q:meter", "q:tracker"} <= set(closure.selected_source_turn_ids)
    assert {identity for candidate in closure.candidates for identity in candidate.identity_keys} >= {"blood glucose meter", "activity tracker"}


def test_relative_question_target_date_is_deterministic() -> None:
    assert _question_relative_target(
        "What did I buy 10 days ago?", "2023/03/25 (Sat) 18:26",
    ).date().isoformat() == "2023-03-15"
    assert _question_relative_target(
        "Who visited last Saturday?", "2023/03/09 (Thu) 15:47",
    ).date().isoformat() == "2023-03-04"


def test_relative_closure_keeps_semantic_preferred_source_without_lexical_alias() -> None:
    turn = replace(
        _turns()[0], node_id="q:contract", session_id="business",
        session_date="2023/03/01 (Wed) 02:43",
        text="I just signed a contract with my first client today.",
    )
    closure = build_source_span_closure(
        build_query_ir(
            "What was the significant buisiness milestone I mentioned four weeks ago?"
        ),
        [turn], {"business"}, question_date="2023/03/28 (Tue) 09:00",
        preferred_source_turn_ids=["q:contract"],
    )
    assert closure.selected_source_turn_ids[0] == "q:contract"
    assert closure.candidates[0].event_time_text == "2023-03-01"


def test_comparison_closure_obeys_independently_bound_session_hints() -> None:
    template = _turns()[0]
    turns = [
        replace(
            template, node_id="q:festival", session_id="festival",
            session_date="2023/05/27 (Sat) 21:39",
            text="I attended a cultural festival yesterday.",
        ),
        replace(
            template, node_id="q:spanish", session_id="spanish",
            session_date="2023/05/27 (Sat) 14:08",
            text="I have been taking Spanish classes for the past three months.",
        ),
        replace(
            template, node_id="q:distractor", session_id="noise",
            session_date="2023/05/27 (Sat) 19:37",
            text="I attended panels and asked when the film festival would start.",
        ),
    ]
    ir = build_query_ir(
        "Which happened first, my attendance at a cultural festival or the start of my Spanish classes?"
    )
    closure = build_source_span_closure(
        ir, turns, {"festival", "spanish", "noise"},
        target_session_hints={
            ir.comparison_targets[0]: "festival",
            ir.comparison_targets[1]: "spanish",
        },
    )
    assert closure.selected_source_turn_ids[:2] == ["q:festival", "q:spanish"]
    assert "q:distractor" not in closure.selected_source_turn_ids


def test_frequency_binding_uses_entity_and_action_not_times_or_phrasal_up() -> None:
    ir = build_query_ir("How many times have I met up with Alex from Germany?")
    target, relation = query_binding_terms(ir)
    assert "time" not in target
    assert "up" not in relation
    assert {"meet", "alex", "germany"} <= relation

    template = _turns()[0]
    relevant = replace(
        template, node_id="q:alex", session_id="social",
        text="I met Alex from Germany for coffee again yesterday.",
    )
    distractor = replace(
        template, node_id="q:asana", session_id="social", turn_index=1,
        text="I use Asana to track my time and keep up with tasks.",
    )
    closure = build_source_span_closure(ir, [relevant, distractor], {"social"})
    assert closure.selected_source_turn_ids == ["q:alex"]
