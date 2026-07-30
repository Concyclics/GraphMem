from __future__ import annotations

from dataclasses import replace

from graphmem_demo.v36.operators import (
    evaluate_operators, query_bound_collection_ledger, counterfactual_dependency_hint, record_time_source_hint, temporal_order_source_hint,
    temporal_source_pair_hint, transaction_sum_from_sources_hint,
    dated_event_count_from_sources_hint, same_unit_state_difference_hint,
    maintenance_entity_count_hint, pending_operation_target_pairs_hint,
)
from graphmem_demo.v36.retrieval import authoritative_operator_answer, build_query_ir
from graphmem_demo.v36.schema import CompletenessCertificate
from test_v36_structural_groups import _index


def _certificate(complete: bool = True) -> CompletenessCertificate:
    return CompletenessCertificate(
        entity_match=complete,
        relation_match=complete,
        scope_match=complete,
        provenance_complete=complete,
        present_roles=["scope", "members", "operations", "source"],
        missing_roles=[] if complete else ["scope"],
        excluded_near_matches=[],
        complete=complete,
    )


def test_collection_operator_is_generic_and_certified() -> None:
    index = _index()
    group = next(
        item for item in index.evidence_groups
        if item.group_kind == "collection"
    )
    hints = evaluate_operators(
        ir=build_query_ir("How many inventory items are there?"),
        index=index,
        frame_ids=group.member_frame_ids,
        group_ids=[group.group_id],
        certificate=_certificate(),
    )
    assert hints
    assert hints[0]["operation"] == "distinct_collection"
    assert hints[0]["certified"] is True


def test_operator_rejects_incomplete_scope() -> None:
    index = _index()
    group = next(
        item for item in index.evidence_groups
        if item.group_kind == "collection"
    )
    assert evaluate_operators(
        ir=build_query_ir("How many inventory items are there?"),
        index=index,
        frame_ids=group.member_frame_ids,
        group_ids=[group.group_id],
        certificate=_certificate(False),
    ) == []


def test_duration_total_normalizes_generic_time_units() -> None:
    index = _index()
    first, second = index.frames[:2]
    first.quantity.value, first.quantity.unit = 2, "weeks"
    first.retrieval_text = "Project Alpha took two weeks"
    second.quantity.value, second.quantity.unit = 7, "days"
    second.retrieval_text = "Project Beta took seven days"
    hints = evaluate_operators(
        ir=build_query_ir("How many weeks did Project Alpha and Project Beta take?"),
        index=index, frame_ids=[first.frame_id, second.frame_id],
        group_ids=[], certificate=_certificate(),
    )
    total = next(item for item in hints if item["operation"] == "duration_total")
    assert total["value"] == 3
    assert total["unit"] == "weeks"


def test_duration_total_deduplicates_adjacent_cross_speaker_echo() -> None:
    index = _index()
    template_turn = index.turns[0]
    index.turns = [
        replace(template_turn, node_id="q:t:0", turn_index=0, speaker="A", speaker_key="a", text="I watched the films in two weeks."),
        replace(template_turn, node_id="q:t:1", turn_index=1, speaker="B", speaker_key="b", text="You watched all the films in two weeks, impressive."),
        replace(template_turn, node_id="q:t:4", turn_index=4, speaker="A", speaker_key="a", text="The other marathon took a week and a half."),
    ]
    first, second, third = index.frames[:3]
    first.quantity.value, first.quantity.unit, first.source_turn_ids = 2, "weeks", ["q:t:0"]
    second.quantity.value, second.quantity.unit, second.source_turn_ids = 2, "weeks", ["q:t:1"]
    third.quantity.value, third.quantity.unit, third.source_turn_ids = 1.5, "weeks", ["q:t:4"]
    hints = evaluate_operators(
        ir=build_query_ir("How many weeks did the films and the other marathon take?"),
        index=index, frame_ids=[first.frame_id, second.frame_id, third.frame_id],
        group_ids=[], certificate=_certificate(),
    )
    total = next(item for item in hints if item["operation"] == "duration_total")
    assert total["value"] == 3.5
    assert total["frame_ids"] == [first.frame_id, third.frame_id]


def test_duration_dedup_keeps_echo_anchor_before_cross_session_merge() -> None:
    index = _index()
    template_turn = index.turns[0]
    index.turns = [
        replace(template_turn, node_id="q:s0:t0", session_id="s0", turn_index=0, speaker_key="a", text="I watched all the films in two weeks."),
        replace(template_turn, node_id="q:s1:t0", session_id="s1", turn_index=0, speaker_key="a", text="The main saga marathon took a week and a half."),
        replace(template_turn, node_id="q:s1:t4", session_id="s1", turn_index=4, speaker_key="a", text="I watched all the films in two weeks."),
        replace(template_turn, node_id="q:s1:t5", session_id="s1", turn_index=5, speaker_key="b", text="You watched all the films in two weeks, impressive."),
    ]
    template = index.frames[0]
    values = [
        ("f0", 2, "weeks", "q:s0:t0"),
        ("f1", 1.5, "weeks", "q:s1:t0"),
        ("f2", 2, "weeks", "q:s1:t4"),
        ("f3", 2, "weeks", "q:s1:t5"),
    ]
    index.frames = [
        replace(
            template, frame_id=frame_id, quantity=replace(
                template.quantity, value=value, unit=unit
            ), source_turn_ids=[source],
        )
        for frame_id, value, unit, source in values
    ]
    hints = evaluate_operators(
        ir=build_query_ir("How many weeks did the films and main saga marathon take?"),
        index=index, frame_ids=[frame.frame_id for frame in index.frames],
        group_ids=[], certificate=_certificate(),
    )
    total = next(item for item in hints if item["operation"] == "duration_total")
    assert total["value"] == 3.5
    assert total["frame_ids"] == ["f0", "f1"]


def test_duration_total_excludes_unbound_nearby_measurement() -> None:
    index = _index()
    template_turn = index.turns[0]
    index.turns = [
        replace(template_turn, node_id="q:t:0", turn_index=0, text="Project Alpha took two weeks."),
        replace(template_turn, node_id="q:t:2", turn_index=2, text="Project Beta took one week."),
        replace(template_turn, node_id="q:t:4", turn_index=4, text="The fermentation process took five days."),
    ]
    first, second, third = index.frames[:3]
    first.quantity.value, first.quantity.unit, first.source_turn_ids = 2, "weeks", ["q:t:0"]
    second.quantity.value, second.quantity.unit, second.source_turn_ids = 1, "week", ["q:t:2"]
    third.quantity.value, third.quantity.unit, third.source_turn_ids = 5, "days", ["q:t:4"]
    hints = evaluate_operators(
        ir=build_query_ir("How many weeks did Project Alpha and Project Beta take?"),
        index=index,
        frame_ids=[first.frame_id, second.frame_id, third.frame_id],
        group_ids=[],
        certificate=_certificate(),
    )
    total = next(item for item in hints if item["operation"] == "duration_total")
    assert total["value"] == 3
    assert total["frame_ids"] == [first.frame_id, second.frame_id]


def test_lossless_temporal_pair_uses_calendar_dates_and_provenance() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:moma", session_id="s1",
            session_date="2023/01/08 (Sun) 12:49",
            text="I just returned from the Museum of Modern Art MoMA visit.",
        ),
        replace(
            template, node_id="q:met", session_id="s2",
            session_date="2023/01/15 (Sun) 00:27",
            text="Today I visited the Ancient Civilizations exhibit at the Metropolitan Museum of Art.",
        ),
    ]
    ir = build_query_ir(
        "How many days passed between my visit to the Museum of Modern Art (MoMA) "
        "and the Ancient Civilizations exhibit at the Metropolitan Museum of Art?"
    )
    hint = temporal_source_pair_hint(
        ir, index, ["q:moma", "q:met"]
    )
    assert hint is not None
    assert hint["value"] == 7 * 24 * 60 * 60
    assert hint["event_a_source_turn_id"] == "q:moma"
    assert hint["event_b_source_turn_id"] == "q:met"
    assert hint["certified"] is True


def test_temporal_order_resolves_relative_dates_for_two_named_events() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:jam", session_id="s1",
            session_date="2023/05/23 (Tue) 05:32",
            text="I met the woman selling jam at the farmer's market two weeks ago.",
        ),
        replace(
            template, node_id="q:tourist", session_id="s2",
            session_date="2023/05/23 (Tue) 01:01",
            text="I met the tourist from Australia last Thursday on the subway.",
        ),
    ]
    ir = build_query_ir(
        "Who did I meet first, the woman selling jam at the farmer's market "
        "or the tourist from Australia?"
    )
    hint = temporal_order_source_hint(
        ir, index, ["q:jam", "q:tourist"]
    )
    assert hint is not None
    assert hint["selected_target"] == "woman selling jam farmer's market"
    assert hint["selected_source_turn_id"] == "q:jam"
    assert hint["event_a_time"] < hint["event_b_time"]


def test_scalar_sum_excludes_planned_and_proposed_amounts() -> None:
    index = _index()
    template_turn = index.turns[0]
    index.turns = [
        replace(template_turn, node_id="q:t:0", text="I spent $200 on a gift for my sibling."),
        replace(template_turn, node_id="q:t:1", text="I paid $100 for another gift for my sibling."),
        replace(template_turn, node_id="q:t:2", text="I might spend $500 on a future gift."),
    ]
    template = index.frames[0]
    index.frames = [
        replace(
            template, frame_id="f0", entity_key="necklace gift",
            predicate_key="cost", lifecycle_status="completed",
            quantity=replace(template.quantity, value=200, unit="$"),
            source_turn_ids=["q:t:0"],
        ),
        replace(
            template, frame_id="f1", entity_key="gift card",
            predicate_key="amount", lifecycle_status="completed",
            quantity=replace(template.quantity, value=100, unit="$"),
            source_turn_ids=["q:t:1"],
        ),
        replace(
            template, frame_id="f2", entity_key="future gift",
            predicate_key="budget", lifecycle_status="planned",
            quantity=replace(template.quantity, value=500, unit="$"),
            source_turn_ids=["q:t:2"],
        ),
    ]
    hints = evaluate_operators(
        ir=build_query_ir("How much did I spend on gifts for my sibling?"),
        index=index, frame_ids=["f0", "f1", "f2"], group_ids=[],
        certificate=_certificate(),
    )
    total = next(item for item in hints if item["operation"] == "scalar_aggregate")
    assert total["aggregate"] == "sum"
    assert total["value"] == 300
    assert total["frame_ids"] == ["f0", "f1"]


def test_duration_total_rejects_unknown_frequency_from_same_topic() -> None:
    index = _index()
    template_turn = index.turns[0]
    index.turns = [
        replace(template_turn, node_id="q:t:0", text="I took a seven-day social media break."),
        replace(template_turn, node_id="q:t:1", text="I took a ten-day social media break."),
        replace(template_turn, node_id="q:t:2", text="I limit social media to fifteen minutes a day."),
    ]
    template = index.frames[0]
    index.frames = [
        replace(
            template, frame_id="f0", lifecycle_status="completed",
            retrieval_text="social media break duration",
            quantity=replace(template.quantity, value=7, unit="days"),
            source_turn_ids=["q:t:0"],
        ),
        replace(
            template, frame_id="f1", lifecycle_status="completed",
            retrieval_text="social media break duration",
            quantity=replace(template.quantity, value=10, unit="days"),
            source_turn_ids=["q:t:1"],
        ),
        replace(
            template, frame_id="f2", lifecycle_status="unknown",
            retrieval_text="social media daily usage limit",
            quantity=replace(template.quantity, value=1, unit="day"),
            source_turn_ids=["q:t:2"],
        ),
    ]
    hints = evaluate_operators(
        ir=build_query_ir("How many days did I take social media breaks in total?"),
        index=index, frame_ids=["f0", "f1", "f2"], group_ids=[],
        certificate=_certificate(),
    )
    total = next(item for item in hints if item["operation"] == "duration_total")
    assert total["value"] == 17
    assert total["frame_ids"] == ["f0", "f1"]


def test_authoritative_state_direction_requires_full_certificate() -> None:
    ir = build_query_ir(
        "Did I switch to more water per tablespoon of coffee, or less?"
    )
    trace = {
        "completeness_certificate": {
            "complete": True,
            "entity_match": True,
            "relation_match": True,
            "scope_match": True,
            "provenance_complete": True,
        },
        "generic_operator_hints": [{
            "operation": "latest_valid_state",
            "value": "1 tablespoon coffee per 5 ounces water",
            "change_direction": "less denominator quantity per numerator",
            "certified": True,
        }],
    }
    answer = authoritative_operator_answer(ir, trace)
    assert answer is not None
    assert answer.startswith("Less")
    trace["completeness_certificate"]["scope_match"] = False
    assert authoritative_operator_answer(ir, trace) is None


def test_query_ir_covers_recommendation_nouns_and_how_much_time() -> None:
    assert build_query_ir("What documentary recommendations would fit my interests?").requested_value_type == "recommendation"
    assert build_query_ir("How much time do I spend coding each day?").requested_value_type == "duration"


def test_dialogue_ordinal_precedes_recommendation_classification() -> None:
    ir = build_query_ir("What was the fifth bottle you recommended in that list?")
    assert ir.requested_value_type == "span"
    assert "ordered_items" in ir.required_roles


def test_currency_sum_normalizes_units_and_deduplicates_same_expense() -> None:
    index = _index()
    template_turn = index.turns[0]
    index.turns = [
        replace(template_turn, node_id="q:t:0", text="I spent $200 on a necklace gift for my sister."),
        replace(template_turn, node_id="q:t:1", text="I paid USD 100 for a spa gift card for my sister."),
        replace(template_turn, node_id="q:t:2", text="The necklace gift cost $200."),
    ]
    template = index.frames[0]
    index.frames = [
        replace(template, frame_id="f0", owner_key="me", entity_key="necklace gift", predicate_key="cost", lifecycle_status="completed", quantity=replace(template.quantity, value=200, unit="$"), source_turn_ids=["q:t:0"]),
        replace(template, frame_id="f1", owner_key="me", entity_key="spa gift card", predicate_key="cost", lifecycle_status="completed", quantity=replace(template.quantity, value=100, unit="USD"), source_turn_ids=["q:t:1"]),
        replace(template, frame_id="f2", owner_key="me", entity_key="necklace gift", predicate_key="cost", lifecycle_status="completed", quantity=replace(template.quantity, value=200, unit="USD"), source_turn_ids=["q:t:2"]),
    ]
    hints = evaluate_operators(
        ir=build_query_ir("How much did I spend on gifts for my sister?"),
        index=index, frame_ids=["f0", "f1", "f2"], group_ids=[], certificate=_certificate(),
    )
    hint = next(item for item in hints if item["operation"] == "scalar_aggregate")
    assert hint["value"] == 300
    trace = {"completeness_certificate": {"complete": True, "entity_match": True, "relation_match": True, "scope_match": True, "provenance_complete": True}, "generic_operator_hints": [hint]}
    assert authoritative_operator_answer(
        build_query_ir("How much did I spend on gifts for my sister?"), trace
    ) is None


def test_average_requires_minimum_plural_collection_cardinality() -> None:
    index = _index()
    template = index.frames[0]
    index.frames = [
        replace(template, frame_id=f"f{position}", entity_key=f"person {position}", lifecycle_status="completed", quantity=replace(template.quantity, value=value, unit="years"))
        for position, value in enumerate((30, 50, 70))
    ]
    hints = evaluate_operators(
        ir=build_query_ir("What is the average age of me, my parents, and my grandparents?"),
        index=index, frame_ids=[frame.frame_id for frame in index.frames], group_ids=[], certificate=_certificate(),
    )
    assert not any(item.get("operation") == "scalar_aggregate" for item in hints)


def test_record_time_source_hint_selects_lowest_elapsed_time() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(template, node_id="q:t:0", text="My personal best time was 27:12."),
        replace(template, node_id="q:t:1", text="I am hoping to beat my personal best time of 25:50."),
    ]
    ir = build_query_ir("What was my personal best time in the run?")
    hint = record_time_source_hint(ir, index, ["q:t:0", "q:t:1"])
    assert hint is not None
    assert hint["value"] == "25:50"
    trace = {
        "completeness_certificate": {"complete": True, "entity_match": True, "relation_match": True, "scope_match": True, "provenance_complete": True},
        "generic_operator_hints": [hint],
    }
    assert authoritative_operator_answer(ir, trace) == "25:50"


def test_collection_ledger_is_owner_bound_and_not_topic_bound() -> None:
    index = _index()
    template_turn = index.turns[0]
    index.turns = [
        replace(
            template_turn, node_id="q:t:dana", speaker="Dana",
            speaker_key="dana", transport_role="user",
            text="I attend sessions on Monday and Thursday.",
        ),
        replace(
            template_turn, node_id="q:t:morgan", speaker="Morgan",
            speaker_key="morgan", transport_role="assistant",
            text="I attend sessions on Tuesday and Wednesday.",
        ),
    ]
    template = index.frames[0]
    index.frames = [
        replace(
            template, frame_id="q:f:dana", owner_key="dana",
            entity_key="weekly sessions", predicate_key="attends",
            object_key="Monday Thursday", source_turn_ids=["q:t:dana"],
            lifecycle_status="ongoing", polarity="positive",
        ),
        replace(
            template, frame_id="q:f:morgan", owner_key="morgan",
            entity_key="weekly sessions", predicate_key="attends",
            object_key="Tuesday Wednesday", source_turn_ids=["q:t:morgan"],
            lifecycle_status="ongoing", polarity="positive",
        ),
    ]
    ir = build_query_ir("How many days a week does Dana attend sessions?")
    ledger = query_bound_collection_ledger(
        ir, index, ["q:t:dana", "q:t:morgan"],
        ["q:f:dana", "q:f:morgan"],
    )
    assert ledger is not None
    assert ledger["derived_value"] == 2
    assert ledger["derived_distinct_weekdays"] == ["monday", "thursday"]
    assert {row["owner"] for row in ledger["frame_candidates"]} == {"dana"}
    assert ledger["candidate_pool_complete"] is False


def test_collection_ledger_derives_generic_weekly_schedule_occurrences() -> None:
    index = _index()
    template_turn = index.turns[0]
    index.turns = [
        replace(
            template_turn, node_id="q:t:choir", speaker="Dana",
            speaker_key="dana", transport_role="user",
            text="Dana attends choir rehearsal on Monday and Wednesday.",
        ),
        replace(
            template_turn, node_id="q:t:dance", speaker="Dana",
            speaker_key="dana", transport_role="user",
            text="Dana attends dance rehearsal every Friday.",
        ),
    ]
    template = index.frames[0]
    index.frames = [
        replace(
            template, frame_id="q:f:choir", owner_key="dana",
            entity_key="choir rehearsal", predicate_key="attends",
            source_turn_ids=["q:t:choir"], lifecycle_status="ongoing",
        ),
        replace(
            template, frame_id="q:f:dance", owner_key="dana",
            entity_key="dance rehearsal", predicate_key="attends",
            source_turn_ids=["q:t:dance"], lifecycle_status="ongoing",
        ),
    ]
    ledger = query_bound_collection_ledger(
        build_query_ir("How many rehearsals does Dana attend in a typical week?"),
        index, ["q:t:choir", "q:t:dance"], ["q:f:choir", "q:f:dance"],
    )
    assert ledger is not None
    assert ledger["derived_weekly_occurrence_days"] == [
        "monday", "wednesday", "friday",
    ]
    assert ledger["derived_weekly_occurrence_value"] == 3


def test_collection_ledger_applies_to_member_location_queries() -> None:
    index = _index()
    template_turn = index.turns[0]
    index.turns = [replace(
        template_turn, node_id="q:t:oslo", speaker="Dana",
        speaker_key="dana", transport_role="user",
        text="Dana presented a workshop in Oslo.",
    )]
    template = index.frames[0]
    index.frames = [replace(
        template, frame_id="q:f:oslo", owner_key="dana",
        entity_key="workshop", predicate_key="presented in", object_key="Oslo",
        source_turn_ids=["q:t:oslo"], lifecycle_status="completed",
    )]
    ir = build_query_ir("Where has Dana presented workshops?")
    assert "members" in ir.required_roles
    ledger = query_bound_collection_ledger(
        ir, index, ["q:t:oslo"], ["q:f:oslo"],
    )
    assert ledger is not None
    assert ledger["frame_candidates"][0]["value"] == "Oslo"


def test_collection_ledger_keeps_semantic_priority_without_lexical_overlap() -> None:
    index = _index()
    template_turn = index.turns[0]
    index.turns = [replace(
        template_turn, node_id="q:t:semantic", transport_role="user",
        text="NebulaNet has been a lifesaver lately.",
    )]
    template = index.frames[0]
    index.frames = [replace(
        template, frame_id="q:f:semantic", owner_key="questioner",
        entity_key="questioner", predicate_key="weekends about",
        object_key="NebulaNet lately", source_turn_ids=["q:t:semantic"],
        lifecycle_status="ongoing",
    )]
    ledger = query_bound_collection_ledger(
        build_query_ir("How many network services have I used recently?"),
        index, ["q:t:semantic"], ["q:f:semantic"],
    )
    assert ledger is not None
    assert ledger["frame_candidates"][0]["frame_id"] == "q:f:semantic"
    assert ledger["lossless_candidates"][0]["source_turn_id"] == "q:t:semantic"



def test_collection_ledger_uses_routed_region_for_named_paraphrase() -> None:
    index = _index()
    template_turn = index.turns[0]
    index.turns = [replace(
        template_turn, node_id="q:t:nebula", session_id="nebula",
        transport_role="user", speaker_key="questioner",
        text="NebulaNet has been a lifesaver on weekends lately.",
    )]
    template = index.frames[0]
    index.frames = [replace(
        template, frame_id="q:f:nebula", session_ids=["nebula"],
        owner_key="questioner", entity_key="NebulaNet",
        predicate_key="weekends about", object_key="lifesaver lately",
        source_turn_ids=["q:t:nebula"], lifecycle_status="ongoing",
    )]
    card = replace(
        index.routing_cards[0], card_id="q:nebula:card",
        session_id="nebula", frame_ids=["q:f:nebula"],
        turn_ids=["q:t:nebula"],
        routing_text="questioner uses NebulaNet relay service on weekends",
    )
    index.routing_cards = [card]
    ledger = query_bound_collection_ledger(
        build_query_ir("How many relay services have I used recently?"),
        index, ["q:t:nebula"], ["q:f:nebula"],
        routed_session_ids=["nebula"],
    )
    assert ledger is not None
    candidate = ledger["lossless_candidates"][0]
    assert candidate["source_turn_id"] == "q:t:nebula"
    assert "routing_context" in candidate


def test_collection_ledger_skips_unscoped_open_activity_list() -> None:
    index = _index()
    template_turn = index.turns[0]
    index.turns = [replace(
        template_turn, node_id="q:t:activity", speaker="Dana",
        speaker_key="dana", transport_role="user",
        text="Dana paints and goes hiking.",
    )]
    template = index.frames[0]
    index.frames = [replace(
        template, frame_id="q:f:activity", owner_key="dana",
        entity_key="painting", predicate_key="does",
        source_turn_ids=["q:t:activity"], lifecycle_status="ongoing",
    )]
    ledger = query_bound_collection_ledger(
        build_query_ir("What activities does Dana partake in?"),
        index, ["q:t:activity"], ["q:f:activity"],
    )
    assert ledger is None


def test_collection_ledger_prioritizes_collection_scope_over_shared_action() -> None:
    index = _index()
    template_turn = index.turns[0]
    index.turns = [
        replace(
            template_turn, node_id="q:t:relevant", transport_role="user",
            text="I learned the Delta routing protocol last month.",
        ),
        replace(
            template_turn, node_id="q:t:distractor", transport_role="user",
            text="I learned project management last month.",
        ),
    ]
    template = index.frames[0]
    index.frames = [
        replace(
            template, frame_id="q:f:distractor", owner_key="questioner",
            entity_key="project management", predicate_key="learned",
            source_turn_ids=["q:t:distractor"], lifecycle_status="completed",
        ),
        replace(
            template, frame_id="q:f:relevant", owner_key="questioner",
            entity_key="Delta routing protocol", predicate_key="learned",
            source_turn_ids=["q:t:relevant"], lifecycle_status="completed",
        ),
    ]
    ledger = query_bound_collection_ledger(
        build_query_ir("How many protocols have I learned in the last month?"),
        index, ["q:t:distractor", "q:t:relevant"],
        ["q:f:distractor", "q:f:relevant"],
    )
    assert ledger is not None
    assert ledger["frame_candidates"][0]["frame_id"] == "q:f:relevant"


def test_collection_ledger_preserves_complementary_operation_sentences() -> None:
    index = _index()
    template_turn = index.turns[0]
    index.turns = [replace(
        template_turn, node_id="q:t:ops", transport_role="user",
        text=(
            "I need to return the old module to the store. "
            "It was defective. The replacement module still needs to be picked up."
        ),
    )]
    template = index.frames[0]
    index.frames = [replace(
        template, frame_id="q:f:ops", owner_key="questioner",
        entity_key="module replacement", predicate_key="return and pick up",
        source_turn_ids=["q:t:ops"], lifecycle_status="planned",
    )]
    ledger = query_bound_collection_ledger(
        build_query_ir(
            "How many equipment items do I need to pick up or return from the store?"
        ),
        index, ["q:t:ops"], ["q:f:ops"],
    )
    assert ledger is not None
    excerpt = ledger["lossless_candidates"][0]["excerpt"].casefold()
    assert "return the old module" in excerpt
    assert "replacement module" in excerpt
    assert "picked up" in excerpt


def test_collection_ledger_derives_generic_bounded_year_span() -> None:
    index = _index()
    template_turn = index.turns[0]
    index.turns = [
        replace(
            template_turn, node_id="q:t:start", speaker="Dana",
            speaker_key="dana", transport_role="user",
            text="Dana began the induction phase in 2011 and finished it in 2013.",
        ),
        replace(
            template_turn, node_id="q:t:end", speaker="Dana",
            speaker_key="dana", transport_role="user",
            text="Dana completed the professional certification in 2019.",
        ),
    ]
    template = index.frames[0]
    index.frames = [
        replace(
            template, frame_id="q:f:start", owner_key="dana",
            entity_key="induction phase", predicate_key="attended",
            source_turn_ids=["q:t:start"], lifecycle_status="completed",
        ),
        replace(
            template, frame_id="q:f:end", owner_key="dana",
            entity_key="professional certification", predicate_key="completed",
            source_turn_ids=["q:t:end"], lifecycle_status="completed",
        ),
    ]
    ledger = query_bound_collection_ledger(
        build_query_ir(
            "How many years did Dana spend from induction to completion of certification?"
        ),
        index, ["q:t:start", "q:t:end"], ["q:f:start", "q:f:end"],
    )
    assert ledger is not None
    assert ledger["derived_bounded_year_span"] == {
        "start_year": 2011, "end_year": 2019, "value": 8, "unit": "years",
    }


def test_first_person_collection_ledger_excludes_assistant_echo() -> None:
    index = _index()
    template_turn = index.turns[0]
    index.turns = [
        replace(
            template_turn, node_id="q:t:user", transport_role="user",
            text="I used the blue route on Monday.",
        ),
        replace(
            template_turn, node_id="q:t:assistant", transport_role="assistant",
            text="You used the blue route on Monday.",
        ),
    ]
    template = index.frames[0]
    index.frames = [
        replace(
            template, frame_id="q:f:user", owner_key="questioner",
            entity_key="blue route", predicate_key="used",
            source_turn_ids=["q:t:user"], lifecycle_status="completed",
        ),
        replace(
            template, frame_id="q:f:assistant", owner_key="assistant",
            entity_key="blue route", predicate_key="repeated",
            source_turn_ids=["q:t:assistant"], lifecycle_status="completed",
        ),
    ]
    ledger = query_bound_collection_ledger(
        build_query_ir("How many routes have I used?"), index,
        ["q:t:user", "q:t:assistant"], ["q:f:user", "q:f:assistant"],
    )
    assert ledger is not None
    assert [row["frame_id"] for row in ledger["frame_candidates"]] == ["q:f:user"]
    assert [row["source_turn_id"] for row in ledger["lossless_candidates"]] == ["q:t:user"]


def test_collection_ledger_reconsiders_all_frames_in_bounded_sources() -> None:
    index = _index()
    template_turn = index.turns[0]
    index.turns = [
        replace(
            template_turn, node_id="q:t:aster", session_id="aster",
            transport_role="user", speaker_key="questioner",
            text="I completed the Aster lattice module.",
        ),
        replace(
            template_turn, node_id="q:t:quill", session_id="quill",
            transport_role="user", speaker_key="questioner",
            text="I also built the Quill lattice module.",
        ),
    ]
    template = index.frames[0]
    index.frames = [
        replace(
            template, frame_id="q:f:aster", session_ids=["aster"],
            owner_key="questioner", entity_key="Aster lattice module",
            predicate_key="completed", source_turn_ids=["q:t:aster"],
            lifecycle_status="completed",
        ),
        replace(
            template, frame_id="q:f:quill", session_ids=["quill"],
            owner_key="questioner", entity_key="Quill lattice module",
            predicate_key="built", source_turn_ids=["q:t:quill"],
            lifecycle_status="completed",
        ),
    ]
    ledger = query_bound_collection_ledger(
        build_query_ir("How many lattice modules have I built?"),
        index, ["q:t:aster", "q:t:quill"], ["q:f:aster"],
    )
    assert ledger is not None
    assert {
        row["frame_id"] for row in ledger["structured_member_candidates"]
    } == {"q:f:aster", "q:f:quill"}


def test_collection_ledger_exposes_unframed_lossless_candidate() -> None:
    index = _index()
    template_turn = index.turns[0]
    index.turns = [replace(
        template_turn, node_id="q:t:orion", session_id="orion",
        transport_role="user", speaker_key="questioner",
        text="I started working on an Orion lattice module.",
    )]
    index.frames = []
    ledger = query_bound_collection_ledger(
        build_query_ir("How many lattice modules have I worked on?"),
        index, ["q:t:orion"], [], routed_session_ids=["orion"],
    )
    assert ledger is not None
    assert ledger["structured_member_candidates"] == []
    assert [
        row["source_turn_id"]
        for row in ledger["unframed_lossless_candidates"]
    ] == ["q:t:orion"]


def test_collection_ledger_builds_explicit_reference_closure() -> None:
    index = _index()
    template_turn = index.turns[0]
    index.turns = [replace(
        template_turn, node_id="q:t:zephyr", session_id="zephyr",
        transport_role="user", speaker_key="questioner",
        text=(
            "I started working on a display featuring a Zephyr lattice module, "
            "and I am refining its finish."
        ),
    )]
    index.frames = []
    ledger = query_bound_collection_ledger(
        build_query_ir("How many lattice modules have I worked on?"),
        index, ["q:t:zephyr"], [], routed_session_ids=["zephyr"],
    )
    assert ledger is not None
    assert ledger["relation_closure_candidates"][0]["target"] == (
        "Zephyr lattice module"
    )
    assert ledger["relation_closure_candidates"][0]["relation"] == "featuring"


def test_date_operator_binds_relation_and_resolves_relative_source_time() -> None:
    index = _index()
    template_turn = index.turns[0]
    index.turns = [
        replace(
            template_turn, node_id="q:t:paint", speaker="Dana",
            speaker_key="dana", transport_role="user",
            session_date="1:56 pm on 8 May, 2023",
            text="I painted the amber sunrise last year.",
        ),
        replace(
            template_turn, node_id="q:t:garden", speaker="Dana",
            speaker_key="dana", transport_role="user",
            session_date="1:56 pm on 8 May, 2024",
            text="I planted the garden today.",
        ),
    ]
    template = index.frames[0]
    index.frames = [
        replace(
            template, frame_id="q:f:paint", owner_key="dana",
            entity_key="dana", predicate_key="painted",
            object_key="amber sunrise", context_key="last year",
            source_turn_ids=["q:t:paint"], lifecycle_status="completed",
            temporal=replace(
                template.temporal, event_time=None, start=None,
                observed_at="1:56 pm on 8 May, 2023",
            ),
        ),
        replace(
            template, frame_id="q:f:garden", owner_key="dana",
            entity_key="garden", predicate_key="planted",
            object_key="flowers", source_turn_ids=["q:t:garden"],
            lifecycle_status="completed",
            temporal=replace(
                template.temporal, event_time="2024-05-08",
                observed_at="1:56 pm on 8 May, 2024",
            ),
        ),
    ]
    hints = evaluate_operators(
        ir=build_query_ir("When did Dana paint the amber sunrise?"),
        index=index, frame_ids=["q:f:paint", "q:f:garden"],
        group_ids=[], certificate=_certificate(),
    )
    event = next(row for row in hints if row["operation"] == "event_time")
    assert event["value"].startswith("2022-05-08")
    assert event["frame_ids"] == ["q:f:paint"]


def test_counterfactual_dependency_requires_same_source_causal_binding() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:t:causal", speaker="Dana",
            speaker_key="dana", transport_role="user",
            text=(
                "The guidance made me realize how important safe systems are, "
                "so I started pursuing robotics as a career."
            ),
        ),
        replace(
            template, node_id="q:t:unbound", speaker="Dana",
            speaker_key="dana", transport_role="user",
            text="I received unrelated advice about gardening.",
        ),
    ]
    ir = build_query_ir(
        "Would Dana still pursue robotics as a career if she had not received guidance?"
    )
    hint = counterfactual_dependency_hint(
        ir, index, ["q:t:causal", "q:t:unbound"]
    )
    assert hint is not None
    assert hint["value"] == "likely_no"
    assert hint["source_turn_ids"] == ["q:t:causal"]
    assert counterfactual_dependency_hint(
        ir, index, ["q:t:unbound"]
    ) is None



def test_routed_schedule_closure_recovers_weekday_outside_fine_pack() -> None:
    index = _index()
    turn_template = index.turns[0]
    index.turns = [
        replace(
            turn_template, node_id="q:s1:t0", session_id="s1",
            transport_role="user", speaker_key="questioner",
            text=(
                "I attend fitness classes on Tuesdays, Thursdays, and "
                "Saturdays."
            ),
        ),
        replace(
            turn_template, node_id="q:s2:t0", session_id="s2",
            transport_role="user", speaker_key="questioner",
            text="I also started taking a yoga class on Wednesdays.",
        ),
    ]
    frame_template = index.frames[0]
    index.frames = [
        replace(
            frame_template, frame_id="q:s1:f0", session_ids=["s1"],
            owner_key="questioner", entity_key="fitness classes",
            predicate_key="attend", object_key="Tuesday Thursday Saturday",
            source_turn_ids=["q:s1:t0"], lifecycle_status="ongoing",
        ),
        replace(
            frame_template, frame_id="q:s2:f0", session_ids=["s2"],
            owner_key="questioner", entity_key="yoga class",
            predicate_key="started taking", object_key="Wednesday",
            source_turn_ids=["q:s2:t0"], lifecycle_status="ongoing",
        ),
    ]
    card_template = index.routing_cards[0]
    index.routing_cards = [
        replace(
            card_template, card_id="q:s1:card", session_id="s1",
            frame_ids=["q:s1:f0"], turn_ids=["q:s1:t0"],
            canonical_entities=["fitness classes"], relations=["attends"],
            routing_text="questioner attends fitness classes every week",
        ),
        replace(
            card_template, card_id="q:s2:card", session_id="s2",
            frame_ids=["q:s2:f0"], turn_ids=["q:s2:t0"],
            canonical_entities=["yoga class"], relations=["started taking"],
            routing_text="questioner takes a yoga class every week",
        ),
    ]
    index.evidence_groups = []
    index.edges = []
    ir = build_query_ir("How many days a week do I attend fitness classes?")
    ledger = query_bound_collection_ledger(
        ir, index, ["q:s1:t0"], ["q:s1:f0"],
        routed_session_ids=["s1", "s2"],
    )
    assert ledger is not None
    assert ledger["derived_distinct_weekdays"] == [
        "tuesday", "wednesday", "thursday", "saturday",
    ]
    assert ledger["certified"] is True
    assert authoritative_operator_answer(
        ir, {"query_bound_collection_ledger": ledger}
    ) == "4 days a week."


def test_local_collection_certificate_requires_all_four_bindings() -> None:
    ir = build_query_ir("How many modules have I built?")
    base = {
        "certified": True,
        "proposed_distinct_target_count": 3,
        "operator_certificate": {
            "entity_match": True,
            "relation_match": True,
            "scope_match": True,
            "provenance_complete": True,
        },
    }
    assert authoritative_operator_answer(
        ir, {"query_bound_collection_ledger": base}
    ) is None
    broken = dict(base)
    broken["operator_certificate"] = {
        **base["operator_certificate"], "scope_match": False,
    }
    assert authoritative_operator_answer(
        ir, {
            "query_bound_collection_ledger": broken,
            "completeness_certificate": {},
        }
    ) is None


def test_transaction_sum_binds_local_clauses_and_unit_prices() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(template, node_id="q:m1", session_id="s1", transport_role="user", text="I sold herbs at the market, earning a total of $120."),
        replace(template, node_id="q:m2", session_id="s2", transport_role="user", text="I sold jam at another market, earning $225."),
        replace(template, node_id="q:m3", session_id="s3", transport_role="user", text="I sold 20 potted plants at the market for $7.5 each."),
        replace(template, node_id="q:noise", session_id="s3", transport_role="user", text="The loyalty program offers points after customers spend $50."),
    ]
    ir = build_query_ir("What is the total amount of money I earned from selling products at the markets?")
    hint = transaction_sum_from_sources_hint(
        ir, index, [turn.node_id for turn in index.turns]
    )
    assert hint is not None
    assert hint["value"] == 495
    assert {row["source_turn_id"] for row in hint["operands"]} == {"q:m1", "q:m2", "q:m3"}


def test_transaction_sum_accepts_thousands_separators() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(template, node_id="q:m1", session_id="s1", transport_role="user", text="I sold equipment at the market and earned $1,200."),
        replace(template, node_id="q:m2", session_id="s2", transport_role="user", text="I sold supplies at another market and earned $350."),
    ]
    ir = build_query_ir("What is the total amount of money I earned from selling products at the markets?")
    hint = transaction_sum_from_sources_hint(
        ir, index, [turn.node_id for turn in index.turns]
    )
    assert hint is not None
    assert hint["value"] == 1550


def test_explicit_currency_operands_bind_nearest_prices() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(template, node_id="q:p1", session_id="s1", transport_role="user", text="I got a new food bowl for $15, and a measuring cup for $5."),
        replace(template, node_id="q:p2", session_id="s2", transport_role="user", text="The dental chews are $10 a pack."),
        replace(template, node_id="q:p3", session_id="s3", transport_role="user", text="I got a flea and tick collar for $20."),
        replace(template, node_id="q:p4", session_id="s3", transport_role="user", text="I also bought a dog bed for $40."),
    ]
    ir = build_query_ir("What is the total cost of the new food bowl, measuring cup, dental chews, and flea and tick collar?")
    hint = transaction_sum_from_sources_hint(
        ir, index, [turn.node_id for turn in index.turns]
    )
    assert hint is not None
    assert hint["value"] == 50
    assert [row["value"] for row in hint["operand_bindings"]] == [15, 5, 10, 20]


def test_temporal_order_supports_article_month_ago() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(template, node_id="q:case", session_id="s1", transport_role="user", session_date="2023/05/26 (Fri) 19:11", text="I received my new phone case about a month ago."),
        replace(template, node_id="q:charger", session_id="s2", transport_role="user", session_date="2023/05/26 (Fri) 18:20", text="I lost my phone charger about two weeks ago."),
    ]
    ir = build_query_ir("Which event happened first, losing my phone charger or receiving my new phone case?")
    hint = temporal_order_source_hint(ir, index, ["q:case", "q:charger"])
    assert hint is not None
    assert hint["selected_target"] == "receiving new phone case"


def test_temporal_pair_rejects_generic_lexical_collision() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(template, node_id="q:noise", session_id="s0", transport_role="user", session_date="2023/05/20 (Sat) 16:32", text="I frequently use the train between New York and Boston."),
        replace(template, node_id="q:binoculars", session_id="s1", transport_role="user", session_date="2023/05/20 (Sat) 20:08", text="I got my new binoculars exactly three weeks ago."),
        replace(template, node_id="q:birds", session_id="s2", transport_role="user", session_date="2023/05/20 (Sat) 22:12", text="A week ago I saw the American goldfinches returning to the area."),
    ]
    ir = build_query_ir("How long did I use my new binoculars before I saw the American goldfinches returning to the area?")
    hint = temporal_source_pair_hint(ir, index, [turn.node_id for turn in index.turns])
    assert hint is not None
    assert hint["event_a_source_turn_id"] == "q:binoculars"
    assert hint["event_b_source_turn_id"] == "q:birds"
    assert hint["value"] == 2
    assert hint["unit"] == "weeks"
    assert round(hint["raw_seconds"] / (7 * 24 * 60 * 60)) == 2



def test_dated_event_count_keeps_provider_abbreviation_attached() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(template, node_id="q:visit1", transport_role="user", text="I had a follow-up appointment with surgeon Dr. Vale on March 20th."),
        replace(template, node_id="q:visit2", transport_role="user", text="I finally went to see my primary care physician, Dr. Reed, on March 3rd."),
        replace(template, node_id="q:future", transport_role="user", text="I am considering an appointment with Dr. Lane on March 28th."),
    ]
    hint = dated_event_count_from_sources_hint(
        build_query_ir("How many doctor's appointments did I go to in March?"),
        index, [turn.node_id for turn in index.turns],
    )
    assert hint is not None
    assert hint["value"] == 2
    assert {row["date"] for row in hint["members"]} == {"March 3", "March 20"}


def test_same_unit_state_difference_binds_old_and_current_values() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(template, node_id="q:old", transport_role="user", text="A few months ago it averaged 30 miles per gallon."),
        replace(template, node_id="q:now", transport_role="user", text="Lately it is getting 28 miles per gallon."),
    ]
    hint = same_unit_state_difference_hint(
        build_query_ir("How much more miles per gallon was it getting a few months ago compared to now?"),
        index, [turn.node_id for turn in index.turns],
    )
    assert hint is not None
    assert hint["value"] == 2
    assert [row["value"] for row in hint["operands"]] == [30, 28]


def test_maintenance_count_collapses_component_to_parent_asset() -> None:
    index = _index()
    template_turn = index.turns[0]
    index.turns = [
        replace(template_turn, node_id="q:tire", transport_role="user", text="I planned to replace the commuter bike front tire."),
        replace(template_turn, node_id="q:road", transport_role="user", text="I serviced the road bike and cleaned its chain."),
    ]
    first, second = index.frames[:2]
    first.owner_key = second.owner_key = "participant 1"
    first.entity_key, first.predicate_key = "commuter bike front tire", "replace"
    first.retrieval_text, first.source_turn_ids = "replace commuter bike front tire", ["q:tire"]
    second.entity_key, second.predicate_key = "road bike", "serviced"
    second.retrieval_text, second.source_turn_ids = "serviced road bike", ["q:road"]
    index.frames = [first, second]
    hint = maintenance_entity_count_hint(
        build_query_ir("How many bikes did I service or plan to service?"),
        index, ["q:tire", "q:road"],
    )
    assert hint is not None
    assert hint["value"] == 2
    assert {row["identity"] for row in hint["members"]} == {"commuter bike", "road bike"}


def test_pending_operation_pairs_keep_return_and_replacement_pickup_distinct() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(template, node_id="q:coat", transport_role="user", text="I still need to pick up my navy coat from the cleaner."),
        replace(template, node_id="q:shoes", transport_role="user", text="I exchanged some shoes at the shop but still need to pick them up."),
    ]
    hint = pending_operation_target_pairs_hint(
        build_query_ir("How many items do I need to pick up or return from a store?"),
        index, [turn.node_id for turn in index.turns],
    )
    assert hint is not None
    assert hint["value"] == 3
    assert {(row["operation"], row["target"]) for row in hint["members"]} == {
        ("pickup", "navy coat"), ("return", "shoes"),
        ("pickup", "replacement shoes"),
    }
