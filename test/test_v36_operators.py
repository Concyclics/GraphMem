from __future__ import annotations

from dataclasses import replace

from graphmem_demo.v36.operators import (
    evaluate_operators, exact_entity_absence_hint, query_bound_collection_ledger, counterfactual_dependency_hint, record_time_source_hint, temporal_order_source_hint,
    open_temporal_sequence_from_sources_hint,
    temporal_source_pair_hint, transaction_sum_from_sources_hint,
    dated_event_count_from_sources_hint, same_unit_state_difference_hint,
    category_acquisition_members_hint, maintenance_entity_count_hint,
    pending_operation_target_pairs_hint, paired_metric_total_from_sources_hint,
    named_event_attendance_count_hint, binary_savings_from_sources_hint,
    temporal_predecessor_entity_hint, relative_anchor_source_hint,
    latest_scalar_state_from_sources_hint,
    threshold_progress_remaining_hint, latest_approx_scalar_state_hint,
    latest_labeled_currency_state_hint, latest_weekly_schedule_time_hint,
    same_unit_acquisition_total_hint,
    age_arithmetic_from_sources_hint,
    advance_booking_recency_from_sources_hint,
    current_role_duration_from_sources_hint,
)
from graphmem_demo.v36.operators import (
    currency_extreme_entity_from_sources_hint,
    dialogue_attribute_match_hint,
    family_relation_total_from_sources_hint,
    latest_category_start_from_sources_hint,
    linked_event_date_from_sources_hint,
    preference_constraints_from_sources_hint,
    presupposed_event_absence_hint,
    weekly_schedule_days_from_sources_hint,
    dialogue_final_choice_from_sources_hint,
    completed_item_metric_total_from_sources_hint,
    scoped_completed_duration_total_from_sources_hint,
    relative_value_multiplier_from_sources_hint,
    relative_duration_at_event_from_sources_hint,
    prior_candidate_count_from_sources_hint,
    completed_carrier_sequence_from_sources_hint,
    event_endpoint_difference_from_sources_hint,
    travel_arrival_time_from_sources_hint,
    completed_work_subtype_total_from_sources_hint,
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
    assert hint["value"] == 7
    assert hint["unit"] == "days"
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


def test_record_time_source_hint_selects_previous_observed_version() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:old", session_date="2023/01/01 08:00",
            transport_role="user",
            text="I finished the charity run with a personal best time of 27:45.",
        ),
        replace(
            template, node_id="q:new", session_date="2023/02/01 08:00",
            transport_role="user",
            text="My latest personal best time for the charity run is 26:30.",
        ),
        replace(
            template, node_id="q:noise", session_date="2023/02/01 08:00",
            transport_role="assistant",
            text="Use an 8:15 pace when training for that personal best.",
        ),
    ]
    hint = record_time_source_hint(
        build_query_ir(
            "What was my previous personal best time for the charity run?"
        ),
        index,
        [turn.node_id for turn in index.turns],
    )
    assert hint is not None
    assert hint["value"] == "27:45"
    assert hint["selection"] == "previous_observed_version"


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


def test_paired_metric_total_parses_thousands_separators() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:first", session_id="s:first",
            transport_role="user",
            text="My YouTube video received 1,456 views.",
        ),
        replace(
            template, node_id="q:second", session_id="s:second",
            transport_role="user",
            text="My TikTok video received 542 views.",
        ),
    ]
    hint = paired_metric_total_from_sources_hint(
        build_query_ir(
            "What was the total number of views on my YouTube and TikTok videos?"
        ),
        index,
        [turn.node_id for turn in index.turns],
    )
    assert hint is not None
    assert [row["value"] for row in hint["operands"]] == [1456, 542]
    assert hint["value"] == 1998


def test_paired_metric_total_binds_named_entities_and_metric_aliases() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:facebook", session_id="s:facebook",
            transport_role="user",
            text="My Facebook ad campaign reached around 2,000 people.",
        ),
        replace(
            template, node_id="q:instagram", session_id="s:instagram",
            transport_role="user",
            text="My Instagram influencer collaboration reached 10,000 followers.",
        ),
        replace(
            template, node_id="q:noise", session_id="s:noise",
            transport_role="user",
            text="I invited 10 people to dinner and discussed social media.",
        ),
    ]
    hint = paired_metric_total_from_sources_hint(
        build_query_ir(
            "What was the total number of people reached by my Facebook "
            "ad campaign and Instagram influencer collaboration?"
        ),
        index,
        ["q:facebook", "q:noise"],
    )
    assert hint is not None
    assert [row["entity"] for row in hint["operands"]] == [
        "Facebook", "Instagram",
    ]
    assert [row["value"] for row in hint["operands"]] == [2000, 10000]
    assert hint["value"] == 12000


def test_named_event_attendance_count_deduplicates_completed_experiences() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:austin", transport_role="user",
            text=(
                "I participated in a challenge at the Austin Film Festival. "
                "I later discussed the Austin Film Festival again."
            ),
        ),
        replace(
            template, node_id="q:seattle", transport_role="user",
            text=(
                "I attended a Q&A after a screening at the "
                "Seattle International Film Festival."
            ),
        ),
        replace(
            template, node_id="q:portland", transport_role="user",
            text="I volunteered at the Portland Film Festival.",
        ),
        replace(
            template, node_id="q:afi", transport_role="user",
            text="I attended AFI Fest and saw a new film.",
        ),
        replace(
            template, node_id="q:recommend", transport_role="assistant",
            text="You should consider Sundance Film Festival.",
        ),
    ]
    hint = named_event_attendance_count_hint(
        build_query_ir("How many movie festivals that I attended?"),
        index,
        ["q:austin"],
    )
    assert hint is not None
    assert hint["value"] == 4
    assert {row["identity"] for row in hint["members"]} == {
        "Austin Film Festival",
        "Seattle International Film Festival",
        "Portland Film Festival",
        "AFI Fest",
    }


def test_attendance_occurrences_deduplicate_people_and_exclude_missed() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:emma1", session_id="s:emma",
            transport_role="user",
            text="I attended my cousin Emma's preschool graduation two months ago.",
        ),
        replace(
            template, node_id="q:emma2", session_id="s:emma",
            transport_role="user",
            text="It feels like yesterday I was attending Emma's graduation ceremony.",
        ),
        replace(
            template, node_id="q:alex", session_id="s:alex",
            transport_role="user",
            text="I attended my colleague Alex's graduation a few weeks ago.",
        ),
        replace(
            template, node_id="q:rachel", session_id="s:rachel",
            transport_role="user",
            text="I attended my friend Rachel's graduation ceremony.",
        ),
        replace(
            template, node_id="q:missed", session_id="s:jack",
            transport_role="user",
            text="I missed my nephew Jack's graduation ceremony.",
        ),
    ]
    hint = named_event_attendance_count_hint(
        build_query_ir(
            "How many graduation ceremonies have I attended "
            "in the past three months?"
        ),
        index,
        [turn.node_id for turn in index.turns],
    )
    assert hint is not None
    assert hint["value"] == 3
    assert {row["identity"] for row in hint["members"]} == {
        "Emma", "Alex", "Rachel",
    }


def test_named_event_attendance_never_treats_auxiliary_have_as_category() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:noise", transport_role="user",
            text="I have volunteered at a local event and assisted guests.",
        ),
        replace(
            template, node_id="q:graduation", transport_role="user",
            text="I attended Emma's preschool graduation ceremony.",
        ),
    ]
    hint = named_event_attendance_count_hint(
        build_query_ir(
            "How many graduation ceremonies have I attended "
            "in the past three months?"
        ),
        index,
        [turn.node_id for turn in index.turns],
    )
    assert hint is None or all(
        row["identity"].casefold() != "have"
        for row in hint["members"]
    )


def test_exact_absence_binds_possessive_relation_role() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:niece", transport_role="user",
            text=(
                "I baked a lemon cake for my niece's birthday party."
            ),
        )
    ]
    hint = exact_entity_absence_hint(
        build_query_ir("What did I bake for my uncle's birthday party?"),
        index,
    )
    assert hint is not None
    assert hint["binding_kind"] == "required_role"
    assert hint["required_marker"] == "uncle"


def test_exact_absence_binds_collection_subtype_not_sibling() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:baseball", transport_role="user",
            text=(
                "I added 20 autographed baseballs to my collection "
                "in the first three months."
            ),
        )
    ]
    hint = exact_entity_absence_hint(
        build_query_ir(
            "How many autographed football have I added to my collection "
            "in the first three months?"
        ),
        index,
    )
    assert hint is not None
    assert hint["binding_kind"] == "required_collection_type"
    assert hint["required_phrase"] == "autograph football"


def test_binary_savings_requires_both_user_provided_costs() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:taxi", transport_role="user",
            text="The taxi from the airport to my hotel will cost $60.",
        ),
        replace(
            template, node_id="q:assistant-bus",
            transport_role="assistant",
            text="A bus often costs $10.",
        ),
    ]
    hint = binary_savings_from_sources_hint(
        build_query_ir(
            "How much will I save by taking the bus from the airport "
            "to my hotel instead of a taxi?"
        ),
        index,
        [turn.node_id for turn in index.turns],
    )
    assert hint is not None
    assert hint["operation"] == "exact_entity_absence"
    assert hint["binding_kind"] == "required_operand"
    assert hint["required_phrase"] == "bus cost"


def test_relative_anchor_extracts_action_with_named_companion() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:rachel", session_id="s:rachel",
            session_date="2023/02/01 (Wed) 14:38",
            turn_index=0, transport_role="user",
            text=(
                "I just started taking ukulele lessons with my friend Rachel "
                "today and it has been fun."
            ),
        )
    ]
    hint = relative_anchor_source_hint(
        build_query_ir(
            "What did I do with Rachel on the Wednesday two months ago?"
        ),
        index,
        "2023/04/01 (Sat) 10:00",
        allowed_session_ids={"s:rachel"},
    )
    assert hint is not None
    assert "started taking ukulele lessons" in hint["answer_candidate"]


def test_maintenance_count_recovers_parent_asset_from_lossless_source() -> None:
    index = _index()
    turn = replace(
        index.turns[0], node_id="q:tire", transport_role="user",
        text="I plan to replace the front tire on my commuter bike this month.",
    )
    index.turns = [turn]
    frame = index.frames[0]
    frame.owner_key = "participant 1"
    frame.entity_key = "front tire"
    frame.predicate_key = "replacement planned"
    frame.retrieval_text = "front tire replacement planned"
    frame.source_turn_ids = [turn.node_id]
    index.frames = [frame]
    hint = maintenance_entity_count_hint(
        build_query_ir("How many bikes did I service or plan to service?"),
        index, [turn.node_id],
    )
    assert hint is not None
    assert [row["identity"] for row in hint["members"]] == ["commuter bike"]


def test_acquisition_splits_compound_members_from_one_source() -> None:
    index = _index()
    turn = replace(
        index.turns[0], node_id="q:plants", transport_role="user",
        text="I bought the peace lily and a succulent plant two weeks ago.",
    )
    index.turns = [turn]
    first, second = index.frames[:2]
    for frame in (first, second):
        frame.owner_key = "participant 1"
        frame.source_turn_ids = [turn.node_id]
        frame.semantic_type_keys = ["plant"]
    first.entity_key = "peace lily"
    second.entity_key = "succulent plant acquired with peace lily"
    index.frames = [first, second]
    hint = category_acquisition_members_hint(
        build_query_ir("How many plants did I acquire?"),
        index, [turn.node_id],
    )
    assert hint is not None
    assert {row["identity"] for row in hint["members"]} == {
        "peace lily", "succulent",
    }


def test_temporal_order_converts_past_duration_to_start_endpoint() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:festival", session_id="s1",
            transport_role="user", session_date="2023/05/27 (Sat) 21:39",
            text="I attended a cultural festival yesterday.",
        ),
        replace(
            template, node_id="q:spanish", session_id="s2",
            transport_role="user", session_date="2023/05/27 (Sat) 14:08",
            text="I have been taking Spanish classes for the past three months.",
        ),
    ]
    ir = build_query_ir(
        "Which event happened first, my attendance at a cultural festival "
        "or the start of my Spanish classes?"
    )
    hint = temporal_order_source_hint(
        ir, index, ["q:festival", "q:spanish"],
    )
    assert hint is not None
    assert hint["selected_target"] == "start spanish classes"
    assert hint["selected_source_turn_id"] == "q:spanish"


def test_exact_entity_absence_uses_named_relation_near_match() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:smith", transport_role="user",
            text="I see Dr. Smith every two weeks.",
        )
    ]
    hint = exact_entity_absence_hint(
        build_query_ir("How often do I see Dr. Johnson?"), index,
    )
    assert hint is not None
    assert hint["required_marker"] == "johnson"
    assert hint["value"] == "insufficient"


def test_exact_entity_absence_ignores_unrelated_named_entity_scene() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:photos", transport_role="user",
            text="I edit my holiday pictures with Google Photos.",
        ),
        replace(
            template, node_id="q:work", transport_role="user",
            text="I have worked professionally for nine years at NovaTech.",
        ),
    ]
    hint = exact_entity_absence_hint(
        build_query_ir(
            "How long had I been working before I started my current job at Google?"
        ),
        index,
    )
    assert hint is not None
    assert hint["required_marker"] == "google"


def test_exact_entity_absence_requires_every_conjoined_component() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:tomatoes", transport_role="user",
            text="I initially planted five tomato plants.",
        )
    ]
    hint = exact_entity_absence_hint(
        build_query_ir(
            "How many plants did I initially plant for tomatoes and chili peppers?"
        ),
        index,
    )
    assert hint is not None
    assert hint["required_phrase"] == "chili pepper"
    assert hint["required_components"] == ["tomatoes", "chili peppers"]


def test_exact_entity_absence_detects_terminal_camelcase_operand() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:headphones", transport_role="user",
            text="I recently purchased headphones for $378.",
        ),
        replace(
            template, node_id="q:generic-ipad", transport_role="assistant",
            text="An iPad is one example of a tablet.",
        ),
    ]
    hint = exact_entity_absence_hint(
        build_query_ir(
            "What is the total cost of my recently purchased headphones and the iPad?"
        ),
        index,
    )
    assert hint is not None
    assert hint["required_phrase"] == "ipad"
    assert hint["binding_kind"] == "required_component"


def test_exact_entity_absence_requires_both_named_comparison_targets() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:alex", transport_role="user",
            text="My friend Alex became a parent in January.",
        )
    ]
    hint = exact_entity_absence_hint(
        build_query_ir("Who became a parent first, Tom or Alex?"),
        index,
    )
    assert hint is not None
    assert hint["required_marker"] == "tom"
    assert hint["binding_kind"] == "named_entity"


def test_exact_entity_absence_does_not_reject_new_recommendation_target() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:hotel", transport_role="user",
            text="I prefer hotels with ocean views and rooftop pools.",
        )
    ]
    assert exact_entity_absence_hint(
        build_query_ir("Can you recommend a hotel in Miami?"), index,
    ) is None


def test_exact_entity_absence_rejects_compound_partial_match() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:tennis", transport_role="user",
            text="I play tennis with my friends at the local park every other week.",
        )
    ]
    hint = exact_entity_absence_hint(
        build_query_ir(
            "How often do I play table tennis with my friends at the local park?"
        ),
        index,
    )
    assert hint is not None
    assert hint["required_phrase"] == "table tenni"
    assert hint["value"] == "insufficient"



def test_temporal_order_requires_target_exclusive_event_identity() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:bake", session_date="2023/03/15",
            turn_index=0, transport_role="user",
            text="I helped organize a charity bake sale at my office today.",
        ),
        replace(
            template, node_id="q:walk", session_date="2023/03/15",
            turn_index=2, transport_role="user",
            text="I participated in a charity walk last month.",
        ),
        replace(
            template, node_id="q:gala", session_date="2023/03/28",
            turn_index=0, transport_role="user",
            text="I am mentioning my charity gala attendance tonight.",
        ),
    ]
    hint = temporal_order_source_hint(
        build_query_ir(
            "Which event did I participate in first, the charity gala or "
            "the charity bake sale?"
        ),
        index, ["q:bake", "q:walk", "q:gala"],
    )
    assert hint is not None
    assert hint["selected_target"] == "charity bake sale"
    assert hint["event_a_source_turn_id"] == "q:gala"
    assert hint["event_b_source_turn_id"] == "q:bake"


def test_latest_scalar_state_uses_newest_user_correction() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:old", session_date="2023/07/11",
            turn_index=10, transport_role="user",
            text="To correct myself, I actually need 125 stars to reach "
                 "the Gold level on my Rewards app, not 400.",
        ),
        replace(
            template, node_id="q:noise", session_date="2023/07/25",
            turn_index=0, transport_role="assistant",
            text="You may need 300 stars to reach the Gold level.",
        ),
        replace(
            template, node_id="q:new", session_date="2023/07/30",
            turn_index=6, transport_role="user",
            text="Actually, I need 120 stars to reach the gold level on "
                 "my Starbucks Rewards app, not 300.",
        ),
    ]
    hint = latest_scalar_state_from_sources_hint(
        build_query_ir(
            "How many stars do I need to reach the gold level on my "
            "Starbucks Rewards app?"
        ),
        index, ["q:old", "q:noise", "q:new"],
    )
    assert hint is not None
    assert hint["value"] == 120
    assert hint["source_turn_ids"] == ["q:new"]


def test_same_unit_acquisition_total_closes_semantic_category() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:feed", session_date="2023/05/22",
            turn_index=0, transport_role="user",
            text="I recently purchased new layer feed. I got a 50-pound "
                 "batch to track my farm expenses.",
        ),
        replace(
            template, node_id="q:grain", session_date="2023/05/22",
            turn_index=0, transport_role="user",
            text="I also bought 20 pounds of organic scratch grains for "
                 "my chickens recently.",
        ),
        replace(
            template, node_id="q:assistant", session_date="2023/05/23",
            turn_index=0, transport_role="assistant",
            text="Consider ordering 100 pounds of feed next season.",
        ),
        replace(
            template, node_id="q:old", session_date="2022/01/01",
            turn_index=0, transport_role="user",
            text="I purchased 200 pounds of feed for the farm.",
        ),
    ]
    hint = same_unit_acquisition_total_hint(
        build_query_ir(
            "What is the total weight of the new feed I purchased in "
            "the past two months?"
        ),
        index, ["q:feed", "q:grain", "q:assistant", "q:old"],
        "2023/06/01",
    )
    assert hint is not None
    assert hint["value"] == 70
    assert set(hint["source_turn_ids"]) == {"q:feed", "q:grain"}



def test_named_event_attendance_deduplicates_pronoun_repeat_in_session() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:emma-1", session_id="s:emma",
            transport_role="user",
            text="I just attended my cousin Emma's preschool graduation.",
        ),
        replace(
            template, node_id="q:emma-2", session_id="s:emma",
            transport_role="user",
            text="It feels like yesterday I was attending her graduation ceremony.",
        ),
        replace(
            template, node_id="q:alex", session_id="s:alex",
            transport_role="user",
            text="I attended my colleague Alex's graduation a few weeks ago.",
        ),
        replace(
            template, node_id="q:rachel", session_id="s:rachel",
            transport_role="user",
            text="I attended my friend Rachel's graduation ceremony recently.",
        ),
        replace(
            template, node_id="q:jack", session_id="s:jack",
            transport_role="user",
            text="I missed my nephew Jack's graduation ceremony.",
        ),
    ]
    hint = named_event_attendance_count_hint(
        build_query_ir(
            "How many graduation ceremonies have I attended in the "
            "past three months?"
        ),
        index, [turn.node_id for turn in index.turns],
    )
    assert hint is not None
    assert hint["value"] == 3
    assert {row["identity"] for row in hint["members"]} == {
        "Emma", "Alex", "Rachel",
    }



def test_relative_anchor_certifies_companion_absence_in_bounded_scene() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:museum", session_id="s:museum",
            session_date="2023/01/11", turn_index=0,
            transport_role="user",
            text="I learned about ancient civilizations in a lecture at "
                 "the History Museum this month.",
        )
    ]
    hint = relative_anchor_source_hint(
        build_query_ir(
            "I mentioned visiting a museum two months ago. "
            "Did I visit with a friend or not?"
        ),
        index, "2023/03/11", allowed_session_ids={"s:museum"},
    )
    assert hint is not None
    assert hint["answer_candidate"].startswith("No,")


def test_relative_anchor_prefers_completed_event_over_future_topic() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:future", session_id="s:future",
            session_date="2023/01/15", turn_index=0,
            transport_role="user",
            text="I would love to attend an art festival in Yunnan.",
        ),
        replace(
            template, node_id="q:met", session_id="s:met",
            session_date="2023/01/15", turn_index=0,
            transport_role="user",
            text='I attended the "Ancient Civilizations" exhibit at the '
                 "Metropolitan Museum of Art today.",
        ),
    ]
    for card, turn in zip(index.routing_cards[:2], index.turns):
        card.session_id = turn.session_id
        card.routing_text = turn.text
    hint = relative_anchor_source_hint(
        build_query_ir(
            "I participated in an art-related event two weeks ago. "
            "Where was that event held at?"
        ),
        index, "2023/02/01",
        allowed_session_ids={"s:future", "s:met"},
    )
    assert hint is not None
    assert hint["source_turn_ids"][0] == "q:met"
    assert "Metropolitan Museum of Art" in hint["answer_candidate"]


def test_exact_absence_does_not_treat_generic_gadget_as_exact_entity() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:pot", transport_role="user",
            text="I invested in an Instant Pot before getting an Air Fryer.",
        )
    ]
    assert exact_entity_absence_hint(
        build_query_ir(
            "What new kitchen gadget did I invest in before getting "
            "the Air Fryer?"
        ),
        index,
    ) is None



def test_threshold_progress_subtracts_current_from_target() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:current", session_date="2023/05/21",
            transport_role="user",
            text="I'm looking for skincare products. I bought an item "
                 "at Sephora and earned 50 points, bringing my total to "
                 "200 points so far.",
        ),
        replace(
            template, node_id="q:target", session_date="2023/05/29",
            transport_role="user",
            text="To redeem a free skincare product from Sephora, I just "
                 "need a total of 300 points and I'm all set.",
        ),
    ]
    hint = threshold_progress_remaining_hint(
        build_query_ir(
            "How many points do I need to earn to redeem a free "
            "skincare product at Sephora?"
        ),
        index, ["q:current", "q:target"],
    )
    assert hint is not None
    assert hint["value"] == 100
    assert [row["role"] for row in hint["operands"]] == [
        "target", "current",
    ]


def test_latest_approx_scalar_accepts_newer_near_value() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:old", session_date="2023/05/25 05:26",
            transport_role="user",
            text="I've got 1250 followers on Instagram now.",
        ),
        replace(
            template, node_id="q:new", session_date="2023/05/25 09:28",
            transport_role="user",
            text="I checked my current Instagram follower count and I "
                 "think I'm close to 1300 now.",
        ),
    ]
    hint = latest_approx_scalar_state_hint(
        build_query_ir("How many followers do I have on Instagram now?"),
        index, ["q:old", "q:new"],
    )
    assert hint is not None
    assert hint["value"] == 1300
    assert hint["source_turn_ids"] == ["q:new"]


def test_latest_labeled_currency_uses_newest_same_provider_state() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:old", session_date="2023/08/11",
            transport_role="user",
            text="I got pre-approved for $350,000 from Wells Fargo.",
        ),
        replace(
            template, node_id="q:new", session_date="2023/11/30",
            transport_role="user",
            text="Remember when I got pre-approved for $400,000 from "
                 "Wells Fargo?",
        ),
    ]
    hint = latest_labeled_currency_state_hint(
        build_query_ir(
            "What was the amount I was pre-approved for when I got my "
            "mortgage from Wells Fargo?"
        ),
        index, ["q:old", "q:new"],
    )
    assert hint is not None
    assert hint["value"] == 400000
    assert hint["source_turn_ids"] == ["q:new"]


def test_latest_weekly_schedule_time_selects_newer_weekday_state() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:old", session_date="2023/05/23",
            transport_role="user",
            text="I've been waking up around 8:30 am on Saturdays.",
        ),
        replace(
            template, node_id="q:new", session_date="2023/05/27",
            transport_role="user",
            text="I like to wake up at 7:30 am on Saturdays.",
        ),
    ]
    hint = latest_weekly_schedule_time_hint(
        build_query_ir("What time do I wake up on Saturday mornings?"),
        index, ["q:old", "q:new"],
    )
    assert hint is not None
    assert hint["value"].casefold() == "7:30 am"
    assert hint["source_turn_ids"] == ["q:new"]


def test_temporal_predecessor_binds_latest_named_acquisition() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:prior", session_date="2023/05/21 05:48",
            turn_index=0, transport_role="user",
            text=(
                "I'm thinking of using my new Instant Pot to make soup "
                "with pressure cooking."
            ),
        ),
        replace(
            template, node_id="q:anchor", session_date="2023/05/21 22:54",
            turn_index=0, transport_role="user",
            text=(
                "I'm using the Air Fryer I got yesterday to make fries."
            ),
        ),
        replace(
            template, node_id="q:future", session_date="2023/05/22 10:00",
            turn_index=0, transport_role="user",
            text="I'm planning to get a new Blender next week.",
        ),
    ]
    hint = temporal_predecessor_entity_hint(
        build_query_ir(
            "What new kitchen gadget did I invest in before getting "
            "the Air Fryer?"
        ),
        index, ["q:prior", "q:anchor", "q:future"],
    )
    assert hint is not None
    assert hint["answer_candidate"] == "Instant Pot"
    assert hint["source_turn_ids"] == ["q:prior", "q:anchor"]


def test_temporal_pair_supports_since_when_and_requested_week_unit() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:recover", session_id="s1",
            transport_role="user", session_date="2023/01/01 10:00",
            text="I finally recovered from the flu today.",
        ),
        replace(
            template, node_id="q:jog", session_id="s2",
            transport_role="user", session_date="2023/01/22 10:00",
            text="I went on my 10th jog outdoors today.",
        ),
    ]
    hint = temporal_source_pair_hint(
        build_query_ir(
            "How many weeks had passed since I recovered from the flu "
            "when I went on my 10th jog outdoors?"
        ),
        index, [turn.node_id for turn in index.turns],
    )
    assert hint is not None
    assert hint["value"] == 3
    assert hint["unit"] == "weeks"


def test_temporal_pair_supports_ago_when_with_relative_source_date() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:class", session_id="s1",
            transport_role="user", session_date="2022/03/21 15:54",
            text="I took an amazing baking class at a local culinary school yesterday.",
        ),
        replace(
            template, node_id="q:cake", session_id="s2",
            transport_role="user", session_date="2022/04/10 14:14",
            text="I made my friend's birthday cake today.",
        ),
    ]
    hint = temporal_source_pair_hint(
        build_query_ir(
            "How many days ago did I attend a baking class at a local "
            "culinary school when I made my friend's birthday cake?"
        ),
        index, [turn.node_id for turn in index.turns],
    )
    assert hint is not None
    assert hint["value"] == 21
    assert hint["unit"] == "days"


def test_temporal_pair_supports_state_when_with_explicit_dates() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:orientation", session_id="s1",
            transport_role="user", session_date="2023/04/19 02:37",
            text="I have been attending pre-departure orientation sessions since 3/27.",
        ),
        replace(
            template, node_id="q:accepted", session_id="s2",
            transport_role="user", session_date="2023/04/19 03:31",
            text="I got accepted into the exchange program on March 20th.",
        ),
    ]
    hint = temporal_source_pair_hint(
        build_query_ir(
            "How many weeks have I been accepted into the exchange program "
            "when I started attending the pre-departure orientation sessions?"
        ),
        index, [turn.node_id for turn in index.turns],
    )
    assert hint is not None
    assert hint["value"] == 1
    assert hint["unit"] == "week"


def test_age_arithmetic_uses_explicit_role_ages_across_sources() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:self", session_id="s1",
            transport_role="user", text="Do you think 32 is considered young?",
        ),
        replace(
            template, node_id="q:grandma", session_id="s2",
            transport_role="user", text="My grandma's 75th birthday was wonderful.",
        ),
    ]
    hint = age_arithmetic_from_sources_hint(
        build_query_ir("How many years older is my grandma than me?"),
        index, [turn.node_id for turn in index.turns],
    )
    assert hint is not None
    assert hint["value"] == 43


def test_age_arithmetic_binds_named_person_pronoun_age() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:self", session_id="s1",
            transport_role="user", text="I just turned 32 last month.",
        ),
        replace(
            template, node_id="q:alex", session_id="s2",
            transport_role="user",
            text="I am mentoring Alex. It is crazy that he's just 21.",
        ),
    ]
    hint = age_arithmetic_from_sources_hint(
        build_query_ir("How old was I when Alex was born?"),
        index, [turn.node_id for turn in index.turns],
    )
    assert hint is not None
    assert hint["value"] == 11


def test_age_arithmetic_supports_explicit_future_offset() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:self", session_id="s1",
            transport_role="user", text="I'm 32 and researching skincare.",
        ),
        replace(
            template, node_id="q:rachel", session_id="s2",
            transport_role="user", text="My friend Rachel is getting married next year.",
        ),
    ]
    hint = age_arithmetic_from_sources_hint(
        build_query_ir("How many years will I be when my friend Rachel gets married?"),
        index, [turn.node_id for turn in index.turns],
    )
    assert hint is not None
    assert hint["value"] == 33


def test_current_role_duration_subtracts_pre_promotion_tenure() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:prior", session_id="s1",
            transport_role="user",
            text="I started as a coordinator and worked my way up to senior specialist after 2 years and 4 months.",
        ),
        replace(
            template, node_id="q:total", session_id="s2",
            transport_role="user",
            text="I have 3 years and 9 months experience in the company.",
        ),
    ]
    hint = current_role_duration_from_sources_hint(
        build_query_ir("How long have I been working in my current role?"),
        index, [turn.node_id for turn in index.turns],
    )
    assert hint is not None
    assert hint["value"] == "1 year and 5 months"
    assert hint["months"] == 17


def test_advance_booking_recency_composes_trip_age_and_lead_time() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(
            template, node_id="q:trip", session_id="s1",
            transport_role="user",
            text="I was in San Francisco exactly two months ago for a wedding.",
        ),
        replace(
            template, node_id="q:booking", session_id="s2",
            transport_role="user",
            text="For that San Francisco Airbnb I had to book three months in advance.",
        ),
    ]
    hint = advance_booking_recency_from_sources_hint(
        build_query_ir("How many months ago did I book the Airbnb in San Francisco?"),
        index, [turn.node_id for turn in index.turns],
    )
    assert hint is not None
    assert hint["value"] == 5
    assert hint["unit"] == "months"


def test_open_temporal_sequence_is_lifecycle_and_action_bound() -> None:
    index = _index()
    template = index.turns[0]
    rows = [
        ("q:watched", "s0", "2023/06/01", "I watched a football game today."),
        ("q:race", "s1", "2023/06/02", "I completed the Spring Sprint triathlon today."),
        ("q:run", "s2", "2023/06/10", "I finished a 5K run today."),
        ("q:soccer", "s3", "2023/06/17", "I participated in a charity soccer tournament today."),
        ("q:plan", "s4", "2023/06/18", "I plan to participate in a tennis tournament next month."),
    ]
    index.turns = [replace(template, node_id=node_id, session_id=session_id,
        transport_role="user", session_date=date, text=text)
        for node_id, session_id, date, text in rows]
    ir = build_query_ir("What is the order of the three sports events I participated in during the past month, from earliest to latest?")
    hint = open_temporal_sequence_from_sources_hint(
        ir, index, [turn.node_id for turn in index.turns], "2023/06/20",
    )
    assert hint is not None
    assert hint["source_turn_ids"] == ["q:race", "q:run", "q:soccer"]
    assert hint["event_times"] == sorted(hint["event_times"])


def test_open_temporal_sequence_prefers_new_event_over_old_reference() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(template, node_id="q:festival", session_id="s1", transport_role="user", session_date="2023/04/01", text="I just got back from a music festival in Brooklyn today."),
        replace(template, node_id="q:queen", session_id="s2", transport_role="user", session_date="2023/04/15", text="I listen to Queen and actually just saw them live with Adam Lambert today. I recently attended a music festival in Brooklyn."),
    ]
    ir = build_query_ir("What is the order of the concerts and musical events I attended, starting from the earliest?")
    hint = open_temporal_sequence_from_sources_hint(
        ir, index, [turn.node_id for turn in index.turns], "2023/04/16",
    )
    assert hint is not None
    assert hint["source_turn_ids"] == ["q:festival", "q:queen"]
    assert "Queen" in hint["ordered_targets"][1]


def test_open_temporal_sequence_resolves_local_venue_pronoun() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(template, node_id="q:science", session_id="s1", transport_role="user", session_date="2023/02/01", text="I visited the Science Museum today."),
        replace(template, node_id="q:modern", session_id="s2", transport_role="user", session_date="2023/02/20", text="I plan to visit the Modern Art Museum again. By the way, I attended their guided tour today."),
        replace(template, node_id="q:old", session_id="s2", transport_role="user", session_date="2023/02/20", text="I recently saw artifacts at the History Museum."),
    ]
    ir = build_query_ir("What is the order of the two museums I visited from earliest to latest?")
    hint = open_temporal_sequence_from_sources_hint(
        ir, index, [turn.node_id for turn in index.turns], "2023/02/21",
    )
    assert hint is not None
    assert hint["source_turn_ids"] == ["q:science", "q:modern"]
    assert "COMPLETED EVENT" in hint["ordered_targets"][1]



def test_weekly_schedule_counts_distinct_days_across_sources() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(template, node_id="q:classes", session_id="s1",
                transport_role="user",
                text="I attend dance classes on Tuesdays and Thursdays."),
        replace(template, node_id="q:yoga", session_id="s2",
                transport_role="user",
                text="I recently started a yoga class on Wednesdays."),
        replace(template, node_id="q:plan", session_id="s3",
                transport_role="user",
                text="I might try a workout class on Friday."),
    ]
    hint = weekly_schedule_days_from_sources_hint(
        build_query_ir("How many days a week do I attend fitness classes?"),
        index, [turn.node_id for turn in index.turns],
    )
    assert hint is not None
    assert hint["value"] == 3
    assert [row["identity"] for row in hint["members"]] == [
        "tuesday", "wednesday", "thursday",
    ]


def test_family_total_sums_sibling_subtypes_only() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(template, node_id="q:sisters", session_id="s1",
                transport_role="user",
                text="I grew up with three sisters."),
        replace(template, node_id="q:brother", session_id="s2",
                transport_role="user",
                text="I have one brother and ten coworkers."),
    ]
    hint = family_relation_total_from_sources_hint(
        build_query_ir("What is the total number of siblings I have?"),
        index, [turn.node_id for turn in index.turns],
    )
    assert hint is not None
    assert hint["value"] == 4


def test_linked_event_date_joins_unique_organization_anchor() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(template, node_id="q:event", session_id="s1",
                transport_role="user",
                text="I submitted my graph reasoning paper to ACL."),
        replace(template, node_id="q:date", session_id="s2",
                transport_role="user",
                text="The ACL submission deadline was February 1st."),
    ]
    hint = linked_event_date_from_sources_hint(
        build_query_ir("When did I submit my graph reasoning paper?"),
        index, [turn.node_id for turn in index.turns],
    )
    assert hint is not None
    assert hint["value"] == "February 1st"
    assert set(hint["source_turn_ids"]) == {"q:event", "q:date"}


def test_latest_category_start_uses_stated_start_duration() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(template, node_id="q:old", session_id="s1",
                session_date="2023/06/01", transport_role="user",
                text="I have been using StreamOne for the past 6 months."),
        replace(template, node_id="q:new", session_id="s2",
                session_date="2023/06/01", transport_role="user",
                text="I started a free trial of Aurora+ last month."),
    ]
    hint = latest_category_start_from_sources_hint(
        build_query_ir("Which streaming service did I start using most recently?"),
        index, [turn.node_id for turn in index.turns],
    )
    assert hint is not None
    assert hint["answer_candidate"] == "Aurora+"


def test_dialogue_attribute_match_prefers_named_entity_plus_attribute() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(template, node_id="q:list1", session_id="s1",
                transport_role="assistant",
                text=("1. Cedar Stew - a regional stew with root vegetables. "
                      "2. Grilled Trout with Peach Relish - trout topped with "
                      "a fruity peach relish.")),
        replace(template, node_id="q:list2", session_id="s2",
                transport_role="assistant",
                text=("1. Orchard Salad - a regional fruit salad. "
                      "2. River Soup - a traditional regional soup.")),
    ]
    hint = dialogue_attribute_match_hint(
        build_query_ir(
            "What was the name from our previous list of the regional dish "
            "with trout that has fruit in it?"
        ),
        index,
    )
    assert hint is not None
    assert hint["answer_candidate"] == "Grilled Trout with Peach Relish"


def test_preference_constraints_do_not_capture_historical_name_lookup() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(template, node_id="q:pref", session_id="s1",
                transport_role="user",
                text="I prefer rooms with a balcony and a quiet view."),
    ]
    sources = [turn.node_id for turn in index.turns]
    assert preference_constraints_from_sources_hint(
        build_query_ir(
            "What was the name from our previous conversation that you recommended?"
        ),
        index, sources,
    ) is None
    hint = preference_constraints_from_sources_hint(
        build_query_ir("Can you suggest a hotel for my upcoming trip?"),
        index, sources,
    )
    assert hint is not None
    assert "balcony" in hint["value"][0]["evidence"]


def test_presupposed_poster_requires_requested_academic_qualifiers() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(template, node_id="q:thesis", session_id="s1",
                transport_role="user",
                text="I presented a poster about my thesis research at North University."),
    ]
    mismatch = presupposed_event_absence_hint(
        build_query_ir(
            "At which university did I present a poster for my undergrad "
            "course research project?"
        ),
        index,
    )
    assert mismatch is not None
    assert mismatch["binding_kind"] == "required_relation"
    assert presupposed_event_absence_hint(
        build_query_ir(
            "At which university did I present a poster for my thesis research?"
        ),
        index,
    ) is None



def test_currency_extreme_entity_binds_all_source_operands() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(template, node_id="q:a", session_id="s1",
                session_date="2023/06/20", transport_role="user",
                text="I bought supplies at North Depot and spent $75 last week."),
        replace(template, node_id="q:b", session_id="s2",
                session_date="2023/06/20", transport_role="user",
                text="I placed an order with Harbor Supply last month and spent $140."),
        replace(template, node_id="q:c", session_id="s3",
                session_date="2023/06/20", transport_role="user",
                text="I ordered from West Mart yesterday and paid $90."),
    ]
    hint = currency_extreme_entity_from_sources_hint(
        build_query_ir(
            "Which supplier did I spend the most money at in the past month?"
        ),
        index, [turn.node_id for turn in index.turns], "2023/06/20",
    )
    assert hint is not None
    assert hint["answer_candidate"] == "Harbor Supply"
    assert hint["selected_amount"] == 140
    assert [row["value"] for row in hint["operands"]] == [75, 90, 140]



def test_dialogue_final_choice_uses_user_endorsement() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(template, node_id="q:choice", session_id="s1",
                session_date="2023/06/01", transport_role="user",
                text="NovaForge is a really cool one. Let's use it for the robot."),
    ]
    hint = dialogue_final_choice_from_sources_hint(
        build_query_ir("What did we finally decide to name the robot?"), index,
    )
    assert hint is not None
    assert hint["answer_candidate"] == "NovaForge"


def test_completed_item_metric_total_deduplicates_scene_and_prior_item() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(template, node_id="q:book1", session_id="s1",
                transport_role="user",
                text=("I just finished a 416-page novel, but before that "
                      "I read a 341-page book.")),
        replace(template, node_id="q:book2", session_id="s2",
                transport_role="user",
                text="I just finished The Night Bird, which was 440 pages."),
        replace(template, node_id="q:book2repeat", session_id="s2",
                transport_role="user",
                text="The Night Bird was 440 pages and I finished it."),
    ]
    hint = completed_item_metric_total_from_sources_hint(
        build_query_ir(
            "What was the page count of the two novels I finished?"
        ), index, [turn.node_id for turn in index.turns],
    )
    assert hint is not None
    assert hint["value"] == 856
    assert len(hint["operands"]) == 2


def test_scoped_completed_duration_excludes_plans_and_habits() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(template, node_id="q:jog", session_id="s1",
                session_date="2023/05/29", transport_role="user",
                text="I went for a 30-minute jog last Saturday."),
        replace(template, node_id="q:yoga", session_id="s2",
                session_date="2023/05/29", transport_role="user",
                text="I used to do 90 minutes of yoga and hope to restart."),
    ]
    hint = scoped_completed_duration_total_from_sources_hint(
        build_query_ir("How many hours of jogging and yoga did I do last week?"),
        index, [turn.node_id for turn in index.turns], "2023/05/30",
    )
    assert hint is not None
    assert hint["value"] == 0.5


def test_relative_value_multiplier_returns_relation_not_prices() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(template, node_id="q:value", session_id="s1",
                transport_role="user",
                text=("I paid $25 for the sculpture and later learned it is "
                      "worth triple what I paid.")),
    ]
    hint = relative_value_multiplier_from_sources_hint(
        build_query_ir(
            "How much is the sculpture worth in terms of the amount I paid?"
        ), index, [turn.node_id for turn in index.turns],
    )
    assert hint is not None
    assert hint["answer_candidate"] == "worth triple what I paid"


def test_relative_duration_at_event_joins_state_start_and_event() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(template, node_id="q:start", session_id="s1",
                session_date="2023/05/20", transport_role="user",
                text=("I started watching stand-up comedy about 3 months ago "
                      "and have watched it regularly.")),
        replace(template, node_id="q:event", session_id="s2",
                session_date="2023/05/20", transport_role="user",
                text="Last month I attended an open mic night at the comedy club."),
    ]
    hint = relative_duration_at_event_from_sources_hint(
        build_query_ir(
            "How long had I been watching stand-up comedy regularly when I "
            "attended the open mic night at the comedy club?"
        ), index, [turn.node_id for turn in index.turns],
    )
    assert hint is not None
    assert hint["value"] == 2
    assert hint["unit"] == "months"


def test_prior_candidate_count_excludes_committed_target() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(template, node_id="q:a", session_id="s1",
                session_date="2023/03/01", transport_role="user",
                text="I saw a bungalow on January 22nd."),
        replace(template, node_id="q:b", session_id="s2",
                session_date="2023/03/01", transport_role="user",
                text="I viewed a condo on February 10th."),
        replace(template, node_id="q:target", session_id="s3",
                session_date="2023/03/01", transport_role="user",
                text=("I saw the Brookside townhouse on February 22nd and "
                      "put in an offer on the Brookside townhouse on February 25th.")),
    ]
    hint = prior_candidate_count_from_sources_hint(
        build_query_ir(
            "How many properties did I view before making an offer on the "
            "Brookside townhouse?"
        ), index, [turn.node_id for turn in index.turns],
    )
    assert hint is not None
    assert hint["value"] == 2


def test_completed_carrier_sequence_excludes_planned_carrier() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(template, node_id="q:first", session_id="s1",
                session_date="2023/01/01", transport_role="user",
                text=("I just got back from a flight on BlueJet. "
                      "I am planning to fly with Future Airlines next month.")),
        replace(template, node_id="q:second", session_id="s2",
                session_date="2023/02/01", transport_role="user",
                text="Today I had a delay on my North Airlines flight."),
    ]
    hint = completed_carrier_sequence_from_sources_hint(
        build_query_ir(
            "What is the order of airlines I flew with from earliest to latest?"
        ), index, [turn.node_id for turn in index.turns], "2023/03/01",
    )
    assert hint is not None
    assert hint["ordered_targets"] == ["BlueJet", "North Airlines"]


def test_event_endpoint_difference_uses_action_and_calendar_dates() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(template, node_id="q:feedback", session_id="s1",
                session_date="2023/03/17 15:43", transport_role="user",
                text="I have been getting feedback that my suspension is too soft."),
        replace(template, node_id="q:test", session_id="s2",
                session_date="2023/04/23 02:44", transport_role="user",
                text="Tomorrow I will be testing my new suspension setup."),
    ]
    hint = event_endpoint_difference_from_sources_hint(
        build_query_ir(
            "How many days passed between the day I received feedback about "
            "my suspension and the day I tested my new suspension setup?"
        ), index, [turn.node_id for turn in index.turns],
    )
    assert hint is not None
    assert hint["value"] == 38


def test_travel_arrival_joins_departure_and_repeated_duration_once() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(template, node_id="q:depart", session_id="s1",
                session_date="2023/06/05 12:00", transport_role="user",
                text="On Monday I left home at 7 AM for the clinic."),
        replace(template, node_id="q:duration1", session_id="s2",
                session_date="2023/06/06 12:00", transport_role="user",
                text="It took me two hours to get to the clinic."),
        replace(template, node_id="q:duration2", session_id="s3",
                session_date="2023/06/07 12:00", transport_role="user",
                text="The trip to the clinic took two hours."),
    ]
    hint = travel_arrival_time_from_sources_hint(
        build_query_ir("What time did I arrive at the clinic on Monday?"),
        index, [turn.node_id for turn in index.turns],
    )
    assert hint is not None
    assert hint["value"] == "9:00 AM"
    assert hint["travel_minutes"] == 120


def test_completed_work_total_combines_requested_subtypes() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(template, node_id="q:stories", session_id="s1",
                transport_role="user",
                text="I have written five short stories."),
        replace(template, node_id="q:poems", session_id="s2",
                transport_role="user",
                text="I completed 17 poems."),
        replace(template, node_id="q:challenge", session_id="s3",
                transport_role="user",
                text="I finished a writing challenge piece called 'Winter'."),
    ]
    hint = completed_work_subtype_total_from_sources_hint(
        build_query_ir(
            "How many stories, poems, and writing pieces have I completed in total?"
        ), index, [turn.node_id for turn in index.turns],
    )
    assert hint is not None
    assert hint["value"] == 23


def test_cookie_preference_ranks_transferable_ingredient_fact_first() -> None:
    index = _index()
    template = index.turns[0]
    index.turns = [
        replace(template, node_id="q:generic", session_id="s1",
                session_date="2023/06/01", transport_role="user",
                text="I like baking cookies with chocolate."),
        replace(template, node_id="q:fact", session_id="s2",
                session_date="2023/05/01", transport_role="user",
                text="I found that turbinado sugar adds a richer flavor to cookies."),
    ]
    hint = preference_constraints_from_sources_hint(
        build_query_ir("What ingredient should I use for richer cookies?"),
        index, [turn.node_id for turn in index.turns],
    )
    assert hint is not None
    assert hint["value"][0]["source_turn_id"] == "q:fact"
