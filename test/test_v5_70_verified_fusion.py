from __future__ import annotations

from graphmem.answer.verified_readout import (
    build_source_focus_prompt, build_unified_source_prompt,
    build_verified_prompt,
)
from graphmem.domain import CandidateScore, SourceTurn
from graphmem.retrieval.flat_plan import (
    SourceFocusPlan,
    append_source_focus_witnesses,
    build_flat_fusion_plan,
    build_source_focus_plan, build_temporal_focus_plan,
    compile_obligation_query_view, compile_query_obligations,
    compile_source_query_views,
    preserve_graph_with_flat_packet,
    query_route,
    source_facing_candidates,
    verification_gate,
)


def turn(index: int, text: str, *, session: str = "s1",
         speaker: str = "Alice") -> SourceTurn:
    return SourceTurn(
        turn_id=f"t{index}", memory_id="m1", session_id=session,
        turn_index=index, speaker=speaker, listener="Bob", role="user",
        timestamp=f"2025-01-{index + 1:02d}", raw_text=text,
        content_hash=f"h{index}")


def score(one: SourceTurn, *, exact: float = 0.0, bm25: float = 0.0,
          dense: float = 0.0, graph: float = 0.0,
          binding: float = 0.0, fused: float = 0.0) -> CandidateScore:
    return CandidateScore(
        turn_id=one.turn_id, session_id=one.session_id,
        exact_score=exact, bm25_score=bm25, dense_score=dense,
        graph_score=graph, role_gain=0.0, slot_gain=0.0,
        token_cost=10, fused_score=fused,
        source_channels=("exact", "bm25", "dense", "graph"),
        binding_score=binding, operand_ids=(("o1",) if binding else ()),
        mandatory=bool(binding),
        graph_path_ids=("e1",), relation_contributions=("shared_entity",),
        relational_consensus_score=1.0)


def test_query_route_is_question_only() -> None:
    assert query_route("How many workshops did I attend?") == "aggregate"
    assert query_route("When did I first visit Lisbon?") == "temporal"
    assert query_route("Where am I currently keeping my old shoes?") == "state"
    assert query_route("What might this suggest about her personality?") == "inference"
    assert query_route("What degree did I graduate with?") == "lookup"


def test_query_obligations_preserve_composition_beyond_primary_route() -> None:
    obligations = compile_query_obligations(
        "Based on their plans, which city did Alice and Bob both visit first?")

    assert obligations.route == "temporal"
    assert {"temporal", "exhaustive_set", "multi_entity", "comparison",
            "causal", "multi_hop", "geographic_resolution",
            "reasoning_chain"} <= set(obligations.tags)
    assert obligations.requires_reasoning is True
    assert obligations.requires_exhaustive is True


def test_query_obligations_detect_plural_wh_answer_head() -> None:
    obligations = compile_query_obligations(
        "What book recommendations has Joanna given to Nate?")

    # The established lookup physical plan remains stable; completeness is an
    # orthogonal QueryIR duty rather than a reason to perturb retrieval.
    assert obligations.route == "lookup"
    assert "exhaustive_set" in obligations.tags
    assert obligations.requires_exhaustive is True
    # The label remains available to retrieval/tracing.  A generic
    # completeness reminder is not repeated in the answer prompt.
    assert obligations.presentation_tags() == ()


def test_presentation_tags_only_extend_generic_routes() -> None:
    aggregate = compile_query_obligations(
        "How many events did Alice attend in 2024?")
    assert aggregate.route == "aggregate"
    assert aggregate.presentation_tags() == ()

    causal = compile_query_obligations(
        "Why did Alice choose the blue pattern?")
    assert causal.route == "lookup"
    assert {"causal", "multi_hop"} <= set(causal.tags)
    assert causal.presentation_tags() == ()

    comparison = compile_query_obligations(
        "Which book did Alice read, Dune or Hamlet?")
    assert comparison.route == "list"
    assert comparison.presentation_tags() == ("comparison",)

    geography = compile_query_obligations(
        "Which city did Alice visit?")
    assert geography.route == "list"
    assert geography.presentation_tags() == ("geographic_resolution",)


def test_geographic_obligation_requires_a_location_answer() -> None:
    assert "geographic_resolution" in compile_query_obligations(
        "Which state might Alice have visited?").tags
    assert "geographic_resolution" in compile_query_obligations(
        "What Canadian province contains Banff?").tags

    # Direct location answer heads and mentions of visits/travel/places do not
    # by themselves ask for an administrative-level derivation.  In
    # particular, ``place`` is also an event verb and an ordinal noun in
    # otherwise non-geographic questions.
    for question in (
            "Where did Alice spend September?",
            "Which outdoor spot did Joanna visit in May?",
            "Which Star Wars-related locations would Tim enjoy?",
            "What are the places Alice visited?",
            "Does John live close to a beach or the mountains?",
            "When did Jon visit networking events for his store?",
            "What happened to the puppy during the clinic visit?",
            "What dance did the team perform to win first place?",
            "When did the training course take place?",
            "What did John receive for achieving second place?",
            "What did John take away from visiting the hospital?",
    ):
        assert "geographic_resolution" not in compile_query_obligations(
            question).tags


def test_obligation_query_view_is_bounded_and_question_only() -> None:
    question = "What personality traits might Alice have?"
    views = compile_obligation_query_view(question)

    assert views[0] == question
    assert len(views) == 2
    assert "direct premises" in views[1]
    assert "Alice" in views[1]


def test_source_query_views_add_only_bounded_relation_wording() -> None:
    question = "Which football club does John support?"
    views = compile_source_query_views(question)
    assert len(views) == 2
    assert views[0] == question
    assert "fan" in views[1] and "team" in views[1]
    assert "John" in views[1]
    assert compile_source_query_views("What color is Alice's coat?") == (
        "What color is Alice's coat?",)


def test_flat_lane_removes_every_graph_feature() -> None:
    one = turn(0, "I graduated with a degree in physics.")
    flattened = source_facing_candidates((score(
        one, exact=1.0, dense=1.0, graph=1.0, binding=0.9, fused=99.0),),
        route="lookup")[0]
    assert flattened.graph_score == 0.0
    assert flattened.graph_path_ids == ()
    assert flattened.relation_contributions == ()
    assert flattened.operand_ids == ()
    assert flattened.binding_score == 0.0
    assert flattened.relational_consensus_score == 0.0
    assert flattened.mandatory is False
    assert set(flattened.source_channels) == {"exact", "bm25", "dense"}


def test_fusion_keeps_graph_head_and_adds_flat_evidence_within_budget() -> None:
    rows = [turn(index, f"unrelated memory {index}") for index in range(8)]
    rows.append(turn(8, "I moved from Sweden before living in Canada.", session="s2"))
    turns = {row.turn_id: row for row in rows}
    candidates = tuple(
        score(row, exact=(1.0 if row.turn_id == "t8" else 0.0),
              bm25=(1.0 if row.turn_id == "t8" else 0.0),
              dense=(1.0 if row.turn_id == "t8" else 0.1),
              graph=(1.0 if row.turn_id in {"t0", "t1"} else 0.0), fused=1.0)
        for row in rows)
    plan = build_flat_fusion_plan(
        question="Where did I move from?", turns=turns,
        candidate_scores=candidates,
        graph_turn_ids=tuple(f"t{i}" for i in range(8)),
        max_turns=6, graph_head=2, flat_head=3)
    assert plan.selected_turn_ids[:2] == ("t0", "t1")
    assert "t8" in plan.selected_turn_ids
    assert "t8" in plan.added_flat_turn_ids
    assert len(plan.selected_turn_ids) <= 6
    preserved = preserve_graph_with_flat_packet(
        plan, tuple(f"t{i}" for i in range(8)), max_extra_turns=1)
    assert preserved.selected_turn_ids[:8] == tuple(f"t{i}" for i in range(8))
    assert preserved.removed_graph_turn_ids == ()
    assert len(preserved.added_flat_turn_ids) <= 1


def test_lookup_gate_freezes_supported_answer_but_verifies_abstention() -> None:
    one = turn(0, "I graduated with a degree in Business Administration.")
    plan = build_flat_fusion_plan(
        question="What degree did I graduate with?", turns={one.turn_id: one},
        candidate_scores=(score(one, exact=1, bm25=1, dense=1),),
        graph_turn_ids=(one.turn_id,), max_turns=1, graph_head=1, flat_head=1)
    frozen = verification_gate(
        question="What degree did I graduate with?",
        previous_answer="Business Administration", plan=plan,
        graph_turns=(one,))
    assert frozen.eligible is False
    verify = verification_gate(
        question="What degree did I graduate with?",
        previous_answer="The information was not mentioned.", plan=plan,
        graph_turns=(one,))
    assert verify.eligible is True
    assert "previous_abstention" in verify.reasons


def test_verified_prompt_groups_sessions_and_marks_previous_as_fallible() -> None:
    first = turn(0, "I bought shoes and put them under the bed.", session="s1")
    second = turn(1, "I moved the old shoes to the closet rack.", session="s2")
    turns = {row.turn_id: row for row in (first, second)}
    candidates = (score(first, dense=0.8), score(second, exact=1, dense=1))
    plan = build_flat_fusion_plan(
        question="Where are my old shoes currently?", turns=turns,
        candidate_scores=candidates, graph_turn_ids=(first.turn_id,),
        max_turns=2, graph_head=1, flat_head=2)
    rendered = build_verified_prompt(
        question="Where are my old shoes currently?",
        question_date="2025-02-01", previous_answer="under the bed",
        plan=plan, turns=turns)
    assert len(rendered.evidence_turn_ids) == 2
    user = rendered.messages[-1]["content"]
    assert "Fallible previous answer" in user
    assert "under the bed" in user
    assert "closet rack" in user
    assert "gold" not in user.casefold()


def test_source_focus_is_graph_independent_and_keeps_dialogue_neighbor() -> None:
    distractors = [turn(index, f"generic travel discussion {index}")
                   for index in range(8)]
    witness = turn(
        20, "The names of my children are Kyle and Sara.", session="s2",
        speaker="John")
    response = turn(
        21, "They both started school this year.", session="s2",
        speaker="Bob")
    turns = {row.turn_id: row for row in (*distractors, witness, response)}
    plan = build_source_focus_plan(
        question="What are the names of John's children?", turns=turns,
        max_turns=4)

    assert witness.turn_id in plan.selected_turn_ids
    assert response.turn_id in plan.selected_turn_ids
    assert plan.trace["uses_graph_scores"] is False
    rendered = build_source_focus_prompt(
        question="What are the names of John's children?",
        question_date="2025-02-01", plan=plan, turns=turns)
    user = rendered.messages[-1]["content"]
    assert "Kyle and Sara" in user
    assert "Fallible previous answer" not in user
    assert "gold" not in user.casefold()


def test_source_focus_reserves_dense_paraphrase_lane() -> None:
    lexical = turn(0, "Alice discussed ordinary weekend plans.", session="s1")
    semantic = turn(1, "She adopted a canine named Luna.", session="s2")
    distractors = [turn(index + 2, f"Alice dog topic {index}", session="s1")
                   for index in range(10)]
    turns = {row.turn_id: row for row in (lexical, semantic, *distractors)}
    plan = build_source_focus_plan(
        question="What is the name of Alice's pet?", turns=turns,
        max_turns=8, dense_scores={semantic.turn_id: 0.99})

    assert semantic.turn_id in plan.seed_turn_ids
    assert plan.trace["uses_dense_scores"] is True
    assert plan.trace["focus_rank_lanes"] == ["lexical", "dense", "hybrid"]


def test_source_focus_can_reserve_sessions_for_exhaustive_queries() -> None:
    repeated = [turn(
        index, f"Alice recommended a travel book volume {index}.",
        session="popular", speaker="Alice") for index in range(8)]
    second_session = turn(
        20, "Alice also recommended the quiet book North Star.",
        session="rare", speaker="Alice")
    turns = {row.turn_id: row for row in (*repeated, second_session)}

    plan = build_source_focus_plan(
        question="What books has Alice recommended?", turns=turns,
        max_turns=4, session_diversity=True)

    assert second_session.turn_id in plan.seed_turn_ids
    assert plan.trace["session_diversity"] is True


def test_source_focus_matches_plural_question_to_singular_name_turns() -> None:
    coco = turn(
        0, "I got a puppy two weeks ago! Her name's Coco.", session="s1")
    shadow = turn(
        1, "Her name is Shadow! She is full of energy.", session="s2")
    distractors = [
        turn(index + 2, f"Maria discussed unrelated topic {index}.",
             session=f"s{index + 3}")
        for index in range(12)
    ]
    turns = {row.turn_id: row for row in (coco, shadow, *distractors)}
    plan = build_source_focus_plan(
        question="What are Maria's dogs' names?", turns=turns, max_turns=4,
        morphological=True)

    assert coco.turn_id in plan.seed_turn_ids
    assert shadow.turn_id in plan.seed_turn_ids


def test_source_focus_auxiliary_packet_is_additive_and_graph_novel() -> None:
    first = turn(0, "Maria discussed her dogs.", session="s1")
    second = turn(1, "Her name is Coco.", session="s2")
    third = turn(2, "Her name is Shadow.", session="s3")
    primary = build_source_focus_plan(
        question="What are Maria's dogs' names?",
        turns={first.turn_id: first}, max_turns=1)
    auxiliary = build_source_focus_plan(
        question="What are Maria's dogs' names?",
        turns={row.turn_id: row for row in (first, second, third)},
        max_turns=3, morphological=True)
    merged = append_source_focus_witnesses(
        primary, auxiliary, excluded_turn_ids=(second.turn_id,),
        max_extra_turns=1)

    assert merged.selected_turn_ids[:1] == primary.selected_turn_ids
    assert second.turn_id not in merged.trace["auxiliary_extra_turn_ids"]
    assert merged.trace["auxiliary_extra_turns"] == 1
    assert len(merged.selected_turn_ids) == len(primary.selected_turn_ids) + 1


def test_obligation_dense_view_preserves_independent_rank_and_provenance() -> None:
    first = turn(0, "Alice discussed travel.", session="s1")
    bridge = turn(1, "Tampa was the destination.", session="s2")
    alternate = turn(2, "Florida contains Tampa.", session="s3")
    primary = SourceFocusPlan(
        route="lookup", selected_turn_ids=(first.turn_id,),
        seed_turn_ids=(first.turn_id,), neighbor_turn_ids=(), trace={})
    auxiliary = SourceFocusPlan(
        route="lookup", selected_turn_ids=(bridge.turn_id, alternate.turn_id),
        seed_turn_ids=(bridge.turn_id, alternate.turn_id),
        neighbor_turn_ids=(), trace={
            "uses_dense_scores": True,
            "view_kind": "obligation_dense",
        })

    merged = append_source_focus_witnesses(
        primary, auxiliary, max_extra_turns=1)

    assert merged.trace["auxiliary_extra_turn_ids"] == [bridge.turn_id]
    assert merged.trace["auxiliary_focus_views"][0]["type"] == (
        "obligation_dense")


def test_auxiliary_focus_prefers_concept_context_and_answer_turns() -> None:
    unrelated = turn(0, "Maria discussed ordinary weekend plans.", session="s0")
    baby = turn(
        1, "That baby is adorable. What's their name?", session="s1",
        speaker="Maria")
    coco = turn(
        2, "I got a puppy two weeks ago! Her name's Coco.", session="s2",
        speaker="Maria")
    dog_question = turn(
        3, "What's her name? Does she like your other dog?", session="s3",
        speaker="John")
    shadow = turn(
        4, "Her name is Shadow! They get along great.", session="s3",
        speaker="Maria")
    turns = {row.turn_id: row for row in (
        unrelated, baby, coco, dog_question, shadow)}
    primary = build_source_focus_plan(
        question="ordinary weekend plans", turns={unrelated.turn_id: unrelated},
        max_turns=1)
    auxiliary = build_source_focus_plan(
        question="What are Maria's dogs' names?", turns=turns,
        max_turns=5, morphological=True)
    merged = append_source_focus_witnesses(
        primary, auxiliary, max_extra_turns=2,
        question="What are Maria's dogs' names?", turns=turns)

    assert set(merged.trace["auxiliary_extra_turn_ids"]) == {
        coco.turn_id, shadow.turn_id}
    assert baby.turn_id not in merged.trace["auxiliary_extra_turn_ids"]
    assert merged.trace["auxiliary_selection"] == (
        "query-concept-dialogue-context")
    view = merged.trace["auxiliary_focus_views"][0]
    assert view["candidate_pool_turns"] >= 2
    assert set(view["selected_scores"]) == {coco.turn_id, shadow.turn_id}


def test_relation_family_focus_bridges_family_and_sports_wording() -> None:
    kyle = turn(
        0, "Our one-year-old is so cute, his name is Kyle!", session="s1",
        speaker="John")
    sara = turn(
        1, "We traveled for my daughter Sara's birthday.", session="s2",
        speaker="John")
    football = turn(
        2, "As a Manchester City fan, my team will win!", session="s3",
        speaker="John")
    distractors = [turn(
        index + 3, f"John discussed an unrelated project {index}.",
        session=f"s{index + 4}", speaker="John") for index in range(8)]
    turns = {row.turn_id: row for row in (
        kyle, sara, football, *distractors)}

    children = build_source_focus_plan(
        question="What are the names of John's children?", turns=turns,
        max_turns=4, relation_families=True)
    club = build_source_focus_plan(
        question="Which football club does John support?", turns=turns,
        max_turns=4, relation_families=True)

    assert kyle.turn_id in children.seed_turn_ids
    assert sara.turn_id in children.seed_turn_ids
    assert football.turn_id in club.seed_turn_ids
    assert children.trace["relation_families"] is True
    assert children.trace["source_closure_risk"] is True
    assert "plural-slot" in children.trace["source_closure_reasons"]

    existence = build_source_focus_plan(
        question="Has John tried surfing?", turns=turns,
        max_turns=4, relation_families=True)
    assert "existence" in existence.trace["source_closure_reasons"]


def test_relation_name_slot_keeps_values_ahead_of_generic_class_mentions() -> None:
    generic = [turn(
        index,
        f"My kids enjoyed an ordinary community activity number {index}.",
        session=f"generic-{index}", speaker="John")
        for index in range(36)
    ]
    kyle = turn(
        40, "Our one-year-old is so cute, his name is Kyle!",
        session="family-one", speaker="John")
    sara = turn(
        41, "We traveled for my daughter Sara's birthday.",
        session="family-two", speaker="John")
    turns = {row.turn_id: row for row in (*generic, kyle, sara)}

    relation = build_source_focus_plan(
        question="What are the names of John's children?", turns=turns,
        max_turns=32, relation_families=True)
    primary = build_source_focus_plan(
        question="unrelated source view", turns=turns, max_turns=1)
    merged = append_source_focus_witnesses(
        primary, relation, max_extra_turns=2,
        question="What are the names of John's children?", turns=turns)

    assert kyle.turn_id in relation.seed_turn_ids
    assert sara.turn_id in relation.seed_turn_ids
    assert set(merged.trace["auxiliary_extra_turn_ids"]) == {
        kyle.turn_id, sara.turn_id}


def test_relation_family_focus_handles_alias_acquisition_and_departure() -> None:
    alias = turn(
        0, "I always call Joanna Jo.", session="s1", speaker="Nate")
    turtle = turn(
        1, "I adopted two turtles in June.", session="s2", speaker="Nate")
    canada = turn(
        2, "I leave for Canada tomorrow.", session="s3", speaker="James")
    turns = {row.turn_id: row for row in (alias, turtle, canada)}

    nickname = build_source_focus_plan(
        question="What nickname does Nate use for Joanna?", turns=turns,
        max_turns=3, relation_families=True)
    acquired = build_source_focus_plan(
        question="When did Nate get his first two turtles?", turns=turns,
        max_turns=3, relation_families=True)
    departed = build_source_focus_plan(
        question="When did James depart for Canada?", turns=turns,
        max_turns=3, relation_families=True)

    assert nickname.seed_turn_ids[0] == alias.turn_id
    assert acquired.seed_turn_ids[0] == turtle.turn_id
    assert departed.seed_turn_ids[0] == canada.turn_id


def test_relation_family_recovers_abbreviated_vocative_alias() -> None:
    alias = turn(
        0, "Hey Jo, guess what I did?", session="alias", speaker="Nate")
    distractors = [turn(
        index + 1, f"Hey Joanna, ordinary update {index}.",
        session=f"other-{index}", speaker="Nate") for index in range(16)]
    turns = {row.turn_id: row for row in (alias, *distractors)}

    plan = build_source_focus_plan(
        question="What nickname does Nate use for Joanna?", turns=turns,
        max_turns=8, relation_families=True,
        relation_concept_coverage=True, expanded_relation_concepts=True)

    assert alias.turn_id in plan.seed_turn_ids
    assert "alias" in plan.trace["relation_concepts_covered_by_seeds"]


def test_relation_concept_lane_binds_recommendation_source_direction() -> None:
    recommendation = turn(
        0, "I recommend the movie Little Women.",
        session="recommendation", speaker="Joanna")
    acknowledgements = [turn(
        index + 1, f"Thanks for the recommendation number {index}.",
        session=f"ack-{index}", speaker="Nate") for index in range(16)]
    turns = {row.turn_id: row for row in (
        recommendation, *acknowledgements)}

    plan = build_source_focus_plan(
        question="What recommendations has Nate received from Joanna?",
        turns=turns, max_turns=8, relation_families=True,
        relation_concept_coverage=True, expanded_relation_concepts=True)

    assert recommendation.turn_id in plan.seed_turn_ids


def test_relation_auxiliary_closes_two_step_elliptical_response() -> None:
    instruction = turn(
        0, "You need to practice a little first, then we can play FIFA together.",
        session="game", speaker="John")
    clarification = turn(
        1, "I hope it is easy to control.", session="game", speaker="James")
    value = turn(
        2, "All you need is a gamepad and a sense of timing.",
        session="game", speaker="John")
    distractors = [turn(
        index + 3, f"John suggested an unrelated activity {index}.",
        session=f"other-{index}", speaker="John") for index in range(12)]
    turns = {row.turn_id: row for row in (
        instruction, clarification, value, *distractors)}
    question = (
        "What did John suggest James practice before playing FIFA together?")
    relation = build_source_focus_plan(
        question=question, turns=turns, max_turns=8,
        relation_families=True)
    primary = build_source_focus_plan(
        question="unrelated primary view", turns=turns, max_turns=1)
    merged = append_source_focus_witnesses(
        primary, relation, excluded_turn_ids=(instruction.turn_id,),
        max_extra_turns=2, question=question, turns=turns)

    assert value.turn_id in merged.trace["auxiliary_extra_turn_ids"]
    assert merged.trace["auxiliary_focus_views"][0][
        "dialogue_packet_radius"] == 2
    assert merged.trace["auxiliary_focus_views"][0]["selected_lanes"][
        value.turn_id] == "dialogue_packet"


def test_relation_auxiliary_bridges_fave_to_preference_value() -> None:
    question_turn = turn(
        0, "Contemporary is my top pick. What's your fave?",
        session="dance", speaker="Jon")
    value = turn(
        1, "Contemporary dance is expressive and really speaks to me.",
        session="dance", speaker="Gina")
    turns = {row.turn_id: row for row in (question_turn, value)}
    question = "What is Gina's favorite style of dance?"
    relation = build_source_focus_plan(
        question=question, turns=turns, max_turns=2,
        relation_families=True)
    primary = build_source_focus_plan(
        question="unrelated primary view", turns=turns, max_turns=1)
    merged = append_source_focus_witnesses(
        primary, relation, excluded_turn_ids=(question_turn.turn_id,),
        max_extra_turns=1, question=question, turns=turns)

    assert merged.trace["auxiliary_extra_turn_ids"] == [value.turn_id]


def test_relation_family_maps_celebration_to_chill_response() -> None:
    event = turn(
        0, "I won an international tournament yesterday!",
        session="tournament", speaker="Nate")
    response = turn(
        8, ("I'm taking some time off to chill with my pets. "
            "Anything cool happening with you?"),
        session="tournament", speaker="Nate")
    distractors = [turn(
        index + 10,
        f"Nate won international video game tournament number {index}.",
        session=f"match-{index}", speaker="Nate") for index in range(30)]
    turns = {row.turn_id: row for row in (event, response, *distractors)}
    plan = build_source_focus_plan(
        question="How did Nate celebrate winning the tournament?",
        turns=turns, max_turns=8, relation_families=True,
        relation_concept_coverage=True)

    assert response.turn_id in plan.seed_turn_ids
    assert "celebrate" in plan.trace[
        "relation_concepts_covered_by_seeds"]


def test_temporal_focus_matches_relative_event_to_absolute_query_date() -> None:
    boston = SourceTurn(
        turn_id="boston", memory_id="m1", session_id="s2", turn_index=1,
        speaker="Calvin", listener="Dave", role="user",
        timestamp="2:44 pm on 4 October, 2023",
        raw_text=("Yesterday I met artists in Boston and discussed a new "
                  "collaboration."), content_hash="hb")
    distractors = [turn(
        index, f"Calvin discussed unrelated city plans {index}.",
        session=f"s{index + 3}", speaker="Calvin") for index in range(12)]
    turns = {row.turn_id: row for row in (boston, *distractors)}

    plan = build_temporal_focus_plan(
        question="Which city was Calvin at on October 3, 2023?",
        turns=turns, max_turns=4)

    assert boston.turn_id in plan.seed_turn_ids
    assert plan.trace["query_time_resolved"] is True
    assert plan.trace["event_time_candidates"] >= 1


def test_temporal_focus_combines_time_bucket_with_existing_dense_score() -> None:
    relevant = SourceTurn(
        turn_id="relevant", memory_id="m1", session_id="s1", turn_index=0,
        speaker="Calvin", listener="Dave", role="user",
        timestamp="2:44 pm on 20 August, 2023",
        raw_text="I filmed material for the album.", content_hash="hr")
    lexical = SourceTurn(
        turn_id="lexical", memory_id="m1", session_id="s2", turn_index=0,
        speaker="Calvin", listener="Dave", role="user",
        timestamp="2:44 pm on 21 August, 2023",
        raw_text="I mentioned a city I might visit.", content_hash="hl")
    plan = build_temporal_focus_plan(
        question="Which city was Calvin visiting in August 2023?",
        turns={row.turn_id: row for row in (relevant, lexical)},
        max_turns=2, dense_scores={relevant.turn_id: 1.0})

    assert plan.seed_turn_ids[0] == relevant.turn_id
    assert plan.trace["query_time_precision"] == "month"
    assert plan.trace["uses_dense_scores"] is True


def test_temporal_focus_uses_expanded_activity_relation_inside_month() -> None:
    kayaking = SourceTurn(
        turn_id="kayaking", memory_id="m1", session_id="s1", turn_index=0,
        speaker="Sam", listener="Evan", role="user",
        timestamp="4:07 pm on 14 October, 2023",
        raw_text="Kayaking looks fun; I am considering giving it a try.",
        content_hash="hk")
    routine = SourceTurn(
        turn_id="routine", memory_id="m1", session_id="s2", turn_index=0,
        speaker="Sam", listener="Evan", role="user",
        timestamp="9:00 am on 15 October, 2023",
        raw_text="My existing health routine is difficult.",
        content_hash="hr")

    plan = build_temporal_focus_plan(
        question="Which new activity does Sam take up in October 2023?",
        turns={row.turn_id: row for row in (kayaking, routine)},
        max_turns=2)

    assert plan.seed_turn_ids[0] == kayaking.turn_id


def test_temporal_focus_does_not_open_broad_when_fallback() -> None:
    one = turn(0, "James will leave for Canada tomorrow.", speaker="James")
    base = build_temporal_focus_plan(
        question="When will James leave for Canada?", turns={one.turn_id: one})
    plan = build_temporal_focus_plan(
        question="When will James leave for Canada?", turns={one.turn_id: one},
        expanded_relation_concepts=True)
    assert base.selected_turn_ids == ()
    assert plan.selected_turn_ids == ()
    assert plan.trace["query_time_resolved"] is False
    assert plan.trace["expanded_temporal_query_triggered"] is False


def test_temporal_focus_recovers_duration_answer_with_entity_context() -> None:
    turtle = turn(
        0, "These little turtles keep me calm.", session="pets",
        speaker="Nate")
    question = turn(
        1, "How long have you had them?", session="pets",
        speaker="Joanna")
    duration = turn(
        2, "I've had them for 3 years now.", session="pets",
        speaker="Nate")
    distractors = [turn(
        index + 3, f"Nate discussed turtle care topic {index}.",
        session=f"other-{index}", speaker="Nate") for index in range(20)]
    turns = {row.turn_id: row for row in (
        turtle, question, duration, *distractors)}

    plan = build_temporal_focus_plan(
        question="When did Nate get his first two turtles?", turns=turns,
        max_turns=8, expanded_relation_concepts=True)

    assert plan.seed_turn_ids[0] == duration.turn_id
    assert plan.trace["duration_candidates"] >= 1


def test_temporal_focus_follows_after_event_anchor() -> None:
    story = turn(
        0, "I read stories about a Himalayan trek.", session="trip",
        speaker="Tim")
    bridge = turn(
        1, "The trek was difficult but worth it.", session="trip",
        speaker="Tim")
    activity = turn(
        2, "I visited a travel agency for my next trip.", session="trip",
        speaker="Tim")
    distractors = [turn(
        index + 3, f"Tim read an unrelated story after dinner {index}.",
        session=f"other-{index}", speaker="Tim") for index in range(20)]
    turns = {row.turn_id: row for row in (
        story, bridge, activity, *distractors)}

    plan = build_temporal_focus_plan(
        question=("What activity did Tim do after reading the stories about "
                  "the Himalayan trek?"),
        turns=turns, max_turns=8, expanded_relation_concepts=True)

    assert activity.turn_id in plan.seed_turn_ids[:2]
    assert plan.trace["transition_candidates"] >= 1


def test_temporal_transition_is_an_independent_additive_view() -> None:
    event = turn(
        0, "I read stories about a Himalayan trek.", session="trip",
        speaker="Tim")
    follow_up = turn(
        1, "I visited a travel agency for my next trip.", session="trip",
        speaker="Tim")
    turns = {row.turn_id: row for row in (event, follow_up)}
    primary = build_source_focus_plan(
        question=("What activity did Tim do after reading stories about "
                  "the Himalayan trek?"),
        turns=turns, max_turns=1)
    transition = build_temporal_focus_plan(
        question=("What activity did Tim do after reading stories about "
                  "the Himalayan trek?"),
        turns=turns, max_turns=2, expanded_relation_concepts=True)

    merged = append_source_focus_witnesses(
        primary, transition, max_extra_turns=1)

    assert transition.trace["expanded_temporal_query_triggered"] is True
    assert merged.selected_turn_ids[:len(primary.selected_turn_ids)] == (
        primary.selected_turn_ids)
    assert merged.trace["auxiliary_focus_views"][-1]["type"] == (
        "temporal_transition")


def test_unified_prompt_is_deduplicated_and_single_payload() -> None:
    first = turn(0, "Alice moved the shoes to the closet.", session="s1")
    second = turn(1, "The closet rack is beside the door.", session="s1")
    third = turn(2, "Bob discussed unrelated travel.", session="s2")
    turns = {row.turn_id: row for row in (first, second, third)}
    focus = build_source_focus_plan(
        question="Where are Alice's shoes?", turns=turns, max_turns=3)
    rendered = build_unified_source_prompt(
        question="Where are Alice's shoes?", question_date="2025-02-01",
        graph_turn_ids=(first.turn_id, third.turn_id), focus_plan=focus,
        turns=turns, graph_limit=2, focus_limit=2)

    assert len(rendered.messages) == 2
    assert len(rendered.evidence_turn_ids) == len(set(rendered.evidence_turn_ids))
    assert rendered.evidence_turn_ids[:2] == (first.turn_id, third.turn_id)
    user = rendered.messages[-1]["content"]
    assert "Graph witnesses" in user
    assert "Source-focus witnesses" in user
    assert "previous answer" not in user.casefold()


def test_unified_prompt_can_promote_focus_overlap_without_duplication() -> None:
    first = turn(0, "Alice moved the shoes to the closet.", session="s1")
    second = turn(1, "The closet rack is beside the door.", session="s1")
    third = turn(2, "Bob discussed unrelated travel.", session="s2")
    turns = {row.turn_id: row for row in (first, second, third)}
    focus = build_source_focus_plan(
        question="Where are Alice's shoes?", turns=turns, max_turns=2)
    rendered = build_unified_source_prompt(
        question="Where are Alice's shoes?", question_date="2025-02-01",
        graph_turn_ids=(first.turn_id, third.turn_id), focus_plan=focus,
        turns=turns, graph_limit=2, focus_limit=2,
        promote_focus_overlap=True)

    assert len(rendered.evidence_turn_ids) == len(set(rendered.evidence_turn_ids))
    assert first.turn_id in rendered.evidence_turn_ids
    assert rendered.trace["focus_overlap_promoted"] is True
    assert rendered.trace["promoted_focus_turns"] >= 1
    user = rendered.messages[-1]["content"]
    assert user.count("Alice moved the shoes to the closet") == 1
    assert "[F01]" in user


def test_unified_prompt_can_rank_focus_sessions_by_seed_relevance() -> None:
    weaker = turn(
        0, "Alice mentioned the cobalt color.", session="s1")
    stronger = turn(
        1, "Alice keeps the cobalt telescope in Lisbon.", session="s9")
    turns = {row.turn_id: row for row in (weaker, stronger)}
    focus = build_source_focus_plan(
        question="Where is Alice's cobalt telescope?", turns=turns,
        max_turns=2)
    ordinary = build_unified_source_prompt(
        question="Where is Alice's cobalt telescope?", question_date="",
        graph_turn_ids=(), focus_plan=focus, turns=turns,
        graph_limit=0, focus_limit=2)
    ranked = build_unified_source_prompt(
        question="Where is Alice's cobalt telescope?", question_date="",
        graph_turn_ids=(), focus_plan=focus, turns=turns,
        graph_limit=0, focus_limit=2, rank_focus_sessions=True)

    assert ordinary.evidence_turn_ids[0] == weaker.turn_id
    assert ranked.evidence_turn_ids[0] == stronger.turn_id
    assert set(ranked.evidence_turn_ids) == set(ordinary.evidence_turn_ids)
    assert ranked.trace["focus_sessions_query_ranked"] is True


def test_unified_prompt_focus_capsule_repeats_without_new_evidence_id() -> None:
    witness = turn(
        0, "Alice keeps the cobalt telescope in Lisbon.", session="s9")
    turns = {witness.turn_id: witness}
    focus = build_source_focus_plan(
        question="Where is Alice's cobalt telescope?", turns=turns,
        max_turns=1)
    rendered = build_unified_source_prompt(
        question="Where is Alice's cobalt telescope?", question_date="",
        graph_turn_ids=(witness.turn_id,), focus_plan=focus, turns=turns,
        graph_limit=1, focus_limit=1, focus_capsule_turns=1)

    assert rendered.evidence_turn_ids == (witness.turn_id,)
    user = rendered.messages[-1]["content"]
    assert user.count("cobalt telescope in Lisbon") == 2
    assert "[P01 repeats=G01]" in user
    assert "count each repeated source turn only once" in (
        rendered.messages[0]["content"])
    assert rendered.trace["focus_capsule_turns"] == 1


def test_unified_prompt_focus_capsule_is_route_gated_and_bounded() -> None:
    witness = turn(
        0,
        "Alice keeps the cobalt telescope in Lisbon. " + "context " * 100,
        session="s9")
    turns = {witness.turn_id: witness}
    focus = build_source_focus_plan(
        question="Where is Alice's cobalt telescope?", turns=turns,
        max_turns=1)
    disabled = build_unified_source_prompt(
        question="Where is Alice's cobalt telescope?", question_date="",
        graph_turn_ids=(witness.turn_id,), focus_plan=focus, turns=turns,
        graph_limit=1, focus_limit=1, focus_capsule_turns=1,
        focus_capsule_routes=("temporal",), focus_capsule_max_chars=180)
    enabled = build_unified_source_prompt(
        question="Where is Alice's cobalt telescope?", question_date="",
        graph_turn_ids=(witness.turn_id,), focus_plan=focus, turns=turns,
        graph_limit=1, focus_limit=1, focus_capsule_turns=1,
        focus_capsule_routes=("lookup",), focus_capsule_max_chars=180)

    assert disabled.trace["focus_capsule_turns"] == 0
    assert disabled.trace["focus_capsule_route_enabled"] is False
    assert enabled.trace["focus_capsule_turns"] == 1
    assert enabled.trace["focus_capsule_max_chars"] == 180
    capsule = enabled.messages[-1]["content"].split(
        "Query-focus capsule", 1)[1]
    assert len(capsule.split("\n\nAnswer", 1)[0]) < 260


def test_unified_prompt_restores_short_lossless_focus_with_bounded_delta() -> None:
    witness = turn(
        0,
        'I loved "Becoming Nicole" by Amy Ellis Nutt. It is a true story '
        'about a trans girl and her family. The writing follows a long and '
        'difficult journey of identity, support, hope, and acceptance. It '
        'made me reflect on many experiences from my own life. The family '
        'members learn how to support one another. I highly recommend it!',
        session="s1")
    distractor = turn(1, "Alice discussed another book.", session="s2")
    turns = {row.turn_id: row for row in (witness, distractor)}
    focus = build_source_focus_plan(
        question="What book did Alice recommend?", turns=turns, max_turns=2)
    rendered = build_unified_source_prompt(
        question="What book did Alice recommend?", question_date="2025-02-01",
        graph_turn_ids=(distractor.turn_id,), focus_plan=focus, turns=turns,
        graph_limit=1, focus_limit=2, focus_lossless_extra_chars=400)

    assert '"Becoming Nicole"' in rendered.messages[-1]["content"]
    assert rendered.trace["lossless_focus_turns"] >= 1
    assert 0 < rendered.trace["lossless_focus_extra_chars"] <= 400


def test_unified_prompt_can_show_relation_routes_without_fact_summaries() -> None:
    first = turn(0, "Alice adopted a dog named Luna.", session="s1")
    second = turn(1, "Luna later moved with Alice.", session="s2")
    turns = {row.turn_id: row for row in (first, second)}
    focus = build_source_focus_plan(
        question="What is Alice's dog's name?", turns=turns, max_turns=2)
    rendered = build_unified_source_prompt(
        question="What is Alice's dog's name?", question_date="2025-02-01",
        graph_turn_ids=(first.turn_id,), focus_plan=focus, turns=turns,
        graph_limit=1, focus_limit=1,
        relation_hints={first.turn_id: " {via=entity>time}"})

    assert "[G01] {via=entity>time}" in rendered.messages[-1]["content"]
    assert "retrieval-route families" in rendered.messages[0]["content"]
    assert rendered.trace["relation_path_labels"] is True
    assert rendered.trace["relation_labeled_turns"] == 1


def test_unified_prompt_maps_auxiliary_views_to_visible_source_ids() -> None:
    first = turn(0, "Alice adopted a dog.", session="s1")
    second = turn(1, "The dog's name is Luna.", session="s1")
    turns = {row.turn_id: row for row in (first, second)}
    primary = SourceFocusPlan(
        route="lookup", selected_turn_ids=(first.turn_id,),
        seed_turn_ids=(first.turn_id,), neighbor_turn_ids=(), trace={})
    auxiliary = SourceFocusPlan(
        route="lookup", selected_turn_ids=(second.turn_id,),
        seed_turn_ids=(second.turn_id,), neighbor_turn_ids=(), trace={
            "relation_families": True,
            "relation_concept_coverage": True,
            "expanded_relation_concepts": True,
        })
    focus = append_source_focus_witnesses(
        primary, auxiliary, max_extra_turns=1)

    rendered = build_unified_source_prompt(
        question="What is Alice's dog's name?", question_date="2025-02-01",
        graph_turn_ids=(first.turn_id,), focus_plan=focus, turns=turns,
        graph_limit=1, focus_limit=1, focus_navigation_map=True)

    user = rendered.messages[-1]["content"]
    assert "relation-synonym/direction: F01" in user
    assert "retrieval pointers, not facts" in rendered.messages[0]["content"]
    assert rendered.trace["focus_navigation_map_ids"] == ["F01"]


def test_unified_prompt_renders_multi_obligation_derivation_contract() -> None:
    witness = turn(
        0, "Alice enjoyed visiting Tampa and its beaches.", session="s1")
    turns = {witness.turn_id: witness}
    focus = build_source_focus_plan(
        question="Which state might Alice have visited?", turns=turns,
        max_turns=1)
    rendered = build_unified_source_prompt(
        question="Which state might Alice have visited?", question_date="",
        graph_turn_ids=(witness.turn_id,), focus_plan=focus, turns=turns,
        graph_limit=1, focus_limit=1,
        query_obligations=(
            "inference", "geographic_resolution", "reasoning_chain"))

    user = rendered.messages[-1]["content"]
    system = rendered.messages[0]["content"]
    assert "Query obligations: inference, geographic_resolution" in user
    assert "resolve only the named source place" in user
    assert "one stable ordinary-knowledge step" in system
    assert rendered.trace["derived_answer_enabled"] is True
