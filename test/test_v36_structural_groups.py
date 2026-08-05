from __future__ import annotations

from graphmem_demo.models import QuestionCase
from graphmem_demo.v36.build import build_index, build_turn_nodes, validate_index
from graphmem_demo.v36.schema import (
    CoverageEntry, EvidenceGroup,
    QuantityValue,
    RoleFrameNode,
    RoutingCard,
    TemporalValue,
)


def _index():
    case = QuestionCase(
        question_id="q", question_type="unknown", question="current state",
        answer="", question_date="2026-02-01",
        haystack_sessions=[[{
            "role": "user", "speaker": "A",
            "content": "The inventory changed.",
        }]],
        haystack_session_ids=["s"], haystack_dates=["2026-01-01"],
        answer_session_ids=[],
    )
    turns = build_turn_nodes(case)
    source = turns[0].node_id

    def frame(frame_id: str, obj: str, op: str, date: str, *, polarity="positive"):
        return RoleFrameNode(
            frame_id=frame_id, question_id="q", session_ids=["s"],
            frame_kind="state", owner_key="a", entity_key="inventory",
            predicate_key="contains", object_key=obj, context_key="main",
            polarity=polarity, state_op=op,
            temporal=TemporalValue(event_time=date, observed_at="2026-01-01"),
            event_identity_key="inventory update",
            quantity=QuantityValue(value=1, unit="item"),
            source_turn_ids=[source], confidence=0.9,
            retrieval_text=f"a inventory contains {obj} {date}",
        )

    frames = [
        frame("q:f:0", "camera", "set", "2026-01-01"),
        frame("q:f:1", "lens", "add", "2026-01-02"),
        frame("q:f:2", "lens", "remove", "2026-01-03", polarity="negative"),
    ]
    card = RoutingCard(
        card_id="q:s:card", question_id="q", session_id="s",
        speaker_keys=["a"], canonical_entities=["inventory"],
        relations=["contains"], key_events=["inventory update"],
        current_states=["camera"], time_range="2026-01-01 to 2026-01-03",
        frame_ids=[frame.frame_id for frame in frames],
        turn_ids=[source], routing_text="inventory contains update",
    )
    return build_index(
        question_id="q", turns=turns, frames=frames,
        routing_cards=[card],
        coverage=[CoverageEntry(source, "memory_frame", [f.frame_id for f in frames])],
    )


def test_state_temporal_collection_groups_are_complete() -> None:
    index = _index()
    assert validate_index(index) == []
    kinds = {group.group_kind for group in index.evidence_groups}
    assert {"state_transition", "collection", "temporal_pair"} <= kinds
    for group in index.evidence_groups:
        if group.group_kind in {"state_transition", "collection", "temporal_pair"}:
            assert all(group.completeness_mask.values()), (
                group.group_kind, group.completeness_mask
            )


def test_only_narrow_directed_relations_exist() -> None:
    index = _index()
    forbidden = {
        "participant", "temporal_scope", "episode_member", "theme_member",
        "operand_projection", "event_frame_member",
    }
    assert all(edge.directed for edge in index.edges)
    assert not ({edge.relation for edge in index.edges} & forbidden)
    temporal = [
        edge for edge in index.edges if edge.relation == "temporal_endpoint"
    ]
    assert temporal
    assert all(
        edge.provenance["local_rule"] == "same_event_identity_endpoint"
        for edge in temporal
    )


def test_identity_candidates_reject_generic_predicate_hubs() -> None:
    from graphmem_demo.v36.runtime import _candidate_pairs

    generic = [
        RoleFrameNode(frame_id=f"q:f:{index}", question_id="q", session_ids=[f"s{index}"], frame_kind="fact", owner_key="a", entity_key="caroline", predicate_key="predicate", object_key=value, source_turn_ids=[f"q:s{index}:turn:0"], retrieval_text=value)
        for index, value in enumerate(("attended", "thought"))
    ]
    assert _candidate_pairs(generic) == []
    for frame in generic:
        frame.event_identity_key = "one grounded event"
    assert _candidate_pairs(generic) == [["q:f:0", "q:f:1"]]


def test_identity_candidates_include_bounded_shared_phrase() -> None:
    from graphmem_demo.v36.runtime import (
        _anchored_identity_phrase, _candidate_pairs, _identity_supported,
    )

    left = RoleFrameNode(
        frame_id="q:f:left", question_id="q", session_ids=["s1"],
        frame_kind="fact", owner_key="dana", entity_key="colleagues",
        predicate_key="known since", object_key="moving former employer",
        context_key="these colleagues", source_turn_ids=["q:s1:turn:0"],
        retrieval_text="dana knew colleagues since moving from former employer",
    )
    right = RoleFrameNode(
        frame_id="q:f:right", question_id="q", session_ids=["s2"],
        frame_kind="fact", owner_key="dana", entity_key="mentor",
        predicate_key="employer", object_key="northwind labs",
        context_key="former employer", source_turn_ids=["q:s2:turn:0"],
        retrieval_text="dana mentor at former employer northwind labs",
    )
    assert _candidate_pairs([left, right]) == [["q:f:left", "q:f:right"]]
    assert _identity_supported(left, right)
    assert _anchored_identity_phrase(left, right) == "former employer"
    right.owner_key = "morgan"
    assert _candidate_pairs([left, right]) == []
    assert not _identity_supported(left, right)


def test_cross_session_completed_activity_forms_bounded_collection() -> None:
    case = QuestionCase(
        question_id="activity", question_type="unknown", question="",
        answer="", question_date="2026-02-01",
        haystack_sessions=[
            [{"role": "user", "speaker": "Dana", "content": "I cycled by the river."}],
            [{"role": "user", "speaker": "Dana", "content": "I went cycling in the park."}],
        ],
        haystack_session_ids=["s1", "s2"],
        haystack_dates=["2026-01-01", "2026-01-02"],
        answer_session_ids=[],
    )
    turns = build_turn_nodes(case)
    frames = [
        RoleFrameNode(
            frame_id="activity:f:0", question_id="activity", session_ids=["s1"],
            frame_kind="event", owner_key="dana", entity_key="cycling outing",
            predicate_key="cycled", object_key="river",
            lifecycle_status="completed", modality="asserted", polarity="positive",
            event_identity_key="dana cycling river",
            source_turn_ids=[turns[0].node_id], confidence=0.95,
            retrieval_text="dana cycling river",
        ),
        RoleFrameNode(
            frame_id="activity:f:1", question_id="activity", session_ids=["s2"],
            frame_kind="state", owner_key="dana", entity_key="cycling outing",
            predicate_key="went cycling", object_key="park",
            lifecycle_status="completed", modality="asserted", polarity="positive",
            event_identity_key="dana cycling park",
            source_turn_ids=[turns[1].node_id], confidence=0.95,
            retrieval_text="dana cycling park",
        ),
    ]
    cards = [
        RoutingCard(
            card_id=f"activity:s{position + 1}:card", question_id="activity",
            session_id=f"s{position + 1}", speaker_keys=["dana"],
            canonical_entities=["cycling outing"], relations=["cycling"],
            key_events=["cycling"], current_states=[], time_range="",
            frame_ids=[frames[position].frame_id],
            turn_ids=[turns[position].node_id],
            routing_text=frames[position].retrieval_text,
        )
        for position in range(2)
    ]
    index = build_index(
        question_id="activity", turns=turns, frames=frames, routing_cards=cards,
        coverage=[
            CoverageEntry(turns[position].node_id, "memory_frame", [frames[position].frame_id])
            for position in range(2)
        ],
    )
    groups = [
        group for group in index.evidence_groups
        if group.group_kind == "collection"
        and set(group.member_frame_ids) == {frame.frame_id for frame in frames}
    ]
    assert groups and all(groups[0].completeness_mask.values())
    edges = [
        edge for edge in index.edges
        if edge.relation == "collection_member" and edge.src == groups[0].group_id
    ]
    assert len(edges) == 2
    assert all(
        edge.provenance["local_rule"] == "bounded_cross_session_activity_collection"
        for edge in edges
    )
    from graphmem_demo.v36.retrieval import _collection_scope_cards, build_query_ir
    selected, trace = _collection_scope_cards(
        index, build_query_ir("Where has Dana cycled?"), [cards[0].card_id]
    )
    assert selected == [cards[0].card_id, cards[1].card_id]
    assert trace and trace[0]["reason"] == "complete_collection_scope"


def test_scalar_location_requires_coherent_reference_group() -> None:
    from graphmem_demo.v36.retrieval import (
        _location_evidence_coherent, build_query_ir,
    )

    move = RoleFrameNode(
        frame_id="q:move", question_id="q", session_ids=["s1"],
        frame_kind="event", owner_key="dana", entity_key="relocation",
        predicate_key="moved from", object_key="former employer",
        source_turn_ids=["q:s1:turn:0"], retrieval_text="dana moved from former employer",
    )
    origin = RoleFrameNode(
        frame_id="q:origin", question_id="q", session_ids=["s2"],
        frame_kind="fact", owner_key="dana", entity_key="former employer",
        predicate_key="origin", object_key="northwind labs",
        context_key="former employer", source_turn_ids=["q:s2:turn:0"],
        retrieval_text="dana former employer origin northwind labs",
    )
    ir = build_query_ir("Where did Dana move from last year?")
    assert not _location_evidence_coherent(ir, [move, origin], [])
    group = EvidenceGroup(
        group_id="q:g", question_id="q", group_kind="reference_chain",
        member_frame_ids=[move.frame_id, origin.frame_id],
        source_turn_ids=[*move.source_turn_ids, *origin.source_turn_ids],
        required_roles=["reference", "identity", "source"],
        completeness_mask={"reference": True, "identity": True, "source": True},
        provenance_complete=True, confidence=0.9,
        retrieval_text=f"{move.retrieval_text} | {origin.retrieval_text}",
        session_ids=["s1", "s2"],
    )
    assert _location_evidence_coherent(ir, [move, origin], [group])
