from __future__ import annotations

from types import SimpleNamespace

from graphmem_demo.v3.coarse_projection import (
    project_reached_episodes,
    project_routed_claim_sources,
)
from graphmem_demo.v3.contrast_relation import contrast_alternative_hint
from graphmem_demo.v3.retrieval import build_query_frame
from graphmem_demo.v3.schema import ClaimNode, EpisodeNode


def test_reached_episode_projects_best_claim_and_lossless_source_turn() -> None:
    episode = EpisodeNode(
        node_id="episode:1",
        question_id="q",
        session_id="s1",
        session_date="2024-01-01",
        label="Games and plans",
        participant_keys=["alex"],
        time_start="2024-01-01",
        time_end="2024-01-01",
        turn_ids=["turn:weak", "turn:answer"],
        claim_ids=["claim:weak", "claim:answer"],
        event_ids=[],
        retrieval_text="Alex discusses games and plans.",
    )
    nodes = {
        episode.node_id: episode,
        "claim:weak": SimpleNamespace(
            node_id="claim:weak", source_turn_ids=["turn:weak"],
            retrieval_text="Alex uses voice chat.",
        ),
        "claim:answer": SimpleNamespace(
            node_id="claim:answer", source_turn_ids=["turn:answer"],
            retrieval_text="Alex plays Star Arena.",
        ),
        "turn:weak": SimpleNamespace(
            node_id="turn:weak", retrieval_text="I use voice chat."
        ),
        "turn:answer": SimpleNamespace(
            node_id="turn:answer", retrieval_text="I play Star Arena."
        ),
    }
    primary = {
        "episode:1": 0.5,
        "claim:weak": 0.2,
        "claim:answer": 0.7,
        "turn:weak": 0.2,
        "turn:answer": 0.8,
    }
    slot = {
        "episode:1": 0.6,
        "claim:weak": 0.1,
        "claim:answer": 0.95,
        "turn:weak": 0.1,
        "turn:answer": 0.9,
    }
    selected, trace = project_reached_episodes(
        nodes=nodes,
        expanded_scores={"episode:1": 0.8},
        primary_similarity=lambda node: primary[node.node_id],
        slot_similarity=lambda node: slot[node.node_id],
        query_overlap=lambda node: 0.5 if "Star Arena" in node.retrieval_text else 0.0,
        episode_limit=1,
        per_episode_limit=2,
        total_limit=2,
    )
    assert selected == ["claim:answer", "turn:answer"]
    assert trace[1]["via_source_node_id"] == "claim:answer"


def test_routed_sessions_project_diverse_answer_compatible_claim_sources() -> None:
    def claim(
        node_id: str, session_id: str, predicate: str, source_id: str,
    ) -> ClaimNode:
        return ClaimNode(
            node_id=node_id,
            question_id="q",
            session_id=session_id,
            subject="Alex",
            subject_key="alex",
            predicate=predicate,
            predicate_key=predicate,
            object=node_id,
            object_key=node_id,
            source_turn_ids=[source_id],
            retrieval_text=f"Alex {predicate} {node_id}",
        )

    answer_a = claim("claim:a", "s1", "recommended", "turn:a")
    distractor = claim("claim:d", "s1", "visited", "turn:d")
    answer_b = claim("claim:b", "s2", "recommended", "turn:b")
    low_information = claim("claim:said", "s2", "said", "turn:said")
    nodes = {
        node.node_id: node
        for node in [answer_a, distractor, answer_b, low_information]
    }
    nodes.update({
        "turn:a": SimpleNamespace(node_id="turn:a", retrieval_text="Read Alpha."),
        "turn:d": SimpleNamespace(node_id="turn:d", retrieval_text="Visited Rome."),
        "turn:b": SimpleNamespace(node_id="turn:b", retrieval_text="Read Beta."),
        "turn:said": SimpleNamespace(node_id="turn:said", retrieval_text="We talked."),
    })
    primary = {"claim:a": 0.8, "claim:d": 0.7, "claim:b": 0.7,
               "turn:a": 0.8, "turn:d": 0.7, "turn:b": 0.7}
    slot = {"claim:a": 0.95, "claim:d": 0.05, "claim:b": 0.90,
            "turn:a": 0.95, "turn:d": 0.05, "turn:b": 0.90}
    overlap = {"claim:a": 0.7, "claim:d": 0.2, "claim:b": 0.6,
               "turn:a": 0.7, "turn:d": 0.2, "turn:b": 0.6}

    selected, trace = project_routed_claim_sources(
        nodes=nodes,
        scope_session_ids=["s1", "s2"],
        primary_similarity=lambda node: primary.get(node.node_id, 0.0),
        slot_similarity=lambda node: slot.get(node.node_id, 0.0),
        query_overlap=lambda node: overlap.get(node.node_id, 0.0),
        per_session_limit=1,
    )

    assert selected == ["turn:a", "turn:b"]
    assert [row["session_id"] for row in trace] == ["s1", "s2"]
    assert all(row["claim_id"] != "claim:said" for row in trace)


def test_contrast_operator_binds_alternative_from_dialogue_adjacency() -> None:
    frame = build_query_frame(
        "What did Alex plan to do rather than resuming running?"
    )
    turns = [
        SimpleNamespace(
            node_id="s1:t1", session_id="s1", turn_index=1,
            speaker="Morgan", speaker_key="morgan",
            text="Have you thought about resuming running?",
        ),
        SimpleNamespace(
            node_id="s1:t2", session_id="s1", turn_index=2,
            speaker="Alex", speaker_key="alex",
            text="We planned to swim at the lake with my partner.",
        ),
        SimpleNamespace(
            node_id="s2:t1", session_id="s2", turn_index=1,
            speaker="Alex", speaker_key="alex",
            text="I later planned to go camping.",
        ),
    ]
    hint = contrast_alternative_hint(frame, turns)
    assert hint is not None
    assert hint["complete"] is True
    assert hint["value"] == "swim at the lake with my partner"
    assert hint["source_turn_ids"] == ["s1:t1", "s1:t2"]


def test_contrast_operator_does_not_fire_without_discourse_marker() -> None:
    frame = build_query_frame("What did Alex plan to do?")
    turns = [
        SimpleNamespace(
            node_id="s1:t1", session_id="s1", turn_index=1,
            speaker="Alex", speaker_key="alex",
            text="I planned to swim.",
        )
    ]
    assert contrast_alternative_hint(frame, turns) is None


def test_contrast_operator_resolves_generic_deferred_action_cause() -> None:
    frame = build_query_frame("Why did Alex put off doing pottery?")
    turns = [
        SimpleNamespace(
            node_id="s1:t4", session_id="s1", turn_index=4,
            speaker="Morgan", speaker_key="morgan",
            text="Have you thought about resuming pottery?",
        ),
        SimpleNamespace(
            node_id="s1:t5", session_id="s1", turn_index=5,
            speaker="Alex", speaker_key="alex",
            text="I am planning to visit the museum with my partner.",
        ),
    ]
    hint = contrast_alternative_hint(frame, turns)
    assert hint is not None
    assert hint["relation_kind"] == "causal_displacement"
    assert hint["value"] == "visit the museum with my partner"
    assert hint["complete"] is True


def test_deferred_action_ignores_target_outside_question_clause() -> None:
    frame = build_query_frame("Why did Alex put off doing pottery?")
    turns = [
        SimpleNamespace(
            node_id="s1:t4", session_id="s1", turn_index=4,
            speaker="Morgan", speaker_key="morgan",
            text=(
                "Pottery can be a relaxing hobby. "
                "What self-care activities have you been doing lately?"
            ),
        ),
        SimpleNamespace(
            node_id="s1:t5", session_id="s1", turn_index=5,
            speaker="Alex", speaker_key="alex",
            text="I am planning to attend a pottery retreat.",
        ),
    ]
    assert contrast_alternative_hint(frame, turns) is None
