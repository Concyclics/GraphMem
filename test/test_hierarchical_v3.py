from __future__ import annotations

import inspect
import json
from pathlib import Path

from graphmem_demo.clients import MockCompressor, MockDeepSeekClient, MockEmbeddingClient
from graphmem_demo.models import QuestionCase
from graphmem_demo.pipeline import DemoConfig, InflightLimiter, run_case
from graphmem_demo.v3.build import (
    build_hypergraph,
    build_turn_nodes,
    parse_session_extraction,
    validate_hypergraph,
)
from graphmem_demo.v3.retrieval import (
    _planned_event_hint,
    _temporal_evidence_ledger,
    build_query_frame,
    retrieve,
)
from graphmem_demo.v3.schema import (
    ClaimNode,
    HyperEdge,
    HyperIncidence,
    TurnNode,
    V3Index,
)


def case(question: str = "What did Alice buy?") -> QuestionCase:
    return QuestionCase(
        question_id="q",
        question_type="unknown",
        question=question,
        answer="a camera",
        question_date="2026-01-03",
        haystack_sessions=[
            [
                {
                    "role": "assistant",
                    "speaker": "Alice",
                    "listener": "Bob",
                    "content": "I bought a camera yesterday.",
                },
                {
                    "role": "user",
                    "speaker": "Bob",
                    "listener": "Alice",
                    "content": "That will be useful.",
                },
            ]
        ],
        haystack_session_ids=["s"],
        haystack_dates=["2026-01-02"],
        answer_session_ids=["s"],
        memory_cache_key="test:q",
    )


def parsed_index(question: str = "What did Alice buy?") -> tuple[QuestionCase, V3Index]:
    item = case(question)
    turns = build_turn_nodes(item)
    payload = {
        "claims": [
            [
                "Alice",
                "bought",
                "a camera",
                "event",
                "positive",
                "asserted",
                "complete",
                "purchase",
                "2026-01-01",
                [turns[0].node_id],
                1,
                "item",
                0.95,
            ]
        ],
        "events": [
            [
                "Alice bought a camera",
                "complete",
                "2026-01-01",
                ["Alice"],
                [0],
                [turns[0].node_id],
                0.95,
            ]
        ],
        "episodes": [
            {
                "label": "Alice purchase",
                "turn_ids": [turn.node_id for turn in turns],
                "claim_indices": [0],
                "event_indices": [0],
            }
        ],
    }
    claims, events, episodes, error = parse_session_extraction(
        json.dumps(payload),
        question_id=item.question_id,
        session_id="s",
        session_date="2026-01-02",
        turns=turns,
    )
    assert error is None
    embedder = MockEmbeddingClient()
    for rows in (turns, claims, events):
        vectors = embedder.embed(
            [row.retrieval_text for row in rows],
            question_id=item.question_id,
            variant="hierarchical_hypergraph_v3",
        )
        for row, vector in zip(rows, vectors):
            row.embedding = vector
    index = build_hypergraph(
        question_id=item.question_id,
        session_ids=["s"],
        session_dates={"s": "2026-01-02"},
        turns=turns,
        claims=claims,
        events=events,
        episode_proposals={"s": episodes},
    )
    return item, index


def test_turns_use_explicit_speaker_not_transport_role() -> None:
    turns = build_turn_nodes(case())
    assert turns[0].speaker == "Alice"
    assert turns[0].speaker_key == "alice"
    assert turns[0].transport_role == "assistant"
    assert turns[1].speaker == "Bob"
    assert "speaker Alice" in turns[0].retrieval_text


def test_bad_or_ungrounded_extraction_falls_back_losslessly() -> None:
    item = case()
    turns = build_turn_nodes(item)
    claims, events, episodes, error = parse_session_extraction(
        '{"claims":[["Alice","bought","camera","event","positive","asserted",'
        '"complete","purchase",null,["missing"]]],"events":[]}',
        question_id="q",
        session_id="s",
        session_date="2026-01-02",
        turns=turns,
    )
    assert error == "empty_claims"
    assert not events and not episodes
    assert {source for claim in claims for source in claim.source_turn_ids} == {
        turn.node_id for turn in turns
    }


def test_truncated_json_salvages_complete_claim_array() -> None:
    item = case()
    turns = build_turn_nodes(item)
    text = json.dumps({
        "claims": [[
            "Alice", "bought", "a camera", "event", "positive", "asserted",
            "complete", "purchase", "2026-01-01", [turns[0].node_id],
            1, "item", 0.9,
        ]],
    })[:-1] + ",\"events\":["
    claims, events, episodes, error = parse_session_extraction(
        text, question_id="q", session_id="s", session_date="2026-01-02",
        turns=turns,
    )
    assert error == "partial_json_salvaged"
    assert len(claims) == 1
    assert not events and not episodes


def test_hypergraph_is_grounded_and_has_no_global_time_chain() -> None:
    _item, index = parsed_index()
    assert validate_hypergraph(index) == []
    assert any(edge.relation == "episode_member" for edge in index.hyperedges)
    assert any(edge.relation == "supports" for edge in index.hyperedges)
    temporal = [edge for edge in index.hyperedges if edge.relation == "temporal_scope"]
    assert all(len({inc.node_id for inc in edge.incidences}) >= 2 for edge in temporal)
    assert not any(edge.relation == "temporal_neighbor" for edge in index.hyperedges)


def test_query_frame_is_topic_invariant() -> None:
    first = build_query_frame("How many cameras did Alice buy?")
    second = build_query_frame("How many orchids did Priya catalog?")
    assert first.requested_operation == second.requested_operation == "count"
    assert first.answer_form == second.answer_form == "number"
    assert first.hypotheses == second.hypotheses


def test_recommendation_intent_is_topic_invariant() -> None:
    first = build_query_frame("Can you recommend resources to learn video editing?")
    second = build_query_frame("Where can I learn advanced orchid propagation?")
    assert first.requested_operation == second.requested_operation == "recommendation"
    assert first.answer_form == second.answer_form == "recommendation"
    assert first.hypotheses == second.hypotheses


def test_temporal_ledger_orders_unseen_values_without_topic_rules() -> None:
    frame = build_query_frame("What is my fastest kiln cycle time?")
    old = TurnNode(
        "old", "q", "s1", "2026-01-01", 0, "A", "a", "B", "user",
        "My fastest kiln cycle time is 31:20.",
        "speaker A listener B My fastest kiln cycle time is 31:20.",
    )
    new = TurnNode(
        "new", "q", "s2", "2026-01-08", 0, "A", "a", "B", "user",
        "I am hoping to beat my fastest kiln cycle time of 28:45.",
        "speaker A listener B I am hoping to beat my fastest kiln cycle time of 28:45.",
    )
    ledger = _temporal_evidence_ledger(
        frame,
        [("turn", old, 0.4, "protected_direct"), ("turn", new, 0.4, "protected_direct")],
    )
    assert [row["observed_at"] for row in ledger[:2]] == ["2026-01-08", "2026-01-01"]
    assert ledger[0]["exact_value_spans"] == ["28:45"]
    assert ledger[1]["exact_value_spans"] == ["31:20"]


def test_query_frame_handles_cross_benchmark_language_forms() -> None:
    assert build_query_frame("Where has Priya camped?").requested_operation == "location"
    assert build_query_frame("What do Priya's children like?").requested_operation == "preference_list"
    assert build_query_frame(
        "Would Priya still volunteer if she had not received support?"
    ).requested_operation == "counterfactual"
    planned = build_query_frame("When is Priya planning on going hiking?")
    assert planned.requested_operation == "planned_date"
    assert "priya" in planned.content_terms


def test_planned_date_hint_falls_back_to_lossless_turn() -> None:
    frame = build_query_frame("When is Priya planning on going hiking?")
    turn = TurnNode(
        "t", "q", "s", "2026-04-12", 0, "Priya", "priya", "Sam", "user",
        "We are thinking about going hiking next month.",
        "speaker Priya listener Sam We are thinking about going hiking next month.",
    )
    hint = _planned_event_hint(frame, [("turn", turn, 0.8, "protected_direct")])
    assert hint is not None
    assert hint["event_time"] == "next month"
    assert hint["anchor_date"] == "2026-04-12"
    assert hint["source_turn_ids"] == ["t"]


def test_retrieval_uses_multilevel_channels_and_bidirectional_hyperedges() -> None:
    item, index = parsed_index()
    embedder = MockEmbeddingClient()
    query_vector = embedder.embed(
        [item.question],
        question_id=item.question_id,
        variant="hierarchical_hypergraph_v3",
    )[0]
    result = retrieve(
        case=item,
        variant="hierarchical_hypergraph_v3",
        index=index,
        query_vector=query_vector,
        token_budget=1600,
    )
    trace = result.retrieval_trace
    assert set(trace["channels"]) == {"dense", "bm25", "exact"}
    assert trace["visited_hyperedge_ids"]
    assert trace["expansion_steps"]
    assert result.fact_node_ids
    assert result.evidence_leaf_ids
    assert "[EPISODE" in result.context_text or "[THEME" in result.context_text


def test_collection_operator_requires_local_closure() -> None:
    item, index = parsed_index("How many items did Alice buy?")
    claim = index.claims[0]
    claim.kind = "quantity"
    claim.quantity = 1
    collection = HyperEdge(
        edge_id="q:hyperedge:collection",
        question_id="q",
        relation="quantity_collection",
        incidences=[
            HyperIncidence(claim.node_id, "operand", 0),
            HyperIncidence(index.episodes[0].node_id, "scope"),
        ],
        retrieval_text="alice bought purchase",
    )
    index.hyperedges.append(collection)
    vector = MockEmbeddingClient().embed(
        [item.question], question_id="q", variant="hierarchical_hypergraph_v3"
    )[0]
    result = retrieve(
        case=item,
        variant="hierarchical_hypergraph_v3",
        index=index,
        query_vector=vector,
        token_budget=1800,
    )
    certificate = result.retrieval_trace["closure_certificate"]
    operator = result.retrieval_trace["operator_result"]
    assert certificate["scope_description"].startswith("local hypergraph")
    if certificate["complete"]:
        assert operator is not None
        assert set(certificate["operand_node_ids"]) <= set(result.fact_node_ids)


def test_mock_pipeline_runs_v3_with_separate_token_budgets(tmp_path: Path) -> None:
    item = case()
    data = tmp_path / "one.json"
    data.write_text(
        json.dumps(
            [
                {
                    "question_id": item.question_id,
                    "question_type": item.question_type,
                    "question": item.question,
                    "answer": item.answer,
                    "question_date": item.question_date,
                    "haystack_sessions": item.haystack_sessions,
                    "haystack_session_ids": item.haystack_session_ids,
                    "haystack_dates": item.haystack_dates,
                    "answer_session_ids": item.answer_session_ids,
                }
            ]
        )
    )
    config = DemoConfig(
        data_path=data,
        output_dir=tmp_path / "out",
        variants=("hierarchical_hypergraph_v3",),
        question_type="all",
        max_questions=1,
        mock_services=True,
        qa_max_tokens=512,
        build_budget_tokens=300_000,
        answer_budget_tokens=10_000,
    )
    run = run_case(
        config,
        item,
        "hierarchical_hypergraph_v3",
        MockDeepSeekClient(),
        MockEmbeddingClient(),
        MockCompressor(1.0),
        InflightLimiter(4),
    )
    assert run.v3_index is not None
    assert run.stats.build_budget_pass and run.stats.answer_budget_pass
    assert run.stats.build_cache_miss_input_tokens > 0
    assert run.stats.answer_cache_miss_input_tokens > 0
    assert all(record.reasoning_tokens == 0 for record in run.llm_records)


def test_v3_retrieval_only_skips_base_answer_call(tmp_path: Path) -> None:
    item = case()
    config = DemoConfig(
        data_path=tmp_path / "unused.json",
        output_dir=tmp_path / "out",
        variants=("hierarchical_hypergraph_v3",),
        question_type="all",
        max_questions=1,
        mock_services=True,
        retrieval_only=True,
        qa_max_tokens=512,
        build_budget_tokens=300_000,
        answer_budget_tokens=10_000,
    )
    run = run_case(
        config,
        item,
        "hierarchical_hypergraph_v3",
        MockDeepSeekClient(),
        MockEmbeddingClient(),
        MockCompressor(1.0),
        InflightLimiter(4),
    )
    assert run.answer == ""
    assert run.retrieval.retrieval_trace["answer_mode"] == "retrieval_only"
    assert run.stats.answer_total_tokens == 0
    assert not any(record.stage == "answer_qa" for record in run.llm_records)


def test_v3_core_contains_no_benchmark_or_category_switches() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "graphmem_demo" / "v3"
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(root.glob("*.py"))
    ).casefold()
    forbidden = ("longmemeval", "locomo", "category_1", "category_2", "question_id ==")
    assert not any(value in text for value in forbidden)
    # The core may use generic linguistic operators, but must not carry a large
    # question/topic dispatch table like the V2 compatibility module.
    assert "_allowed" not in text
    assert "topic_rules" not in text
