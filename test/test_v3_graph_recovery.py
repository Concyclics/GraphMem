import json

from graphmem_demo.clients import rough_token_count
from graphmem_demo.v3.graph_recovery import PersistedGraphStore
from graphmem_demo.v3.llm_navigation import (
    NavigationPlan,
    recovered_evidence_text,
)


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_graph_recovery_aliases_shared_index_and_fills_missing_slot(tmp_path) -> None:
    owner = "dialogue02_0111"
    question_id = "dialogue02_0049"
    _write_jsonl(
        tmp_path / "nodes.jsonl",
        [
            {
                "node_id": f"{owner}:session_2:turn:0",
                "question_id": owner,
                "node_type": "turn",
                "text": "I delivered the fired pieces to the workshop.",
                "retrieval_text": "speaker Ada | delivered fired pieces to workshop",
            },
            {
                "node_id": f"{owner}:session_2:claim:0",
                "question_id": owner,
                "node_type": "claim",
                "retrieval_text": "Ada | delivered | fired pieces | workshop",
                "source_turn_ids": [f"{owner}:session_2:turn:0"],
            },
            {
                "node_id": f"{owner}:session_1:turn:4",
                "question_id": owner,
                "node_type": "turn",
                "text": "The fired pieces were ceramic tiles made in the solar kiln.",
                "retrieval_text": "speaker Ada | fired pieces were ceramic tiles in solar kiln",
            },
        ],
    )
    _write_jsonl(
        tmp_path / "hyperedges.jsonl",
        [
            {
                "edge_id": f"{owner}:edge:0",
                "question_id": owner,
                "relation": "supports",
                "confidence": 0.95,
                "incidences": [
                    {"node_id": f"{owner}:session_2:turn:0", "role": "source"},
                    {"node_id": f"{owner}:session_2:claim:0", "role": "claim"},
                ],
            }
        ],
    )
    store = PersistedGraphStore(tmp_path)
    result = store.recover(
        question_id=question_id,
        question="What material were the fired pieces made from?",
        selected_ids=[f"{question_id}:session_2:turn:0"],
        missing_slots=["exact material used for the fired pieces"],
        needed_relations=["supports", "source"],
    )
    ids = {row["node_id"] for row in result.rows}
    assert f"{question_id}:session_2:claim:0" in ids
    assert f"{question_id}:session_1:turn:4" in ids
    assert all(node_id.startswith(f"{question_id}:") for node_id in ids)
    assert result.graph_rows >= 1
    assert result.lexical_rows >= 1


def test_recovered_evidence_preserves_selected_recovery_and_fallback() -> None:
    ledger = [
        {
            "node_id": "selected",
            "node_type": "turn",
            "selection_source": "focused_provenance_expansion",
            "score": 0.9,
            "text": "The shipment reached the depot.",
        },
        {
            "node_id": "fallback",
            "node_type": "claim",
            "selection_source": "protected_catalog",
            "score": 0.8,
            "text": "The shipment left on Monday.",
        },
    ]
    recovery = [
        {
            "node_id": "recovered",
            "node_type": "turn",
            "selection_source": "navigator_graph_recovery",
            "score": 1.0,
            "text": "The shipment arrived on Thursday.",
            "relation_path": ["after"],
        }
    ]
    text, ids = recovered_evidence_text(
        question="How long did the shipment take?",
        evidence_ledger=ledger,
        plan=NavigationPlan(("selected",), "duration", (), ("after",)),
        recovery_rows=recovery,
        max_rough_tokens=500,
    )
    assert ids[:2] == ["selected", "recovered"]
    assert "fallback" in ids
    assert "arrived on Thursday" in text
    assert rough_token_count(text) <= 500


def test_graph_recovery_blocks_unrequested_cross_theme_bridge(tmp_path) -> None:
    question_id = "dialogue03_0001"
    nodes = [
        {
            "node_id": f"{question_id}:session_1:turn:0",
            "question_id": question_id,
            "node_type": "turn",
            "retrieval_text": "speaker Nate | career setback in September 2022",
        },
        {
            "node_id": f"{question_id}:episode:1",
            "question_id": question_id,
            "node_type": "episode",
            "retrieval_text": "Nate career setback | September 2022",
        },
        {
            "node_id": f"{question_id}:theme:1",
            "question_id": question_id,
            "node_type": "theme",
            "retrieval_text": "career updates",
        },
        {
            "node_id": f"{question_id}:episode:2",
            "question_id": question_id,
            "node_type": "episode",
            "retrieval_text": "Nate career success | September 2023",
        },
    ]
    _write_jsonl(tmp_path / "nodes.jsonl", nodes)
    _write_jsonl(
        tmp_path / "hyperedges.jsonl",
        [
            {
                "question_id": question_id,
                "relation": "episode_member",
                "confidence": 0.9,
                "incidences": [
                    {"node_id": nodes[0]["node_id"]},
                    {"node_id": nodes[1]["node_id"]},
                ],
            },
            {
                "question_id": question_id,
                "relation": "theme_member",
                "confidence": 0.9,
                "incidences": [
                    {"node_id": nodes[1]["node_id"]},
                    {"node_id": nodes[2]["node_id"]},
                    {"node_id": nodes[3]["node_id"]},
                ],
            },
        ],
    )
    store = PersistedGraphStore(tmp_path)
    result = store.recover(
        question_id=question_id,
        question="Was September 2022 good career-wise for Nate?",
        selected_ids=[nodes[0]["node_id"]],
        missing_slots=[],
        needed_relations=["episode_member", "source"],
        operation="compare status",
    )
    ids = {row["node_id"] for row in result.rows}
    assert nodes[1]["node_id"] in ids
    assert nodes[3]["node_id"] not in ids
    assert result.relation_filtered_edges >= 1


def test_graph_store_reports_empty_or_populated_assets(tmp_path) -> None:
    empty = PersistedGraphStore(tmp_path)
    assert empty.node_count == 0
    assert empty.edge_count == 0
    assert not empty.has_scope("dialogue01_0001")


def test_graph_store_keeps_rich_node_when_auxiliary_projection_repeats_it(tmp_path) -> None:
    question_id = "dialogue01_0001"
    node_id = f"{question_id}:operand:1"
    _write_jsonl(tmp_path / "nodes.jsonl", [{
        "node_id": node_id,
        "question_id": question_id,
        "node_type": "operand",
        "retrieval_text": "Ada | owns | a red bicycle",
        "source_turn_ids": [f"{question_id}:session_1:turn:0"],
    }])
    _write_jsonl(tmp_path / "operands.jsonl", [{
        "operand_id": node_id,
        "question_id": question_id,
        "retrieval_text": "Ada | owns | a red bicycle",
    }])
    store = PersistedGraphStore(tmp_path)
    assert store.nodes_for(question_id)["operand:1"]["node_type"] == "operand"
    assert store.nodes_for(question_id)["operand:1"]["source_turn_ids"]


def test_fallback_reserves_slots_for_session_diversity() -> None:
    ledger = []
    for index in range(4):
        ledger.append({
            "node_id": f"q:session_1:turn:{index}",
            "node_type": "turn",
            "session_id": "session_1",
            "score": 1.0 - index * 0.01,
            "text": f"Ada discussed the shipment detail {index}.",
        })
    ledger.extend([
        {
            "node_id": "q:session_2:turn:0", "node_type": "turn",
            "session_id": "session_2", "score": 0.2,
            "text": "Ada discussed the shipment at the depot.",
        },
        {
            "node_id": "q:session_3:turn:0", "node_type": "turn",
            "session_id": "session_3", "score": 0.1,
            "text": "Ada discussed the shipment on Thursday.",
        },
    ])
    _text, ids = recovered_evidence_text(
        question="What happened to Ada's shipment?",
        evidence_ledger=ledger,
        plan=NavigationPlan((), "lookup", (), ()),
        recovery_rows=[],
        fallback_rows=4,
        max_rough_tokens=500,
    )
    assert any(":session_2:" in node_id for node_id in ids)
    assert any(":session_3:" in node_id for node_id in ids)


def test_lossless_first_evidence_keeps_turns_and_only_selected_routes() -> None:
    ledger = [
        {"node_id": "turn", "node_type": "turn", "text": "Exact source reply."},
        {"node_id": "selected_claim", "node_type": "claim", "text": "Selected route."},
        {"node_id": "noise_claim", "node_type": "claim", "text": "Noisy extraction."},
    ]
    text, ids = recovered_evidence_text(
        question="What was the reply?",
        evidence_ledger=ledger,
        plan=NavigationPlan(("selected_claim",), "lookup", (), ()),
        recovery_rows=[ledger[0], ledger[2]],
        evidence_profile="lossless-first",
    )
    assert ids == ["turn", "selected_claim"]
    assert "Exact source reply" in text
    assert "Noisy extraction" not in text
