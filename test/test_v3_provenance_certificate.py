from graphmem_demo.v3.retrieval import _annotate_packed_provenance


def test_graph_completeness_survives_bounded_prompt_packing() -> None:
    hint = {
        "operation": "partitioned_scalar_total",
        "complete": True,
        "operand_ids": ["o1", "o2"],
        "source_turn_ids": ["t1", "t2"],
    }
    result = _annotate_packed_provenance(hint, {"o1", "o2", "t1"})
    assert result["complete"] is True
    assert result["packed_operand_coverage"] == 1.0
    assert result["packed_source_coverage"] == 0.5
    assert result["packed_provenance_complete"] is True


def test_packing_coverage_is_reported_without_claiming_prompt_completeness() -> None:
    hint = {
        "operation": "partitioned_scalar_total",
        "complete": True,
        "operand_ids": ["o1", "o2"],
        "source_turn_ids": ["t1", "t2"],
    }
    result = _annotate_packed_provenance(hint, {"o1", "t1"})
    assert result["complete"] is True
    assert result["packed_operand_coverage"] == 0.5
    assert result["packed_source_coverage"] == 0.5
    assert result["packed_provenance_complete"] is False
