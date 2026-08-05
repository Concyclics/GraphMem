from __future__ import annotations

from graphmem_demo.v3.catalog_arithmetic import arithmetic_hint
from graphmem_demo.v3.catalog_schema import OperandRecordV3
from graphmem_demo.v3.retrieval import build_query_frame


def _operand(
    operand_id: str,
    predicate: str,
    obj: str,
    quantity: float | None,
    session: str,
    day: str,
) -> OperandRecordV3:
    return OperandRecordV3(
        operand_id=operand_id,
        question_id="q",
        subject_key="participant 1",
        predicate_key=predicate,
        object_key=obj.casefold(),
        object_text=obj,
        quantity=quantity,
        unit="courses" if quantity is not None else "",
        observed_at=day,
        session_ids=[session],
        source_turn_ids=[f"{session}:{operand_id}"],
        retrieval_text=f"participant 1 | {predicate} | {obj}",
    )


def test_total_sums_latest_snapshots_across_distinct_contexts() -> None:
    frame = build_query_frame(
        "What is the total number of online courses I've completed?"
    )
    operands = [
        _operand("o1", "completed courses", "edX", 8, "s1", "2023-05-21"),
        _operand("o2", "completed courses", "edX", 8, "s1", "2023-05-21"),
        _operand("o3", "completed courses", "Coursera", None, "s2", "2023-05-27"),
        _operand("o4", "completed courses count", "12", 12, "s2", "2023-05-27"),
        _operand("o5", "has", "specialization with 9 courses", 9, "s3", "2023-05-24"),
    ]
    result = arithmetic_hint(
        frame,
        operands,
        [],
        lambda query, text: sum(
            term in text.casefold() for term in query.content_terms
        ) / max(1, len(query.content_terms)),
    )
    assert result is not None
    assert result["operation"] == "partitioned_scalar_total"
    assert result["value"] == 20
    assert {item["context"] for item in result["parts"]} == {"edx", "coursera"}
