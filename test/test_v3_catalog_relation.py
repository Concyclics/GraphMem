from graphmem_demo.v3.catalog_relation import latest_relation_hint
from graphmem_demo.v3.catalog_schema import OperandRecordV3
from graphmem_demo.v3.retrieval import build_query_frame


def _overlap(frame, text):
    return len(set(frame.content_terms) & set(text.casefold().split()))


def _item(index, predicate, value, day, *, modality="asserted"):
    return OperandRecordV3(
        f"q:operand:{index}", "q", "owner", predicate, value, value,
        polarity="positive", modality=modality, event_time=day,
        observed_at=day, source_claim_ids=[f"c{index}"],
        source_turn_ids=[f"t{index}"], retrieval_text=f"{predicate} {value}",
    )


def test_latest_relation_prefers_newer_storage_intent_over_old_location() -> None:
    frame = build_query_frame("Where is the current kiln manual kept?")
    hint = latest_relation_hint(frame, [
        _item(0, "has stored", "kiln manual under desk", "2026-01-01"),
        _item(1, "wants to store", "kiln manual in wall cabinet", "2026-01-08", modality="planned"),
        _item(2, "wants to discard", "kiln manual", "2026-01-09", modality="planned"),
    ], _overlap)
    assert hint is not None
    assert "wall cabinet" in hint["value"]


def test_latest_initiation_prefers_recent_trial_over_unrelated_capacity() -> None:
    frame = build_query_frame("Which tool service did I start using most recently?")
    hint = latest_relation_hint(frame, [
        _item(0, "allows", "two simultaneous tool streams", "2026-02-10"),
        _item(1, "had free trial", "Delta service last month", "2026-02-01"),
        _item(2, "has used", "Gamma service for six months", "2026-02-01"),
    ], _overlap)
    assert hint is not None
    assert "Delta service" in hint["value"]
