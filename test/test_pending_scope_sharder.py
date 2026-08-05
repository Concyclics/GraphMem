from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "shard_pending_by_scope.py"
SPEC = importlib.util.spec_from_file_location("pending_scope_sharder", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_pending_scope_shards_are_disjoint_and_scope_local(tmp_path: Path) -> None:
    rows = [
        {"question_id": f"q{index}", "scope": "a" if index <= 5 else "b"}
        for index in range(1, 9)
    ]
    shards, manifest = MODULE.shard_pending(
        rows, {"q1", "q6"}, scope_field="scope", shards_per_scope=2,
    )

    flattened = [row["question_id"] for shard in shards for row in shard]
    assert len(flattened) == len(set(flattened)) == 6
    assert set(flattened) == {"q2", "q3", "q4", "q5", "q7", "q8"}
    assert all(len({row["scope"] for row in shard}) == 1 for shard in shards)
    assert manifest["pending_questions"] == 6
    assert manifest["shard_count"] == 4


def test_completed_question_ids_reads_only_selected_variant(tmp_path: Path) -> None:
    selected = tmp_path / "shard_0" / "chosen"
    selected.mkdir(parents=True)
    (selected / "question_stats.jsonl").write_text(
        json.dumps({"question_id": "q1"}) + "\n",
        encoding="utf-8",
    )
    other = tmp_path / "shard_1" / "other"
    other.mkdir(parents=True)
    (other / "question_stats.jsonl").write_text(
        json.dumps({"question_id": "q2"}) + "\n",
        encoding="utf-8",
    )

