from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from graphmem.eval import load_gold_turns


def test_gold_annotation_requires_second_review_for_non_high_confidence(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    path.write_text(json.dumps({
        "question_id": "q", "session_id": "s", "turn_index": 0,
        "span_start": 0, "span_end": 1, "support_role": "fact",
        "confidence": "low", "first_review": "accepted",
        "second_review": "not_required", "adjudication": "accepted",
    }) + "\n")
    with pytest.raises(ValueError, match="second review"):
        load_gold_turns(path)


def test_gold_annotations_are_offline_only() -> None:
    online_roots = [
        Path("src/graphmem/build"), Path("src/graphmem/retrieval"),
        Path("src/graphmem/runtime"),
    ]
    for root in online_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert "graphmem.eval" not in text
            assert "gold_turn" not in text


def test_committed_lme_annotations_are_git_safe_and_match_manifest() -> None:
    root = Path(__file__).resolve().parents[1] / "eval_annotations"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    path = root / manifest["annotation_file"]
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == manifest["annotation_sha256"]
    rows = [json.loads(line) for line in payload.decode().splitlines() if line]
    assert len({row["question_id"] for row in rows}) == manifest["questions"] == 100
    assert len(rows) == manifest["annotations"] == 217
    forbidden = {"question", "answer", "question_type", "text", "review_excerpt"}
    assert all(not forbidden.intersection(row) for row in rows)
