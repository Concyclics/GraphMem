from __future__ import annotations

import json
import sys

import pytest

from scripts.select_benchmark_subset import main


def test_manifest_proves_remaining_unseen_count(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.json"
    excluded = tmp_path / "excluded.json"
    output = tmp_path / "batch.json"
    source.write_text(
        json.dumps([{"question_id": f"q{index}"} for index in range(5)]),
        encoding="utf-8",
    )
    excluded.write_text(json.dumps([{"question_id": "q0"}]), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_benchmark_subset.py",
            "--data",
            str(source),
            "--output",
            str(output),
            "--exclude-data",
            str(excluded),
            "--limit",
            "2",
            "--seed",
            "7",
        ],
    )
    main()
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["source_question_count"] == 5
    assert manifest["excluded_source_question_count"] == 1
    assert manifest["eligible_question_count"] == 4
    assert manifest["question_count"] == 2
    assert manifest["remaining_after_selection"] == 2
    assert manifest["exclude_sources"] == [str(excluded)]
    assert len(manifest["question_ids_sha256"]) == 64


def test_duplicate_source_question_ids_are_rejected(tmp_path, monkeypatch) -> None:
    source = tmp_path / "duplicates.json"
    output = tmp_path / "batch.json"
    source.write_text(
        json.dumps([{"question_id": "q0"}, {"question_id": "q0"}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_benchmark_subset.py",
            "--data",
            str(source),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="duplicate question_id"):
        main()
