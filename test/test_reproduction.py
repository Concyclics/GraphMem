"""Prevent score/usage claims from silently accepting partial or stale artifacts."""
import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "reproduction", Path(__file__).resolve().parents[1] / "scripts/reproduce_locomo.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def fixture_run(tmp_path):
    write(tmp_path / "answers.jsonl", [{"question_id": "q", "prediction": "Luna",
          "cached": True, "api_prompt_tokens": 1000, "completion_tokens": 2,
          "api_total_tokens": 1002, "finish_reason": "stop"}])
    write(tmp_path / "judge.jsonl", [{"question_id": "q", "category": 1, "correct": True,
          "prediction_sha256": hashlib.sha256(b"Luna").hexdigest()}])
    write(tmp_path / "api_calls.jsonl", [{"phase": "judge", "kind": "chat", "error": None,
          "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}}])


def test_actual_ledger_does_not_charge_a_cached_answer_again(tmp_path):
    fixture_run(tmp_path)
    result = RUNNER.report(tmp_path, {"q"})
    assert result["checks_pass"]
    assert result["cached_answers"] == 1
    assert "answer/chat" not in result["actual_api_usage"]
    assert result["answer_tokens_per_question"]["api_total_tokens"]["sum"] == 1002


def test_missing_question_fails_even_when_existing_answers_are_all_correct(tmp_path):
    fixture_run(tmp_path)
    result = RUNNER.report(tmp_path, {"q", "missing"})
    assert not result["checks_pass"]
    assert result["accuracy"] == 0.5


def test_judge_of_an_old_answer_cannot_pass_alignment(tmp_path):
    fixture_run(tmp_path)
    rows = RUNNER.read_rows(tmp_path / "answers.jsonl")
    rows[0]["prediction"] = "Oliver"
    write(tmp_path / "answers.jsonl", rows)
    assert not RUNNER.report(tmp_path, {"q"})["checks_pass"]


@pytest.mark.parametrize("usage,finish", [(None, "stop"), ({"total_tokens": 60}, "length")])
def test_missing_usage_or_truncated_answer_fails_validation(tmp_path, usage, finish):
    fixture_run(tmp_path)
    rows = RUNNER.read_rows(tmp_path / "answers.jsonl")
    rows[0]["finish_reason"] = finish
    write(tmp_path / "answers.jsonl", rows)
    calls = RUNNER.read_rows(tmp_path / "api_calls.jsonl")
    calls[0]["usage"] = usage
    write(tmp_path / "api_calls.jsonl", calls)
    assert not RUNNER.report(tmp_path, {"q"})["checks_pass"]


def test_resume_rejects_duplicate_checkpoint_rows(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    write(path, [{"question_id": "q"}, {"question_id": "q"}])
    with pytest.raises(ValueError, match="duplicate"):
        RUNNER.checkpoint([("q", None)], "question_id", path, lambda _: {}, 1)


def test_timed_out_request_is_not_treated_as_a_zero_cost_call(tmp_path):
    fixture_run(tmp_path)
    RUNNER.append(tmp_path / "api_calls.jsonl", {
        "phase": "build", "kind": "chat", "error": "timeout before usage", "usage": None})
    result = RUNNER.report(tmp_path, {"q"})
    assert not result["api_token_accounting_complete"]
    assert not result["checks_pass"]


def test_graph_snapshot_keeps_build_cache_but_never_imports_answers(tmp_path):
    source, target = tmp_path / "source.sqlite", tmp_path / "target.sqlite"
    with sqlite3.connect(source) as db:
        db.execute("CREATE TABLE llm_cache(stage TEXT, value TEXT)")
        db.execute("CREATE TABLE llm_calls(stage TEXT)")
        db.executemany("INSERT INTO llm_cache VALUES (?,?)", [
            ("scene_semantic", "reusable graph"), ("answer", "must regenerate")])
        db.executemany("INSERT INTO llm_calls VALUES (?)", [("scene_semantic",), ("answer",)])
    RUNNER.copy_graph(source, target)
    with sqlite3.connect(target) as db:
        assert db.execute("SELECT * FROM llm_cache").fetchall() == [("scene_semantic", "reusable graph")]
        assert db.execute("SELECT * FROM llm_calls").fetchall() == [("scene_semantic",)]
    with sqlite3.connect(source) as db:
        assert db.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0] == 2
