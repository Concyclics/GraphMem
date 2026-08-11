import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "paired_judge_delta.py"
SPEC = importlib.util.spec_from_file_location("paired_judge_delta", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_changed_ids_can_pair_by_prompt_identity() -> None:
    baseline = [
        {"question_id": "same", "prediction": "old", "prompt_payload_hash": "h1"},
        {"question_id": "changed", "prediction": "old", "prompt_payload_hash": "h2"},
    ]
    candidate = [
        {"question_id": "same", "prediction": "new", "prompt_payload_hash": "h1"},
        {"question_id": "changed", "prediction": "old", "prompt_payload_hash": "h3"},
    ]

    assert MODULE.changed_ids(baseline, candidate) == ("same",)
    assert MODULE.changed_ids(
        baseline, candidate, identity_field="prompt_payload_hash") == (
            "same", "changed")


def test_merge_carries_verdict_only_for_identical_prompt_and_prediction(
        tmp_path: Path) -> None:
    def write(name: str, rows: list[dict]) -> Path:
        path = tmp_path / name
        MODULE.write_rows(path, rows)
        return path

    baseline_answers = write("baseline_answers.jsonl", [
        {"question_id": "same", "prediction": "a", "prompt_payload_hash": "h1"},
        {"question_id": "changed", "prediction": "a", "prompt_payload_hash": "h2"},
    ])
    candidate_answers = write("candidate_answers.jsonl", [
        {"question_id": "same", "prediction": "a", "prompt_payload_hash": "h1"},
        {"question_id": "changed", "prediction": "b", "prompt_payload_hash": "h2"},
    ])
    baseline_judge = write("baseline_judge.jsonl", [
        {"question_id": "same", "correct": True},
        {"question_id": "changed", "correct": False},
    ])
    delta_judge = write("delta_judge.jsonl", [
        {"question_id": "changed", "correct": True},
    ])
    output = tmp_path / "merged.jsonl"
    result = MODULE.merge(
        baseline_answers, candidate_answers, baseline_judge, delta_judge,
        output, tmp_path / "manifest.json", identity_field="prompt_payload_hash")

    assert result["correct"] == 2
    assert result["changed_fresh"] == 1
    rows = MODULE.read_rows(output)
    assert rows[0]["paired_verdict_source"] == (
        "carried_identical_prompt_payload_hash")
    assert rows[1]["paired_verdict_source"] == (
        "fresh_changed_prompt_payload_hash")


def test_merge_rejects_declared_prediction_hash_mismatch(tmp_path: Path) -> None:
    def write(name: str, rows: list[dict]) -> Path:
        path = tmp_path / name
        MODULE.write_rows(path, rows)
        return path

    answer = {"question_id": "q", "prediction": "current"}
    answers = write("answers.jsonl", [answer])
    judge = write("judge.jsonl", [{
        "question_id": "q", "correct": True,
        "prediction_sha256": MODULE.prediction_sha256({
            "prediction": "different"}),
    }])
    try:
        MODULE.merge(
            answers, answers, judge, write("empty.jsonl", []),
            tmp_path / "output.jsonl", tmp_path / "manifest.json")
    except ValueError as error:
        assert "prediction hash mismatch" in str(error)
    else:
        raise AssertionError("mismatched prediction digest was accepted")
