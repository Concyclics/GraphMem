from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_locomo_category5_abstain_policy_only_changes_category5(tmp_path: Path) -> None:
    data = [
        {
            "question_id": "q1",
            "question_type": "category_5",
            "question": "What did Caroline realize?",
            "answer": "",
        },
        {
            "question_id": "q2",
            "question_type": "category_1",
            "question": "What did Caroline research?",
            "answer": "Adoption agencies",
        },
    ]
    answers = [
        {"question_id": "q1", "question_type": "category_5", "prediction": "Final answer: invented"},
        {"question_id": "q2", "question_type": "category_1", "prediction": "Final answer: Adoption agencies"},
    ]
    data_path = tmp_path / "data.json"
    answers_path = tmp_path / "answers.jsonl"
    out_dir = tmp_path / "out"
    data_path.write_text(json.dumps(data), encoding="utf-8")
    answers_path.write_text(
        "".join(json.dumps(row) + "\n" for row in answers),
        encoding="utf-8",
    )

    subprocess.run(
        [
            "python",
            str(ROOT / "scripts" / "apply_locomo_policy.py"),
            "--data",
            str(data_path),
            "--answers",
            str(answers_path),
            "--output-dir",
            str(out_dir),
            "--variant",
            "probe",
            "--category5-abstain",
        ],
        check=True,
    )

    rows = [json.loads(line) for line in (out_dir / "answers.jsonl").read_text().splitlines()]

    assert rows[0]["locomo_policy_applied"] is True
    assert "not mentioned" in rows[0]["prediction"]
    assert rows[1]["locomo_policy_applied"] is False
    assert rows[1]["prediction"] == "Final answer: Adoption agencies"
