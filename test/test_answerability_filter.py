from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "answerability_filter_from_retrieval",
    ROOT / "scripts" / "answerability_filter_from_retrieval.py",
)
assert SPEC is not None
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

POLICY_SPEC = importlib.util.spec_from_file_location(
    "apply_answerability_policy",
    ROOT / "scripts" / "apply_answerability_policy.py",
)
assert POLICY_SPEC is not None
policy_module = importlib.util.module_from_spec(POLICY_SPEC)
assert POLICY_SPEC.loader is not None
sys.modules[POLICY_SPEC.name] = policy_module
POLICY_SPEC.loader.exec_module(policy_module)


def test_parse_payload_extracts_json_object() -> None:
    payload = module.parse_payload(
        'prefix {"decision":"abstain","final_answer":"Insufficient.","reason":"wrong speaker"}'
    )

    assert payload == {
        "decision": "abstain",
        "final_answer": "Insufficient.",
        "reason": "wrong speaker",
    }


def test_render_prediction_keeps_original_when_decision_keep() -> None:
    assert module.render_prediction("original answer", {"decision": "keep"}) == "original answer"


def test_render_prediction_abstain_only_ignores_revisions() -> None:
    rendered = module.render_prediction(
        "original answer",
        {
            "decision": "revise",
            "final_answer": "new answer",
            "reason": "better evidence",
        },
        policy="abstain_only",
    )

    assert rendered == "original answer"


def test_render_prediction_abstains_with_default_final_answer() -> None:
    rendered = module.render_prediction(
        "original answer",
        {"decision": "abstain", "reason": "Evidence belongs to another speaker."},
    )

    assert "Evidence belongs to another speaker." in rendered
    assert "Final answer:" in rendered
    assert "insufficient" in rendered.lower()


def test_evidence_owner_diagnostics_separates_other_speaker_photo_evidence() -> None:
    diagnostics = module.evidence_owner_diagnostics(
        "Did Caroline make the black and white bowl in the photo?",
        "\n".join(
            [
                "[Session conv_session_1 | 2023/07/03 | turn 6]",
                "User: That bowl is gorgeous! The black and white design looks so fancy. Did you make it?",
                "Assistant: [Melanie shared an image: a photo of a black and white bowl] Thanks, Caroline! Yeah, I made this bowl in my class.",
            ]
        ),
    )

    assert "Target person from question: Caroline" in diagnostics
    assert "Target-owned candidate evidence: none found" in diagnostics
    assert "Other-speaker candidate evidence:" in diagnostics
    assert "Melanie:" in diagnostics
    assert "Photo/image candidate evidence:" in diagnostics


def test_classifier_messages_include_owner_diagnostics() -> None:
    messages = module.classifier_messages(
        question="Is Oscar Melanie's pet?",
        question_type="category_5",
        question_date="2023/10/22",
        prediction="Final answer: Yes.",
        evidence="User: Caroline has a guinea pig named Oscar.",
        diagnostics="Evidence-owner diagnostics (heuristic, non-authoritative):\n- Target person from question: Oscar",
    )

    user_message = messages[1]["content"]
    assert "Retrieved memory evidence:" in user_message
    assert "Evidence-owner diagnostics" in user_message
    assert "Target person from question: Oscar" in user_message
    assert "directly supports the asked person's own item" in messages[0]["content"]
    assert "false premise" in messages[0]["content"]


def test_evidence_owner_diagnostics_marks_target_response_about_other_person() -> None:
    diagnostics = module.evidence_owner_diagnostics(
        "What is Melanie excited about in her adoption process?",
        "Assistant: [Melanie shared an image: a photo of family figurines] "
        "Congrats, Caroline! Adoption sounds awesome. I'm so happy for you.",
    )

    assert "Target person from question: Melanie" in diagnostics
    assert "Target-owned candidate evidence: none found" in diagnostics
    assert "Target-speaker response about another person/object:" in diagnostics


def test_answerability_policy_keeps_skipped_rows_under_all_policy(
    tmp_path: Path, monkeypatch
) -> None:
    original = tmp_path / "original.jsonl"
    classified = tmp_path / "classified.jsonl"
    output_dir = tmp_path / "out"
    original.write_text(
        json.dumps({"question_id": "q1", "prediction": "original answer"}) + "\n",
        encoding="utf-8",
    )
    classified.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "prediction": "classifier-side text",
                "answerability_decision": "skipped",
                "answerability_reason": "question_type_not_selected",
                "answerability_final_answer": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "apply_answerability_policy.py",
            "--original-answers",
            str(original),
            "--answerability-answers",
            str(classified),
            "--output-dir",
            str(output_dir),
            "--decision-policy",
            "all",
        ],
    )

    policy_module.main()

    row = json.loads((output_dir / "answers.jsonl").read_text(encoding="utf-8"))
    assert row["prediction"] == "original answer"
    assert row["answerability_effective_decision"] == "keep_original"
