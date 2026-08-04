from __future__ import annotations

from pathlib import Path

from graphmem.eval import calibration40, load_dev_questions, load_gold_turns
from graphmem.eval.devset import parse_locomo_evidence


ROOT = Path("/ssd3/chenhan/Spark_MemGraph_Dev")
DEV = ROOT / "artifacts/development_sets/hard_lme_multisession_temporal_locomo_cat1_cat2_200_20260804"


def test_locomo_evidence_parser_is_exact_and_deduplicated() -> None:
    rows = parse_locomo_evidence(["D4:8; D6:6", "D4:8"])
    assert [(row.session_id, row.turn_index) for row in rows] == [
        ("session_4", 7), ("session_6", 5)
    ]


def test_real_devset_has_four_equal_strata_and_stable_calibration() -> None:
    questions = load_dev_questions(
        DEV / "longmemeval_hard_multisession50_temporal50.json",
        DEV / "locomo_hard_cat1_multihop50_cat2_temporal50.json",
        load_gold_turns(Path("eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl")),
    )
    counts = {name: sum(row.stratum == name for row in questions) for name in {
        row.stratum for row in questions
    }}
    assert counts == {
        "lme_multi_session": 50, "lme_temporal": 50,
        "locomo_multihop": 50, "locomo_temporal": 50,
    }
    first = calibration40(questions)
    second = calibration40(list(reversed(questions)))
    assert [row.question_id for row in first] == [row.question_id for row in second]
    assert len(first) == len({row.question_id for row in first}) == 40
