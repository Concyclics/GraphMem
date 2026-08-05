#!/usr/bin/env python3
"""Apply the explicit semantic-review decisions for the V5 LongMemEval dev set.

The override table contains source references only, never dialogue text. Cases
without an override retain the question-level reduced candidates after exact-ref
deduplication. This script produces the decisions consumed by the finalizer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


Ref = tuple[str, int, str]

# question_id -> (session_id, user turn index, support role)
OVERRIDES: dict[str, list[Ref]] = {
    "09ba9854_abs": [("answer_96c743d0_abs_1", 6, "negative_scope")],
    "2788b940": [("answer_8f6b938d_1", 0, "aggregation_member"),
                 ("answer_8f6b938d_2", 0, "aggregation_member"),
                 ("answer_8f6b938d_3", 0, "aggregation_member"),
                 ("answer_8f6b938d_4", 0, "aggregation_member")],
    "28dc39ac": [("answer_8d015d9d_1", 0, "aggregation_member"),
                 ("answer_8d015d9d_2", 0, "aggregation_member"),
                 ("answer_8d015d9d_3", 6, "aggregation_member"),
                 ("answer_8d015d9d_4", 0, "aggregation_member"),
                 ("answer_8d015d9d_5", 0, "aggregation_member")],
    "2ebe6c90": [("answer_c9d35c00_1", 0, "temporal_endpoint"),
                 ("answer_c9d35c00_2", 0, "temporal_endpoint")],
    "370a8ff4": [("answer_61d1be50_1", 0, "temporal_endpoint"),
                 ("answer_61d1be50_2", 0, "temporal_endpoint")],
    "3fdac837": [("answer_419d21d5_1", 0, "aggregation_member"),
                 ("answer_419d21d5_2", 2, "aggregation_member")],
    "4f54b7c9": [("answer_50940cb7_1", 0, "aggregation_member"),
                 ("answer_50940cb7_2", 0, "aggregation_member")],
    "60bf93ed_abs": [("answer_e0956e0a_abs_1", 2, "negative_scope"),
                     ("answer_e0956e0a_abs_2", 0, "negative_scope")],
    "6e984301": [("answer_88841f26_1", 0, "temporal_endpoint"),
                 ("answer_88841f26_2", 2, "temporal_endpoint")],
    "80ec1f4f_abs": [("answer_990c8992_abs_1", 0, "negative_scope"),
                     ("answer_990c8992_abs_2", 0, "negative_scope")],
    "88432d0a": [("answer_733e443a_1", 0, "aggregation_member"),
                 ("answer_733e443a_2", 2, "aggregation_member"),
                 ("answer_733e443a_4", 0, "aggregation_member"),
                 ("answer_733e443a_4", 8, "aggregation_member")],
    "88432d0a_abs": [("answer_733e443a_abs_1", 0, "negative_scope")],
    "982b5123": [("answer_ab603dd5_1", 0, "temporal_endpoint"),
                 ("answer_ab603dd5_2", 0, "temporal_endpoint")],
    "982b5123_abs": [("answer_ab603dd5_abs_1", 0, "negative_scope")],
    "9aaed6a3": [("answer_353d3c6d_1", 0, "fact"),
                 ("answer_353d3c6d_2", 0, "fact")],
    "a3045048": [("answer_016f6bd4_1", 0, "temporal_endpoint"),
                 ("answer_016f6bd4_2", 4, "temporal_endpoint")],
    "a346bb18": [("answer_4934b2d7_1", 0, "fact"),
                 ("answer_4934b2d7_2", 0, "fact")],
    "a3838d2b": [("answer_4ffa04a2_1", 0, "temporal_endpoint"),
                 ("answer_4ffa04a2_2", 0, "aggregation_member"),
                 ("answer_4ffa04a2_4", 0, "aggregation_member"),
                 ("answer_4ffa04a2_5", 0, "aggregation_member"),
                 ("answer_4ffa04a2_6", 8, "aggregation_member")],
    "a4996e51": [("answer_feb5f98a_1", 0, "fact"),
                 ("answer_feb5f98a_2", 2, "fact")],
    "a9f6b44c": [("answer_cc021f81_1", 0, "aggregation_member"),
                 ("answer_cc021f81_2", 0, "aggregation_member")],
    "aae3761f": [("answer_526354c8_1", 0, "aggregation_member"),
                 ("answer_526354c8_2", 0, "aggregation_member"),
                 ("answer_526354c8_3", 0, "aggregation_member")],
    "b3c15d39": [("answer_05d808e6_1", 0, "temporal_endpoint"),
                 ("answer_05d808e6_2", 0, "temporal_endpoint")],
    "ba358f49_abs": [("answer_cbd08e3c_abs_1", 0, "negative_scope")],
    "bf659f65": [("answer_7726e7e9_1", 0, "aggregation_member"),
                 ("answer_7726e7e9_1", 10, "aggregation_member"),
                 ("answer_7726e7e9_3", 6, "aggregation_member")],
    "c8090214": [("answer_70dc7d08_1", 0, "temporal_endpoint"),
                 ("answer_70dc7d08_2", 0, "temporal_endpoint")],
    "c8090214_abs": [("answer_70dc7d08_abs_1", 0, "negative_scope"),
                     ("answer_70dc7d08_abs_2", 0, "negative_scope")],
    "cc6d1ec1": [("answer_be73098b_1", 0, "temporal_endpoint"),
                 ("answer_be73098b_2", 0, "temporal_endpoint")],
    "dd2973ad": [("answer_f9de4602_1", 0, "temporal_endpoint"),
                 ("answer_f9de4602_2", 0, "temporal_endpoint")],
    "e4e14d04": [("answer_cf425855_1", 0, "temporal_endpoint"),
                 ("answer_cf425855_2", 0, "temporal_endpoint")],
    "e831120c": [("answer_86c505e7_1", 0, "aggregation_member"),
                 ("answer_86c505e7_2", 0, "aggregation_member")],
    "edced276": [("answer_60e8941a_1", 2, "aggregation_member"),
                 ("answer_60e8941a_2", 0, "aggregation_member")],
    "edced276_abs": [("answer_60e8941a_abs_1", 2, "negative_scope")],
    "ef9cf60a": [("answer_87e3a1cb_1", 0, "aggregation_member"),
                 ("answer_87e3a1cb_2", 0, "aggregation_member")],
    "gpt4_15e38248": [("answer_8858d9dc_2", 0, "aggregation_member"),
                       ("answer_8858d9dc_3", 0, "aggregation_member"),
                       ("answer_8858d9dc_4", 2, "aggregation_member")],
    "gpt4_2c50253f": [("answer_9af4e346_1", 0, "fact"),
                       ("answer_9af4e346_2", 0, "fact")],
    "gpt4_2f8be40d": [("answer_e7b0637e_1", 2, "aggregation_member"),
                       ("answer_e7b0637e_2", 0, "aggregation_member"),
                       ("answer_e7b0637e_3", 0, "aggregation_member")],
    "gpt4_372c3eed_abs": [("answer_35c5419d_abs_1", 2, "negative_scope"),
                           ("answer_35c5419d_abs_2", 0, "negative_scope"),
                           ("answer_35c5419d_abs_3", 0, "negative_scope")],
    "gpt4_4fc4f797": [("answer_be07688f_1", 10, "temporal_endpoint"),
                       ("answer_be07688f_2", 0, "temporal_endpoint")],
    "gpt4_59c863d7": [("answer_593bdffd_1", 0, "aggregation_member"),
                       ("answer_593bdffd_2", 0, "aggregation_member"),
                       ("answer_593bdffd_3", 0, "aggregation_member"),
                       ("answer_593bdffd_4", 0, "aggregation_member")],
    "gpt4_7a0daae1": [("answer_4d5490f1_1", 6, "temporal_endpoint"),
                       ("answer_4d5490f1_2", 0, "temporal_endpoint")],
    "gpt4_7abb270c": [("answer_7093d898_1", 0, "temporal_endpoint"),
                       ("answer_7093d898_1", 6, "temporal_endpoint"),
                       ("answer_7093d898_3", 0, "temporal_endpoint"),
                       ("answer_7093d898_4", 0, "temporal_endpoint"),
                       ("answer_7093d898_5", 0, "temporal_endpoint"),
                       ("answer_7093d898_6", 0, "temporal_endpoint")],
    "gpt4_88806d6e": [("answer_e60a93ff_1", 0, "temporal_endpoint"),
                       ("answer_e60a93ff_2", 0, "temporal_endpoint")],
    "gpt4_93159ced": [("answer_e5131a1b_1", 2, "fact"),
                       ("answer_e5131a1b_2", 4, "fact")],
    "gpt4_a1b77f9c": [("answer_e9ad5914_1", 0, "temporal_endpoint"),
                       ("answer_e9ad5914_2", 0, "temporal_endpoint"),
                       ("answer_e9ad5914_3", 6, "temporal_endpoint"),
                       ("answer_e9ad5914_4", 0, "temporal_endpoint"),
                       ("answer_e9ad5914_5", 6, "temporal_endpoint"),
                       ("answer_e9ad5914_6", 0, "temporal_endpoint")],
    "gpt4_a56e767c": [("answer_cf9e3940_1", 0, "aggregation_member"),
                       ("answer_cf9e3940_2", 0, "aggregation_member"),
                       ("answer_cf9e3940_2", 4, "aggregation_member"),
                       ("answer_cf9e3940_3", 0, "aggregation_member")],
    "gpt4_ab202e7f": [("answer_728deb4d_1", 0, "aggregation_member"),
                       ("answer_728deb4d_2", 6, "aggregation_member"),
                       ("answer_728deb4d_3", 0, "aggregation_member"),
                       ("answer_728deb4d_4", 0, "aggregation_member"),
                       ("answer_728deb4d_5", 0, "aggregation_member")],
    "gpt4_af6db32f": [("answer_184c8f56_1", 0, "temporal_endpoint")],
    "gpt4_cd90e484": [("answer_aa930b56_1", 4, "temporal_endpoint"),
                       ("answer_aa930b56_2", 0, "temporal_endpoint")],
    "gpt4_d84a3211": [("answer_2880eb6c_1", 6, "aggregation_member"),
                       ("answer_2880eb6c_2", 2, "aggregation_member")],
    "gpt4_e072b769": [("answer_c19bd2bf_1", 0, "temporal_endpoint")],
    "gpt4_f420262d": [("answer_d8a1af6c_5", 0, "temporal_endpoint")],
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reduced", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = {str(row["question_id"]): row for row in json.loads(args.data.read_text(encoding="utf-8"))}
    reduced = {str(row["question_id"]): row for row in read_jsonl(args.reduced)}
    if set(cases) != set(reduced):
        raise ValueError("reduced candidate coverage mismatch")
    unknown = set(OVERRIDES) - set(cases)
    if unknown:
        raise ValueError(f"unknown override questions: {sorted(unknown)}")
    decisions = []
    for qid in sorted(cases):
        case = cases[qid]
        sessions = {str(sid): messages for sid, messages in zip(
            case["haystack_session_ids"], case["haystack_sessions"]
        )}
        if qid in OVERRIDES:
            evidence = []
            for session_id, turn_index, support_role in OVERRIDES[qid]:
                turn = sessions[session_id][turn_index]
                if turn.get("role") != "user":
                    raise ValueError(f"override is not a user turn: {qid}/{session_id}/{turn_index}")
                evidence.append({
                    "session_id": session_id, "turn_index": turn_index,
                    "span_start": 0, "span_end": len(str(turn.get("content") or "")),
                    "support_role": support_role, "confidence": "high",
                    "disagreement": "candidate_set_changed",
                })
        else:
            evidence = []
            seen: set[tuple[str, int]] = set()
            for row in reduced[qid]["evidence"]:
                key = (str(row["session_id"]), int(row["turn_index"]))
                if key not in seen:
                    evidence.append({key: value for key, value in row.items() if key != "question_id"})
                    seen.add(key)
        if not evidence:
            raise ValueError(f"manual adjudication left no evidence: {qid}")
        decisions.append({
            "question_id": qid, "status": "accepted", "changed": qid in OVERRIDES,
            "reviewer": "codex-semantic-review-r1",
            "second_reviewer": "codex-semantic-adjudication-r2",
            "disagreement": "candidate_set_changed" if qid in OVERRIDES else "none",
            "replacement_evidence": evidence,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in decisions),
        encoding="utf-8",
    )
    print(json.dumps({
        "questions": len(decisions), "changed": len(OVERRIDES),
        "evidence": sum(len(row["replacement_evidence"]) for row in decisions),
    }))


if __name__ == "__main__":
    main()
