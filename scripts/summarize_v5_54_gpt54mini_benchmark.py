#!/usr/bin/env python3
"""Summarize the unified GPT-5.4-mini build and 32/64-turn benchmark."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def accuracy(data: Iterable[dict[str, Any]]) -> dict[str, Any]:
    verdicts = list(data)
    correct = sum(bool(row.get("correct")) for row in verdicts)
    return {
        "questions": len(verdicts), "correct": correct,
        "accuracy": correct / max(1, len(verdicts)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    build = read(args.root / "build_report.json")
    result: dict[str, Any] = {
        "schema_version": "graphmem-v5.54-gpt54mini-unified-v1",
        "root": str(args.root),
        "model_contract": {
            "build_model": "gpt-5.4-mini",
            "answer_model": "gpt-5.4-mini",
            "judge_model": "gpt-5.6-luna",
            "packing_tokenizer": "Qwen3-30B frozen tokenizer",
        },
        "build": build.get("summary", {}),
        "budgets": {},
    }
    for turns in (32, 64):
        answer_root = args.root / f"turn{turns}" / "answer"
        manifest = read(answer_root / "run_manifest.json")
        answers = rows(answer_root / "answers.jsonl")
        lme = rows(answer_root / "judge_lme" / "auto_eval.jsonl")
        locomo = rows(answer_root / "judge_locomo" / "auto_eval.jsonl")
        if len(answers) != 2040 or len(lme) != 500 or len(locomo) != 1540:
            raise ValueError(
                f"turn{turns} incomplete: answers={len(answers)}, "
                f"lme={len(lme)}, locomo={len(locomo)}")
        answer_by_id = {str(row["question_id"]): row for row in answers}
        strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for verdict in (*lme, *locomo):
            question = answer_by_id[str(verdict["question_id"])]
            if question.get("benchmark") == "longmemeval":
                label = "lme:" + str(
                    question.get("question_type") or question.get("stratum"))
            else:
                label = "locomo:cat" + str(
                    question.get("category") or question.get("stratum", "").split("cat")[-1])
            strata[label].append(verdict)
        result["budgets"][str(turns)] = {
            "answer_manifest": manifest,
            "accuracy": {
                "longmemeval": accuracy(lme),
                "locomo": accuracy(locomo),
                "by_stratum": {
                    key: accuracy(value) for key, value in sorted(strata.items())
                },
            },
            "artifacts": {"answer_root": str(answer_root)},
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({
        turns: value["accuracy"] for turns, value in result["budgets"].items()
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
