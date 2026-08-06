#!/usr/bin/env python3
"""Full-benchmark report: accuracy by stratum, with the caveats made structural.

Three things this refuses to do, because each produces a number that looks fine
and is wrong:

* **average turn-level metrics over unannotated questions.** Turn gold covers
  100 of 500 LongMemEval questions and ``gold <= predicted`` is vacuously true
  where there is no gold, so those rows are excluded rather than counted as hits.
* **mix abstention questions into retrieval metrics.** The 30 ``_abs`` questions
  have no correct evidence; their retrieval numbers are meaningless even though
  their answers are scorable.
* **print a bare accuracy with no interval.** At n=200 the 95% CI is about
  +/-5.5pp, which is wider than every effect measured so far; a point estimate
  without it invites reading noise as progress.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def wilson(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval: behaves at the extremes where the normal approximation does not."""
    if total == 0:
        return (0.0, 0.0)
    phat = correct / total
    denom = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denom
    margin = z * ((phat * (1 - phat) / total + z * z / (4 * total * total)) ** 0.5) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="an answer run directory")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    judged: dict[str, bool] = {}
    for name in ("judge_lme", "judge_locomo"):
        path = args.run / name / "auto_eval.jsonl"
        if path.exists():
            judged.update({str(row["question_id"]): bool(row["correct"])
                           for row in read_jsonl(path)})
    retrieval = {str(row["dev_question_id"]): row
                 for row in read_jsonl(args.run / "retrieval.jsonl")}
    rows = [{**row, "correct": judged[qid]} for qid, row in retrieval.items() if qid in judged]
    if not rows:
        raise SystemExit("no judged questions found; run the judges first")

    def block(subset: list[dict]) -> dict:
        total = len(subset)
        correct = sum(row["correct"] for row in subset)
        low, high = wilson(correct, total)
        turn_rows = [row for row in subset if row.get("has_turn_gold", True)]
        return {
            "n": total, "correct": correct,
            "accuracy": correct / total if total else 0.0,
            "ci95": [round(low, 4), round(high, 4)],
            "turn_all_hit_defined_on": len(turn_rows),
            "turn_all_hit": (sum(bool(row.get("turn_all_hit")) for row in turn_rows) / len(turn_rows)
                             if turn_rows else None),
            "prompt_tokens_mean": sum(int(row.get("prompt_tokens", 0)) for row in subset) / total,
        }

    report = {"run": str(args.run), "overall": block(rows), "by_benchmark": {}, "by_stratum": {}}
    for benchmark in sorted({row["benchmark"] for row in rows}):
        report["by_benchmark"][benchmark] = block(
            [row for row in rows if row["benchmark"] == benchmark])
    for stratum in sorted({row["stratum"] for row in rows}):
        report["by_stratum"][stratum] = block([row for row in rows if row["stratum"] == stratum])

    abstention = [row for row in rows if row.get("is_abstention")]
    if abstention:
        report["abstention"] = block(abstention)
        report["non_abstention"] = block([row for row in rows if not row.get("is_abstention")])

    tokens = sorted(int(row.get("prompt_tokens", 0)) for row in rows)
    report["answer_tokens"] = {
        "mean": sum(tokens) / len(tokens), "p50": tokens[len(tokens) // 2],
        "p95": tokens[max(0, int(0.95 * len(tokens)) - 1)], "max": max(tokens),
        "over_10k": sum(1 for value in tokens if value > 10_000),
    }
    report["judge_model_caveat"] = (
        "Judged by a local Qwen3-30B under mem0's official prompts. Every historical "
        "GraphMem number (V3.7 89.0%/86.2%, V4.1 72.6%) was judged by gpt-5.4-mini, so "
        "these are not directly comparable until the cross-judge calibration is run.")

    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    def line(name: str, block_row: dict) -> None:
        hit = block_row["turn_all_hit"]
        print(f"{name:28s} {block_row['n']:5d} {block_row['accuracy']:7.3f} "
              f"[{block_row['ci95'][0]:.3f},{block_row['ci95'][1]:.3f}] "
              f"{(f'{hit:.3f}' if hit is not None else '-'):>13s}")

    print(f"{'subset':28s} {'n':>5s} {'acc':>7s} {'CI95':>18s} {'turn_all_hit':>13s}")
    line("OVERALL", report["overall"])
    for group in ("by_benchmark", "by_stratum"):
        print()
        for name, block_row in report[group].items():
            line(name, block_row)
    print(f"\nanswer tokens: {report['answer_tokens']}")
    print(f"\n{report['judge_model_caveat']}")


if __name__ == "__main__":
    main()
