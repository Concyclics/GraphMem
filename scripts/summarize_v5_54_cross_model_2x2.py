#!/usr/bin/env python3
"""Summarize the frozen-evidence GraphMem build/answer-model 2x2."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def accuracy(path: Path) -> dict:
    verdicts = rows(path)
    correct = sum(bool(row["correct"]) for row in verdicts)
    return {"questions": len(verdicts), "correct": correct,
            "accuracy": correct / max(1, len(verdicts))}


def exact_mcnemar(a: list[bool], b: list[bool]) -> dict:
    gains = sum(not left and right for left, right in zip(a, b))
    losses = sum(left and not right for left, right in zip(a, b))
    discordant = gains + losses
    if not discordant:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, index)
                   for index in range(0, min(gains, losses) + 1)) / 2 ** discordant
        p_value = min(1.0, 2 * tail)
    return {"gains": gains, "losses": losses,
            "delta_correct": gains - losses, "mcnemar_exact_p": p_value}


def index(path: Path) -> dict[str, dict]:
    return {str(row["question_id"]): row for row in rows(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native_root = args.root.parent
    arms = {
        "q30_graph_q30": native_root / "index_structure_ablation",
        "q30_graph_gpt54": args.root / "q30_graph_gpt54",
        "gpt54_graph_q30": args.root / "gpt54_graph_q30",
        "gpt54_graph_gpt54": native_root / "gpt54mini_unified_full",
    }
    output: dict = {
        "schema_version": "graphmem-v5.54-cross-model-2x2-v1",
        "protocol": {
            "judge_model": "gpt-5.6-luna",
            "frozen_evidence_within_each_graph": True,
            "turn_budgets": [32, 64],
        },
        "arms": {}, "paired_answer_model_effect": {},
    }
    verdict_maps: dict[tuple[str, int, str], dict[str, dict]] = {}
    prepared_hashes: dict[tuple[str, int], dict[str, str]] = {}
    for name, root in arms.items():
        output["arms"][name] = {}
        for budget in (32, 64):
            if name == "q30_graph_q30":
                answer = root / f"turn{budget}/full/answer"
                prepare = root / f"turn{budget}/full/prepare/prepared_answers.jsonl"
                lme = answer / "judge_longmemeval/paired_verdicts.jsonl"
                locomo = answer / "judge_locomo/paired_verdicts.jsonl"
            else:
                answer = root / f"turn{budget}/answer"
                prepare = (
                    native_root / "index_structure_ablation"
                    / f"turn{budget}/full/prepare/prepared_answers.jsonl"
                    if name == "q30_graph_gpt54"
                    else native_root / "gpt54mini_unified_full"
                    / f"turn{budget}/answer/prepared_answers.jsonl")
                lme = answer / "judge_lme/auto_eval.jsonl"
                locomo = answer / "judge_locomo/auto_eval.jsonl"
            prepared_hashes[(name, budget)] = {
                str(row["question_id"]): str(row["prompt_payload_hash"])
                for row in rows(prepare)}
            output["arms"][name][str(budget)] = {
                "longmemeval": accuracy(lme), "locomo": accuracy(locomo),
                "answer_root": str(answer),
            }
            verdict_maps[(name, budget, "longmemeval")] = index(lme)
            verdict_maps[(name, budget, "locomo")] = index(locomo)
    graph_pairs = {
        "q30_graph": ("q30_graph_q30", "q30_graph_gpt54"),
        "gpt54_graph": ("gpt54_graph_gpt54", "gpt54_graph_q30"),
    }
    for graph, (baseline, candidate) in graph_pairs.items():
        output["paired_answer_model_effect"][graph] = {}
        for budget in (32, 64):
            prompt_match = (prepared_hashes[(baseline, budget)]
                            == prepared_hashes[(candidate, budget)])
            result = {"prompt_hashes_exactly_match": prompt_match}
            for benchmark in ("longmemeval", "locomo"):
                left = verdict_maps[(baseline, budget, benchmark)]
                right = verdict_maps[(candidate, budget, benchmark)]
                ids = sorted(set(left) & set(right))
                result[benchmark] = exact_mcnemar(
                    [bool(left[item]["correct"]) for item in ids],
                    [bool(right[item]["correct"]) for item in ids])
                result[benchmark]["questions"] = len(ids)
            output["paired_answer_model_effect"][graph][str(budget)] = result
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
