#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
sys.path.insert(0, str(ROOT / "src"))

from graphmem_demo.clients import OpenAICompatibleClient  # noqa: E402
from graphmem_demo.data import load_longmemeval_cases  # noqa: E402
from graphmem_demo.v3.llm_navigation import (  # noqa: E402
    navigated_answer_messages,
    navigation_messages,
    parse_navigation_plan,
    selected_evidence_text,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-answer saved V3 retrieval with a token-bounded LLM path selector."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--retrieval-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("SGAO_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--base-url", default=os.environ.get("SGAO_BASE_URL", "https://sub2api.sgao.me/v1/"))
    parser.add_argument("--api-key-env", default="SGAO_API_KEY")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-questions", type=int, default=500)
    parser.add_argument("--navigation-max-tokens", type=int, default=256)
    parser.add_argument("--answer-max-tokens", type=int, default=256)
    parser.add_argument("--answer-budget-tokens", type=int, default=10000)
    return parser.parse_args()


def _run_one(case: Any, retrieval: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    client = OpenAICompatibleClient(
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        request_profile="openai",
    )
    nav_messages, valid_ids = navigation_messages(
        question=case.question,
        question_date=case.question_date,
        evidence_ledger=list(retrieval.get("evidence_ledger") or []),
    )
    nav_result = client.chat(
        question_id=case.question_id,
        variant="v3_llm_navigator",
        stage="answer_navigation",
        thinking_mode="none",
        messages=nav_messages,
        max_tokens=args.navigation_max_tokens,
        json_mode=True,
    )
    plan = parse_navigation_plan(nav_result.text, valid_ids)
    evidence = selected_evidence_text(
        question=case.question,
        evidence_ledger=list(retrieval.get("evidence_ledger") or []),
        plan=plan,
        max_rough_tokens=2800,
    )
    answer_result = client.chat(
        question_id=case.question_id,
        variant="v3_llm_navigator",
        stage="answer_qa",
        thinking_mode="none",
        messages=navigated_answer_messages(
            question=case.question,
            question_date=case.question_date,
            evidence_text=evidence,
            plan=plan,
        ),
        max_tokens=args.answer_max_tokens,
    )
    records = [asdict(nav_result.record), asdict(answer_result.record)]
    answer_total = sum(int(row.get("total_tokens") or 0) for row in records)
    if answer_total > args.answer_budget_tokens:
        raise RuntimeError(
            f"answer budget exceeded for {case.question_id}: "
            f"{answer_total}>{args.answer_budget_tokens}"
        )
    if any(int(row.get("reasoning_tokens") or 0) != 0 for row in records):
        raise RuntimeError(f"reasoning tokens were returned for {case.question_id}")
    answer = {
        "question_id": case.question_id,
        "variant": "v3_llm_navigator",
        "question": case.question,
        "gold_answer": case.answer,
        "prediction": answer_result.text,
        "answer_session_ids": case.answer_session_ids,
        "retrieved_answer_session_hit": retrieval.get("answer_session_hit", False),
        "retrieved_answer_session_all_hit": retrieval.get("answer_session_all_hit", False),
        "retrieved_answer_session_recall": retrieval.get("answer_session_recall", 0.0),
        "navigation_operation": plan.operation,
        "navigation_parse_error": plan.parse_error,
        "navigation_selected_ids": list(plan.selected_ids),
        "navigation_missing_slots": list(plan.missing_slots),
        "answer_total_tokens": answer_total,
    }
    navigation = {
        "question_id": case.question_id,
        "plan": asdict(plan),
        "valid_candidate_ids": valid_ids,
        "navigation_response": nav_result.text,
        "selected_evidence": evidence,
    }
    return answer, records, navigation


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_longmemeval_cases(args.data, "all", args.max_questions)
    retrieval_by_id = {
        str(row["question_id"]): row
        for row in _read_jsonl(args.retrieval_results)
    }
    missing = [case.question_id for case in cases if case.question_id not in retrieval_by_id]
    if missing:
        raise RuntimeError(f"missing retrieval rows: {missing[:8]}")

    order = {case.question_id: index for index, case in enumerate(cases)}
    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_run_one, case, retrieval_by_id[case.question_id], args): case
            for case in cases
        }
        for future in as_completed(futures):
            case = futures[future]
            answer, records, navigation = future.result()
            completed.append((order[case.question_id], answer, records, navigation))
            print(
                f"navigator question={case.question_id} "
                f"tokens={answer['answer_total_tokens']} "
                f"selected={len(answer['navigation_selected_ids'])}",
                flush=True,
            )
    completed.sort(key=lambda row: row[0])
    answers = [row[1] for row in completed]
    records = [record for row in completed for record in row[2]]
    navigations = [row[3] for row in completed]
    _write_jsonl(args.output_dir / "answers.jsonl", answers)
    _write_jsonl(args.output_dir / "llm_calls.jsonl", records)
    _write_jsonl(args.output_dir / "navigation_results.jsonl", navigations)
    _write_jsonl(
        args.output_dir / "hypothesis.jsonl",
        [
            {"question_id": row["question_id"], "hypothesis": row["prediction"]}
            for row in answers
        ],
    )
    per_question = [int(row["answer_total_tokens"]) for row in answers]
    stats = {
        "question_count": len(answers),
        "call_count": len(records),
        "prompt_cache_miss_tokens": sum(int(row.get("prompt_cache_miss_tokens") or 0) for row in records),
        "prompt_cache_hit_tokens": sum(int(row.get("prompt_cache_hit_tokens") or 0) for row in records),
        "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in records),
        "reasoning_tokens": sum(int(row.get("reasoning_tokens") or 0) for row in records),
        "answer_total_tokens": sum(per_question),
        "answer_max_tokens": max(per_question, default=0),
        "answer_over_budget_count": sum(value > args.answer_budget_tokens for value in per_question),
        "navigation_parse_error_count": sum(bool(row["navigation_parse_error"]) for row in answers),
    }
    (args.output_dir / "token_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
