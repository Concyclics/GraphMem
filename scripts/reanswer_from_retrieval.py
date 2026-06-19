#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem_demo.clients import DeepSeekClient  # noqa: E402
from graphmem_demo.data import load_longmemeval_cases  # noqa: E402
from graphmem_demo.pipeline import _answer_messages  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-answer LongMemEval questions from a saved retrieval_results.jsonl file."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--retrieval-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", default="reanswer_from_retrieval")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--question-type", default="all")
    parser.add_argument("--max-questions", type=int, default=60)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--enhanced-qa", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def answer_one(
    *,
    case: Any,
    retrieval: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    llm = DeepSeekClient(model=args.model, base_url=args.base_url)
    started = time.perf_counter()
    result = llm.chat(
        question_id=case.question_id,
        variant=args.variant,
        stage="answer_qa",
        thinking_mode="none",
        messages=_answer_messages(case, retrieval["context_text"], enhanced=args.enhanced_qa),
        max_tokens=args.max_tokens,
    )
    answer_row = {
        "question_id": case.question_id,
        "variant": args.variant,
        "question": case.question,
        "gold_answer": case.answer,
        "prediction": result.text,
        "answer_session_ids": case.answer_session_ids,
        "retrieved_answer_session_hit": retrieval.get("answer_session_hit", False),
        "retrieved_answer_session_any_hit": retrieval.get("answer_session_hit", False),
        "retrieved_answer_session_all_hit": retrieval.get("answer_session_all_hit", False),
        "retrieved_answer_session_recall": retrieval.get("answer_session_recall", 0.0),
        "source_retrieval_variant": retrieval.get("variant"),
        "answer_latency_sec": time.perf_counter() - started,
    }
    return answer_row, asdict(result.record)


def aggregate(records: list[dict[str, Any]], answers: list[dict[str, Any]]) -> dict[str, Any]:
    total_prompt = sum(int(row.get("prompt_tokens") or 0) for row in records)
    total_completion = sum(int(row.get("completion_tokens") or 0) for row in records)
    total_reasoning = sum(int(row.get("reasoning_tokens") or 0) for row in records)
    total_tokens = sum(int(row.get("total_tokens") or 0) for row in records)
    qa_totals = [
        int(row.get("prompt_tokens") or 0) + int(row.get("completion_tokens") or 0)
        for row in records
    ]
    return {
        "question_count": len(answers),
        "answer_prompt_tokens": total_prompt,
        "answer_completion_tokens": total_completion,
        "reasoning_tokens": total_reasoning,
        "answer_total_tokens": total_tokens,
        "avg_answer_tokens_per_question": total_tokens / len(records) if records else 0.0,
        "max_query_answer_tokens": max(qa_totals) if qa_totals else 0,
        "query_answer_over_10k_count": sum(1 for value in qa_totals if value > 10000),
        "empty_answer_count": sum(1 for row in answers if not str(row.get("prediction") or "").strip()),
        "length_finish_count": sum(1 for row in records if row.get("finish_reason") == "length"),
        "retrieval_answer_session_all_hit_rate": (
            sum(1 for row in answers if row.get("retrieved_answer_session_all_hit")) / len(answers)
            if answers
            else 0.0
        ),
        "avg_retrieved_answer_session_recall": (
            sum(float(row.get("retrieved_answer_session_recall") or 0.0) for row in answers)
            / len(answers)
            if answers
            else 0.0
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = load_longmemeval_cases(args.data, args.question_type, args.max_questions)
    retrieval_by_id = {row["question_id"]: row for row in read_jsonl(args.retrieval_results)}
    missing = [case.question_id for case in cases if case.question_id not in retrieval_by_id]
    if missing:
        raise RuntimeError(f"Missing retrieval results for {len(missing)} questions: {missing[:5]}")

    answers_path = args.output_dir / "answers.jsonl"
    calls_path = args.output_dir / "llm_calls.jsonl"
    hypothesis_path = args.output_dir / f"{args.variant}_hypothesis.jsonl"
    if not args.resume:
        answers_path.write_text("", encoding="utf-8")
        calls_path.write_text("", encoding="utf-8")
        hypothesis_path.write_text("", encoding="utf-8")

    completed = {row["question_id"] for row in read_jsonl(answers_path)} if args.resume else set()
    pending = [case for case in cases if case.question_id not in completed]
    results: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    index_by_id = {case.question_id: index for index, case in enumerate(cases)}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                answer_one,
                case=case,
                retrieval=retrieval_by_id[case.question_id],
                args=args,
            ): case
            for case in pending
        }
        for future in as_completed(futures):
            case = futures[future]
            answer_row, call_row = future.result()
            results.append((index_by_id[case.question_id], answer_row, call_row))
            print(
                f"{args.variant}: question={case.question_id} "
                f"tokens={call_row.get('total_tokens')} finish={call_row.get('finish_reason')}",
                flush=True,
            )

    results.sort(key=lambda item: item[0])
    append_jsonl(answers_path, [item[1] for item in results])
    append_jsonl(calls_path, [item[2] for item in results])

    all_answers = read_jsonl(answers_path)
    all_records = read_jsonl(calls_path)
    hypothesis_rows = [
        {"question_id": row["question_id"], "hypothesis": row.get("prediction", "")}
        for row in all_answers
    ]
    hypothesis_path.write_text("", encoding="utf-8")
    append_jsonl(hypothesis_path, hypothesis_rows)
    write_json(args.output_dir / "query_stats.json", aggregate(all_records, all_answers))
    print(f"answers={answers_path}")
    print(f"hypothesis={hypothesis_path}")
    print(f"query_stats={args.output_dir / 'query_stats.json'}")


if __name__ == "__main__":
    main()
