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

from graphmem_demo.clients import DeepSeekClient, rough_token_count  # noqa: E402
from graphmem_demo.data import load_longmemeval_cases  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-stage Qwen answer: extract evidence notes, then answer from notes."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--retrieval-results", type=Path, required=True)
    parser.add_argument("--assistant-retrieval-results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", default="two_stage_answer")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--question-type", default="all")
    parser.add_argument("--max-questions", type=int, default=60)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--note-max-tokens", type=int, default=512)
    parser.add_argument("--answer-max-tokens", type=int, default=512)
    parser.add_argument("--note-context-rough-tokens", type=int, default=6800)
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


def trim_rough_tokens(text: str, token_limit: int) -> tuple[str, bool]:
    if rough_token_count(text) <= token_limit:
        return text, False
    lines = text.splitlines()
    kept: list[str] = []
    total = 0
    for line in lines:
        line_tokens = rough_token_count(line)
        if kept and total + line_tokens > token_limit:
            break
        kept.append(line)
        total += line_tokens
    return "\n".join(kept), True


def note_messages(case: Any, context: str) -> list[dict[str, str]]:
    system = (
        "Extract compact evidence notes for answering a memory question. Use only the supplied "
        "evidence. Do not answer from general knowledge. Return concise bullets under these exact "
        "headings: Relevant facts, Calculations or ordering, Missing or conflicts, Answer hint. "
        "Scan all evidence before writing notes. Preserve exact names, dates, times, amounts, "
        "counts, services, brands, negative preferences, cancellations, and latest updates. For "
        "previous-chat questions, later user acceptance or repeated use of a name overrides earlier "
        "draft suggestions; later named tables override earlier Agent-number tables. For preference "
        "questions, extract user-specific likes, dislikes, constraints, and acceptable categories "
        "even if no live local listing is present. For totals and counts, list every candidate item "
        "with its value before summing. For count questions, mark whether each candidate satisfies "
        "the wording of the question. Do not count recommendations, examples, budgets, price ranges, "
        "or future plans unless the question asks about planned items. For currently-own or "
        "currently-use questions, exclude items only considered, suggested, replaced, canceled, "
        "returned, or not yet acquired. For attended/visited/completed questions, exclude missed, "
        "planned, suggested, or merely discussed events. Count separately named people, babies, "
        "items, devices, appointments, trips, and events when evidence states separate entities. "
        "For temporal questions, resolve relative dates using session "
        "headers and show the arithmetic needed."
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"Question date: {case.question_date or 'unknown'}\n"
                f"Question type: {case.question_type}\n"
                f"Question: {case.question}\n\n"
                f"Retrieved memory evidence:\n{context}"
            ),
        },
    ]


def final_messages(case: Any, notes: str) -> list[dict[str, str]]:
    system = (
        "Answer the memory question using only the evidence notes. First write one short "
        "'Evidence facts:' section, then 'Final answer:'. If the notes contain the answer, do not "
        "abstain. For preference or recommendation questions, give a useful personalized answer "
        "grounded in the notes. For counts, totals, time, and ordering, perform the calculation "
        "explicitly. For count questions, first list the counted items and exclude recommendations, "
        "examples, budgets, price ranges, missed events, canceled/replaced/returned items, and future "
        "plans unless the question asks for them. If a required value is missing from the notes, say "
        "what is missing."
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"Question date: {case.question_date or 'unknown'}\n"
                f"Question type: {case.question_type}\n"
                f"Question: {case.question}\n\nEvidence notes:\n{notes}"
            ),
        },
    ]


def choose_retrieval(case: Any, base: dict[str, Any], assistant: dict[str, Any] | None) -> dict[str, Any]:
    if case.question_type == "single-session-assistant" and assistant is not None:
        return assistant
    return base


def answer_one(
    *,
    case: Any,
    base_retrieval: dict[str, Any],
    assistant_retrieval: dict[str, Any] | None,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    retrieval = choose_retrieval(case, base_retrieval, assistant_retrieval)
    context, truncated = trim_rough_tokens(
        retrieval["context_text"], args.note_context_rough_tokens
    )
    llm = DeepSeekClient(model=args.model, base_url=args.base_url)
    started = time.perf_counter()
    note_result = llm.chat(
        question_id=case.question_id,
        variant=args.variant,
        stage="answer_qa",
        thinking_mode="none",
        messages=note_messages(case, context),
        max_tokens=args.note_max_tokens,
    )
    final_result = llm.chat(
        question_id=case.question_id,
        variant=args.variant,
        stage="answer_qa",
        thinking_mode="none",
        messages=final_messages(case, note_result.text),
        max_tokens=args.answer_max_tokens,
    )
    answer_row = {
        "question_id": case.question_id,
        "variant": args.variant,
        "question": case.question,
        "gold_answer": case.answer,
        "prediction": final_result.text,
        "evidence_notes": note_result.text,
        "answer_session_ids": case.answer_session_ids,
        "retrieved_answer_session_hit": retrieval.get("answer_session_hit", False),
        "retrieved_answer_session_any_hit": retrieval.get("answer_session_hit", False),
        "retrieved_answer_session_all_hit": retrieval.get("answer_session_all_hit", False),
        "retrieved_answer_session_recall": retrieval.get("answer_session_recall", 0.0),
        "source_retrieval_variant": retrieval.get("variant"),
        "context_truncated_for_notes": truncated,
        "answer_latency_sec": time.perf_counter() - started,
    }
    note_record = asdict(note_result.record)
    note_record["stage_detail"] = "evidence_notes"
    final_record = asdict(final_result.record)
    final_record["stage_detail"] = "final_answer"
    return answer_row, [note_record, final_record]


def aggregate(records: list[dict[str, Any]], answers: list[dict[str, Any]]) -> dict[str, Any]:
    by_question: dict[str, int] = {}
    for row in records:
        qid = row["question_id"]
        by_question[qid] = by_question.get(qid, 0) + int(row.get("total_tokens") or 0)
    total_prompt = sum(int(row.get("prompt_tokens") or 0) for row in records)
    total_completion = sum(int(row.get("completion_tokens") or 0) for row in records)
    total_reasoning = sum(int(row.get("reasoning_tokens") or 0) for row in records)
    total_tokens = sum(int(row.get("total_tokens") or 0) for row in records)
    return {
        "question_count": len(answers),
        "llm_call_count": len(records),
        "answer_prompt_tokens": total_prompt,
        "answer_completion_tokens": total_completion,
        "reasoning_tokens": total_reasoning,
        "answer_total_tokens": total_tokens,
        "avg_answer_tokens_per_question": total_tokens / len(answers) if answers else 0.0,
        "max_query_answer_tokens": max(by_question.values()) if by_question else 0,
        "query_answer_over_10k_count": sum(1 for value in by_question.values() if value > 10000),
        "empty_answer_count": sum(1 for row in answers if not str(row.get("prediction") or "").strip()),
        "length_finish_count": sum(1 for row in records if row.get("finish_reason") == "length"),
        "context_truncated_count": sum(1 for row in answers if row.get("context_truncated_for_notes")),
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
    assistant_by_id = (
        {row["question_id"]: row for row in read_jsonl(args.assistant_retrieval_results)}
        if args.assistant_retrieval_results
        else {}
    )
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
    index_by_id = {case.question_id: index for index, case in enumerate(cases)}
    results: list[tuple[int, dict[str, Any], list[dict[str, Any]]]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                answer_one,
                case=case,
                base_retrieval=retrieval_by_id[case.question_id],
                assistant_retrieval=assistant_by_id.get(case.question_id),
                args=args,
            ): case
            for case in pending
        }
        for future in as_completed(futures):
            case = futures[future]
            answer_row, call_rows = future.result()
            total_tokens = sum(int(row.get("total_tokens") or 0) for row in call_rows)
            results.append((index_by_id[case.question_id], answer_row, call_rows))
            print(
                f"{args.variant}: question={case.question_id} "
                f"tokens={total_tokens} truncated={answer_row['context_truncated_for_notes']}",
                flush=True,
            )

    results.sort(key=lambda item: item[0])
    append_jsonl(answers_path, [item[1] for item in results])
    append_jsonl(calls_path, [row for item in results for row in item[2]])
    all_answers = read_jsonl(answers_path)
    all_records = read_jsonl(calls_path)
    hypothesis_path.write_text("", encoding="utf-8")
    append_jsonl(
        hypothesis_path,
        [
            {"question_id": row["question_id"], "hypothesis": row.get("prediction", "")}
            for row in all_answers
        ],
    )
    write_json(args.output_dir / "query_stats.json", aggregate(all_records, all_answers))
    print(f"answers={answers_path}")
    print(f"hypothesis={hypothesis_path}")
    print(f"query_stats={args.output_dir / 'query_stats.json'}")


if __name__ == "__main__":
    main()
