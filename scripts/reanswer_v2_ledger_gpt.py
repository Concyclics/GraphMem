#!/usr/bin/env python3
"""Answer persisted V2 evidence ledgers with the current GPT backbone."""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
sys.path.insert(0, str(ROOT / "src"))

from graphmem_demo.clients import OpenAICompatibleClient  # noqa: E402
from graphmem_demo.data import load_longmemeval_cases  # noqa: E402
from graphmem_demo.hierarchical_v2 import answer_messages  # noqa: E402

VARIANT = "hierarchical_hybrid_graph_v3_7_v2_ledger"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--retrieval-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--answer-budget-tokens", type=int, default=12100)
    parser.add_argument("--model", default=os.environ.get("SGAO_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--base-url", default=os.environ.get("SGAO_BASE_URL", "https://sub2api.sgao.me/v1/"))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    answers_path = args.output_dir / "answers.jsonl"
    calls_path = args.output_dir / "llm_calls.jsonl"
    if not args.resume:
        answers_path.write_text("", encoding="utf-8")
        calls_path.write_text("", encoding="utf-8")
    existing = rows(answers_path)
    completed = {str(row["question_id"]) for row in existing}
    cases = [
        case for case in load_longmemeval_cases(args.data, "all")
        if case.question_id not in completed
    ]
    retrievals = {
        str(row["question_id"]): row for row in rows(args.retrieval_results)
    }
    missing = [case.question_id for case in cases if case.question_id not in retrievals]
    if missing:
        raise RuntimeError(f"missing V2 retrieval results: {missing[:10]}")
    client = OpenAICompatibleClient(
        model=args.model, base_url=args.base_url,
        api_key_env="SGAO_API_KEY", request_profile="openai",
    )

    def answer(case):
        retrieval = retrievals[case.question_id]
        view = SimpleNamespace(
            context_text=str(retrieval["context_text"]),
            query_kind=str(retrieval.get("query_kind") or ""),
        )
        messages = answer_messages(case, view)
        result = client.chat(
            question_id=case.question_id,
            variant=VARIANT,
            stage="answer_qa",
            messages=messages,
            thinking_mode="none",
            max_tokens=args.max_tokens,
        )
        if result.record.total_tokens > args.answer_budget_tokens:
            raise RuntimeError(
                f"{case.question_id}: {result.record.total_tokens}>"
                f"{args.answer_budget_tokens}"
            )
        if result.record.reasoning_tokens:
            raise RuntimeError(f"{case.question_id}: reasoning tokens returned")
        return {
            "question_id": case.question_id,
            "variant": VARIANT,
            "question": case.question,
            "question_type": case.question_type,
            "question_date": case.question_date,
            "gold_answer": case.answer,
            "prediction": result.text.strip(),
            "answer_session_ids": case.answer_session_ids,
            "answer_mode": "gpt_from_persisted_v2_evidence_ledger",
        }, asdict(result.record)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(answer, case): case for case in cases}
        done = len(existing)
        for future in as_completed(futures):
            answer_row, call_row = future.result()
            append(answers_path, answer_row)
            append(calls_path, call_row)
            done += 1
            print(
                f"[{done}/{done + len(futures) - (done - len(existing))}] "
                f"{answer_row['question_id']} {answer_row['prediction'][:100]}",
                flush=True,
            )


if __name__ == "__main__":
    main()
