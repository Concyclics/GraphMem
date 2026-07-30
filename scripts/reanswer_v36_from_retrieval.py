#!/usr/bin/env python3
"""Run the single V3.6 answer call again without changing frozen retrieval."""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
sys.path.insert(0, str(ROOT / "src"))

from graphmem_demo.clients import OpenAICompatibleClient
from graphmem_demo.data import load_longmemeval_cases
from graphmem_demo.models import RetrievedContext
from graphmem_demo.v36.retrieval import answer_messages


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--retrieval-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-answer-tokens", type=int, default=512)
    parser.add_argument("--model", default=os.environ.get("SGAO_MODEL", "gpt-5.4-mini"))
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SGAO_BASE_URL", "https://sub2api.sgao.me/v1/"),
    )
    return parser.parse_args()


def _rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = _args()
    cases = load_longmemeval_cases(args.data, question_type="all")
    retrievals = {
        row["question_id"]: RetrievedContext(**row)
        for row in _rows(args.retrieval_results)
    }
    missing = [case.question_id for case in cases if case.question_id not in retrievals]
    if missing:
        raise RuntimeError(f"missing frozen retrieval results: {missing}")
    client = OpenAICompatibleClient(
        model=args.model,
        base_url=args.base_url,
        api_key_env="SGAO_API_KEY",
        request_profile="openai",
    )

    def run(case):
        retrieval = retrievals[case.question_id]
        response = client.chat(
            question_id=case.question_id,
            variant=retrieval.variant,
            stage="answer_qa",
            messages=answer_messages(case, retrieval),
            thinking_mode="none",
            max_tokens=args.max_answer_tokens,
        )
        answer = {
            "question_id": case.question_id,
            "variant": retrieval.variant,
            "question": case.question,
            "question_type": case.question_type,
            "question_date": case.question_date,
            "gold_answer": case.answer,
            "prediction": response.text.strip(),
            "answer_session_ids": case.answer_session_ids,
        }
        return answer, asdict(response.record)

    answers, calls = [], []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run, case): case for case in cases}
        for future in as_completed(futures):
            answer, call = future.result()
            answers.append(answer)
            calls.append(call)
            print(
                f"[{len(answers)}/{len(cases)}] {answer['question_id']} "
                f"{answer['prediction'][:100]}",
                flush=True,
            )
    order = {case.question_id: index for index, case in enumerate(cases)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("answers.jsonl", answers), ("llm_calls.jsonl", calls)):
        rows.sort(key=lambda row: order[row["question_id"]])
        with (args.output_dir / name).open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
