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
from graphmem_demo.v36.retrieval import answer_messages as v36_answer_messages
from graphmem_demo.v41.compact_answer import (
    answer_messages as v41_compact_answer_messages,
    binding_verifier_messages,
    needs_binding_verifier,
    validate_binding_verdict,
)
from graphmem_demo.v41.retrieval import answer_messages as v41_answer_messages
from graphmem_demo.v41.agentic_answer import (
    answer_messages as v41_agentic_answer_messages,
    planner_messages as v41_agentic_planner_messages,
    validate_plan as validate_agentic_plan,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--retrieval-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-answer-tokens", type=int, default=512)
    parser.add_argument(
        "--question-ids",
        help="Optional comma-separated question IDs for a targeted replay.",
    )
    parser.add_argument(
        "--answer-policy", choices=("v36", "v41", "v41_compact", "v41_agentic"), default="v36"
    )
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
    if args.question_ids:
        requested = {item.strip() for item in args.question_ids.split(",") if item.strip()}
        cases = [case for case in cases if case.question_id in requested]
        found = {case.question_id for case in cases}
        missing_requested = sorted(requested - found)
        if missing_requested:
            raise RuntimeError(f"unknown requested question IDs: {missing_requested}")
    retrievals = {
        row["question_id"]: RetrievedContext(**row)
        for row in _rows(args.retrieval_results)
    }
    missing = [case.question_id for case in cases if case.question_id not in retrievals]
    if missing:
        raise RuntimeError(f"missing frozen retrieval results: {missing}")
    answer_message_builder = {
        "v36": v36_answer_messages,
        "v41": v41_answer_messages,
        "v41_compact": v41_compact_answer_messages,
        "v41_agentic": v41_agentic_answer_messages,
    }[args.answer_policy]
    client = OpenAICompatibleClient(
        model=args.model,
        base_url=args.base_url,
        api_key_env=os.environ.get("ANSWER_API_KEY_ENV", "SGAO_API_KEY"),
        request_profile=os.environ.get("ANSWER_REQUEST_PROFILE", "openai"),
    )

    def run(case):
        retrieval = retrievals[case.question_id]
        call_records = []
        binding_verdict = None
        agentic_plan = None
        if args.answer_policy == "v41_agentic":
            planner = client.chat(
                question_id=case.question_id,
                variant=os.environ.get("ANSWER_VARIANT", retrieval.variant),
                stage="answer_search_planner",
                messages=v41_agentic_planner_messages(case, retrieval),
                thinking_mode="none",
                max_tokens=192,
                json_mode=True,
            )
            call_records.append(asdict(planner.record))
            agentic_plan = validate_agentic_plan(planner.text, retrieval, case)
        if args.answer_policy == "v41_compact" and needs_binding_verifier(retrieval):
            verifier = client.chat(
                question_id=case.question_id,
                variant=os.environ.get("ANSWER_VARIANT", retrieval.variant),
                stage="answer_binding_planner",
                messages=binding_verifier_messages(case, retrieval),
                thinking_mode="none",
                max_tokens=192,
            )
            call_records.append(asdict(verifier.record))
            binding_verdict = validate_binding_verdict(verifier.text, retrieval)
        messages = (
            v41_compact_answer_messages(case, retrieval, binding_verdict)
            if args.answer_policy == "v41_compact"
            else v41_agentic_answer_messages(case, retrieval, agentic_plan or {})
            if args.answer_policy == "v41_agentic"
            else answer_message_builder(case, retrieval)
        )
        response = client.chat(
            question_id=case.question_id,
            variant=os.environ.get("ANSWER_VARIANT", retrieval.variant),
            stage="answer_qa",
            messages=messages,
            thinking_mode="none",
            max_tokens=args.max_answer_tokens,
        )
        answer = {
            "question_id": case.question_id,
            "variant": os.environ.get("ANSWER_VARIANT", retrieval.variant),
            "question": case.question,
            "question_type": case.question_type,
            "question_date": case.question_date,
            "gold_answer": case.answer,
            "prediction": response.text.strip(),
            "answer_session_ids": case.answer_session_ids,
            "binding_verdict": binding_verdict,
            "agentic_plan": agentic_plan,
        }
        call_records.append(asdict(response.record))
        return answer, call_records

    args.output_dir.mkdir(parents=True, exist_ok=True)
    answer_path = args.output_dir / "answers.jsonl"
    calls_path = args.output_dir / "llm_calls.jsonl"
    existing_answers = _rows(answer_path) if answer_path.exists() else []
    completed = {str(row["question_id"]) for row in existing_answers}
    pending_cases = [case for case in cases if case.question_id not in completed]
    failures: list[dict[str, str]] = []
    processed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run, case): case for case in pending_cases}
        for future in as_completed(futures):
            case = futures[future]
            try:
                answer, call_records = future.result()
            except Exception as exc:
                failures.append({"question_id": case.question_id, "error": repr(exc)})
                print(f"[error] {case.question_id}: {exc}", flush=True)
                continue
            with answer_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(answer, ensure_ascii=False) + "\n")
            with calls_path.open("a", encoding="utf-8") as handle:
                for call in call_records:
                    handle.write(json.dumps(call, ensure_ascii=False) + "\n")
            processed += 1
            print(
                "[%d/%d] %s %s" % (len(completed) + processed, len(cases), case.question_id, str(answer.get("prediction", ""))[:100]),
                flush=True,
            )
    if failures:
        with (args.output_dir / "answer_failures.jsonl").open("a", encoding="utf-8") as handle:
            for row in failures:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        raise RuntimeError(f"{len(failures)} answer calls failed; rerun the same command to resume")


if __name__ == "__main__":
    main()
