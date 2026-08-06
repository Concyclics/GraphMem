#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
sys.path.insert(0, str(ROOT / "src"))

from graphmem.judging import OpenAICompatibleClient  # noqa: E402


DEFAULT_REPO = Path(
    "/mnt/ssd1/yongan/Resources/RefRepos/general/memory-benchmarks"
)
PINNED_COMMIT = "4b61c5d31b9c668a12b4f5e78064248a02c82d2b"
PROMPT_SHA256 = "8ebac1ef60e9ab5caf99079fdaac038b85472e81491ed35e2d2655f3927c76c2"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        # JSON permits U+2028/U+2029 inside strings. str.splitlines() treats
        # those characters as record separators, while JSONL uses physical LF.
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_prompts(repo: Path) -> Any:
    path = repo / "benchmarks/locomo/prompts.py"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != PROMPT_SHA256:
        raise RuntimeError(
            f"memory-benchmarks LoCoMo prompt hash changed: {digest} != {PROMPT_SHA256}"
        )
    spec = importlib.util.spec_from_file_location("memory_benchmarks_locomo_prompts", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_json(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("judge response is not a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Judge existing LoCoMo answers with mem0ai/memory-benchmarks prompts."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--memory-benchmarks-repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--model", default=os.environ.get("SGAO_MODEL", "gpt-5.4-mini"))
    parser.add_argument(
        "--base-url", default=os.environ.get("SGAO_BASE_URL", "https://sub2api.sgao.me/v1/")
    )
    parser.add_argument("--api-key-env", default="SGAO_API_KEY")
    parser.add_argument(
        "--request-profile", choices=["deepseek", "openai", "omit"],
        default="openai",
    )
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    prompts = _load_prompts(args.memory_benchmarks_repo)
    cases = {
        row["question_id"]: row
        for row in json.loads(args.data.read_text(encoding="utf-8"))
    }
    answers = {row["question_id"]: row for row in _read_jsonl(args.answers)}
    selected = [
        (question_id, case, answers[question_id])
        for question_id, case in cases.items()
        if int(case["locomo_category"]) in prompts.CATEGORIES_TO_EVALUATE
        and question_id in answers
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_path = args.output_dir / "auto_eval.jsonl"
    calls_path = args.output_dir / "judge_calls.jsonl"
    if not args.resume:
        eval_path.write_text("", encoding="utf-8")
        calls_path.write_text("", encoding="utf-8")
    completed = {
        row["question_id"] for row in _read_jsonl(eval_path)
    } if eval_path.exists() else set()
    selected = [item for item in selected if item[0] not in completed]
    client = OpenAICompatibleClient(
        model=args.model, base_url=args.base_url,
        api_key_env=args.api_key_env, request_profile=args.request_profile,
    )

    def judge(item: tuple[str, dict[str, Any], dict[str, Any]]):
        question_id, case, answer_row = item
        category = int(case["locomo_category"])
        gold = prompts.preprocess_answer(category, str(case.get("answer") or ""))
        prompt = prompts.get_judge_prompt(
            category,
            str(case["question"]),
            gold,
            str(answer_row.get("prediction") or ""),
        )
        result = client.chat(
            question_id=question_id,
            variant="memory_benchmarks_locomo_judge",
            stage="judge_locomo",
            messages=[
                {"role": "system", "content": prompts.JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            thinking_mode="none",
            max_tokens=args.max_tokens,
            json_mode=True,
            temperature=0.0,
            seed=0,
        )
        result.record.excluded_from_budget = True
        payload = _parse_json(result.text)
        label = str(payload.get("label") or "").upper()
        if label not in {"CORRECT", "WRONG"}:
            raise ValueError(f"invalid LoCoMo judge label for {question_id}: {label!r}")
        if result.record.reasoning_tokens != 0:
            raise RuntimeError(f"judge reasoning_tokens must be 0 for {question_id}")
        return case, result, payload, label

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(judge, item) for item in selected]
        for future in as_completed(futures):
            case, result, payload, label = future.result()
            _append_jsonl(calls_path, asdict(result.record))
            _append_jsonl(
                eval_path,
                {
                    "question_id": case["question_id"],
                    "conversation_id": case["locomo_sample_id"],
                    "category": int(case["locomo_category"]),
                    "correct": label == "CORRECT",
                    "label": label,
                    "reasoning": str(payload.get("reasoning") or ""),
                    "judge_model": result.record.model,
                    "judge_prompt_commit": PINNED_COMMIT,
                    "judge_prompt_sha256": PROMPT_SHA256,
                    "with_evidence": False,
                },
            )
            print(f"judge {case['question_id']}: {label}", flush=True)

    evaluations = _read_jsonl(eval_path)
    calls = _read_jsonl(calls_path)
    by_category: dict[str, dict[str, Any]] = {}
    for row in evaluations:
        key = str(row["category"])
        bucket = by_category.setdefault(key, {"correct": 0, "total": 0})
        bucket["correct"] += int(row["correct"])
        bucket["total"] += 1
    for bucket in by_category.values():
        bucket["accuracy"] = bucket["correct"] / bucket["total"] if bucket["total"] else 0.0
    correct = sum(int(row["correct"]) for row in evaluations)
    stats = {
        "benchmark": "locomo",
        "judge_source": "mem0ai/memory-benchmarks",
        "judge_prompt_commit": PINNED_COMMIT,
        "judge_prompt_sha256": PROMPT_SHA256,
        "categories_evaluated": [1, 2, 3, 4],
        "category_5_excluded_by_repository": True,
        "with_evidence": False,
        "model": args.model,
        "thinking_request_profile": args.request_profile,
        "thinking": {"type": "disabled"} if args.request_profile == "deepseek" else None,
        "reasoning_effort": "none" if args.request_profile == "openai" else None,
        "reasoning_effort_field_sent": args.request_profile == "openai",
        "excluded_from_build_and_answer_budgets": True,
        "question_count": len(evaluations),
        "correct": correct,
        "accuracy": correct / len(evaluations) if evaluations else 0.0,
        "by_category": by_category,
        "prompt_cache_miss_tokens": sum(
            int(row.get("prompt_cache_miss_tokens") or 0) for row in calls
        ),
        "prompt_cache_hit_tokens": sum(
            int(row.get("prompt_cache_hit_tokens") or 0) for row in calls
        ),
        "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in calls),
        "reasoning_tokens": sum(int(row.get("reasoning_tokens") or 0) for row in calls),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in calls),
    }
    (args.output_dir / "judge_token_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"accuracy": stats["accuracy"], "questions": len(evaluations)}))


if __name__ == "__main__":
    main()
