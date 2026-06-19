#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from openai import APIError, OpenAI, RateLimitError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel LongMemEval-compatible QA judge using official prompts."
    )
    parser.add_argument("metric_model", choices=["deepseek-v4-flash", "deepseek-v4-pro"])
    parser.add_argument("hyp_file", type=Path)
    parser.add_argument("ref_file", type=Path)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    eval_dir = Path(__file__).resolve().parents[1] / "LongMemEval" / "src" / "evaluation"
    sys.path.insert(0, str(eval_dir))
    from evaluate_qa import get_anscheck_prompt  # type: ignore

    hypotheses = load_json_or_jsonl(args.hyp_file)
    references = load_json_or_jsonl(args.ref_file)
    qid2ref = {entry["question_id"]: entry for entry in references}
    output_file = args.output_file or args.hyp_file.with_name(
        args.hyp_file.name + f".eval-results-{args.metric_model}.parallel"
    )

    client_state = threading.local()

    def get_client() -> OpenAI:
        client = getattr(client_state, "client", None)
        if client is None:
            client = OpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
                timeout=120,
            )
            client_state.client = client
        return client

    def judge_one(index: int, entry: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        ref = qid2ref[entry["question_id"]]
        prompt = get_anscheck_prompt(
            ref["question_type"],
            ref["question"],
            ref["answer"],
            entry["hypothesis"],
            abstention="_abs" in entry["question_id"],
        )
        last_error: Exception | None = None
        for attempt in range(6):
            try:
                completion = get_client().chat.completions.create(
                    model=args.metric_model,
                    messages=[{"role": "user", "content": prompt}],
                    n=1,
                    temperature=0,
                    max_tokens=512,
                )
                eval_response = (completion.choices[0].message.content or "").strip()
                result = dict(entry)
                result["autoeval_label"] = {
                    "model": args.metric_model,
                    "label": "yes" in eval_response.lower(),
                }
                result["autoeval_response"] = eval_response
                return index, result
            except (RateLimitError, APIError, Exception) as error:
                last_error = error
                time.sleep(min(30, 2**attempt))

        result = dict(entry)
        result["autoeval_label"] = {"model": args.metric_model, "label": False}
        result["autoeval_error"] = repr(last_error)
        return index, result

    results: list[tuple[int, dict[str, Any]]] = []
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(judge_one, index, entry): index
            for index, entry in enumerate(hypotheses)
            if entry["question_id"] in qid2ref
        }
        for done_count, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            results.append(future.result())
            if done_count % 50 == 0 or done_count == len(futures):
                correct = sum(
                    1 for _, row in results if row.get("autoeval_label", {}).get("label")
                )
                print(
                    f"progress {done_count}/{len(futures)} "
                    f"correct_so_far={correct} elapsed_sec={time.time() - started:.1f}",
                    flush=True,
                )

    results.sort(key=lambda pair: pair[0])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as out_f:
        for _, row in results:
            print(json.dumps(row, ensure_ascii=False), file=out_f)

    correct = sum(1 for _, row in results if row.get("autoeval_label", {}).get("label"))
    total = len(results)
    print("Accuracy:", round(correct / total, 4) if total else 0.0)
    print("Saved to", output_file)


if __name__ == "__main__":
    main()
