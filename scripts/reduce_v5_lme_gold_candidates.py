#!/usr/bin/env python3
"""Suggest a minimal question-level subset from per-session gold-turn candidates.

This is candidate generation only. The output deliberately remains outside the
online package and must be accepted or replaced by semantic review before use.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI


SYSTEM = """You reduce LongMemEval evidence candidates to the smallest jointly
sufficient set of source turns for the reference answer. Select only candidate_id
values supplied below. Remove repeated mentions, generic requests, future plans,
    and facts unrelated to the exact operands or temporal endpoints. A gold session
may be over-inclusive and contribute no selected turn. For an unanswerable item,
retain only stated operands or contrasts that establish the missing-information
boundary. Return JSON only: {"candidate_ids":[int],"ambiguous":bool,"notes":str}.
An empty set is allowed only when none of the supplied candidates establishes a
real operand or missing-information boundary; it then requires manual resolution."""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    args = parser.parse_args()
    cases = json.loads(args.packet.read_text(encoding="utf-8"))
    existing: dict[str, dict[str, Any]] = {}
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[str(row["question_id"])] = row
    local = threading.local()

    def reduce(case: dict[str, Any]) -> dict[str, Any]:
        if not hasattr(local, "client"):
            local.client = OpenAI(base_url=args.base_url, api_key="local-vllm", timeout=300)
        candidates = [
            {
                "candidate_id": index,
                "session_id": row["session_id"],
                "turn_index": row["turn_index"],
                "support_role": row["support_role"],
                "text": row["review_excerpt"],
            }
            for index, row in enumerate(case["evidence"])
        ]
        payload = {
            "question": case["question"],
            "reference_answer": case["reference_answer"],
            "official_gold_sessions": case["official_gold_sessions"],
            "candidates": candidates,
        }
        for attempt in range(3):
            try:
                response = local.client.chat.completions.create(
                    model=args.model,
                    temperature=0,
                    max_tokens=1000,
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    response_format={"type": "json_object"},
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                message = response.choices[0].message
                if getattr(message, "reasoning_content", None):
                    raise ValueError("reasoning content must be empty")
                result = json.loads(message.content or "{}")
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        selected = [int(value) for value in result.get("candidate_ids") or []]
        if len(set(selected)) != len(selected) or any(
            value < 0 or value >= len(candidates) for value in selected
        ):
            raise ValueError(f"invalid candidate selection: {case['question_id']}: {result}")
        return {
            "question_id": str(case["question_id"]),
            "evidence": [
                {key: value for key, value in case["evidence"][index].items()
                 if key != "review_excerpt"}
                for index in selected
            ],
            "ambiguous": bool(result.get("ambiguous")),
            "notes": str(result.get("notes") or ""),
            "usage": response.usage.model_dump() if response.usage else {},
        }

    pending = [case for case in cases if str(case["question_id"]) not in existing]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(reduce, case): case for case in pending}
        with args.output.open("a", encoding="utf-8") as handle:
            for position, future in enumerate(as_completed(futures), 1):
                row = future.result()
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"[{position}/{len(pending)}] {row['question_id']}", flush=True)
    print(json.dumps({"questions": len(cases), "new": len(pending)}))


if __name__ == "__main__":
    main()
