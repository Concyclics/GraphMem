#!/usr/bin/env python3
"""Select candidate user turns independently inside every official gold session."""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI


SYSTEM = """You select source-turn candidates for LongMemEval annotation.
The supplied session is an official gold session. Select the smallest set of USER
turn indices in this session that materially supplies an operand, temporal endpoint,
fact, or negative-scope boundary for the reference answer. Never select generic
requests merely echoing the question when a factual user statement exists. Never
select assistant turns. An official gold session can occasionally provide only
background or be over-inclusive; in that case return an empty turns list instead
of forcing an unrelated turn. For unanswerable questions select a stated operand
or contrast only when it genuinely establishes what information is missing. Return JSON
only: {"turns":[{"turn_index":int,"support_role":"fact"|"temporal_endpoint"|
"aggregation_member"|"negative_scope","confidence":"high"|"medium"|"low"}],
"notes":str}."""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    args = parser.parse_args()
    cases = json.loads(args.data.read_text(encoding="utf-8"))
    jobs: list[tuple[dict[str, Any], str, list[dict[str, Any]]]] = []
    for case in cases:
        official = set(map(str, case["answer_session_ids"]))
        for sid, messages in zip(case["haystack_session_ids"], case["haystack_sessions"]):
            if str(sid) in official:
                jobs.append((case, str(sid), messages))
    completed: set[tuple[str, str]] = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line); completed.add((row["question_id"], row["session_id"]))
    local = threading.local()

    def run(job: tuple[dict[str, Any], str, list[dict[str, Any]]]) -> dict[str, Any]:
        case, sid, messages = job
        if not hasattr(local, "client"):
            local.client = OpenAI(base_url=args.base_url, api_key="local-vllm", timeout=300)
        user_turns = [
            {"turn_index": index, "text": message.get("content", "")}
            for index, message in enumerate(messages) if message.get("role") == "user"
        ]
        payload = {
            "question_id": str(case["question_id"]), "question": case["question"],
            "reference_answer": case.get("answer"), "session_id": sid,
            "session_date": next(
                date for current, date in zip(case["haystack_session_ids"], case["haystack_dates"])
                if str(current) == sid
            ),
            "user_turns": user_turns,
        }
        for attempt in range(3):
            try:
                response = local.client.chat.completions.create(
                    model=args.model, temperature=0, max_tokens=1200,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
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
        allowed = {turn["turn_index"]: turn["text"] for turn in user_turns}
        selected = result.get("turns") or []
        if any(int(row["turn_index"]) not in allowed for row in selected):
            raise ValueError(f"invalid turn selection: {case['question_id']}/{sid}: {result}")
        valid_roles = {"fact", "temporal_endpoint", "aggregation_member", "negative_scope"}
        valid_confidence = {"high", "medium", "low"}
        if any(row.get("support_role") not in valid_roles or
               row.get("confidence") not in valid_confidence for row in selected):
            raise ValueError(f"invalid evidence metadata: {case['question_id']}/{sid}: {result}")
        return {
            "question_id": str(case["question_id"]), "session_id": sid,
            "evidence": [{
                "turn_index": int(row["turn_index"]), "span_start": 0,
                "span_end": len(allowed[int(row["turn_index"])]),
                "support_role": row["support_role"], "confidence": row["confidence"],
            } for row in selected],
            "notes": result.get("notes", ""),
            "usage": response.usage.model_dump() if response.usage else {},
        }

    pending = [job for job in jobs if (str(job[0]["question_id"]), job[1]) not in completed]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run, job): job for job in pending}
        with args.output.open("a", encoding="utf-8") as handle:
            for position, future in enumerate(as_completed(futures), 1):
                row = future.result()
                handle.write(json.dumps(row, ensure_ascii=False) + "\n"); handle.flush()
                print(f"[{position}/{len(pending)}] {row['question_id']}/{row['session_id']}", flush=True)
    print(json.dumps({"gold_sessions": len(jobs), "new": len(pending)}))


if __name__ == "__main__":
    main()
