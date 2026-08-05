#!/usr/bin/env python3
"""Create two-pass semantic annotation drafts; final human acceptance is separate.

Raw dialogue is written only to the ignored run log. The Git-safe draft contains
source references, offsets, review decisions, and no copied dialogue text.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


SYSTEM = """You annotate sufficient evidence for LongMemEval retrieval evaluation.
Select the smallest set of turns that jointly supports the reference answer. Use
only supplied official gold sessions. For unanswerable (_abs) items, select turns
that establish the known operands or the missing-information boundary. Do not
select generic advice or a turn merely because it repeats the question.
Prefer the user's autobiographical statements. Assistant general knowledge is
not memory evidence unless the question explicitly asks what the assistant said.
Each quote must contain at most 24 words and notes at most 30 words.
JSON only: {\"evidence\":[{\"session_id\":str,\"turn_index\":int,
\"quote\":exact substring,\"support_role\":one of fact|temporal_endpoint|
aggregation_member|negative_scope,\"confidence\":high|medium|low}],
\"ambiguous\":bool,\"notes\":str}. Quotes must be exact non-empty substrings."""


def conversation_payload(case: dict[str, Any]) -> dict[str, Any]:
    gold = set(map(str, case.get("answer_session_ids") or []))
    sessions = []
    for sid, date, messages in zip(
        case["haystack_session_ids"], case["haystack_dates"], case["haystack_sessions"]
    ):
        if str(sid) in gold:
            sessions.append({
                "session_id": str(sid), "date": date,
                "turns": [{"turn_index": i, "role": m.get("role"), "text": m.get("content", "")}
                          for i, m in enumerate(messages)],
            })
    return {
        "question_id": str(case["question_id"]), "question": case["question"],
        "question_date": case.get("question_date"), "reference_answer": case.get("answer"),
        "gold_sessions": sessions,
    }


def ask(client: OpenAI, model: str, payload: dict[str, Any], review: dict[str, Any] | None) -> dict[str, Any]:
    suffix = "\nIndependently review the proposed evidence and return a corrected final set:\n" + json.dumps(
        review, ensure_ascii=False
    ) if review else ""
    response = client.chat.completions.create(
        model=model, temperature=0, max_tokens=3000,
        messages=[{"role": "system", "content": SYSTEM}, {
            "role": "user", "content": json.dumps(payload, ensure_ascii=False) + suffix,
        }],
        response_format={"type": "json_object"},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    message = response.choices[0].message
    if getattr(message, "reasoning_content", None):
        raise ValueError("thinking/reasoning content must be empty")
    result = json.loads(message.content or "{}")
    result["_usage"] = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }
    return result


def locate(payload: dict[str, Any], evidence: dict[str, Any]) -> tuple[int, int, bool]:
    sid, index, quote = str(evidence["session_id"]), int(evidence["turn_index"]), str(evidence["quote"])
    for session in payload["gold_sessions"]:
        if session["session_id"] != sid:
            continue
        for turn in session["turns"]:
            if turn["turn_index"] == index:
                start = turn["text"].find(quote)
                if start < 0:
                    start = turn["text"].casefold().find(quote.casefold())
                if start < 0:
                    # A generated quote may normalize Markdown or whitespace.
                    # Retain a valid full-turn reference and force low confidence
                    # so the final human pass must inspect it.
                    return 0, len(turn["text"]), True
                return start, start + len(quote), False
    raise ValueError(f"unknown source turn: {sid}/{index}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-log", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    cases = json.loads(args.data.read_text(encoding="utf-8"))
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require 0 <= shard-index < num-shards")
    cases = [case for index, case in enumerate(cases) if index % args.num_shards == args.shard_index]
    client = OpenAI(base_url=args.base_url, api_key="local-vllm", timeout=300)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_log.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, list[dict[str, Any]]] = {}
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line); existing.setdefault(row["question_id"], []).append(row)
    with args.audit_log.open("a", encoding="utf-8") as audit:
        for position, case in enumerate(cases, 1):
            qid = str(case["question_id"])
            if qid in existing:
                continue
            payload = conversation_payload(case)
            for attempt in range(3):
                try:
                    first = ask(client, args.model, payload, None)
                    second = ask(client, args.model, payload, first)
                    break
                except Exception as exc:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
            audit.write(json.dumps({"question_id": qid, "first": first, "second": second}, ensure_ascii=False) + "\n")
            first_refs = {(str(e["session_id"]), int(e["turn_index"])) for e in first.get("evidence", [])}
            second_refs = {(str(e["session_id"]), int(e["turn_index"])) for e in second.get("evidence", [])}
            disagreement = "none" if first_refs == second_refs else "evidence_set_changed"
            rows = []
            for evidence in second.get("evidence", []):
                start, end, quote_mismatch = locate(payload, evidence)
                rows.append({
                    "question_id": qid, "session_id": str(evidence["session_id"]),
                    "turn_index": int(evidence["turn_index"]), "span_start": start, "span_end": end,
                    "support_role": evidence["support_role"],
                    "confidence": "low" if quote_mismatch else evidence["confidence"],
                    "first_review": "accepted", "second_review": "changed" if disagreement != "none" else "accepted",
                    "adjudication": "changed" if disagreement != "none" else "accepted",
                    "first_reviewer": "qwen30-candidate-r1", "second_reviewer": "qwen30-independent-r2",
                    "disagreement": "quote_mismatch" if quote_mismatch else disagreement,
                    "annotation_version": "lme-v5-dev100-draft-r1",
                })
            if not rows:
                raise ValueError(f"no evidence selected for {qid}")
            with args.output.open("a", encoding="utf-8") as out:
                for row in rows:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{position}/{len(cases)}] {qid}: {len(rows)} evidence refs", flush=True)


if __name__ == "__main__":
    main()
