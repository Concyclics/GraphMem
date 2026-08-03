#!/usr/bin/env python3
"""Answer persisted V2 evidence ledgers with the current GPT backbone."""
from __future__ import annotations

import argparse
import json
import os
import re
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
    parser.add_argument("--planner", action="store_true")
    parser.add_argument("--planner-max-tokens", type=int, default=192)
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



_QUERY_STOP = {
    "the", "a", "an", "i", "me", "my", "of", "to", "in", "on",
    "at", "for", "and", "or", "did", "do", "does", "was", "were",
    "is", "are", "what", "which", "who", "when", "how", "many",
    "much", "have", "has", "had", "been", "from", "with",
}


def _term(token: str) -> str:
    token = token.casefold().strip("-_")
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            token = token[:-len(suffix)]
            break
    return token


def _focused_source_rows(
    context: str, question: str, limit: int = 10,
) -> list[dict[str, str]]:
    source_rows = re.findall(
        r"\[SOURCE ([^ |\]]+)[^\]]*\]\n(.*?)(?=\n\n\[|\Z)",
        context, re.S,
    )
    query_terms = {
        _term(token) for token in re.findall(r"[A-Za-z0-9_-]+", question)
        if token.casefold() not in _QUERY_STOP and len(token) >= 3
    }
    document_terms = [
        {_term(token) for token in re.findall(r"[A-Za-z0-9_-]+", text)}
        for _source_id, text in source_rows
    ]
    frequencies = {
        term: sum(term in terms for terms in document_terms)
        for term in query_terms
    }
    ranked = []
    for (source_id, text), terms in zip(source_rows, document_terms):
        overlap = query_terms & terms
        if not overlap:
            continue
        score = sum(8.0 / (1 + frequencies[term]) for term in overlap)
        score += 2.0 * len(overlap)
        ranked.append((score, source_id, text.strip()))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return [
        {"source_id": source_id, "text": text[:320]}
        for _score, source_id, text in ranked[:limit]
    ]


def _planner_messages(case, focused: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{
        "role": "system",
        "content": (
            "You are a lightweight evidence planner. Thinking is disabled. "
            "Return compact JSON only. Bind the exact requested owner, entity, "
            "relation, lifecycle and time scope. For totals list every distinct "
            "operand; for temporal questions bind both endpoints and resolve "
            "relative dates from each cited source date; never transfer a value "
            "from a sibling entity. A proposed answer is advisory, not evidence."
        ),
    }, {
        "role": "user",
        "content": json.dumps({
            "question_date": case.question_date,
            "question": case.question,
            "candidate_sources": focused,
            "output_schema": {
                "answer_algebra": "string",
                "selected_source_ids": ["source id"],
                "operands_or_endpoints": ["short source-bound item"],
                "candidate_answer": "short answer or null",
                "missing_evidence": ["role"],
            },
        }, ensure_ascii=False),
    }]


def _validated_planner_payload(text: str, offered_ids: set[str]) -> dict:
    try:
        payload = json.loads(text)
    except Exception:
        return {"valid": False, "error": "invalid_json"}
    if not isinstance(payload, dict):
        return {"valid": False, "error": "not_object"}
    payload["selected_source_ids"] = [
        source_id for source_id in payload.get("selected_source_ids", [])
        if source_id in offered_ids
    ][:10]
    missing = [
        str(value).strip() for value in payload.get("missing_evidence", [])
        if str(value).strip()
    ][:8]
    payload["missing_evidence"] = missing
    if missing:
        # A proposed scalar cannot coexist with a declared missing operand or
        # binding. Keep the diagnosis and force the final LLM to abstain from
        # transferring a sibling value.
        payload["candidate_answer"] = None
        payload["evidence_gate"] = "insufficient"
    else:
        payload["evidence_gate"] = "complete"
    payload["valid"] = True
    return payload


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
        context_text = str(retrieval["context_text"])
        call_records = []
        planner_payload = None
        focused = []
        run_variant = VARIANT + ("_planner" if args.planner else "")
        if args.planner:
            focused = _focused_source_rows(
                context_text, case.question, limit=12,
            )
            planner_call = client.chat(
                question_id=case.question_id,
                variant=run_variant,
                stage="answer_query_planner",
                messages=_planner_messages(case, focused),
                thinking_mode="none", max_tokens=args.planner_max_tokens,
                json_mode=True, temperature=0.0, seed=0,
            )
            planner_payload = _validated_planner_payload(
                planner_call.text, {row["source_id"] for row in focused},
            )
            call_records.append(asdict(planner_call.record))
            context_text = (
                "[LIGHTWEIGHT PLANNER EVIDENCE GATE]\n"
                "For factual, temporal, count, list, and aggregate questions, "
                "evidence_gate=insufficient is mandatory: answer that the exact "
                "requested binding is unavailable and do not output a sibling "
                "entity value, candidate arithmetic result, or partial subtotal. "
                "Recommendations remain answerable from supported preferences.\n"
                + json.dumps(planner_payload, ensure_ascii=False)
                + "\n\n[PLANNER FOCUSED LOSSLESS SOURCES]\n"
                + json.dumps(focused, ensure_ascii=False)
                + "\n\n" + context_text
            )
        view = SimpleNamespace(
            context_text=context_text,
            query_kind=str(retrieval.get("query_kind") or ""),
        )
        messages = answer_messages(case, view)
        result = client.chat(
            question_id=case.question_id,
            variant=run_variant,
            stage="answer_qa",
            messages=messages,
            thinking_mode="none",
            max_tokens=args.max_tokens,
        )
        call_records.append(asdict(result.record))
        combined_tokens = sum(row["total_tokens"] for row in call_records)
        if combined_tokens > args.answer_budget_tokens:
            raise RuntimeError(
                f"{case.question_id}: {combined_tokens}>"
                f"{args.answer_budget_tokens}"
            )
        if any(row.get("reasoning_tokens") for row in call_records):
            raise RuntimeError(f"{case.question_id}: reasoning tokens returned")
        return {
            "question_id": case.question_id,
            "variant": run_variant,
            "question": case.question,
            "question_type": case.question_type,
            "question_date": case.question_date,
            "gold_answer": case.answer,
            "prediction": result.text.strip(),
            "answer_session_ids": case.answer_session_ids,
            "answer_mode": (
                "gpt_lightweight_planner_from_persisted_v2_evidence_ledger"
                if args.planner else "gpt_from_persisted_v2_evidence_ledger"
            ),
            "planner_applied": bool(args.planner),
            "planner_payload": planner_payload,
            "query_total_tokens": combined_tokens,
        }, call_records

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(answer, case): case for case in cases}
        done = len(existing)
        for future in as_completed(futures):
            answer_row, call_rows = future.result()
            append(answers_path, answer_row)
            for call_row in call_rows:
                append(calls_path, call_row)
            done += 1
            print(
                f"[{done}/{done + len(futures) - (done - len(existing))}] "
                f"{answer_row['question_id']} {answer_row['prediction'][:100]}",
                flush=True,
            )


if __name__ == "__main__":
    main()
