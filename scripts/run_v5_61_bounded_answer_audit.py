#!/usr/bin/env python3
"""Run a bounded, conservative audit over V5.61 relation witnesses.

The primary answer is produced exactly once by the frozen V5.54 readout.  This
stage is deliberately much smaller: it sees the question, a compact view of
the primary answer, and source-backed witness rows that were derived only from
turns already present in the evidence pack.  It may replace the primary answer
only when those rows form a complete operand set.  Otherwise it emits KEEP.

The script enforces the *actual API input + output* budget per routed question;
it is not enough for a locally estimated prompt to be below the limit.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import sys
import threading
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.judging.clients import OpenAICompatibleClient  # noqa: E402
from graphmem.tokenization import resolve_token_counter  # noqa: E402


SYSTEM = """You are a conservative answer auditor. Use only the source-backed
witness rows below; they may be incomplete. Output exactly KEEP unless the rows
directly bind every required item, operand, or endpoint and therefore prove the
baseline answer wrong. Only then output REPLACE: followed by a concise corrected
answer. Exclude near-matches, plans, negations, and assistant suggestions.
Absence is not zero. Deduplicate repeated mentions but retain distinct events.
Do exact arithmetic. Give no explanation."""

WORKSPACE_MARKER = "Source-backed relation workspace"
WORKSPACE_END = "Bind only the exact queried"
ROW_RE = re.compile(r"^R\d+\s", re.M)
DECISION_RE = re.compile(r"^\s*(KEEP|REPLACE\s*:\s*(.+))\s*$", re.I | re.S)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(
        json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8")


def index(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {str(row["question_id"]): row for row in rows}
    return result


def workspace_rows(prepared: dict[str, Any]) -> list[str]:
    messages = prepared.get("messages") or []
    user = str(messages[-1].get("content") or "") if messages else ""
    if WORKSPACE_MARKER not in user:
        return []
    section = user.split(WORKSPACE_MARKER, 1)[1]
    section = section.split(WORKSPACE_END, 1)[0]
    matches = list(ROW_RE.finditer(section))
    rows: list[str] = []
    for pos, match in enumerate(matches):
        end = matches[pos + 1].start() if pos + 1 < len(matches) else len(section)
        row = " ".join(section[match.start():end].split())
        if row:
            rows.append(row)
    return rows


def answer_synopsis(text: str, limit: int = 220) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", clean)
                 if part.strip()]
    numeric = [row for row in sentences if re.search(
        r"[$£€]?\d+(?:[,.]\d+)*(?:\s*(?:days?|weeks?|months?|years?|hours?))?",
        row, re.I)]
    chosen = numeric[-1] if numeric else sentences[-1] if sentences else clean
    prefix = sentences[0] if sentences else clean
    value = prefix if chosen == prefix else prefix + " … " + chosen
    if len(value) > limit:
        value = value[:limit].rsplit(" ", 1)[0] + "…"
    return value


def make_messages(
    *, question: str, operation: str, baseline: str, witnesses: list[str],
    counter: Any, max_input_tokens: int,
) -> list[dict[str, str]]:
    synopsis = answer_synopsis(baseline, limit=120)

    def render_head(value: str) -> str:
        return (
            f"Question: {question}\n"
            f"Operation: {operation or 'lookup'}\n"
            f"Baseline answer: {value}\n"
            "Source-backed witness rows:\n"
        )

    head = render_head(synopsis)
    if counter.count(SYSTEM) + counter.count(head) > max_input_tokens:
        words = synopsis.split()
        low, high, best = 0, len(words), ""
        while low <= high:
            middle = (low + high) // 2
            clipped = " ".join(words[:middle])
            candidate = render_head(clipped)
            if counter.count(SYSTEM) + counter.count(candidate) <= max_input_tokens:
                best = clipped; low = middle + 1
            else:
                high = middle - 1
        head = render_head(best)
    rows: list[str] = []
    for witness in witnesses:
        candidate = rows + [witness]
        user = head + "\n".join(candidate)
        if counter.count(SYSTEM) + counter.count(user) <= max_input_tokens:
            rows = candidate
            continue
        # Keep a clipped form of the next witness when there is still useful
        # room.  Word-boundary clipping makes the token gate deterministic.
        words = witness.split()
        low, high, best = 0, len(words), ""
        while low <= high:
            middle = (low + high) // 2
            clipped = " ".join(words[:middle])
            user = head + "\n".join(rows + ([clipped] if clipped else []))
            if counter.count(SYSTEM) + counter.count(user) <= max_input_tokens:
                best = clipped; low = middle + 1
            else:
                high = middle - 1
        if best and len(best.split()) >= 8:
            # Do not append an ellipsis after the binary-search gate: even one
            # extra punctuation token can invalidate an exact 400-token cap.
            rows.append(best)
        break
    user = head + ("\n".join(rows) if rows else "(none)")
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user}]
    if sum(counter.count(str(row["content"])) for row in messages) > max_input_tokens:
        raise RuntimeError("bounded audit prompt exceeds local input gate")
    return messages


def parse_decision(text: str) -> tuple[str, str]:
    match = DECISION_RE.match(text)
    if not match:
        return "invalid", ""
    if match.group(1).casefold() == "keep":
        return "keep", ""
    answer = " ".join((match.group(2) or "").split()).strip()
    if not answer or len(answer) > 500:
        return "invalid", ""
    return "replace", answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--baseline-answers", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="SGAO_API_KEY")
    parser.add_argument("--packing-model", required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--max-input-tokens", type=int, default=400)
    parser.add_argument("--max-output-tokens", type=int, default=64)
    parser.add_argument("--max-total-tokens", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=args.resume)
    calls_path = args.output_root / "audit_calls.jsonl"
    prepared_rows = read_jsonl(args.prepared)
    prepared = index(prepared_rows)
    baseline_rows = read_jsonl(args.baseline_answers)
    baseline = index(baseline_rows)
    if set(prepared) != set(baseline):
        raise ValueError("prepared and baseline question IDs differ")

    completed_rows = read_jsonl(calls_path) if args.resume else []
    completed = index(completed_rows)
    counter = resolve_token_counter(args.packing_model, require_exact=True)
    local = threading.local()

    def client() -> OpenAICompatibleClient:
        if not hasattr(local, "client"):
            local.client = OpenAICompatibleClient(
                model=args.model, base_url=args.base_url,
                api_key_env=args.api_key_env, request_profile="qwen",
                max_retries=12, timeout_sec=180.0)
        return local.client

    routed: list[tuple[str, dict[str, Any], list[str]]] = []
    for question_id, row in prepared.items():
        trace = row.get("trace") or {}
        witnesses = workspace_rows(row)
        if trace.get("relation_workspace") and witnesses:
            routed.append((question_id, row, witnesses))

    lock = threading.Lock()

    def run(item: tuple[str, dict[str, Any], list[str]]) -> dict[str, Any]:
        question_id, row, witnesses = item
        operation = str((row.get("trace") or {}).get(
            "relation_workspace_operation") or "")
        messages = make_messages(
            question=str(baseline[question_id].get("question") or ""),
            operation=operation,
            baseline=str(baseline[question_id].get("prediction") or ""),
            witnesses=witnesses, counter=counter,
            max_input_tokens=args.max_input_tokens)
        prompt_hash = hashlib.sha256(json.dumps(
            messages, sort_keys=True, ensure_ascii=True).encode()).hexdigest()
        result = client().chat(
            question_id=question_id, variant="v5_61_bounded_answer_audit",
            stage="answer", messages=messages, thinking_mode="disabled",
            max_tokens=args.max_output_tokens, temperature=0.0, seed=0)
        decision, replacement = parse_decision(result.text)
        record = asdict(result.record)
        actual_total = int(record.get("total_tokens") or 0)
        if actual_total > args.max_total_tokens:
            decision, replacement = "budget_reject", ""
        return {
            "question_id": question_id,
            "operation": operation,
            "decision": decision,
            "replacement": replacement,
            "raw_response": result.text,
            "prompt_hash": prompt_hash,
            "local_input_tokens": sum(
                counter.count(str(message["content"])) for message in messages),
            "witness_rows_rendered": sum(
                1 for line in messages[-1]["content"].splitlines()
                if ROW_RE.match(line)),
            **record,
        }

    pending = [item for item in routed if item[0] not in completed]
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(run, item): item[0] for item in pending}
            for future in as_completed(futures):
                result = future.result()
                with lock:
                    completed[result["question_id"]] = result
                    write_jsonl(calls_path, (
                        completed[qid] for qid, _row, _witnesses in routed
                        if qid in completed))
                    print(
                        f"audited {len(completed)}/{len(routed)} "
                        f"{result['question_id']}={result['decision']}",
                        flush=True)

    output_rows: list[dict[str, Any]] = []
    replacements = 0
    for row in baseline_rows:
        question_id = str(row["question_id"])
        output = dict(row)
        audit = completed.get(question_id)
        if audit and audit["decision"] == "replace":
            output["prediction"] = audit["replacement"]
            output["bounded_audit_replaced"] = True
            replacements += 1
        else:
            output["bounded_audit_replaced"] = False
        output_rows.append(output)
    write_jsonl(args.output_root / "answers.jsonl", output_rows)
    for benchmark in ("longmemeval", "locomo"):
        write_jsonl(
            args.output_root / f"answers_{benchmark}.jsonl",
            (row for row in output_rows if row.get("benchmark") == benchmark))

    calls = [completed[qid] for qid, _row, _witnesses in routed]
    totals = [int(row.get("total_tokens") or 0) for row in calls]
    manifest = {
        "schema_version": "graphmem-v5.61-bounded-answer-audit-v1",
        "questions": len(output_rows),
        "routed": len(routed),
        "completed": len(calls),
        "replacements": replacements,
        "decisions": {
            key: sum(row["decision"] == key for row in calls)
            for key in ("keep", "replace", "invalid", "budget_reject")},
        "max_input_tokens": args.max_input_tokens,
        "max_output_tokens": args.max_output_tokens,
        "max_total_tokens": args.max_total_tokens,
        "actual_total_tokens": {
            "sum": sum(totals),
            "mean_routed": sum(totals) / max(1, len(totals)),
            "mean_all_questions": sum(totals) / max(1, len(output_rows)),
            "max": max(totals, default=0),
        },
        "uses_gold_or_judge": False,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
